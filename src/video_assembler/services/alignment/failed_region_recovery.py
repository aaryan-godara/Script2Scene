"""Generic INTERNAL failed-region local recovery for transcription gaps.

Complements the existing EOF tail recovery: when alignment finds FAILED scenes
in the middle of the narration, they are grouped into contiguous regions and
each region is re-transcribed from a wider, contextual audio window (anchored
on the nearest reliable HIGH scenes). Recovered words are shifted to global
timestamps, validated, merged into the transcript with the shared dedup logic,
and the whole transcript is re-aligned by the unchanged AlignmentService.

Design rules (from the production spec):
    * do NOT recover individual scenes blindly -> group contiguous FAILED scenes
      into ONE region and make ONE contextual transcription per region;
    * do NOT trust FAILED scene timestamps -> window is anchored on nearest HIGH
      scenes (or 0 / audio_duration at the boundaries);
    * recovery passes are bounded (default 2) and windows widen on later passes;
    * acoustic energy is recorded as a diagnostic guard, never as the sole
      reason to skip recovery;
    * recovered words are validated (finite, ordered, within audio) and merged
      with the same dedup used by chunked transcription;
    * re-alignment must use the UNCHANGED AlignmentService; automatic recovery
      never manually promotes a scene to HIGH.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .chunked_transcription_provider import ChunkedTranscriptionProvider
from .merge_utils import merge_chunk, normalize_window_words
from .persistent_transcription_cache import TranscriptionIdentityConfig
from .provider_base import TranscribedWord, TranscriptionResult


def rebuild_transcription(base: TranscriptionResult,
                          words: List[TranscribedWord]) -> TranscriptionResult:
    """Returns a copy of ``base`` with new words/segments (metadata preserved)."""
    segments = ChunkedTranscriptionProvider._build_segments(words)
    return base.model_copy(update={
        "words": words,
        "segments": segments,
        "processing_seconds": base.processing_seconds,
    })


class FailedRegionRecoveryEngine:
    """Transcribes missing internal regions and merges them back in."""

    def __init__(self, transcribe_fn, audio_service, audio_path: Path,
                 chunk_dir: Path, config: TranscriptionIdentityConfig):
        """``transcribe_fn(audio_path) -> TranscriptionResult`` for one clip.

        Passing a callable (rather than a provider object) lets the pipeline
        share its single loaded Whisper model while also supporting injected
        fake providers in tests.
        """
        self.transcribe_fn = transcribe_fn
        self.audio_service = audio_service
        self.audio_path = Path(audio_path)
        self.chunk_dir = Path(chunk_dir)
        self.cfg = config

    # ------------------------------------------------------------ grouping
    def group_failed_regions(self, scenes, statuses: Dict[int, str]) -> List[List[int]]:
        """Groups contiguous FAILED scenes into one region per run.

        Grouping is based on scene ORDER (not arithmetic on scene ids), so a
        project whose scene ids are not strictly sequential still groups
        correctly. Conservative: a FAILED scene separated from the previous
        group by a non-adjacent position starts a new group.
        """
        positions = {s.scene_id: i for i, s in enumerate(scenes)}
        failed = sorted((s.scene_id for s in scenes
                         if statuses.get(s.scene_id) == "FAILED"),
                        key=lambda sid: positions[sid])
        groups: List[List[int]] = []
        for sid in failed:
            if groups and positions[sid] == positions[groups[-1][-1]] + 1:
                groups[-1].append(sid)
            else:
                groups.append([sid])
        return groups

    # ------------------------------------------------------------ windows
    def estimate_window(self, group: List[int], scenes, statuses: Dict[int, str],
                        diagnostics, audio_duration: float, pass_no: int
                        ) -> Tuple[float, float, Dict]:
        """Estimates the contextual recovery window for one FAILED group.

        Anchors on the nearest HIGH scene before and after the group. FAILED
        timestamps are never trusted. Window length is clamped to
        [min_window_seconds, max_window_seconds] and widened for later passes.
        """
        positions = {s.scene_id: i for i, s in enumerate(scenes)}
        order = [s.scene_id for s in scenes]
        group_positions = sorted(positions[sid] for sid in group)
        g_start_pos, g_end_pos = group_positions[0], group_positions[-1]

        prev_high = next_high = None
        for i in range(g_start_pos - 1, -1, -1):
            if statuses.get(order[i]) == "HIGH":
                prev_high = order[i]
                break
        for i in range(g_end_pos + 1, len(order)):
            if statuses.get(order[i]) == "HIGH":
                next_high = order[i]
                break

        def _anchor_time(sid: int, key: str) -> float:
            d = diagnostics.diagnostics.get(sid, {})
            val = d.get(key)
            if val is None:
                val = getattr(scenes[positions[sid]], key, None)
            return float(val) if val is not None else 0.0

        gap_start = (_anchor_time(prev_high, "speech_end")
                     - self.cfg.context_before_seconds) if prev_high else 0.0
        gap_end = (_anchor_time(next_high, "speech_start")
                   + self.cfg.context_after_seconds) if next_high else audio_duration
        gap_start = max(0.0, gap_start)
        gap_end = min(audio_duration, gap_end)

        span = gap_end - gap_start
        target = (self.cfg.preferred_window_seconds if pass_no <= 1
                  else self.cfg.max_window_seconds)
        length = min(max(span, self.cfg.min_window_seconds), self.cfg.max_window_seconds)
        length = max(length, target) if span >= self.cfg.min_window_seconds else length
        if span <= 0:
            # no anchors: fall back to full audio
            window_start, window_end = 0.0, audio_duration
        else:
            center = (gap_start + gap_end) / 2.0
            window_start = max(0.0, center - length / 2.0)
            window_end = min(audio_duration, center + length / 2.0)
            if window_end - window_start < self.cfg.min_window_seconds - 1e-6:
                window_start, window_end = 0.0, audio_duration

        return (round(window_start, 3), round(window_end, 3),
                {"prev_high": prev_high, "next_high": next_high,
                 "gap_start": round(gap_start, 3), "gap_end": round(gap_end, 3)})

    # ------------------------------------------------------------ recovery
    def recover_pass(self, words: List[TranscribedWord], scenes, statuses,
                     diagnostics, audio_duration: float, pass_no: int
                     ) -> Tuple[List[TranscribedWord], List[Dict]]:
        """One recovery pass over all FAILED groups.

        Returns (merged words, audit log entries). Raises no exception on
        transcription failure: a failed group simply contributes nothing.
        """
        groups = self.group_failed_regions(scenes, statuses)
        if not groups:
            return words, []
        merged = list(words)
        audit: List[Dict] = []
        for group in groups:
            entry, merged = self._recover_group(group, merged, scenes, statuses,
                                                diagnostics, audio_duration, pass_no)
            audit.append(entry)
            merged.sort(key=lambda w: (w.start, w.end))
        return merged, audit

    def _recover_group(self, group: List[int], merged: List[TranscribedWord],
                       scenes, statuses, diagnostics, audio_duration: float,
                       pass_no: int) -> Tuple[Dict, List[TranscribedWord]]:
        window_start, window_end, anchors = self.estimate_window(
            group, scenes, statuses, diagnostics, audio_duration, pass_no)
        reason = "TRANSCRIPTION_GAP"
        entry: Dict = {
            "scene_group": group,
            "reason": reason,
            "window_start": window_start,
            "window_end": window_end,
            "pass": pass_no,
            "anchors": anchors,
            "acoustic_energy_detected": None,
            "words_before": len(merged),
            "recovery_raw_words": 0,
            "duplicates_removed": 0,
            "words_added": 0,
            "before_status": {str(sid): statuses.get(sid) for sid in group},
        }

        # acoustic diagnostic guard (never the sole reason to skip recovery)
        if self.cfg.acoustic_check_enabled:
            try:
                has_energy = self.audio_service.tail_has_acoustic_energy(
                    self.audio_path, window_start, window_end)
                entry["acoustic_energy_detected"] = bool(has_energy)
            except Exception:  # noqa: BLE001 - diagnostic only
                entry["acoustic_energy_detected"] = None

        clip_path = self.chunk_dir / (
            f"failed_region_p{pass_no}_g{group[0]}_"
            f"{window_start:07.3f}_{window_end:07.3f}.wav")
        try:
            self.audio_service.extract_chunk(self.audio_path, clip_path,
                                             window_start, window_end)
        except Exception:  # noqa: BLE001 - never crash recovery
            return entry, merged

        try:
            result = self.transcribe_fn(str(clip_path))
        except Exception:  # noqa: BLE001 - transcription failure = no recovery
            return entry, merged

        rec_words = normalize_window_words(result.words, window_start, audio_duration)
        entry["recovery_raw_words"] = len(rec_words)
        if not rec_words:
            return entry, merged

        # prev boundary: last merged word before the window start
        prev_before = [w for w in merged if w.end <= window_start + 1e-6]
        prev_start = prev_before[0].start if prev_before else window_start
        prev_end = prev_before[-1].end if prev_before else window_start

        new_merged, dups = merge_chunk(merged, rec_words, window_start,
                                       prev_start, prev_end)
        entry["duplicates_removed"] = dups
        entry["words_added"] = len(new_merged) - len(merged)
        return entry, new_merged