"""Chunked long-form transcription.

Wraps any :class:`TranscriptionProvider` (normally StableWhisperProvider) so
that long narrations are transcribed in overlapping chunks instead of a single
pass. Short audios keep the existing single-pass path.

Pipeline:

    probe duration
    if duration <= long_audio_threshold_seconds  -> single-pass inner.transcribe
    else:
        slice narration into overlapping 16kHz mono PCM chunks (FFmpeg)
        transcribe each chunk (local timestamps)
        shift to global timestamps
        reconcile overlapping regions (deterministic dedup)
        return one merged TranscriptionResult with GLOBAL timestamps

The downstream consumer (AlignmentService) never learns whether the source was
one chunk or many: it only sees a single global word timeline.
"""

from __future__ import annotations

import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel

from .provider_base import (TranscribedSegment, TranscribedWord,
                            TranscriptionProvider, TranscriptionResult)
from .text_normalizer import TextNormalizer


class ChunkingConfig(BaseModel):
    chunking_enabled: bool = True
    chunk_duration_seconds: float = 180.0
    overlap_seconds: float = 10.0
    long_audio_threshold_seconds: float = 300.0
    dedup_time_tolerance: float = 1.5
    keep_chunks: bool = True
    sample_rate: int = 16000
    channels: int = 1
    # EOF tail recovery: when the merged transcript ends significantly before the
    # audio EOF but the remaining tail contains speech-like acoustic energy, the
    # tail is re-transcribed from a wider EOF context window and merged back in.
    tail_recovery_enabled: bool = True
    tail_gap_trigger_seconds: float = 2.0
    tail_recovery_context_seconds: float = 90.0
    tail_energy_threshold_db: float = -40.0
    tail_energy_window_ms: int = 10

    def tail_recovery_identity(self) -> Dict[str, object]:
        """The tail-recovery configuration that forms part of the cache identity."""
        return {
            "tail_recovery_enabled": self.tail_recovery_enabled,
            "tail_gap_trigger_seconds": self.tail_gap_trigger_seconds,
            "tail_recovery_context_seconds": self.tail_recovery_context_seconds,
        }


def audio_sha256(path: str | Path) -> str:
    """Deterministic SHA-256 of the narration file for cache identity."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def shift_word(word: TranscribedWord, offset: float) -> TranscribedWord:
    return TranscribedWord(word=word.word, start=round(word.start + offset, 6),
                           end=round(word.end + offset, 6), confidence=word.confidence)


class ChunkedTranscriptionProvider(TranscriptionProvider):
    """Transcribes long narration in overlapping chunks, returns global words."""

    def __init__(self, inner_provider: TranscriptionProvider,
                 config: Optional[ChunkingConfig] = None,
                 audio_service=None,
                 normalizer: Optional[TextNormalizer] = None):
        self.inner = inner_provider
        self.config = config or ChunkingConfig()
        self.audio_service = audio_service
        self.normalizer = normalizer or TextNormalizer()
        self.last_diagnostics: Dict = {}

    # ------------------------------------------------------------- API
    def transcribe(self, audio_path: str, chunk_dir: Optional[str] = None) -> TranscriptionResult:
        start = time.time()
        audio_path = Path(audio_path)

        audio_service = self.audio_service
        if audio_service is None:
            from video_assembler.services.audio_service import AudioService
            audio_service = AudioService(audio_path.parent)
            self.audio_service = audio_service

        metadata = audio_service.get_audio_metadata(audio_path)
        duration = float(metadata["duration"])

        if not self.config.chunking_enabled or duration <= self.config.long_audio_threshold_seconds:
            return self._single_pass(audio_path, duration, start)

        if chunk_dir is None:
            chunk_dir = str(audio_path.parent / "transcription_chunks")

        chunks = audio_service.extract_chunks(
            audio_path, Path(chunk_dir),
            self.config.chunk_duration_seconds, self.config.overlap_seconds)

        chunk_results: List[Tuple[float, float, TranscriptionResult]] = []
        for chunk_path, g_start, g_end in chunks:
            result = self.inner.transcribe(str(chunk_path))
            chunk_results.append((g_start, g_end, result))

        merged, boundaries, per_chunk, dups = self._merge(chunk_results)
        merged, tail_diag = self._maybe_recover_tail(
            merged, audio_path, duration, Path(chunk_dir), boundaries)

        processing = round(time.time() - start, 3)
        sha = audio_sha256(audio_path)
        result = TranscriptionResult(
            provider=getattr(self.inner, "provider", "chunked"),
            model=getattr(self.inner, "model_name",
                          getattr(self.inner, "model", "unknown")),
            device=getattr(self.inner, "device", "unknown"),
            language=self._majority_language(chunk_results),
            audio_duration=duration,
            processing_seconds=processing,
            words=merged,
            segments=self._build_segments(merged),
            audio_sha256=sha,
            chunking_enabled=True,
            chunk_duration=self.config.chunk_duration_seconds,
            overlap=self.config.overlap_seconds,
            created_at=datetime.now(timezone.utc).isoformat(),
            chunk_count=len(chunks),
            chunk_boundaries=[[b[0], b[1]] for b in boundaries],
            words_per_chunk=per_chunk,
            duplicates_removed=dups,
            tail_recovery_enabled=self.config.tail_recovery_enabled,
            tail_gap_trigger_seconds=self.config.tail_gap_trigger_seconds,
            tail_recovery_context_seconds=self.config.tail_recovery_context_seconds,
            tail_recovery=tail_diag,
        )
        self.last_diagnostics = {
            "chunking_enabled": True,
            "chunk_count": len(chunks),
            "boundaries": [[b[0], b[1]] for b in boundaries],
            "words_per_chunk": per_chunk,
            "duplicates_removed": dups,
            "merged_word_count": len(merged),
            "processing_seconds": processing,
            "tail_recovery": tail_diag,
        }
        return result

    # ------------------------------------------------------------- single pass
    def _single_pass(self, audio_path: Path, duration: float, start: float) -> TranscriptionResult:
        result = self.inner.transcribe(str(audio_path))
        sha = audio_sha256(audio_path)
        result.audio_duration = duration
        result.audio_sha256 = sha
        result.chunking_enabled = False
        result.chunk_duration = self.config.chunk_duration_seconds
        result.overlap = self.config.overlap_seconds
        result.created_at = datetime.now(timezone.utc).isoformat()
        result.processing_seconds = round(time.time() - start, 3)
        result.chunk_count = 1
        # Stamp tail-recovery identity for cache consistency even though single
        # pass never runs tail recovery (tail recovery is chunked-path only).
        result.tail_recovery_enabled = self.config.tail_recovery_enabled
        result.tail_gap_trigger_seconds = self.config.tail_gap_trigger_seconds
        result.tail_recovery_context_seconds = self.config.tail_recovery_context_seconds
        self.last_diagnostics = {
            "chunking_enabled": False,
            "chunk_count": 1,
            "duplicates_removed": 0,
            "merged_word_count": len(result.words),
            "processing_seconds": result.processing_seconds,
        }
        return result

    # ------------------------------------------------------------- merge
    def _merge(self, chunk_results: List[Tuple[float, float, TranscriptionResult]]
               ) -> Tuple[List[TranscribedWord], List[Tuple[float, float]], List[int], int]:
        merged: List[TranscribedWord] = []
        boundaries: List[Tuple[float, float]] = []
        per_chunk: List[int] = []
        duplicates_removed = 0

        prev_start: Optional[float] = None
        prev_end: Optional[float] = None

        for g_start, g_end, result in chunk_results:
            words = [shift_word(w, g_start) for w in result.words]
            boundaries.append((g_start, g_end))
            per_chunk.append(len(words))
            if not words:
                continue

            if merged and prev_start is not None and prev_end is not None:
                merged, dups = self._merge_chunk(
                    merged, words, g_start, prev_start, prev_end)
                duplicates_removed += dups
            else:
                merged = words

            prev_start = g_start
            prev_end = g_end

        # final sort guard: merged must stay globally ordered
        merged.sort(key=lambda w: (w.start, w.end))
        return merged, boundaries, per_chunk, duplicates_removed

    def _merge_chunk(self, merged: List[TranscribedWord], words: List[TranscribedWord],
                     g_start: float, prev_start: float, prev_end: float
                     ) -> Tuple[List[TranscribedWord], int]:
        """Reconciles one new chunk's words against the merged transcript.

        The current chunk re-transcribes the same spoken content that the
        previous chunk already captured in the overlap window. Words are matched
        sequentially by normalized text against the tail of the merged
        transcript; a matched pair is one spoken word seen by both chunks and is
        kept only once (preferring the transcription whose word sits farther
        from a chunk boundary). No merged word is dropped unless a strictly
        better duplicate replaces it, so legitimate repeated words survive.
        """
        overlap_end = prev_end

        # only the tail of the merged transcript can be re-captured by this chunk
        prev_candidates = merged[-len(words):]

        matches, _ = self._match_sequences(prev_candidates, words)

        dup_cur_ids = set()
        for pi, cj in matches:
            dup_cur_ids.add(cj)
            pw = prev_candidates[pi]
            cw = words[cj]
            pw_dist = min(pw.start - prev_start, prev_end - pw.end)
            cw_dist = min(cw.start - g_start, overlap_end - cw.end)
            if cw_dist > pw_dist:
                for idx, w in enumerate(merged):
                    if w is pw:
                        merged[idx] = cw
                        break

        # keep non-duplicate words from this chunk
        to_add = [w for j, w in enumerate(words) if j not in dup_cur_ids]
        return merged + to_add, len(matches)

    # -------------------------------------------------------- EOF tail recovery
    def _tail_energy_detected(self, audio_path: Path, tail_start: float,
                              duration: float) -> bool:
        """Whether the audio tail [tail_start, duration) has acoustic content.

        Prefers an injected audio_service capability; otherwise falls back to the
        standard ffmpeg/numpy energy check used across the codebase.
        """
        svc = self.audio_service
        check = getattr(svc, "tail_has_acoustic_energy", None) if svc is not None else None
        if check is None:
            from video_assembler.services.audio_service import tail_has_acoustic_energy
            check = tail_has_acoustic_energy
        try:
            return bool(check(audio_path, tail_start, duration))
        except Exception:  # noqa: BLE001 - fail open: never crash transcription
            # Acoustic analysis is only a trigger guard. If it cannot be
            # performed the tail is left untouched rather than recovered.
            return False

    def _maybe_recover_tail(self, merged: List[TranscribedWord], audio_path: Path,
                            duration: float, chunk_dir: Path,
                            boundaries: List[Tuple[float, float]]
                            ) -> Tuple[List[TranscribedWord], Dict]:
        """Recovers narration Whisper dropped from the final short chunk.

        Genuinely missing narration is recovered by re-transcribing a wide EOF
        context window (tail_recovery_context_seconds) and merging the results
        back with the existing deduplication logic. Recovery triggers ONLY when
        the transcript ends significantly before EOF AND the remaining tail
        contains speech-like acoustic energy, so legitimate trailing silence is
        never harmed.
        """
        cfg = self.config
        diag: Dict = {
            "tail_recovery_triggered": False,
            "tail_gap_seconds": round(duration - (merged[-1].end if merged else 0.0), 3),
            "tail_energy_detected": False,
            "tail_recovery_start": None,
            "tail_recovery_end": None,
            "tail_recovery_raw_words": 0,
            "tail_recovery_words_added": 0,
            "final_word_end_before_recovery": merged[-1].end if merged else None,
            "final_word_end_after_recovery": None,
        }
        if not cfg.tail_recovery_enabled or not merged:
            return merged, diag

        last_end = merged[-1].end
        diag["final_word_end_before_recovery"] = round(last_end, 6)
        diag["tail_gap_seconds"] = round(duration - last_end, 3)

        if duration - last_end <= cfg.tail_gap_trigger_seconds:
            return merged, diag

        has_energy = self._tail_energy_detected(audio_path, last_end, duration)
        diag["tail_energy_detected"] = bool(has_energy)
        if not has_energy:
            return merged, diag

        recovery_start = max(0.0, duration - cfg.tail_recovery_context_seconds)
        recovery_end = duration
        diag["tail_recovery_start"] = round(recovery_start, 3)
        diag["tail_recovery_end"] = round(recovery_end, 3)

        clip_path = chunk_dir / "tail_recovery.wav"
        self.audio_service.extract_chunk(audio_path, clip_path, recovery_start, recovery_end)
        rec = self.inner.transcribe(str(clip_path))
        rec_words = [shift_word(w, recovery_start) for w in rec.words]
        diag["tail_recovery_raw_words"] = len(rec_words)

        if not rec_words:
            return merged, diag

        prev_start, prev_end = boundaries[-1] if boundaries else (0.0, duration)
        recovered, _ = self._merge_chunk(merged, rec_words, recovery_start,
                                         prev_start, prev_end)
        diag["tail_recovery_triggered"] = True
        diag["tail_recovery_words_added"] = len(recovered) - len(merged)
        diag["final_word_end_after_recovery"] = round(recovered[-1].end, 6)
        recovered.sort(key=lambda w: (w.start, w.end))
        return recovered, diag

    @staticmethod
    def _match_sequences(prev_list: List[TranscribedWord],
                         cur_list: List[TranscribedWord]
                         ) -> Tuple[List[Tuple[int, int]], List[int]]:
        """Sequential token match over the overlap window.

        Walks both lists in order; a pair matches when normalized text is
        equal. Returns (matched (prev,cur) index pairs, unmatched cur indices).
        """
        normalizer = TextNormalizer()
        i = 0
        j = 0
        matches: List[Tuple[int, int]] = []
        unmatched_cur: List[int] = []
        while j < len(cur_list):
            if i < len(prev_list) and normalizer.normalize(prev_list[i].word) \
                    == normalizer.normalize(cur_list[j].word) \
                    and prev_list[i].word.strip():
                matches.append((i, j))
                i += 1
                j += 1
            elif i < len(prev_list) and prev_list[i].end < cur_list[j].start:
                # prev captured a word the current chunk missed -> keep prev
                i += 1
            else:
                # current chunk has a word the previous chunk missed -> keep cur
                unmatched_cur.append(j)
                j += 1
        return matches, unmatched_cur

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _majority_language(chunk_results) -> str:
        from collections import Counter
        langs = Counter(r.language for *_, r in chunk_results if r.language)
        return (langs.most_common(1)[0][0] if langs else "en")

    @staticmethod
    def _build_segments(words: List[TranscribedWord],
                        max_gap: float = 2.0) -> List[TranscribedSegment]:
        segments: List[TranscribedSegment] = []
        current: List[TranscribedWord] = []
        for w in words:
            if current and w.start - current[-1].end > max_gap:
                segments.append(TranscribedSegment(
                    text=" ".join(x.word for x in current),
                    start=current[0].start, end=current[-1].end, words=list(current)))
                current = []
            current.append(w)
        if current:
            segments.append(TranscribedSegment(
                text=" ".join(x.word for x in current),
                start=current[0].start, end=current[-1].end, words=list(current)))
        return segments
