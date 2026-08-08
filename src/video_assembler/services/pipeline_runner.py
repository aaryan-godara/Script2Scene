"""PipelineRunner: the single shared pipeline function used by BOTH the CLI and
the WebUI. There is exactly one implementation of the generate flow:

    validate -> audio analysis -> transcription -> alignment -> refinement
    -> timeline -> render -> output validation

The WebUI never duplicates this logic. Heavy dependencies (stable_whisper /
torch) are imported lazily so unit tests and the CLI can run without them.

All backend failures are converted into user-friendly :class:`PipelineError`
messages; full technical detail (tracebacks, ffmpeg stderr) is written to the
job log file instead of being shown to the user.
"""

from __future__ import annotations

import json
import math
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from video_assembler.models import ProjectInput, Timeline
from video_assembler.services.alignment.alignment_service import AlignmentService
from video_assembler.services.alignment.failed_region_recovery import (
    FailedRegionRecoveryEngine, rebuild_transcription)
from video_assembler.services.alignment.persistent_transcription_cache import (
    PersistentTranscriptionCache, TranscriptionIdentityConfig, audio_sha256)
from video_assembler.services.audio_service import AudioService
from video_assembler.services.project_validator import ProjectValidator
from video_assembler.services.render_service import RenderConfig, RenderService
from video_assembler.services.timeline_service import TimelineService


class PipelineError(RuntimeError):
    """User-friendly pipeline failure. ``str(error)`` is safe to show in the UI."""


@dataclass
class PipelineResult:
    project_name: str
    output_video: Optional[Path]
    timeline: Timeline
    metadata: Dict = field(default_factory=dict)


ProgressFn = Callable[[str], None]


def _write_timeline_report(timeline: Timeline, dest: Path) -> None:
    lines = []
    header = (f"{'Scene':>5} | {'Image':>24} | {'Speech Start':>13} {'Speech End':>12} | "
              f"{'Visual Start':>13} {'Visual End':>11} {'Visual Dur':>10}")
    lines.append(header)
    lines.append("-" * len(header))
    for sc in timeline.scenes:
        for i, slot in enumerate(sc.images):
            img = Path(slot.path).name
            dur = slot.visual_end - slot.visual_start
            label = img if len(sc.images) == 1 else f"{img} [{i + 1}/{len(sc.images)}]"
            lines.append(
                f"{sc.scene_id:>5} | {label:>24} | "
                f"{sc.speech_start:>13.3f} {sc.speech_end:>12.3f} | "
                f"{slot.visual_start:>13.3f} {slot.visual_end:>11.3f} {dur:>10.3f}"
            )
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")


class PipelineRunner:
    STAGES = [
        "Preparing project...",
        "Validating assets...",
        "Analyzing narration...",
        "Transcribing narration...",
        "Aligning scenes...",
        "Refining speech boundaries...",
        "Creating timeline...",
        "Rendering video...",
        "Validating output...",
        "Complete.",
    ]

    def __init__(self, model_name: str = "base", transcriber: Optional[Callable[[str], object]] = None,
                 chunking_config: object = None, max_review_ratio: float = 0.05,
                 cache_root: Optional[Path | str] = None,
                 identity: Optional[TranscriptionIdentityConfig] = None,
                 auto_recovery: bool = True,
                 collapse_region: Optional[Tuple[float, float]] = None):
        self.model_name = model_name
        # Optional alignment-only ASR stutter collapse window (e.g. a Whisper
        # stutter-damaged region). Default None keeps production behavior.
        self.collapse_region = collapse_region
        # Optional injection point (used by tests to avoid running Whisper).
        self._transcriber = transcriber
        self._chunking_config = chunking_config
        # REVIEW scenes are allowed to render (with a warning) when their share of
        # the project stays at or below this ratio. Above it, manual review is
        # required. FAILED scenes are always hard blockers regardless of this.
        self.max_review_ratio = float(max_review_ratio)
        # Persistent transcription cache location (keyed by audio+config identity).
        self.cache_root = Path(cache_root) if cache_root else Path("workspace/cache/transcriptions")
        # Explicit identity override (tests); otherwise derived from model+chunking.
        self._identity = identity
        # Automatic failed-region recovery between alignment and the render gate.
        self.auto_recovery = auto_recovery
        # Lazily-created real whisper provider shared across transcription and
        # recovery so the model is not reloaded for every failed region.
        self._provider = None

    # ----------------------------------------------------------------- identity
    def transcription_identity(self) -> TranscriptionIdentityConfig:
        if self._identity is not None:
            return self._identity
        cfg = self._chunking_config
        return TranscriptionIdentityConfig(
            provider="stable_whisper",
            model=self.model_name,
            no_speech_threshold=0.9,
            chunking_enabled=bool(getattr(cfg, "chunking_enabled", True)),
            chunk_duration_seconds=float(getattr(cfg, "chunk_duration_seconds", 180.0)),
            overlap_seconds=float(getattr(cfg, "overlap_seconds", 10.0)),
            long_audio_threshold_seconds=float(getattr(cfg, "long_audio_threshold_seconds", 300.0)),
            dedup_time_tolerance=float(getattr(cfg, "dedup_time_tolerance", 1.5)),
            tail_recovery_enabled=bool(getattr(cfg, "tail_recovery_enabled", True)),
            tail_gap_trigger_seconds=float(getattr(cfg, "tail_gap_trigger_seconds", 2.0)),
            tail_recovery_context_seconds=float(getattr(cfg, "tail_recovery_context_seconds", 90.0)),
        )

    def persistent_cache(self) -> PersistentTranscriptionCache:
        return PersistentTranscriptionCache(self.cache_root, self.transcription_identity())

    # ------------------------------------------------------------------ helpers
    def _stage(self, name: str, friendly: str, fn: Callable, progress: Optional[ProgressFn],
               log: Optional[Path], stages: List[str]) -> object:
        stages.append(name)
        if progress is not None:
            progress(name)
        try:
            return fn()
        except PipelineError as e:
            self._write_log(log, name, friendly, e)
            raise
        except Exception as e:  # noqa: BLE001 - boundary for user-facing errors
            self._write_log(log, name, friendly, e)
            raise PipelineError(f"{friendly} failed. See the job log for details.")

    def _write_log(self, log: Optional[Path], name: str, friendly: str,
                   exc: Exception, scene_id: Optional[int] = None) -> None:
        """Writes a stage-failure entry to the job pipeline.log.

        Any stage failure (including PipelineError raised by alignment) produces
        a log record with stage, timestamp and traceback so the failure is never
        silent in the job workspace.
        """
        if log is None:
            return
        import datetime
        ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        lines = [
            f"[{name}] {friendly} failed.",
            f"timestamp: {ts}",
            f"stage: {name}",
            f"error: {type(exc).__name__}: {exc}",
        ]
        if scene_id is not None:
            lines.append(f"scene_id: {scene_id}")
        lines.append(traceback.format_exc())
        log.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _get_provider(self):
        """Lazily builds the real whisper provider ONCE (shared with recovery)."""
        if self._provider is None and self._transcriber is None:
            from video_assembler.services.alignment.stable_whisper_provider import StableWhisperProvider
            self._provider = StableWhisperProvider(model_name=self.model_name)
        return self._provider

    def _transcribe_provider(self, audio_path: str) -> object:
        """Runs transcription for a single clip via the injected or real provider."""
        if self._transcriber is not None:
            return self._transcriber(audio_path)
        return self._get_provider().transcribe(audio_path)

    def _transcribe(self, narration: Path, intermediate_dir: Path) -> object:
        if self._transcriber is not None:
            return self._transcriber(str(narration))
        from video_assembler.services.alignment.chunked_transcription_provider import (
            ChunkedTranscriptionProvider, ChunkingConfig)
        from video_assembler.services.alignment.stable_whisper_provider import StableWhisperProvider
        inner = self._get_provider()
        config = self._chunking_config or ChunkingConfig()
        provider = ChunkedTranscriptionProvider(inner, config=config)
        chunk_dir = intermediate_dir / "transcription_chunks"
        return provider.transcribe(str(narration), chunk_dir=str(chunk_dir))

    def _resolve_transcription(self, narration: Path, intermediate_dir: Path):
        """Persistent-cache-aware transcription resolution.

        Returns (transcription, source, cache_meta) where source is one of
        "persistent_cache" | "fresh_whisper". When a persistent cache hit exists
        for the identical audio+config, Whisper is not re-run.
        """
        sha = audio_sha256(narration)
        cache = self.persistent_cache()
        cached = cache.load(sha)
        if cached is not None:
            transcription = cached
            source = "persistent_cache"
            meta = {
                "source": source,
                "persistent_cache_key": cache.entry_dir(sha).name,
                "cache_version": cache.schema_version,
                "audio_sha256": sha,
            }
            return transcription, source, meta, cache

        transcription = self._transcribe(narration, intermediate_dir)
        transcription.audio_sha256 = sha
        cache.save(transcription, sha, source="fresh_whisper")
        meta = {
            "source": "fresh_whisper",
            "persistent_cache_key": cache.entry_dir(sha).name,
            "cache_version": cache.schema_version,
            "audio_sha256": sha,
        }
        return transcription, "fresh_whisper", meta, cache

    def _persist_recovered(self, cache: PersistentTranscriptionCache,
                           transcription: object, sha: str,
                           recovered_regions: List, recovered_scene_ids: List,
                           tail_triggered: bool) -> None:
        try:
            cache.save(
                transcription, sha, source="recovered_cache",
                metadata_extra={
                    "recovery_applied": True,
                    "recovery_passes": len(recovered_regions),
                    "recovered_regions": recovered_regions,
                    "recovered_scene_ids": recovered_scene_ids,
                    "tail_recovery_triggered": tail_triggered,
                })
        except Exception:  # noqa: BLE001 - cache write is best-effort
            pass

    def _auto_recover(self, scenes, transcription: object, narration: Path,
                      intermediate_dir: Path, audio_duration: float):
        """Runs bounded failed-region recovery passes, re-aligning after each.

        Returns (aligned_scenes, transcription, diagnostics, statuses,
                 audit_log, succeeded). Automatic recovery never promotes
        scenes; the unchanged AlignmentService re-evaluates every scene.
        Returns None when recovery is disabled (callers keep base alignment).
        """
        cfg = self.transcription_identity()
        if not cfg.failed_region_recovery_enabled or not self.auto_recovery:
            return None
        svc = AudioService(narration.parent)
        engine = FailedRegionRecoveryEngine(
            self._transcribe_provider, svc, narration,
            Path(intermediate_dir) / "transcription_chunks", cfg)
        audit: List[Dict] = []
        transcription = rebuild_transcription(transcription, list(transcription.words))

        for pass_no in range(1, cfg.max_recovery_passes + 1):
            aligned, diag, statuses = self._align_scenes(scenes, transcription)
            failed = [sid for sid, st in statuses.items() if st == "FAILED"]
            if not failed:
                return aligned, transcription, diag, statuses, audit, True
            new_words, pass_audit = engine.recover_pass(
                list(transcription.words), scenes, statuses, diag,
                audio_duration, pass_no)
            audit.extend(pass_audit)
            added = len(new_words) - len(transcription.words)
            if added <= 0:
                # nothing recovered this pass; do not spin.
                break
            transcription = rebuild_transcription(transcription, new_words)

        aligned, diag, statuses = self._align_scenes(scenes, transcription)
        return aligned, transcription, diag, statuses, audit, bool(audit)

    def _align_scenes(self, scenes, transcription: object):
        """Runs the unchanged AlignmentService and returns (aligned, diag, statuses)."""
        if self.collapse_region is not None:
            aligned, diag = AlignmentService().align_scenes(
                scenes, transcription, collapse_region=self.collapse_region)
        else:
            aligned, diag = AlignmentService().align_scenes(scenes, transcription)
        statuses = {sid: d.get("status") for sid, d in diag.diagnostics.items()}
        return aligned, diag, statuses

    def run(
        self,
        project_input: ProjectInput,
        narration: Path | str,
        images_dir: Path | str,
        *,
        intermediate_dir: Path | str,
        output_dir: Path | str,
        logs_dir: Path | str,
        transcribe: bool = True,
        transcription: object = None,
        render: bool = True,
        output_name: str = "final_video.mp4",
        progress: Optional[ProgressFn] = None,
    ) -> PipelineResult:
        """Runs the full pipeline for one project. Both CLI and WebUI call this."""
        start = time.time()
        narration = Path(narration)
        images_dir = Path(images_dir)
        intermediate_dir = Path(intermediate_dir)
        output_dir = Path(output_dir)
        logs_dir = Path(logs_dir)

        for d in (intermediate_dir, output_dir, logs_dir):
            d.mkdir(parents=True, exist_ok=True)
        log = logs_dir / "pipeline.log"
        stages: List[str] = []

        self._stage("Validating assets...", "Project validation", lambda: None, progress, log, stages)
        outcome = ProjectValidator().validate(project_input, narration, images_dir)
        if not outcome.valid:
            raise PipelineError("Project validation failed.\n" + "\n".join(f"- {e}" for e in outcome.errors))

        audio_meta = self._stage(
            "Analyzing narration...", "Narration analysis",
            lambda: AudioService(narration.parent).get_audio_metadata(narration),
            progress, log, stages)
        audio_duration = float(audio_meta["duration"])
        if audio_duration <= 0:
            raise PipelineError("Narration could not be decoded (zero duration).")

        cache_meta = {}
        recovered_scene_ids: List[int] = []
        recovery_regions: List[Dict] = []
        tail_recovery_triggered = False
        transcription_source = "external"

        if transcribe:
            transcription, transcription_source, cache_meta, persistent_cache = \
                self._stage(
                    "Transcribing narration...", "Whisper transcription",
                    lambda: self._resolve_transcription(narration, intermediate_dir),
                    progress, log, stages)
            from video_assembler.services.alignment.transcription_cache import save_transcription
            save_transcription(transcription, intermediate_dir / "transcription.json",
                               audio_path=narration)
            (intermediate_dir / "transcription_cache.json").write_text(
                json.dumps(cache_meta, indent=2), encoding="utf-8")

        def _align():
            aligned, diagnostics, statuses = self._align_scenes(
                project_input.scenes, transcription)
            return aligned, diagnostics, statuses

        aligned_scenes, diagnostics, statuses = self._stage(
            "Aligning scenes...", "Scene alignment",
            lambda: _align(), progress, log, stages)

        # --- automatic failed-region local recovery + numeric validation -------
        recovery_audit: List[Dict] = []
        if transcription_source != "external":
            recovered = self._auto_recover(
                project_input.scenes, transcription, narration,
                intermediate_dir, audio_duration)
            if recovered is not None:
                aligned_scenes, transcription, diagnostics, statuses, recovery_audit, _ = recovered
                self._stage(
                    "Recovering failed regions...", "Local recovery",
                    lambda: None, progress, log, stages)
                if recovery_audit:
                    for entry in recovery_audit:
                        recovery_regions.append({
                            "window_start": entry.get("window_start"),
                            "window_end": entry.get("window_end"),
                            "pass": entry.get("pass"),
                            "scene_group": entry.get("scene_group"),
                        })
                        recovered_scene_ids.extend(entry.get("scene_group") or [])
                    recovered_scene_ids = sorted(set(recovered_scene_ids))
                    tail_recovery_triggered = bool(
                        (transcription.tail_recovery or {}).get("tail_recovery_triggered"))
                    self._persist_recovered(
                        persistent_cache, transcription,
                        cache_meta.get("audio_sha256") or audio_sha256(narration),
                        recovery_regions, recovered_scene_ids, tail_recovery_triggered)
                    # refresh the job copy with the recovered transcription
                    save_transcription(
                        transcription, intermediate_dir / "transcription.json",
                        audio_path=narration)
                    (intermediate_dir / "recovery_log.json").write_text(
                        json.dumps(recovery_audit, indent=2), encoding="utf-8")
                    cache_meta["source"] = "recovered_cache"
                    cache_meta["recovery_applied"] = True
                    (intermediate_dir / "transcription_cache.json").write_text(
                        json.dumps(cache_meta, indent=2), encoding="utf-8")

        # Write an informational alignment report before the gate runs so a
        # manual reviewer always has the evidence even when generation is
        # stopped by a hard FAILED block or an over-limit REVIEW share.
        try:
            from video_assembler.services.alignment.review_report import write_review_report
            write_review_report(intermediate_dir / "alignment_review.json",
                                aligned_scenes, statuses, diagnostics, transcription)
        except Exception:  # noqa: BLE001 - a review report must never block the run
            pass

        alignment_warnings = self._stage(
            "Aligning scenes...", "Scene alignment",
            lambda: self._check_alignment_gate(statuses, diagnostics, audio_duration),
            progress, log, stages)

        (intermediate_dir / "alignment.json").write_text(
            json.dumps([{
                "scene_id": s.scene_id,
                "speech_start": s.speech_start,
                "speech_end": s.speech_end,
                "status": statuses.get(s.scene_id, "FAILED"),
                "confidence": s.match_confidence,
                **{k: v for k, v in diagnostics.diagnostics.get(s.scene_id, {}).items()
                   if k in ("numeric_match", "warning_type",
                            "canonical_numeric_values", "asr_numeric_values")},
            } for s in aligned_scenes], indent=2), encoding="utf-8")

        review_scene_ids = {sid for sid, st in statuses.items() if st == "REVIEW"}
        for sc in aligned_scenes:
            if sc.scene_id in review_scene_ids:
                diag = diagnostics.diagnostics.get(sc.scene_id, {})
                warning = diag.get("warning_type")
                sc.warning = (
                    f"Aligned with only REVIEW confidence but has usable timestamps. "
                    f"Rendered anyway - please verify."
                    + (f" ({warning})" if warning else ""))

        refinements = self._stage(
            "Refining speech boundaries...", "Acoustic refinement",
            lambda: self._refine(narration, aligned_scenes), progress, log, stages)
        for s, r in zip(aligned_scenes, refinements[0]):
            s.raw_speech_end = s.speech_end
            s.speech_end = r.refined_speech_end

        for sc in aligned_scenes:
            if sc.speech_start is None or sc.speech_end is None:
                raise PipelineError(f"Scene {sc.scene_id} has no speech timestamps.")

        timeline = self._stage(
            "Creating timeline...", "Timeline creation",
            lambda: TimelineService(audio_duration=audio_duration, fps=30,
                                    images_dir=images_dir).build_timeline(aligned_scenes),
            progress, log, stages)
        timeline.project = project_input.project
        (intermediate_dir / "timeline.json").write_text(
            json.dumps(timeline.model_dump(), indent=2), encoding="utf-8")
        _write_timeline_report(timeline, intermediate_dir / "timeline_report.txt")

        metadata = {
            "project": project_input.project,
            "duration_s": round(audio_duration, 3),
            "scene_count": len(timeline.scenes),
            "stages": list(stages),
        }
        if cache_meta:
            metadata["transcription_cache"] = {
                "source": cache_meta.get("source"),
                "persistent_cache_key": cache_meta.get("persistent_cache_key"),
                "cache_version": cache_meta.get("cache_version"),
                "audio_sha256": cache_meta.get("audio_sha256"),
            }
        metadata["transcription_source"] = transcription_source
        if recovery_regions:
            metadata["recovery"] = {
                "recovered_scene_ids": recovered_scene_ids,
                "regions": recovery_regions,
                "tail_recovery_triggered": tail_recovery_triggered,
            }
        if alignment_warnings:
            metadata["alignment_warnings"] = alignment_warnings
        metadata["alignment_statuses"] = {
            st: sum(1 for v in statuses.values() if v == st)
            for st in sorted(set(statuses.values()))
        }

        output_video: Optional[Path] = None
        if render:
            output_video = output_dir / output_name
            probe = self._stage(
                "Rendering video...", "FFmpeg rendering",
                lambda: RenderService(RenderConfig(), temp_dir=intermediate_dir / "render_temp")
                .render(timeline, narration, output_video),
                progress, log, stages)
            self._stage(
                "Validating output...", "Output validation",
                lambda: self._validate_output(probe, audio_duration),
                progress, log, stages)
            metadata.update({
                "resolution": f"{probe['width']}x{probe['height']}",
                "fps": round(probe.get("fps") or 0, 2),
                "video_codec": probe.get("video_codec"),
                "audio_codec": probe.get("audio_codec"),
                "video_path": str(output_video),
            })

        metadata["processing_time_s"] = round(time.time() - start, 2)
        if log is not None:
            log.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        stages.append("Complete.")
        if progress is not None:
            progress("Complete.")
        metadata["stages"] = stages

        return PipelineResult(project_name=project_input.project, output_video=output_video,
                              timeline=timeline, metadata=metadata)

    # ----------------------------------------------------------- inner helpers
    def _check_alignment_gate(self, statuses: Dict, diagnostics: object,
                              audio_duration: float) -> List[str]:
        """Applies the alignment generation gate and returns warnings.

        HIGH scenes always pass. FAILED scenes are hard blockers. REVIEW scenes
        are allowed only when every one of them has usable timestamps and their
        share of the project stays at or below ``max_review_ratio``. Allowed
        REVIEW scenes stay REVIEW and produce warnings for later manual review.
        """
        failed = [sid for sid, st in statuses.items() if st == "FAILED"]
        review = [sid for sid, st in statuses.items() if st == "REVIEW"]
        if failed:
            raise PipelineError(
                f"Scene {failed[0]} could not be aligned confidently. Generation stopped.")
        if not review:
            return []
        # REVIEW scenes are allowed only when every one has usable timestamps
        # and the review share stays within the configured ratio. Their status
        # is never promoted to HIGH.
        invalid = [sid for sid in review
                   if not self._review_timestamps_usable(
                       diagnostics.diagnostics.get(sid, {}), audio_duration)]
        if invalid:
            raise PipelineError(
                f"Scene {invalid[0]} aligned with only REVIEW confidence but has "
                "invalid/unsafe timestamps. Generation stopped.")
        ratio = len(review) / max(len(statuses), 1)
        if ratio > self.max_review_ratio:
            raise PipelineError(
                f"{len(review)} of {len(statuses)} scenes aligned with only REVIEW "
                f"confidence ({ratio:.1%} exceeds the allowed "
                f"{self.max_review_ratio:.1%}). Manual review required before "
                "generation.")
        return self._review_warnings(review, diagnostics, statuses)

    @staticmethod
    def _review_timestamps_usable(diag: Dict, audio_duration: float) -> bool:
        """True when a REVIEW scene's matched window is safe to render.

        The window must be finite, ordered (start < end) and inside the audio.
        This mirrors what the timeline service enforces for every scene, so a
        REVIEW scene allowed here is guaranteed to build a valid timeline entry.
        """
        start = diag.get("speech_start")
        end = diag.get("speech_end")
        try:
            start = float(start)
            end = float(end)
        except (TypeError, ValueError):
            return False
        if not (math.isfinite(start) and math.isfinite(end)):
            return False
        if not (0.0 <= start < end <= audio_duration + 1e-6):
            return False
        return True

    @staticmethod
    def _review_warnings(review: List[int], diagnostics: object,
                         statuses: Dict) -> List[str]:
        """Human-readable warnings for scenes that were only REVIEW-aligned.

        The scenes are allowed to render because their timestamps are usable,
        but their status stays REVIEW and each one gets a warning so the
        operator can manually verify the match after the fact.
        """
        warnings: List[str] = []
        for sid in review:
            diag = diagnostics.diagnostics.get(sid, {})
            confidence = diag.get("confidence")
            warnings.append(
                f"Scene {sid} aligned with only REVIEW confidence"
                f" (match {confidence if confidence is not None else 'unknown'}). "
                "Rendered anyway because its timestamps are usable - please verify.")
        warnings.append(
            f"{len(statuses) - len(review)} HIGH, {len(review)} REVIEW, "
            f"{sum(1 for st in statuses.values() if st == 'FAILED')} FAILED.")
        return warnings

    def _refine(self, narration: Path, aligned_scenes: List) -> Tuple[List, object]:
        from video_assembler.services.alignment.acoustic_boundary_refiner import (
            AcousticBoundaryRefiner, RefinerConfig)
        refiner = AcousticBoundaryRefiner(str(narration), RefinerConfig())
        return refiner.refine([
            {"scene_id": s.scene_id, "speech_start": s.speech_start, "speech_end": s.speech_end}
            for s in aligned_scenes])

    def _validate_output(self, probe: Dict, audio_duration: float) -> None:
        expected = RenderConfig()
        problems = []
        if not probe.get("has_video"):
            problems.append("no video stream")
        if not probe.get("has_audio"):
            problems.append("no audio stream")
        if probe.get("width") != expected.width or probe.get("height") != expected.height:
            problems.append(
                f"resolution {probe.get('width')}x{probe.get('height')} != "
                f"{expected.width}x{expected.height}")
        tolerance = 1.0 / expected.fps + 0.05
        if abs((probe.get("video_duration") or 0) - audio_duration) > tolerance:
            problems.append(
                f"video duration {probe.get('video_duration'):.3f}s != audio {audio_duration:.3f}s")
        if abs((probe.get("audio_duration") or 0) - audio_duration) > 0.1:
            problems.append(
                f"audio stream duration {probe.get('audio_duration'):.3f}s != "
                f"narration {audio_duration:.3f}s")
        if abs((probe.get("container_duration") or 0) - audio_duration) > tolerance:
            problems.append(
                f"container duration {probe.get('container_duration'):.3f}s != "
                f"audio {audio_duration:.3f}s")
        if problems:
            raise PipelineError("Output validation failed: " + "; ".join(problems))
