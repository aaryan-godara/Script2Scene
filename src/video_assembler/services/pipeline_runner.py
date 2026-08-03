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
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from video_assembler.models import ProjectInput, Timeline
from video_assembler.services.alignment.alignment_service import AlignmentService
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

    def __init__(self, model_name: str = "base", transcriber: Optional[Callable[[str], object]] = None):
        self.model_name = model_name
        # Optional injection point (used by tests to avoid running Whisper).
        self._transcriber = transcriber

    # ------------------------------------------------------------------ helpers
    def _stage(self, name: str, friendly: str, fn: Callable, progress: Optional[ProgressFn],
               log: Optional[Path], stages: List[str]) -> object:
        stages.append(name)
        if progress is not None:
            progress(name)
        try:
            return fn()
        except PipelineError:
            raise
        except Exception as e:  # noqa: BLE001 - boundary for user-facing errors
            if log is not None:
                log.write_text(f"[{name}] {friendly} failed.\n{traceback.format_exc()}",
                               encoding="utf-8")
            raise PipelineError(f"{friendly} failed. See the job log for details.")

    def _transcribe(self, narration: Path) -> object:
        if self._transcriber is not None:
            return self._transcriber(str(narration))
        from video_assembler.services.alignment.stable_whisper_provider import StableWhisperProvider
        provider = StableWhisperProvider(model_name=self.model_name)
        return provider.transcribe(str(narration))

    # --------------------------------------------------------------------- run
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
        allow_review: bool = False,
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

        if transcribe:
            transcription = self._stage(
                "Transcribing narration...", "Whisper transcription",
                lambda: self._transcribe(narration), progress, log, stages)
            (intermediate_dir / "transcription.json").write_text(
                json.dumps(transcription.model_dump(), indent=2), encoding="utf-8")

        aligned_scenes, diagnostics = self._stage(
            "Aligning scenes...", "Scene alignment",
            lambda: AlignmentService().align_scenes(project_input.scenes, transcription),
            progress, log, stages)

        statuses = {sid: d.get("status") for sid, d in diagnostics.diagnostics.items()}
        failed = [sid for sid, st in statuses.items() if st == "FAILED"]
        review = [sid for sid, st in statuses.items() if st == "REVIEW"]
        if failed:
            raise PipelineError(
                f"Scene {failed[0]} could not be aligned confidently. Generation stopped.")
        if review and not allow_review:
            raise PipelineError(
                f"Scenes {', '.join(str(s) for s in review)} aligned with only REVIEW confidence. "
                "Generation stopped. Re-record the narration or review these scenes.")

        (intermediate_dir / "alignment.json").write_text(
            json.dumps([{
                "scene_id": s.scene_id,
                "speech_start": s.speech_start,
                "speech_end": s.speech_end,
                "status": statuses.get(s.scene_id, "FAILED"),
                "confidence": s.match_confidence,
            } for s in aligned_scenes], indent=2), encoding="utf-8")

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
