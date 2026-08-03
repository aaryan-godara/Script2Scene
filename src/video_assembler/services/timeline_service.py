"""TimelineService: converts aligned scenes into a visual timeline.

Speech timing and visual timing are kept strictly separate:
  - speech_start / speech_end come from AlignmentService + AcousticBoundaryRefiner
    and are never modified here.
  - visual_start / visual_end decide how long each image stays on screen.

MVP visual timing policy (deterministic):
  - First scene:            visual_start = 0.0
  - Other scenes:           visual_start = speech_start
  - Non-final scenes:       visual_end = next scene's visual_start
  - Final scene:            visual_end = audio_duration

The current image therefore stays visible through the silence until the next
scene begins; silence belongs only to visual duration, never to speech_end.

Multi-image support: a scene's visual duration is divided equally between its
images (no AI-based sub-timing yet).

The service validates the timeline and fails clearly instead of guessing.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from video_assembler.models import Scene, Timeline, TimelineImage, TimelineScene


class TimelineValidationError(ValueError):
    pass


class TimelineService:
    def __init__(self, audio_duration: float, fps: int = 30, images_dir: Optional[Path] = None):
        self.audio_duration = float(audio_duration)
        self.fps = int(fps)
        self.images_dir = Path(images_dir) if images_dir else None
        self._tolerance = 1e-3

    # ------------------------------------------------------------ path resolution
    def _resolve_image(self, image_ref: str) -> str:
        p = Path(image_ref)
        if p.is_absolute() and p.exists():
            return str(p)
        if self.images_dir is not None:
            candidate = self.images_dir / image_ref
            if candidate.exists():
                return str(candidate)
        raise TimelineValidationError(f"Image file not found for scene: {image_ref}")

    # ---------------------------------------------------------------- validation
    def _validate_order(self, scenes: List[Scene]) -> None:
        if not scenes:
            raise TimelineValidationError("Timeline requires at least one scene.")

        previous_visual_end = None
        for i, sc in enumerate(scenes):
            if sc.speech_start is None or sc.speech_end is None:
                raise TimelineValidationError(
                    f"Scene {sc.scene_id} has no speech timestamps; alignment must run before timeline.")

            visual_start = 0.0 if i == 0 else sc.speech_start
            visual_end = scenes[i + 1].speech_start if i < len(scenes) - 1 else self.audio_duration

            if visual_start < 0:
                raise TimelineValidationError(f"Scene {sc.scene_id}: visual_start < 0.")
            if visual_end <= visual_start:
                raise TimelineValidationError(
                    f"Scene {sc.scene_id}: visual_end ({visual_end:.3f}s) is not greater than "
                    f"visual_start ({visual_start:.3f}s); scene has zero/negative visual duration.")
            if visual_end + self._tolerance < sc.speech_end:
                raise TimelineValidationError(
                    f"Scene {sc.scene_id}: visual_end ({visual_end:.3f}s) ends before its own "
                    f"speech_end ({sc.speech_end:.3f}s); scene ordering is invalid "
                    f"(next scene's speech_start is too early).")
            if visual_end > self.audio_duration + self._tolerance:
                raise TimelineValidationError(
                    f"Scene {sc.scene_id}: visual_end ({visual_end:.3f}s) exceeds audio_duration "
                    f"({self.audio_duration:.3f}s).")
            if i == 0 and abs(visual_start) > self._tolerance:
                raise TimelineValidationError("First scene visual_start must be 0.0.")
            if i == len(scenes) - 1 and abs(visual_end - self.audio_duration) > self._tolerance:
                raise TimelineValidationError(
                    f"Final scene visual_end ({visual_end:.3f}s) must equal audio_duration "
                    f"({self.audio_duration:.3f}s).")
            if previous_visual_end is not None and abs(previous_visual_end - visual_start) > self._tolerance:
                raise TimelineValidationError(
                    f"Visual gap or overlap between scene {i} (end {previous_visual_end:.3f}s) and "
                    f"scene {sc.scene_id} (start {visual_start:.3f}s).")
            previous_visual_end = visual_end

            if not sc.images:
                raise TimelineValidationError(f"Scene {sc.scene_id} has no images.")

    # ------------------------------------------------------------------ building
    def build_timeline(self, scenes: List[Scene]) -> Timeline:
        scenes = sorted(scenes, key=lambda s: s.scene_id)
        self._validate_order(scenes)

        timeline_scenes: List[TimelineScene] = []
        for i, sc in enumerate(scenes):
            visual_start = 0.0 if i == 0 else sc.speech_start
            visual_end = scenes[i + 1].speech_start if i < len(scenes) - 1 else self.audio_duration

            resolved = [self._resolve_image(img) for img in sc.images]
            k = len(resolved)
            duration = visual_end - visual_start
            slots = []
            for j, path in enumerate(resolved):
                slot_start = visual_start + j * duration / k
                slot_end = visual_start + (j + 1) * duration / k
                if j == k - 1:
                    slot_end = visual_end
                slots.append(TimelineImage(path=str(path), visual_start=slot_start, visual_end=slot_end))

            timeline_scenes.append(
                TimelineScene(
                    scene_id=sc.scene_id,
                    script_text=sc.script_text,
                    speech_start=sc.speech_start,
                    speech_end=sc.speech_end,
                    visual_start=visual_start,
                    visual_end=visual_end,
                    images=slots,
                )
            )

        return Timeline(project="", audio_duration=self.audio_duration, fps=self.fps, scenes=timeline_scenes)
