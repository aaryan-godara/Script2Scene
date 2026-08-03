"""Tests for TimelineService (visual timeline policy).

Policy under test:
  - first scene visual_start == 0.0 (image leads narration)
  - other scenes visual_start == their speech_start
  - non-final visual_end == next scene's speech_start (image stays visible
    through any silence between scenes)
  - final visual_end == total audio duration
  - silence belongs to the *visual* duration, never to speech_end
  - multi-image scenes split their visual duration equally between images
"""

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from video_assembler.models import Scene
from video_assembler.services.timeline_service import TimelineService, TimelineValidationError


def make_image(path: Path, color=(255, 0, 0)) -> None:
    Image.new("RGB", (320, 180), color).save(path)


class TimelineServiceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.images = Path(self.tmp.name) / "img"
        self.images.mkdir()
        for name in ("a.png", "b.png", "c.png", "d.png"):
            make_image(self.images / name)

    def tearDown(self):
        self.tmp.cleanup()

    def _scene(self, sid, start, end, names):
        return Scene(
            scene_id=sid,
            script_text=f"scene {sid}",
            images=[str(self.images / n) for n in names],
            speech_start=start,
            speech_end=end,
        )

    def test_normal_three_scene_timeline(self):
        scenes = [
            self._scene(1, 0.0, 3.0, ["a.png"]),
            self._scene(2, 3.5, 8.0, ["b.png"]),
            self._scene(3, 8.5, 12.0, ["c.png"]),
        ]
        t = TimelineService(audio_duration=20.0, fps=30, images_dir=self.images).build_timeline(scenes)

        self.assertEqual(len(t.scenes), 3)
        s1, s2, s3 = t.scenes
        self.assertEqual(s1.images[0].visual_start, 0.0)
        self.assertEqual(s1.images[0].visual_end, 3.5)
        self.assertEqual(s2.images[0].visual_start, 3.5)
        self.assertEqual(s2.images[0].visual_end, 8.5)
        self.assertEqual(s3.images[0].visual_start, 8.5)
        self.assertEqual(s3.images[0].visual_end, 20.0)

    def test_silence_between_scenes_belongs_to_visual_duration(self):
        scenes = [
            self._scene(1, 0.0, 3.0, ["a.png"]),
            self._scene(2, 5.0, 8.0, ["b.png"]),
        ]
        t = TimelineService(audio_duration=10.0, fps=30, images_dir=self.images).build_timeline(scenes)
        s1, s2 = t.scenes
        self.assertEqual(s1.images[0].visual_end, 5.0)
        self.assertEqual(s1.speech_end, 3.0)
        self.assertEqual(s2.images[0].visual_start, 5.0)
        self.assertEqual(s2.speech_start, 5.0)

    def test_first_scene_starts_at_zero_even_when_speech_starts_later(self):
        scenes = [self._scene(1, 0.5, 4.0, ["a.png"])]
        t = TimelineService(audio_duration=6.0, fps=30, images_dir=self.images).build_timeline(scenes)
        s1 = t.scenes[0]
        self.assertEqual(s1.images[0].visual_start, 0.0)
        self.assertEqual(s1.speech_start, 0.5)

    def test_final_scene_extends_to_audio_duration(self):
        scenes = [
            self._scene(1, 0.0, 2.0, ["a.png"]),
            self._scene(2, 3.0, 4.0, ["b.png"]),
        ]
        t = TimelineService(audio_duration=9.5, fps=30, images_dir=self.images).build_timeline(scenes)
        self.assertEqual(t.scenes[-1].images[0].visual_end, 9.5)

    def test_multi_image_scene_splits_duration_equally(self):
        scenes = [self._scene(1, 0.0, 5.0, ["a.png", "b.png", "c.png"])]
        t = TimelineService(audio_duration=6.0, fps=30, images_dir=self.images).build_timeline(scenes)
        slots = t.scenes[0].images
        self.assertEqual(len(slots), 3)
        expected = [(0.0, 2.0), (2.0, 4.0), (4.0, 6.0)]
        for slot, (vs, ve) in zip(slots, expected):
            self.assertAlmostEqual(slot.visual_start, vs, places=6)
            self.assertAlmostEqual(slot.visual_end, ve, places=6)

    def test_scene_order_must_respect_speech_boundaries(self):
        scenes = [
            self._scene(1, 0.0, 3.0, ["a.png"]),
            self._scene(2, 2.0, 5.0, ["b.png"]),
        ]
        with self.assertRaises(TimelineValidationError):
            TimelineService(audio_duration=10.0, fps=30, images_dir=self.images).build_timeline(scenes)

    def test_zero_visual_duration_rejected(self):
        scenes = [
            self._scene(1, 0.0, 1.0, ["a.png"]),
            self._scene(2, 0.0, 2.0, ["b.png"]),
        ]
        with self.assertRaises(TimelineValidationError):
            TimelineService(audio_duration=5.0, fps=30, images_dir=self.images).build_timeline(scenes)

    def test_missing_image_rejected(self):
        scenes = [self._scene(1, 0.0, 2.0, ["does_not_exist.png"])]
        with self.assertRaises(TimelineValidationError):
            TimelineService(audio_duration=5.0, fps=30, images_dir=self.images).build_timeline(scenes)

    def test_scene_with_no_images_rejected(self):
        scenes = [Scene(scene_id=1, script_text="x", images=[], speech_start=0.0, speech_end=2.0)]
        with self.assertRaises(TimelineValidationError):
            TimelineService(audio_duration=5.0, fps=30, images_dir=self.images).build_timeline(scenes)


if __name__ == "__main__":
    unittest.main()
