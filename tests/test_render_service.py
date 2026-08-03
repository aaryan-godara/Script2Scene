"""Integration tests for RenderService (real FFmpeg encode, ffprobe validation).

These tests synthesize a short silent-free narration clip with ffmpeg, build a
two-scene timeline, render it, and verify the container/probe output.
"""

import subprocess
import tempfile
import unittest
from pathlib import Path

from video_assembler.models import Timeline, TimelineScene, TimelineImage
from video_assembler.services.render_service import RenderConfig, RenderService

try:
    from PIL import Image
    HAVE_PIL = True
except ImportError:
    HAVE_PIL = False


def make_image(path: Path, color=(0, 0, 255)) -> None:
    Image.new("RGB", (1920, 1080), color).save(path)


def synth_audio(path: Path, seconds: float = 2.0) -> None:
    proc = subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
         "-c:a", "aac", "-b:a", "128k", str(path)],
        capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"audio synth failed: {(proc.stderr or '')[-2000:]}")


class RenderServiceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not HAVE_PIL:
            raise unittest.SkipTest("PIL not available")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.images = self.dir / "img"
        self.images.mkdir()
        make_image(self.images / "a.png", color=(230, 25, 75))
        make_image(self.images / "b.png", color=(60, 180, 75))
        self.audio = self.dir / "narration.m4a"
        synth_audio(self.audio, 2.0)

    def tearDown(self):
        self.tmp.cleanup()

    def _timeline(self):
        return Timeline(
            project="render-test",
            audio_duration=2.0,
            scenes=[
                TimelineScene(
                    scene_id=1, script_text="scene one",
                    images=[TimelineImage(path=str(self.images / "a.png"),
                                          visual_start=0.0, visual_end=1.0)],
                    speech_start=0.0, speech_end=0.8, visual_start=0.0, visual_end=1.0),
                TimelineScene(
                    scene_id=2, script_text="scene two",
                    images=[TimelineImage(path=str(self.images / "b.png"),
                                          visual_start=1.0, visual_end=2.0)],
                    speech_start=1.0, speech_end=1.6, visual_start=1.0, visual_end=2.0),
            ],
        )

    def test_renders_synchronized_mp4(self):
        temp_out = self.dir / "render_temp"
        out = self.dir / "out.mp4"
        svc = RenderService(RenderConfig(), temp_dir=temp_out)
        probe = svc.render(self._timeline(), self.audio, out)

        self.assertTrue(out.exists())
        self.assertTrue(probe["has_video"])
        self.assertTrue(probe["has_audio"])
        self.assertEqual((probe["width"], probe["height"]), (1920, 1080))
        self.assertEqual(probe["video_codec"], "h264")
        self.assertEqual(probe["audio_codec"], "aac")
        self.assertAlmostEqual(probe["video_duration"], 2.0, delta=0.1)
        self.assertAlmostEqual(probe["audio_duration"], 2.0, delta=0.1)
        self.assertAlmostEqual(probe["container_duration"], 2.0, delta=0.1)
        self.assertAlmostEqual(probe["fps"], 30.0, delta=0.1)

    def test_temp_dir_cleaned_up_after_success(self):
        temp_out = self.dir / "render_temp2"
        svc = RenderService(RenderConfig(), temp_dir=temp_out)
        svc.render(self._timeline(), self.audio, self.dir / "out2.mp4")
        self.assertTrue(temp_out.exists())
        self.assertEqual(list(temp_out.iterdir()), [])

    def test_missing_image_slot_raises(self):
        tl = self._timeline()
        tl.scenes[0].images[0].path = str(self.images / "nope.png")
        svc = RenderService(RenderConfig(), temp_dir=self.dir / "render_temp3")
        with self.assertRaises(Exception):
            svc.render(tl, self.audio, self.dir / "out3.mp4")


if __name__ == "__main__":
    unittest.main()
