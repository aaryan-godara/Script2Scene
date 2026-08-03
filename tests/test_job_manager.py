"""Tests for JobManager: isolated per-run workspaces."""

import tempfile
import unittest
from pathlib import Path

from video_assembler.services.job_manager import JobError, JobManager

from tests._helpers import make_image, synth_audio


class JobManagerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.manager = JobManager(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_create_job_builds_standard_layout(self):
        job = self.manager.create_job()
        self.assertTrue(job.input_dir.is_dir())
        self.assertTrue(job.images_dir.is_dir())
        self.assertTrue(job.intermediate_dir.is_dir())
        self.assertTrue(job.output_dir.is_dir())
        self.assertTrue(job.logs_dir.is_dir())

    def test_two_jobs_are_distinct(self):
        a = self.manager.create_job()
        b = self.manager.create_job()
        self.assertNotEqual(a.job_id, b.job_id)
        self.assertNotEqual(a.root, b.root)
        self.assertFalse((a.root / "x").exists() and (b.root / "x").exists())

    def test_write_images_preserves_original_names_in_any_order(self):
        job = self.manager.create_job()
        src1 = make_image(self.root / "s1.png")
        src2 = make_image(self.root / "s2.png")
        src3 = make_image(self.root / "s3.png")
        # Deliberately uploaded in shuffled order.
        job.images_dir.mkdir(parents=True, exist_ok=True)
        written = self.manager.write_images(job, [
            ("scene_003.png", src3),
            ("scene_001.png", src1),
            ("scene_002.png", src2),
        ])
        self.assertEqual(sorted(p.name for p in written),
                         ["scene_001.png", "scene_002.png", "scene_003.png"])
        for name in ("scene_001.png", "scene_002.png", "scene_003.png"):
            self.assertTrue((job.images_dir / name).is_file())

    def test_duplicate_image_filename_rejected(self):
        job = self.manager.create_job()
        src1 = make_image(self.root / "a.png")
        src2 = make_image(self.root / "b.png")
        with self.assertRaises(JobError):
            self.manager.write_images(job, [("scene_001.png", src1), ("scene_001.png", src2)])

    def test_write_narration_preserves_name(self):
        job = self.manager.create_job()
        src = synth_audio(self.root / "src.wav", 0.4)
        dest = self.manager.write_narration(job, src, "my_narration.wav")
        self.assertEqual(dest.name, "my_narration.wav")
        self.assertTrue(dest.is_file())

    def test_unsafe_filenames_are_sanitized(self):
        job = self.manager.create_job()
        src = make_image(self.root / "x.png")
        dest = self.manager.write_images(job, [("../../escape.png", src)])[0]
        self.assertEqual(dest.name, "escape.png")
        self.assertNotIn("..", str(dest.relative_to(job.images_dir)))


if __name__ == "__main__":
    unittest.main()
