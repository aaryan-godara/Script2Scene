"""Tests for ProjectValidator: fast preflight checks (no Whisper)."""

import json
import tempfile
import unittest
from pathlib import Path

from video_assembler.services.job_manager import JobManager
from video_assembler.services.project_validator import ProjectValidator

from tests._helpers import make_image, synth_audio, write_project_json


def scenes(*items):
    return list(items)


class ProjectValidatorTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.images = self.root / "images"
        self.images.mkdir(parents=True, exist_ok=True)
        self.narration = synth_audio(self.root / "narration.wav", 1.0)
        self.validator = ProjectValidator()

    def tearDown(self):
        self.tmp.cleanup()

    def _project(self, scene_list, name="food_truck"):
        return write_project_json(self.root, scene_list, name)

    def test_valid_project_with_unordered_image_uploads(self):
        # Uploaded images land in the images dir by ORIGINAL name (any order).
        for sid in (3, 1, 2):
            make_image(self.images / f"scene_{sid:03d}.png")
        pj = self._project([
            {"scene_id": 1, "script_text": "hello world", "images": ["scene_001.png"]},
            {"scene_id": 2, "script_text": "second scene", "images": ["scene_002.png"]},
            {"scene_id": 3, "script_text": "third scene", "images": ["scene_003.png"]},
        ])
        outcome = self.validator.parse_and_validate(pj, self.narration, self.images)
        self.assertTrue(outcome.valid, outcome.errors)
        self.assertEqual(outcome.scene_count, 3)
        self.assertEqual(outcome.image_count, 3)
        mapping = {r["scene_id"]: r["images"] for r in outcome.rows}
        self.assertEqual(mapping[1], "scene_001.png")
        self.assertEqual(mapping[2], "scene_002.png")
        self.assertEqual(mapping[3], "scene_003.png")

    def test_missing_image_reported_with_friendly_message(self):
        make_image(self.images / "scene_006.png")
        pj = self._project([
            {"scene_id": 6, "script_text": "six", "images": ["scene_006.png"]},
            {"scene_id": 7, "script_text": "seven", "images": ["scene_007.png"]},
        ])
        outcome = self.validator.parse_and_validate(pj, self.narration, self.images)
        self.assertFalse(outcome.valid)
        self.assertTrue(any("Scene 7 expects: scene_007.png" in e for e in outcome.errors))

    def test_invalid_json(self):
        bad = self.root / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        outcome = self.validator.parse_and_validate(bad, self.narration, self.images)
        self.assertFalse(outcome.valid)
        self.assertTrue(any("Invalid JSON" in e for e in outcome.errors))

    def test_missing_script_text(self):
        make_image(self.images / "scene_001.png")
        pj = self._project([{"scene_id": 1, "images": ["scene_001.png"]}])
        outcome = self.validator.parse_and_validate(pj, self.narration, self.images)
        self.assertFalse(outcome.valid)
        self.assertTrue(any("script_text" in e for e in outcome.errors))

    def test_empty_scenes(self):
        pj = self._project([])
        outcome = self.validator.parse_and_validate(pj, self.narration, self.images)
        self.assertFalse(outcome.valid)
        self.assertTrue(any("no scenes" in e.lower() for e in outcome.errors))

    def test_duplicate_scene_ids(self):
        pj = self._project([
            {"scene_id": 1, "script_text": "a"},
            {"scene_id": 1, "script_text": "b"},
        ])
        outcome = self.validator.parse_and_validate(pj, self.narration, self.images)
        self.assertFalse(outcome.valid)
        self.assertTrue(any("Duplicate scene_id 1" in e for e in outcome.errors))

    def test_unsupported_narration_format(self):
        txt = self.root / "narration.txt"
        txt.write_text("hi", encoding="utf-8")
        pj = self._project([{"scene_id": 1, "script_text": "a"}])
        outcome = self.validator.parse_and_validate(pj, txt, self.images)
        self.assertFalse(outcome.valid)
        self.assertTrue(any(".mp3, .wav or .m4a" in e for e in outcome.errors))

    def test_zero_byte_narration(self):
        empty = self.root / "narration.wav"
        empty.write_bytes(b"")
        pj = self._project([{"scene_id": 1, "script_text": "a"}])
        outcome = self.validator.parse_and_validate(pj, empty, self.images)
        self.assertFalse(outcome.valid)
        self.assertTrue(any("empty (0 bytes)" in e for e in outcome.errors))

    def test_undecodable_narration(self):
        bad = self.root / "narration.mp3"
        bad.write_bytes(b"\x00\xff\xfe" * 200)
        pj = self._project([{"scene_id": 1, "script_text": "a"}])
        outcome = self.validator.parse_and_validate(pj, bad, self.images)
        self.assertFalse(outcome.valid)
        self.assertTrue(any("Narration could not be decoded" in e for e in outcome.errors))

    def test_zero_byte_image(self):
        make_image(self.images / "scene_001.png")
        (self.images / "scene_001.png").write_bytes(b"")
        pj = self._project([{"scene_id": 1, "script_text": "a", "images": ["scene_001.png"]}])
        outcome = self.validator.parse_and_validate(pj, self.narration, self.images)
        self.assertFalse(outcome.valid)
        self.assertTrue(any("empty (0 bytes)" in e for e in outcome.errors))

    def test_corrupt_image_file(self):
        (self.images / "scene_001.png").write_bytes(b"not an image")
        pj = self._project([{"scene_id": 1, "script_text": "a", "images": ["scene_001.png"]}])
        outcome = self.validator.parse_and_validate(pj, self.narration, self.images)
        self.assertFalse(outcome.valid)
        self.assertTrue(any("not a valid image" in e for e in outcome.errors))

    def test_unsupported_image_format_referenced(self):
        (self.images / "scene_001.txt").write_text("x", encoding="utf-8")
        pj = self._project([{"scene_id": 1, "script_text": "a", "images": ["scene_001.txt"]}])
        outcome = self.validator.parse_and_validate(pj, self.narration, self.images)
        self.assertFalse(outcome.valid)
        self.assertTrue(any("unsupported format" in e for e in outcome.errors))

    def test_unused_image_warning(self):
        make_image(self.images / "scene_001.png")
        make_image(self.images / "unused.png")
        pj = self._project([{"scene_id": 1, "script_text": "a", "images": ["scene_001.png"]}])
        outcome = self.validator.parse_and_validate(pj, self.narration, self.images)
        self.assertTrue(outcome.valid)
        self.assertTrue(any("unused.png" in w for w in outcome.warnings))

    def test_webui_style_write_then_validate(self):
        """Simulates the UI flow: JobManager writes uploads, then validate."""
        manager = JobManager(self.root / "workspace")
        job = manager.create_job()
        srcs = [make_image(self.root / f"u{i}.png") for i in range(3)]
        # shuffled upload order
        manager.write_images(job, [
            ("scene_003.png", srcs[2]),
            ("scene_001.png", srcs[0]),
            ("scene_002.png", srcs[1]),
        ])
        manager.write_narration(job, self.narration, "narration.wav")
        pj = write_project_json(job.input_dir, [
            {"scene_id": 1, "script_text": "one", "images": ["scene_001.png"]},
            {"scene_id": 2, "script_text": "two", "images": ["scene_002.png"]},
            {"scene_id": 3, "script_text": "three", "images": ["scene_003.png"]},
        ])
        outcome = self.validator.parse_and_validate(pj, job.input_dir / "narration.wav",
                                                    job.images_dir)
        self.assertTrue(outcome.valid, outcome.errors)
        mapping = {r["scene_id"]: r["images"] for r in outcome.rows}
        self.assertEqual(mapping[3], "scene_003.png")
        self.assertEqual(mapping[1], "scene_001.png")


if __name__ == "__main__":
    unittest.main()
