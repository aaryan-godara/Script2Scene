"""Tests for PipelineRunner: the shared generate function (Whisper mocked)."""

import tempfile
import unittest
from pathlib import Path

from video_assembler.services.alignment.provider_base import (
    TranscribedSegment, TranscribedWord, TranscriptionResult)
from video_assembler.services.job_manager import JobManager
from video_assembler.services.pipeline_runner import PipelineError, PipelineRunner

from tests._helpers import make_image, synth_audio, write_project_json


def fake_transcriber(words, duration=1.0, error=None):
    def transcribe(audio_path):
        if error is not None:
            raise error
        return TranscriptionResult(
            provider="fake", model="fake", device="cpu", language="en",
            audio_duration=duration, processing_seconds=0.0,
            words=[TranscribedWord(word=w, start=s, end=e, confidence=0.99)
                   for w, s, e in words],
            segments=[
                TranscribedSegment(text=" ".join(w for w, _, _ in words),
                                   start=words[0][1], end=words[-1][2],
                                   words=[TranscribedWord(word=w, start=s, end=e, confidence=0.99)
                                          for w, s, e in words]),
            ],
        )
    return transcribe


def one_scene_project(images_dir, script="hello world", scene_id=1, name="food_truck"):
    make_image(images_dir / f"scene_{scene_id:03d}.png")
    return write_project_json(images_dir.parent, [
        {"scene_id": scene_id, "script_text": script, "images": [f"scene_{scene_id:03d}.png"]},
    ], name=name)


class PipelineRunnerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.workspace = self.root / "workspace"
        self.manager = JobManager(self.workspace)
        self.narration = synth_audio(self.root / "narration.wav", 1.0)

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, runner, scenes_project, images_dir, job, **kwargs):
        from video_assembler.services.parser_service import ParserService
        project_input = ParserService().parse_input_json(scenes_project)
        return runner.run(
            project_input=project_input,
            narration=self.narration,
            images_dir=images_dir,
            intermediate_dir=job.intermediate_dir,
            output_dir=job.output_dir,
            logs_dir=job.logs_dir,
            render=True,
            **kwargs,
        )

    def test_run_produces_synchronized_video(self):
        job = self.manager.create_job()
        pj = one_scene_project(job.images_dir)
        runner = PipelineRunner(transcriber=fake_transcriber(
            [("hello", 0.0, 0.3), ("world", 0.3, 0.6)]))
        result = self._run(runner, pj, job.images_dir, job)

        self.assertTrue(result.output_video.is_file())
        meta = result.metadata
        self.assertEqual(meta["resolution"], "1920x1080")
        self.assertEqual(meta["scene_count"], 1)
        self.assertAlmostEqual(meta["fps"], 30.0, delta=0.1)
        self.assertAlmostEqual(meta["duration_s"], 1.0, delta=0.2)
        self.assertIn("Complete.", meta["stages"])
        self.assertTrue((job.intermediate_dir / "timeline.json").is_file())
        self.assertTrue((job.intermediate_dir / "alignment.json").is_file())
        self.assertTrue((job.logs_dir / "pipeline.log").is_file())

    def test_two_jobs_do_not_share_artifacts(self):
        job_a = self.manager.create_job()
        pj_a = one_scene_project(job_a.images_dir, name="truck_a")
        job_b = self.manager.create_job()
        pj_b = one_scene_project(job_b.images_dir, script="bye world", name="truck_b")

        runner = PipelineRunner(transcriber=fake_transcriber(
            [("hello", 0.0, 0.3), ("world", 0.3, 0.6)]))
        res_a = self._run(runner, pj_a, job_a.images_dir, job_a)

        runner_b = PipelineRunner(transcriber=fake_transcriber(
            [("bye", 0.0, 0.3), ("world", 0.3, 0.6)]))
        res_b = self._run(runner_b, pj_b, job_b.images_dir, job_b)

        self.assertNotEqual(res_a.output_video, res_b.output_video)
        self.assertTrue(res_a.output_video.exists())
        self.assertTrue(res_b.output_video.exists())
        # Jobs run in separate directories - no shared artifacts.
        self.assertNotEqual(job_a.root, job_b.root)
        self.assertNotEqual(job_a.intermediate_dir, job_b.intermediate_dir)
        # Each job has its own timeline; the contents are not shared/reused.
        import json as _json
        ta = _json.loads((job_a.intermediate_dir / "timeline.json").read_text(encoding="utf-8"))
        tb = _json.loads((job_b.intermediate_dir / "timeline.json").read_text(encoding="utf-8"))
        self.assertNotEqual(ta["project"], tb["project"])
        # Nothing leaked outside the job dirs into a shared path.
        self.assertEqual(set(self.manager.jobs_root.glob("*/output/final_video.mp4")),
                         {res_a.output_video, res_b.output_video})

    def test_alignment_failure_stops_generation(self):
        job = self.manager.create_job()
        pj = one_scene_project(job.images_dir)
        runner = PipelineRunner(transcriber=fake_transcriber([("zzz", 0.0, 0.5)]))
        with self.assertRaises(PipelineError) as ctx:
            self._run(runner, pj, job.images_dir, job)
        self.assertIn("could not be aligned confidently", str(ctx.exception))

    def test_review_stops_unless_allowed(self):
        job = self.manager.create_job()
        pj = one_scene_project(job.images_dir, script="hello earth")
        runner = PipelineRunner(transcriber=fake_transcriber(
            [("hello", 0.0, 0.3), ("world", 0.3, 0.6)]))
        with self.assertRaises(PipelineError) as ctx:
            self._run(runner, pj, job.images_dir, job)
        self.assertIn("REVIEW", str(ctx.exception))

        # allow_review lets the pipeline continue.
        result = self._run(runner, pj, job.images_dir, job, allow_review=True)
        self.assertTrue(result.output_video.is_file())

    def test_backend_exception_maps_to_friendly_error(self):
        job = self.manager.create_job()
        pj = one_scene_project(job.images_dir)
        runner = PipelineRunner(
            transcriber=fake_transcriber([("hello", 0.0, 0.3)],
                                         error=RuntimeError("boom")))
        with self.assertRaises(PipelineError) as ctx:
            self._run(runner, pj, job.images_dir, job)
        self.assertIn("Whisper transcription failed", str(ctx.exception))
        # Full technical detail is in the job log, not the user-facing message.
        log = (job.logs_dir / "pipeline.log").read_text(encoding="utf-8")
        self.assertIn("boom", log)

    def test_validation_error_is_user_friendly(self):
        job = self.manager.create_job()
        pj = write_project_json(job.images_dir, [
            {"scene_id": 1, "script_text": "a", "images": ["missing.png"]},
        ])
        runner = PipelineRunner(transcriber=fake_transcriber([("a", 0.0, 0.3)]))
        with self.assertRaises(PipelineError) as ctx:
            self._run(runner, pj, job.images_dir, job)
        self.assertIn("Project validation failed", str(ctx.exception))
        self.assertIn("missing.png", str(ctx.exception))

    def test_no_render_writes_timeline_only(self):
        job = self.manager.create_job()
        pj = one_scene_project(job.images_dir)
        runner = PipelineRunner(transcriber=fake_transcriber(
            [("hello", 0.0, 0.3), ("world", 0.3, 0.6)]))
        result = runner.run(
            project_input=__import__("video_assembler.services.parser_service",
                                     fromlist=["ParserService"]).ParserService()
            .parse_input_json(pj),
            narration=self.narration,
            images_dir=job.images_dir,
            intermediate_dir=job.intermediate_dir,
            output_dir=job.output_dir,
            logs_dir=job.logs_dir,
            render=False,
        )
        self.assertIsNone(result.output_video)
        self.assertTrue((job.intermediate_dir / "timeline.json").is_file())
        self.assertFalse(any(job.output_dir.iterdir()))


if __name__ == "__main__":
    unittest.main()
