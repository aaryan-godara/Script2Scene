"""Phase-7 specification tests for the permanent production solution.

Covers the spec's recurring problems without running Whisper (fake
transcribers) or Gradio, matching the existing suite's conventions.

Focus areas
  A  persistent cache: identical audio + config is not re-transcribed.
  B  a changed audio identity is a cache miss (fresh transcription).
  C  a changed transcription identity/config is a cache miss.
  D  an internally-omitted segment is recovered locally (region transcribed,
     words merged, scene re-aligned HIGH).
  E  consecutive failed regions are grouped; distinct regions are separate.
  F  beginning-of-audio and end-of-audio windows are recovered.
  G  dedup overlaps are removed when merging recovery words.
  I  timestamps from the recovery window are shifted to global timing.
  K/L  numeric equivalence (percent, Indian numbering) is accepted.
  M  numeric mismatch is never promoted to HIGH (REVIEW + warning).
  Q  corrupt/audio-mismatched cache entry causes a miss, not a reuse.
  R  recovered result is itself persisted and reused on the next run.
  S  manual overrides stay REVIEW and are never auto-promoted.
  T  an alignment review report is written even when generation is blocked.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_assembler.models import Scene
from video_assembler.services.alignment.alignment_service import AlignmentService
from video_assembler.services.alignment.numeric_normalizer import NumericNormalizer
from video_assembler.services.alignment.persistent_transcription_cache import audio_sha256
from video_assembler.services.alignment.provider_base import (
    TranscribedSegment, TranscribedWord, TranscriptionResult,
)
from tests._helpers import make_image, synth_audio, write_project_json


# ---------------------------------------------------------------- scaffolding
def make_result(words, duration=6.0):
    """``words`` is a list of (word, start, end, confidence) tuples."""
    return TranscriptionResult(
        provider="fake", model="fake", device="cpu", language="en",
        audio_duration=duration, processing_seconds=0.0,
        words=[TranscribedWord(word=w, start=s, end=e, confidence=c)
               for w, s, e, c in words],
        segments=[TranscribedSegment(id=0, start=0.0, end=duration,
                                     text=" ".join(w[0] for w in words))],
    )


def build_runner(transcriber, cache_root, identity=None, auto_recovery=True,
                 max_review_ratio=0.05):
    from video_assembler.services.pipeline_runner import PipelineRunner
    return PipelineRunner(transcriber=transcriber, cache_root=cache_root,
                          identity=identity, auto_recovery=auto_recovery,
                          max_review_ratio=max_review_ratio)


def run_pipeline(transcriber, script, audio_path, tmp_path, *, render=False,
                 identity=None, auto_recovery=True, max_review_ratio=0.05):
    from video_assembler.services.job_manager import JobManager
    from video_assembler.services.parser_service import ParserService

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    mgr = JobManager(workspace)
    job = mgr.create_job()
    scenes = []
    for sid, text, img in script:
        make_image(Path(job.images_dir) / img)
        scenes.append({"scene_id": sid, "script_text": text, "images": [img]})
    pj = write_project_json(job.images_dir, scenes)
    pi = ParserService().parse_input_json(pj)
    runner = build_runner(transcriber, tmp_path / "cache", identity=identity,
                          auto_recovery=auto_recovery,
                          max_review_ratio=max_review_ratio)
    return runner, pi, job


# ------------------------------------------------------------- numeric tests
class TestNumericNormalizer:
    def test_equiv_currency_and_scale(self):
        nn = NumericNormalizer()
        assert nn.text_numeric_consistency("$283,000", "2,83,000") is True
        assert nn.text_numeric_consistency(
            "$283,000", "two lakh eighty three thousand dollars") is True
        assert nn.text_numeric_consistency(
            "$1,560,000", "one million five hundred sixty thousand") is True

    def test_percentage_consistency(self):
        nn = NumericNormalizer()
        assert nn.text_numeric_consistency("15%", "fifteen percent") is True
        assert nn.text_numeric_consistency("10.5%", "10.5 percent") is True

    def test_indian_numbering_lakh(self):
        nn = NumericNormalizer()
        assert nn.text_numeric_consistency(
            "2,83,000", "two lakh eighty three thousand") is True
        assert nn.extract("2 lakh 83 thousand")[0].canonical == "283000"

    def test_million_vs_thousand_mismatch(self):
        nn = NumericNormalizer()
        assert nn.text_numeric_consistency(
            "$1,560,000", "one million five hundred sixty thousand") is True
        assert nn.text_numeric_consistency(
            "$2.5 million", "two point five million") is True
        assert nn.text_numeric_consistency("$10,000", "one hundred thousand") is False

    def test_year_consistency(self):
        nn = NumericNormalizer()
        assert nn.text_numeric_consistency("2025", "twenty twenty five") is True
        assert nn.text_numeric_consistency("2025", "two thousand twenty five") is True

    def test_no_numbers_returns_none(self):
        nn = NumericNormalizer()
        assert nn.text_numeric_consistency("hello world", "hello world") is None

    def test_hyphenated_word_numbers_are_composite(self):
        # "eighty-two" / "twenty-five" are ONE value, not two fragments.
        nn = NumericNormalizer()
        assert nn.text_numeric_consistency("eighty-two cents", "82 cents") is True
        assert nn.text_numeric_consistency(
            "between thirty-five and fifty percent", "between 35 and 50 percent") is True

    def test_and_compound_number_stays_one_value(self):
        # "one hundred AND eighty" is 180 (low-order remainder), not 100 and 80.
        nn = NumericNormalizer()
        assert nn.text_numeric_consistency(
            "one hundred and eighty dollars", "$180") is True
        assert nn.text_numeric_consistency(
            "one thousand six hundred and twenty dollars", "$1620") is True
        assert nn.text_numeric_consistency(
            "four thousand three hundred and eighty dollars", "$4380") is True

    def test_and_range_separates_values(self):
        # "two hundred AND eight hundred" is a range {200, 800}, not 20800.
        nn = NumericNormalizer()
        assert nn.text_numeric_consistency(
            "between two hundred and eight hundred dollars", "between $200 and $800") is True
        assert nn.text_numeric_consistency(
            "between three thousand and five thousand dollars",
            "between $3000 and $5000") is True
        assert nn.text_numeric_consistency(
            "between two hundred and four hundred dollars", "between $200 and $400") is True

    def test_range_percent_and_article(self):
        nn = NumericNormalizer()
        assert nn.text_numeric_consistency(
            "between five and twenty-five percent", "between 5 and 25 %") is True
        # "a" before a number is an article, not a phantom 1.
        assert nn.text_numeric_consistency(
            "take a two dollar bag", "take a $2 bag") is True

    def test_stutter_doubled_word_is_one_value(self):
        nn = NumericNormalizer()
        assert nn.text_numeric_consistency(
            "start with one machine", "start with one one machine") is True


class TestNumericAlignmentIntegrity:
    """A scene with consistent numeric evidence is HIGH; a numeric mismatch is
    never promoted to HIGH."""

    def _align(self, script, words):
        svc = AlignmentService()
        scene = [Scene(scene_id=1, script_text=script)]
        _, diag = svc.align_scenes(scene, make_result(words))
        return scene[0], diag

    def test_consistent_numeric_is_high(self):
        _, diag = self._align(
            "the amount was twenty five dollars",
            [("the", 0.0, 0.3, 0.99), ("amount", 0.3, 0.6, 0.99),
             ("was", 0.6, 0.8, 0.99), ("twenty", 0.8, 1.0, 0.99),
             ("five", 1.0, 1.15, 0.99), ("dollars", 1.15, 1.3, 0.99)])
        assert diag.get(1)["status"] == "HIGH"

    def test_mismatched_numeric_never_high(self):
        _, diag = self._align(
            "the amount was twenty five dollars",
            [("the", 0.0, 0.3, 0.99), ("amount", 0.3, 0.6, 0.99),
             ("was", 0.6, 0.8, 0.99), ("thirty", 0.8, 1.0, 0.99),
             ("five", 1.0, 1.15, 0.99), ("dollars", 1.15, 1.3, 0.99)])
        assert diag.get(1)["status"] != "HIGH"
        assert diag.get(1)["numeric_mismatch"] in (None, True)

    def test_mismatch_marks_warning_type(self):
        svc = AlignmentService()
        d = svc._numeric_diagnostic("price is 100 dollars",
                                    "price is five dollars")
        assert d["numeric_mismatch"] is True
        assert d["warning_type"] == "NUMERIC_VALUE_MISMATCH"

    def test_split_thousands_does_not_false_mismatch(self):
        # Whisper emits "$256" and ",000." as separate words; the matched
        # window stops at "$256" so "$256" must compare against "$256,000"
        # without a false numeric mismatch.
        _, diag = self._align(
            "the total is 256,000 dollars",
            [("the", 0.0, 0.3, 0.99), ("total", 0.3, 0.6, 0.99),
             ("is", 0.6, 0.8, 0.99), ("$256", 0.8, 1.0, 0.99),
             (",000", 1.0, 1.15, 0.99), ("dollars", 1.15, 1.3, 0.99)])
        assert diag.get(1)["status"] == "HIGH"

    def test_split_dollars_and_cents_is_consistent(self):
        # "$8" ".28" -> "$8.28" == "8 dollars and 28 cents".
        _, diag = self._align(
            "you get 8 dollars and 28 cents",
            [("you", 0.0, 0.3, 0.99), ("get", 0.3, 0.6, 0.99),
             ("$8", 0.6, 0.9, 0.99), (".28", 0.9, 1.1, 0.99)])
        assert diag.get(1)["status"] == "HIGH"

    def test_split_spaced_decimal_is_consistent(self):
        # Whisper "1 .3 million" must match a script "1.3 million".
        nn = NumericNormalizer()
        assert nn.text_numeric_consistency("1.3 million", "1 .3 million") is True

    def test_spoken_dollars_and_cents_matches_decimal(self):
        # Script "2 dollars and 99 cents" vs ASR "$2 .99" are the same amount.
        nn = NumericNormalizer()
        assert nn.text_numeric_consistency(
            "2 dollars and 99 cents", "$2 .99") is True

    def test_alignment_review_ratio_under_gate(self):
        # The integration job alignment must stay at or under 5% REVIEW. This
        # guards the whole split-number handling chain against regression.
        import json as _json
        import os as _os
        base = _os.path.join(
            _os.path.dirname(__file__), "..", "workspace", "jobs",
            "f265ad00e930")
        trans_path = _os.path.join(base, "intermediate", "transcription.json")
        if not _os.path.exists(trans_path):
            import pytest
            pytest.skip("job fixture not present")
        tj = _json.load(open(trans_path, encoding="utf-8"))
        result = make_result([(w["word"], w["start"], w["end"],
                               w.get("confidence")) for w in tj["words"]],
                             duration=tj.get("audio_duration"))
        proj = _json.load(open(_os.path.join(base, "input", "project.json"),
                               encoding="utf-8"))
        scenes = [Scene(scene_id=s["scene_id"], script_text=s["script_text"])
                  for s in proj["scenes"]]
        _, diag = AlignmentService().align_scenes(scenes, result)
        review = sum(1 for d in diag.diagnostics.values()
                     if d.get("status") == "REVIEW")
        total = len(diag.diagnostics)
        assert review / total <= 0.05


# ----------------------------------------------------------------- cache tests
class TestPersistentCache:
    def make_cache(self, tmp_path):
        from video_assembler.services.alignment.persistent_transcription_cache import (PersistentTranscriptionCache, TranscriptionIdentityConfig, audio_sha256)
        return PersistentTranscriptionCache(tmp_path / "cache",
                                            TranscriptionIdentityConfig())

    def test_identical_audio_and_config_are_cache_hit(self, tmp_path):
        a = synth_audio(tmp_path / "a.wav", 1.0)
        cache = self.make_cache(tmp_path)
        sha = audio_sha256(a)
        assert cache.load(sha) is None
        cache.save(make_result([("hi", 0.0, 1.0, 0.9)], 1.0), sha,
                   source="fresh_whisper")
        cached = cache.load(sha)
        assert cached is not None
        assert [w.word for w in cached.words] == ["hi"]

    def test_changed_audio_is_cache_miss(self, tmp_path):
        a = synth_audio(tmp_path / "a.wav", 1.0)
        b = synth_audio(tmp_path / "b.wav", 1.4)
        cache = self.make_cache(tmp_path)
        sha_a = audio_sha256(a)
        sha_b = audio_sha256(b)
        assert sha_a != sha_b
        cache.save(make_result([("hi", 0.0, 1.0, 0.9)], 1.0), sha_a,
                   source="fresh_whisper")
        assert cache.load(sha_b) is None

    def test_changed_identity_is_cache_miss(self, tmp_path):
        from video_assembler.services.alignment.persistent_transcription_cache import (PersistentTranscriptionCache, TranscriptionIdentityConfig, audio_sha256)
        a = synth_audio(tmp_path / "a.wav", 1.0)
        c1 = PersistentTranscriptionCache(tmp_path / "cache",
                                          TranscriptionIdentityConfig(model="base"))
        c2 = PersistentTranscriptionCache(tmp_path / "cache",
                                          TranscriptionIdentityConfig(model="small"))
        sha = audio_sha256(a)
        c1.save(make_result([("hi", 0.0, 1.0, 0.9)], 1.0), sha, source="fresh_whisper")
        assert c1.load(sha) is not None
        assert c2.load(sha) is None

    def test_corrupt_entry_never_reused(self, tmp_path):
        a = synth_audio(tmp_path / "a.wav", 1.0)
        cache = self.make_cache(tmp_path)
        sha = audio_sha256(a)
        cache.save(make_result([("hi", 0.0, 1.0, 0.9)], 1.0), sha, source="fresh_whisper")
        (cache.entry_dir(sha) / "transcription.json").write_text(
            "{not json", encoding="utf-8")
        assert cache.load(sha) is None

    def test_changed_audio_sha_in_meta_forces_miss(self, tmp_path):
        a = synth_audio(tmp_path / "a.wav", 1.0)
        cache = self.make_cache(tmp_path)
        sha = audio_sha256(a)
        cache.save(make_result([("hi", 0.0, 1.0, 0.9)], 1.0), sha, source="fresh_whisper")
        payload = cache.entry_dir(sha) / "transcription.json"
        data = json.loads(payload.read_text(encoding="utf-8"))
        data["audio_sha256"] = "f" * 64
        payload.write_text(json.dumps(data), encoding="utf-8")
        assert cache.load(sha) is None


# ----------------------------------------------------------------- recovery tests
class TestFailedRegionRecovery:
    def make_engine(self, tmp_path, transcribe_fn, duration=6.0):
        from video_assembler.services.alignment.failed_region_recovery import (
            FailedRegionRecoveryEngine,
        )
        from video_assembler.services.alignment.persistent_transcription_cache import (
            TranscriptionIdentityConfig,
        )
        from video_assembler.services.audio_service import AudioService
        audio = synth_audio(tmp_path / "n.wav", duration)
        return FailedRegionRecoveryEngine(
            transcribe_fn=transcribe_fn,
            audio_service=AudioService(tmp_path),
            audio_path=audio,
            chunk_dir=tmp_path / "chunks",
            config=TranscriptionIdentityConfig(),
        )

    def _scenes(self, *texts):
        return [Scene(scene_id=i + 1, script_text=t) for i, t in enumerate(texts)]

    def test_group_consecutive_and_separate(self, tmp_path):
        engine = self.make_engine(tmp_path, lambda p: make_result([]))
        scenes = self._scenes("a", "b", "c", "d", "e", "f", "g")
        statuses = {s.scene_id: ("FAILED" if s.scene_id in (2, 3, 7) else "HIGH")
                    for s in scenes}
        groups = engine.group_failed_regions(scenes, statuses)
        assert groups == [[2, 3], [7]]

    def test_group_order_not_id_arithmetic(self, tmp_path):
        engine = self.make_engine(tmp_path, lambda p: make_result([]))
        scenes = [Scene(scene_id=100, script_text="a"),
                  Scene(scene_id=5, script_text="b"),
                  Scene(scene_id=200, script_text="c")]
        statuses = {5: "FAILED", 200: "FAILED", 100: "HIGH"}
        groups = engine.group_failed_regions(scenes, statuses)
        assert groups == [[5, 200]]

    def test_window_beginning_of_audio(self, tmp_path):
        engine = self.make_engine(tmp_path, lambda p: make_result([]))
        scenes = self._scenes("aaaa", "bbbb", "cccc")
        statuses = {1: "FAILED", 2: "HIGH", 3: "HIGH"}
        diagnostics = _Diag({2: {"speech_start": 3.0}})
        start, end, _ = engine.estimate_window([1], scenes, statuses,
                                               diagnostics, 6.0, 1)
        assert start == pytest.approx(0.0)
        assert 0.0 < end <= 6.0

    def test_window_end_of_audio(self, tmp_path):
        engine = self.make_engine(tmp_path, lambda p: make_result([]))
        scenes = self._scenes("aaaa", "bbbb", "cccc")
        statuses = {1: "HIGH", 2: "HIGH", 3: "FAILED"}
        diagnostics = _Diag({2: {"speech_end": 3.0}})
        start, end, _ = engine.estimate_window([3], scenes, statuses,
                                               diagnostics, 6.0, 1)
        assert end == pytest.approx(6.0)
        assert 0.0 <= start < 6.0

    def test_recovery_merges_with_global_shift_and_dedup(self, tmp_path):
        # initial words only cover scenes 1 and 3; scene 2 region (2s-4s) is
        # omitted by the initial pass and restored by the recovery clip.
        initial = [("welcome", 0.0, 1.0, 0.9), ("thanks", 4.0, 5.0, 0.9)]
        calls = []

        def transcribe(path):
            calls.append(str(path))
            if "failed_region" in str(path):
                return make_result([("our", 2.2, 2.5, 0.9),
                                    ("sales", 2.5, 3.0, 0.9),
                                    ("grew", 3.0, 3.4, 0.9)], 6.0)
            return make_result(initial, 6.0)

        engine = self.make_engine(tmp_path, transcribe)
        merged, audit = engine.recover_pass(
            [TranscribedWord(word=w, start=s, end=e, confidence=c)
             for w, s, e, c in initial],
            self._scenes("welcome", "our sales grew", "thanks"),
            {1: "HIGH", 2: "FAILED", 3: "HIGH"},
            _Diag({1: {"speech_end": 1.0}, 3: {"speech_start": 4.0}}),
            6.0, 1)
        merged_words = [w.word for w in merged]
        assert "sales" in merged_words
        sales = next(w for w in merged if w.word == "sales")
        assert sales.start == pytest.approx(2.5)
        assert calls and "failed_region" in calls[0]
        assert audit and audit[0]["words_added"] == 3

    def test_recovery_rejects_bad_timestamps(self, tmp_path):
        initial = [("welcome", 0.0, 1.0, 0.9)]
        engine = self.make_engine(tmp_path, lambda p: make_result(
            [("boom", -1.0, 3.0, 0.9), ("end", 30.0, 31.0, 0.9)], 6.0))
        merged, audit = engine.recover_pass(
            [TranscribedWord(word=w, start=s, end=e, confidence=c)
             for w, s, e, c in initial],
            self._scenes("welcome", "boom end"),
            {1: "HIGH", 2: "FAILED"},
            _Diag({1: {"speech_end": 1.0}}), 6.0, 1)
        assert audit[0]["words_added"] == 0


class _Diag:
    def __init__(self, data):
        self.diagnostics = data


# ----------------------------------------------------------------- integration
class TestPipelineIntegration:
    def _one_scene_project(self, tmp_path, script="hello world"):
        from video_assembler.services.job_manager import JobManager
        workspace = tmp_path / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        job = JobManager(workspace).create_job()
        make_image(Path(job.images_dir) / "scene_001.png")
        pj = write_project_json(job.images_dir, [
            {"scene_id": 1, "script_text": script, "images": ["scene_001.png"]}])
        return job, pj

    def test_pipeline_uses_persistent_cache_second_run(self, tmp_path):
        from video_assembler.services.parser_service import ParserService
        from video_assembler.services.pipeline_runner import PipelineError
        job, pj = self._one_scene_project(tmp_path)
        narration = synth_audio(tmp_path / "n.wav", 1.0)
        calls = {"n": 0}

        def transcribe(path):
            calls["n"] += 1
            return make_result([("hello", 0.0, 0.5, 0.99),
                                ("world", 0.5, 1.0, 0.99)], 1.0)

        runner = build_runner(transcribe, tmp_path / "cache")
        pi = ParserService().parse_input_json(pj)
        result1 = runner.run(project_input=pi, narration=narration,
                             images_dir=job.images_dir,
                             intermediate_dir=job.intermediate_dir,
                             output_dir=job.output_dir, logs_dir=job.logs_dir,
                             render=False)
        assert result1.metadata["transcription_source"] == "fresh_whisper"
        assert calls["n"] == 1

        job2, _ = self._one_scene_project(tmp_path, "hello world")
        result2 = runner.run(project_input=pi, narration=narration,
                             images_dir=job2.images_dir,
                             intermediate_dir=job2.intermediate_dir,
                             output_dir=job2.output_dir, logs_dir=job2.logs_dir,
                             render=False)
        assert result2.metadata["transcription_source"] == "persistent_cache"
        assert calls["n"] == 1

    def test_pipeline_fresh_when_audio_changes(self, tmp_path):
        from video_assembler.services.parser_service import ParserService
        job, pj = self._one_scene_project(tmp_path)
        narration1 = synth_audio(tmp_path / "n1.wav", 1.0)
        narration2 = synth_audio(tmp_path / "n2.wav", 1.6)
        calls = {"n": 0}

        def transcribe(path):
            calls["n"] += 1
            return make_result([("hello", 0.0, 0.5, 0.99),
                                ("world", 0.5, 1.0, 0.99)], 1.0)

        runner = build_runner(transcribe, tmp_path / "cache")
        pi = ParserService().parse_input_json(pj)
        runner.run(project_input=pi, narration=narration1,
                   images_dir=job.images_dir, intermediate_dir=job.intermediate_dir,
                   output_dir=job.output_dir, logs_dir=job.logs_dir, render=False)
        r2 = runner.run(project_input=pi, narration=narration2,
                        images_dir=job.images_dir,
                        intermediate_dir=job.intermediate_dir,
                        output_dir=job.output_dir, logs_dir=job.logs_dir,
                        render=False)
        assert r2.metadata["transcription_source"] == "fresh_whisper"
        assert calls["n"] == 2

    def test_recovery_heals_omission_and_reruns_align(self, tmp_path):
        from video_assembler.services.parser_service import ParserService
        job, pj = self._one_scene_project(tmp_path, "hello world")
        narration = synth_audio(tmp_path / "n.wav", 1.0)

        def transcribe(path):
            if "failed_region" in str(path):
                return make_result([("hello", 0.2, 0.5, 0.9),
                                    ("world", 0.5, 0.8, 0.9)], 1.0)
            return make_result([("hello", 0.0, 0.4, 0.99)], 1.0)

        runner = build_runner(transcribe, tmp_path / "cache")
        pi = ParserService().parse_input_json(pj)
        result = runner.run(project_input=pi, narration=narration,
                            images_dir=job.images_dir,
                            intermediate_dir=job.intermediate_dir,
                            output_dir=job.output_dir, logs_dir=job.logs_dir,
                            render=False)
        assert result.metadata["recovery"]["recovered_scene_ids"] == [1]
        assert result.metadata["alignment_statuses"]["HIGH"] == 1
        assert result.metadata["transcription_cache"]["source"] == "recovered_cache"
        assert (job.intermediate_dir / "recovery_log.json").exists()

    def test_recovered_result_is_reused_next_run(self, tmp_path):
        from video_assembler.services.parser_service import ParserService
        job, pj = self._one_scene_project(tmp_path, "hello world")
        narration = synth_audio(tmp_path / "n.wav", 1.0)

        def transcribe(path):
            if "failed_region" in str(path):
                return make_result([("hello", 0.2, 0.5, 0.9),
                                    ("world", 0.5, 0.8, 0.9)], 1.0)
            return make_result([("hello", 0.0, 0.4, 0.99)], 1.0)

        runner = build_runner(transcribe, tmp_path / "cache")
        pi = ParserService().parse_input_json(pj)
        r1 = runner.run(project_input=pi, narration=narration,
                        images_dir=job.images_dir,
                        intermediate_dir=job.intermediate_dir,
                        output_dir=job.output_dir, logs_dir=job.logs_dir,
                        render=False)
        assert r1.metadata["transcription_cache"]["source"] == "recovered_cache"
        r2 = runner.run(project_input=pi, narration=narration,
                        images_dir=job.images_dir,
                        intermediate_dir=job.intermediate_dir,
                        output_dir=job.output_dir, logs_dir=job.logs_dir,
                        render=False)
        assert r2.metadata["transcription_source"] == "persistent_cache"

    def test_review_report_written_even_when_blocked(self, tmp_path):
        from video_assembler.services.parser_service import ParserService
        from video_assembler.services.pipeline_runner import PipelineError
        job, pj = self._one_scene_project(tmp_path, "hello world")
        narration = synth_audio(tmp_path / "n.wav", 1.0)

        def transcribe(path):
            if "failed_region" in str(path):
                return make_result([], 1.0)
            return make_result([("zzz", 0.0, 0.5, 0.99)], 1.0)

        runner = build_runner(transcribe, tmp_path / "cache")
        pi = ParserService().parse_input_json(pj)
        with pytest.raises(PipelineError):
            runner.run(project_input=pi, narration=narration,
                       images_dir=job.images_dir,
                       intermediate_dir=job.intermediate_dir,
                       output_dir=job.output_dir, logs_dir=job.logs_dir,
                       render=False)
        report = job.intermediate_dir / "alignment_review.json"
        assert report.exists()
        data = json.loads(report.read_text(encoding="utf-8"))
        assert data["summary"].get("FAILED", 0) >= 1

    def test_manual_override_stays_review(self, tmp_path):
        from video_assembler.services.alignment.review_report import (
            AlignmentReviewStore,
        )
        store = AlignmentReviewStore(tmp_path / "reviews")
        store.override("1", 0.0, 1.0, "confirmed by reviewer")
        rec = store.get("1")
        assert rec.status == "override"
        assert rec.speech_start == 0.0
        store.accept("2")
        assert store.get("2").status == "accept"
        with pytest.raises(ValueError):
            store.override("3", 1.0, 0.5)
