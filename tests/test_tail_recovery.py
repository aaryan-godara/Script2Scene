"""Focused tests for EOF tail-recovery transcription.

Covers: no recovery when the transcript reaches audio EOF, no recovery when the
remaining tail is silent, recovery when the tail has acoustic content,
overlap deduplication of the recovery clip, local->global timestamp conversion,
final ordering, and cache-identity rejection of pre-tail-recovery caches.
"""

import unittest
from pathlib import Path

from video_assembler.services.alignment.chunked_transcription_provider import (
    ChunkedTranscriptionProvider, ChunkingConfig)
from video_assembler.services.alignment.provider_base import (
    TranscribedWord, TranscriptionProvider, TranscriptionResult)
from video_assembler.services.alignment.transcription_cache import (
    TranscriptionCacheError, load_transcription, save_transcription,
    transcription_is_current)


def _tw(word, start, end):
    return TranscribedWord(word=word, start=start, end=end, confidence=0.99)


class TailRecoveryInner(TranscriptionProvider):
    """Scripted inner provider: per-chunk words + optional recovery clip words.

    chunk_words: {global_chunk_start: [(word, local_start, local_end), ...]}
    recovery_words: [(word, local_start, local_end), ...] inside the tail clip.
    """

    def __init__(self, chunk_words, recovery_words=None, duration=1.0):
        self.chunk_words = chunk_words
        self.recovery_words = recovery_words or []
        self.duration = duration
        self.calls = []
        self.provider = "fake"
        self.model = "fake"
        self.model_name = "fake"
        self.device = "cpu"

    def transcribe(self, audio_path: str) -> TranscriptionResult:
        name = Path(audio_path).name
        self.calls.append(name)
        if "tail_recovery" in name:
            words = self.recovery_words
        elif "chunk_" in name:
            parts = name.split("_")
            g_start = float(parts[2])
            words = self.chunk_words.get(g_start, [])
        else:
            words = []
        tws = [_tw(w, s, e) for w, s, e in words]
        return TranscriptionResult(
            provider="fake", model="fake", device="cpu", language="en",
            audio_duration=self.duration, processing_seconds=0.0,
            words=tws, segments=[])


class TailRecoveryAudioService:
    """Mimics AudioService chunk slicing + energy check without FFmpeg."""

    def __init__(self, duration, chunk_duration=180.0, overlap=10.0,
                 energy_result=True):
        self.duration = duration
        self.chunk_duration = chunk_duration
        self.overlap = overlap
        self.energy_result = energy_result
        self.recovery_extract = None

    def get_audio_metadata(self, audio_path):
        return {"duration": self.duration}

    def extract_chunks(self, source_audio, chunk_dir, chunk_duration, overlap):
        chunk_dir = Path(chunk_dir)
        step = chunk_duration - overlap
        start = 0.0
        idx = 0
        chunks = []
        while start < self.duration - 1e-6:
            end = min(start + chunk_duration, self.duration)
            name = f"chunk_{idx:03d}_{start:07.3f}_{end:07.3f}.wav"
            chunks.append((chunk_dir / name, start, end))
            idx += 1
            start += step
        return chunks

    def extract_chunk(self, source_audio, output_path, start, end):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"tail-recovery-clip")
        self.recovery_extract = (float(start), float(end))
        return output_path

    def tail_has_acoustic_energy(self, audio_path, tail_start, end_seconds=None):
        return self.energy_result


class TailRecoveryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path("tmp_tail_recovery_tests")
        self.tmp.mkdir(exist_ok=True)
        self.narration = self.tmp / "nar.mp3"
        self.narration.write_bytes(b"fake-audio-bytes")

    def _run(self, duration, chunk_words, recovery_words=None, energy=True):
        inner = TailRecoveryInner(chunk_words, recovery_words)
        svc = TailRecoveryAudioService(duration, energy_result=energy)
        cfg = ChunkingConfig(long_audio_threshold_seconds=0)
        provider = ChunkedTranscriptionProvider(inner, config=cfg, audio_service=svc)
        result = provider.transcribe(str(self.narration),
                                     chunk_dir=str(self.tmp / "chunks"))
        return provider, inner, svc, result

    def test_transcript_reaching_audio_end_no_recovery(self):
        # chunk0: 0-180, chunk1: 170-190; chunk1 reaches to ~189.9
        _, inner, _, result = self._run(
            190.0,
            {0.0: [("a", 0.0, 0.4), ("b", 0.5, 1.0)],
             170.0: [("c", 19.0, 19.5), ("d", 19.6, 19.9)]})
        tail = result.tail_recovery
        self.assertFalse(tail["tail_recovery_triggered"])
        self.assertAlmostEqual(result.words[-1].end, 189.9, places=3)
        self.assertNotIn("tail_recovery.wav", inner.calls)
        self.assertFalse(tail["tail_energy_detected"])

    def test_early_end_with_silent_tail_no_recovery(self):
        # merged ends at 170.4; tail [170.4, 190] is silent -> no recovery
        _, inner, _, result = self._run(
            190.0,
            {0.0: [("a", 0.0, 0.4), ("b", 0.5, 1.0), ("c", 1.1, 1.5)],
             170.0: [("z", 0.0, 0.4)]},
            energy=False)
        tail = result.tail_recovery
        self.assertGreater(tail["tail_gap_seconds"], 2.0)
        self.assertFalse(tail["tail_recovery_triggered"])
        self.assertFalse(tail["tail_energy_detected"])
        self.assertNotIn("tail_recovery.wav", inner.calls)
        self.assertEqual(result.words[-1].end, 170.4)

    def test_early_end_with_acoustic_tail_triggers_recovery(self):
        # merged ends at 170.4; tail has energy -> recovery over [100, 190]
        _, inner, svc, result = self._run(
            190.0,
            {0.0: [("a", 0.0, 0.4), ("b", 0.5, 1.0), ("c", 1.1, 1.5)],
             170.0: [("m", 0.0, 0.4)]},
            recovery_words=[("x", 0.0, 0.5), ("y", 0.6, 1.0), ("z", 88.0, 88.5)],
            energy=True)
        tail = result.tail_recovery
        self.assertTrue(tail["tail_recovery_triggered"])
        self.assertTrue(tail["tail_energy_detected"])
        self.assertEqual(tail["tail_recovery_start"], 100.0)
        self.assertEqual(tail["tail_recovery_end"], 190.0)
        self.assertEqual(tail["tail_recovery_raw_words"], 3)
        self.assertEqual(tail["tail_recovery_words_added"], 3)
        self.assertEqual(tail["final_word_end_before_recovery"], 170.4)
        self.assertEqual(tail["final_word_end_after_recovery"], 188.5)
        self.assertIn("tail_recovery.wav", inner.calls)
        self.assertEqual(svc.recovery_extract, (100.0, 190.0))

    def test_overlap_duplicates_removed_and_tail_added(self):
        # chunk1 returns nothing; merged = a b c d ends at 7s; recovery re-captures
        # c d (overlap) and adds e f (missing tail) -> no duplicated c d
        _, inner, _, result = self._run(
            190.0,
            {0.0: [("a", 0.0, 1.0), ("b", 2.0, 3.0), ("c", 4.0, 5.0), ("d", 6.0, 7.0)]},
            recovery_words=[("c", 0.5, 1.5), ("d", 2.0, 3.0), ("e", 4.0, 5.0), ("f", 6.0, 7.0)],
            energy=True)
        tail = result.tail_recovery
        self.assertTrue(tail["tail_recovery_triggered"])
        self.assertEqual([w.word for w in result.words], ["a", "b", "c", "d", "e", "f"])
        self.assertEqual(tail["tail_recovery_words_added"], 2)
        self.assertIn("tail_recovery.wav", inner.calls)

    def test_recovery_timestamps_converted_to_global(self):
        # recovery clip starts at 100; local word at 76.6 -> global 176.6
        _, inner, _, result = self._run(
            190.0,
            {0.0: [("a", 0.0, 1.0)]},
            recovery_words=[("w", 76.6, 77.0)],
            energy=True)
        tail = result.tail_recovery
        self.assertTrue(tail["tail_recovery_triggered"])
        w = [x for x in result.words if x.word == "w"][0]
        self.assertAlmostEqual(w.start, 176.6, places=3)
        self.assertAlmostEqual(w.end, 177.0, places=3)

    def test_final_merged_timestamps_remain_ordered(self):
        _, _, _, result = self._run(
            190.0,
            {0.0: [("a", 0.0, 1.0), ("b", 2.0, 3.0), ("c", 4.0, 5.0), ("d", 6.0, 7.0)]},
            recovery_words=[("c", 0.5, 1.5), ("d", 2.0, 3.0), ("e", 4.0, 5.0), ("f", 6.0, 7.0)],
            energy=True)
        starts = [w.start for w in result.words]
        self.assertEqual(starts, sorted(starts))
        ends = [w.end for w in result.words]
        for s, e in zip(starts, ends):
            self.assertLess(s, e)


class TailRecoveryCacheIdentityTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path("tmp_tail_recovery_cache_tests")
        self.tmp.mkdir(exist_ok=True)
        self.audio = self.tmp / "nar.wav"
        self.audio.write_bytes(b"some-audio-bytes")
        self.identity = ChunkingConfig().tail_recovery_identity()

    def _result(self, **overrides):
        base = dict(
            provider="fake", model="fake", device="cpu", language="en",
            audio_duration=0.5, processing_seconds=0.0,
            words=[TranscribedWord(word="hi", start=0.0, end=0.4, confidence=0.9)],
            segments=[])
        base.update(overrides)
        return TranscriptionResult(**base)

    def test_pre_tail_recovery_cache_rejected_when_enabled(self):
        # legacy cache: no tail-recovery identity fields (all None)
        cache = self.tmp / "legacy.json"
        save_transcription(self._result(), cache, audio_path=self.audio)
        with self.assertRaises(TranscriptionCacheError):
            load_transcription(cache, audio_path=self.audio, expected_config=self.identity)
        self.assertFalse(
            transcription_is_current(cache, self.audio, expected_config=self.identity))

    def test_cache_with_matching_identity_accepted(self):
        cache = self.tmp / "current.json"
        save_transcription(
            self._result(tail_recovery_enabled=True, tail_gap_trigger_seconds=2.0,
                         tail_recovery_context_seconds=90.0),
            cache, audio_path=self.audio)
        loaded = load_transcription(cache, audio_path=self.audio,
                                    expected_config=self.identity)
        self.assertEqual(loaded.words[0].word, "hi")
        self.assertTrue(
            transcription_is_current(cache, self.audio, expected_config=self.identity))

    def test_wrong_tail_recovery_config_rejected(self):
        cache = self.tmp / "wrong.json"
        save_transcription(
            self._result(tail_recovery_enabled=False, tail_gap_trigger_seconds=2.0,
                         tail_recovery_context_seconds=90.0),
            cache, audio_path=self.audio)
        with self.assertRaises(TranscriptionCacheError):
            load_transcription(cache, audio_path=self.audio, expected_config=self.identity)


if __name__ == "__main__":
    unittest.main()
