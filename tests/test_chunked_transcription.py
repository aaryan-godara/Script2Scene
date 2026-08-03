"""Tests for robust long-form chunked transcription and cache safety."""

import json
import unittest
from pathlib import Path

from video_assembler.services.alignment.chunked_transcription_provider import (
    ChunkedTranscriptionProvider, ChunkingConfig, audio_sha256, shift_word)
from video_assembler.services.alignment.provider_base import (
    TranscribedSegment, TranscribedWord, TranscriptionProvider, TranscriptionResult)
from video_assembler.services.alignment.transcription_cache import (
    TranscriptionCacheError, load_transcription, save_transcription,
    transcription_is_current)
from video_assembler.services.audio_service import AudioService

from tests._helpers import synth_audio


class FakeInnerProvider(TranscriptionProvider):
    """Returns scripted per-chunk transcriptions.

    responses: list of (global_chunk_start, [(word, local_start, local_end), ...]).
    The provider looks up the response by the chunk's global start, which the
    FakeAudioService embeds in the chunk path.
    """

    def __init__(self, responses, duration=1.0):
        self.responses = responses
        self.duration = duration
        self.calls = []
        self.provider = "fake"
        self.model = "fake"
        self.model_name = "fake"
        self.device = "cpu"

    def transcribe(self, audio_path: str) -> TranscriptionResult:
        self.calls.append(audio_path)
        name = Path(audio_path).name
        if "chunk_" in name:
            # chunk path carries the global start: chunk_<idx>_<start>_<end>.wav
            parts = name.split("_")
            g_start = float(parts[2])
            g_end = float(parts[3].replace(".wav", ""))
        else:
            g_start = 0.0
            g_end = self.duration
        words = self.responses.get(g_start, [])
        tws = [TranscribedWord(word=w, start=s, end=e, confidence=0.99)
               for w, s, e in words]
        return TranscriptionResult(
            provider="fake", model="fake", device="cpu", language="en",
            audio_duration=g_end - g_start, processing_seconds=0.0,
            words=tws,
            segments=[TranscribedSegment(text=" ".join(x.word for x in tws),
                                         start=tws[0].start, end=tws[-1].end,
                                         words=tws)] if tws else [])


class FakeAudioService:
    """Mimics AudioService chunk slicing without invoking FFmpeg."""

    def __init__(self, duration, chunk_duration, overlap):
        self.duration = duration
        self.chunk_duration = chunk_duration
        self.overlap = overlap
        self.extracted = []

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
        self.extracted = chunks
        return chunks


def make_provider(inner, config=None, audio_service=None):
    return ChunkedTranscriptionProvider(inner, config=config, audio_service=audio_service)


class ChunkBoundaryTest(unittest.TestCase):
    def test_correct_chunk_boundaries_via_audio_service(self):
        tmp = Path(self._testMethodName)
        src = synth_audio(tmp / "nar.wav", 1.0)
        svc = AudioService(tmp)
        chunks = svc.extract_chunks(src, tmp / "chunks", chunk_duration=0.6, overlap=0.2)
        # step = 0.4: 0-0.6, 0.4-1.0, 0.8-1.0
        self.assertEqual([(round(c[1], 3), round(c[2], 3)) for c in chunks],
                         [(0.0, 0.6), (0.4, 1.0), (0.8, 1.0)])
        for c in chunks:
            self.assertTrue(c[0].is_file())

    def test_final_chunk_shorter_than_configured(self):
        tmp = Path(self._testMethodName)
        src = synth_audio(tmp / "nar.wav", 2.0)
        svc = AudioService(tmp)
        chunks = svc.extract_chunks(src, tmp / "chunks", chunk_duration=0.9, overlap=0.3)
        starts = [c[1] for c in chunks]
        # step = 0.6: 0, 0.6, 1.2, 1.8
        self.assertEqual([round(s, 3) for s in starts], [0.0, 0.6, 1.2, 1.8])
        last_start, last_end = chunks[-1][1], chunks[-1][2]
        self.assertLess(last_end - last_start, 0.9)
        self.assertAlmostEqual(last_end, 2.0, delta=0.01)

    def test_chunk_duration_is_configurable(self):
        tmp = Path(self._testMethodName)
        src = synth_audio(tmp / "nar.wav", 5.0)
        svc = AudioService(tmp)
        chunks = svc.extract_chunks(src, tmp / "chunks", chunk_duration=3.0, overlap=1.0)
        # step = 2: 0-3, 2-5, 4-5
        self.assertEqual([(round(c[1], 3), round(c[2], 3)) for c in chunks],
                         [(0.0, 3.0), (2.0, 5.0), (4.0, 5.0)])


class ChunkedTranscriptionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path("tmp_chunked_tests")
        self.tmp.mkdir(exist_ok=True)
        self.narration = self.tmp / "nar.mp3"
        self.narration.write_bytes(b"fake-audio-bytes-for-hash")

    def _run(self, duration, chunk_duration, overlap, responses, threshold=0.0):
        inner = FakeInnerProvider(responses)
        svc = FakeAudioService(duration, chunk_duration, overlap)
        cfg = ChunkingConfig(chunk_duration_seconds=chunk_duration,
                             overlap_seconds=overlap,
                             long_audio_threshold_seconds=threshold)
        provider = make_provider(inner, config=cfg, audio_service=svc)
        result = provider.transcribe(str(self.narration),
                                     chunk_dir=str(self.tmp / "chunks"))
        return provider, inner, result

    def test_short_audio_uses_single_pass(self):
        inner = FakeInnerProvider({})
        svc = FakeAudioService(duration=10.0, chunk_duration=180, overlap=10)
        cfg = ChunkingConfig(long_audio_threshold_seconds=300)
        provider = make_provider(inner, config=cfg, audio_service=svc)
        inner.responses[0.0] = [("hi", 0.0, 0.4), ("there", 0.4, 0.8)]
        result = provider.transcribe(str(self.narration))
        self.assertEqual(len(inner.calls), 1)
        self.assertFalse(result.chunking_enabled)
        self.assertEqual(result.chunk_count, 1)
        self.assertEqual([w.word for w in result.words], ["hi", "there"])
        # single-pass must receive the ORIGINAL path, not a chunk
        self.assertNotIn("chunks", inner.calls[0])

    def test_long_audio_activates_chunking(self):
        # audio 600s, chunk 180s, overlap 10 -> 4 chunks
        inner = FakeInnerProvider({
            0.0: [("a", 0.0, 0.5)],
            170.0: [("b", 0.0, 0.5)],
            340.0: [("c", 0.0, 0.5)],
            510.0: [("d", 0.0, 0.5)],
        })
        svc = FakeAudioService(duration=600.0, chunk_duration=180, overlap=10)
        cfg = ChunkingConfig(long_audio_threshold_seconds=300)
        provider = make_provider(inner, config=cfg, audio_service=svc)
        result = provider.transcribe(str(self.narration))
        self.assertTrue(result.chunking_enabled)
        self.assertEqual(result.chunk_count, 4)
        self.assertEqual(len(inner.calls), 4)

    def test_local_to_global_timestamp_conversion(self):
        inner = FakeInnerProvider({
            0.0: [("a", 12.40, 12.78), ("b", 12.80, 13.10)],
        })
        svc = FakeAudioService(duration=20.0, chunk_duration=180, overlap=10)
        cfg = ChunkingConfig(long_audio_threshold_seconds=0)
        provider = make_provider(inner, config=cfg, audio_service=svc)
        result = provider.transcribe(str(self.narration))
        self.assertEqual(result.words[0].start, 12.40)
        self.assertEqual(result.words[0].end, 12.78)

    def test_chunk_start_offset_applied(self):
        # second chunk starts at global 170; local 12.4 -> global 182.4
        inner = FakeInnerProvider({
            170.0: [("a", 12.40, 12.78), ("b", 12.80, 13.10)],
        })
        svc = FakeAudioService(duration=400.0, chunk_duration=180, overlap=10)
        cfg = ChunkingConfig(long_audio_threshold_seconds=0)
        provider = make_provider(inner, config=cfg, audio_service=svc)
        result = provider.transcribe(str(self.narration))
        self.assertAlmostEqual(result.words[0].start, 182.40, places=3)
        self.assertAlmostEqual(result.words[0].end, 182.78, places=3)

    def test_overlapping_duplicates_removed(self):
        # chunk1 covers a sentence; chunk2 overlaps the same tail
        responses = {
            0.0: [("hello", 0.0, 0.4), ("world", 0.5, 1.0), ("this", 1.2, 1.5),
                  ("is", 1.6, 1.8), ("a", 1.9, 2.1), ("test", 2.2, 2.6)],
            # chunk2 starts at 170, overlaps [170, 180] which contains "test"
            170.0: [("test", 170.0 - 170.0 + 0.3, 0.7), ("again", 1.0, 1.4),
                    ("word", 1.6, 2.0)],
        }
        # rebuild with realistic local coords for chunk2
        responses[170.0] = [("test", 0.3, 0.7), ("again", 1.0, 1.4), ("word", 1.6, 2.0)]
        inner = FakeInnerProvider(responses)
        svc = FakeAudioService(duration=190.0, chunk_duration=180, overlap=10)
        cfg = ChunkingConfig(long_audio_threshold_seconds=0)
        provider = make_provider(inner, config=cfg, audio_service=svc)
        result = provider.transcribe(str(self.narration))
        words = [w.word for w in result.words]
        self.assertEqual(words, ["hello", "world", "this", "is", "a", "test",
                                 "again", "word"])
        self.assertEqual(result.duplicates_removed, 1)

    def test_legitimate_repeated_words_preserved(self):
        # "very very good" must keep both "very"
        responses = {
            0.0: [("very", 0.0, 0.4), ("very", 0.5, 0.9), ("good", 1.0, 1.4),
                  ("okay", 1.6, 2.0)],
            170.0: [("very", 0.3, 0.7), ("very", 0.8, 1.2), ("good", 1.3, 1.7),
                    ("okay", 1.8, 2.2)],
        }
        inner = FakeInnerProvider(responses)
        svc = FakeAudioService(duration=190.0, chunk_duration=180, overlap=10)
        cfg = ChunkingConfig(long_audio_threshold_seconds=0)
        provider = make_provider(inner, config=cfg, audio_service=svc)
        result = provider.transcribe(str(self.narration))
        words = [w.word for w in result.words]
        self.assertEqual(words, ["very", "very", "good", "okay"])

    def test_sentence_spanning_chunk_boundary_reconstructed_once(self):
        # chunk1 ends mid-sentence; chunk2 restarts before that sentence
        responses = {
            0.0: [("the", 0.0, 0.3), ("quick", 0.4, 0.8), ("brown", 0.9, 1.3),
                  ("fox", 1.4, 1.7)],
            170.0: [("fox", 0.2, 0.5), ("jumps", 0.6, 1.0), ("over", 1.1, 1.5),
                    ("the", 1.6, 1.9), ("lazy", 2.0, 2.3), ("dog", 2.4, 2.8)],
        }
        inner = FakeInnerProvider(responses)
        svc = FakeAudioService(duration=190.0, chunk_duration=180, overlap=10)
        cfg = ChunkingConfig(long_audio_threshold_seconds=0)
        provider = make_provider(inner, config=cfg, audio_service=svc)
        result = provider.transcribe(str(self.narration))
        words = [w.word for w in result.words]
        self.assertEqual(words, ["the", "quick", "brown", "fox", "jumps",
                                 "over", "the", "lazy", "dog"])

    def test_no_timestamp_ordering_regression(self):
        responses = {
            0.0: [("a", 0.0, 0.3), ("b", 0.4, 0.7)],
            170.0: [("b", 0.1, 0.4), ("c", 0.5, 0.8), ("d", 0.9, 1.2)],
            340.0: [("d", 0.1, 0.4), ("e", 0.5, 0.8)],
        }
        inner = FakeInnerProvider(responses)
        svc = FakeAudioService(duration=360.0, chunk_duration=180, overlap=10)
        cfg = ChunkingConfig(long_audio_threshold_seconds=0)
        provider = make_provider(inner, config=cfg, audio_service=svc)
        result = provider.transcribe(str(self.narration))
        starts = [w.start for w in result.words]
        self.assertEqual(starts, sorted(starts))

    def test_metadata_diagnostics_written(self):
        inner = FakeInnerProvider({
            0.0: [("a", 0.0, 0.5)],
            170.0: [("b", 0.0, 0.5)],
        })
        svc = FakeAudioService(duration=190.0, chunk_duration=180, overlap=10)
        cfg = ChunkingConfig(long_audio_threshold_seconds=0)
        provider = make_provider(inner, config=cfg, audio_service=svc)
        result = provider.transcribe(str(self.narration))
        self.assertTrue(result.chunking_enabled)
        self.assertEqual(result.chunk_count, 2)
        self.assertEqual(result.words_per_chunk, [1, 1])
        self.assertEqual(len(result.chunk_boundaries), 2)
        self.assertIsNotNone(result.duplicates_removed)
        self.assertIsNotNone(result.audio_sha256)
        self.assertIsNotNone(result.created_at)

    def test_shift_word_helper(self):
        w = TranscribedWord(word="hi", start=1.0, end=1.5, confidence=0.9)
        shifted = shift_word(w, 170.0)
        self.assertEqual(shifted.start, 171.0)
        self.assertEqual(shifted.end, 171.5)
        self.assertEqual(shifted.word, "hi")


class CacheSafetyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path("tmp_cache_tests")
        self.tmp.mkdir(exist_ok=True)
        self.audio = self.tmp / "nar.wav"
        synth_audio(self.audio, 0.5)
        self.cache_path = self.tmp / "intermediate" / "transcription.json"

    def _result(self):
        return TranscriptionResult(
            provider="fake", model="fake", device="cpu", language="en",
            audio_duration=0.5, processing_seconds=0.0,
            words=[TranscribedWord(word="hi", start=0.0, end=0.4, confidence=0.9)],
            segments=[])

    def test_audio_hash_stored(self):
        result = self._result()
        save_transcription(result, self.cache_path, audio_path=self.audio)
        data = json.loads(self.cache_path.read_text(encoding="utf-8"))
        self.assertEqual(data["audio_sha256"], audio_sha256(self.audio))

    def test_current_transcription_accepted_when_hash_matches(self):
        save_transcription(self._result(), self.cache_path, audio_path=self.audio)
        loaded = load_transcription(self.cache_path, audio_path=self.audio)
        self.assertEqual(loaded.words[0].word, "hi")
        self.assertTrue(transcription_is_current(self.cache_path, self.audio))

    def test_stale_transcription_rejected_when_audio_hash_differs(self):
        save_transcription(self._result(), self.cache_path, audio_path=self.audio)
        other = self.tmp / "other.wav"
        synth_audio(other, 0.5, )  # different file content
        # force different content
        import subprocess
        subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i",
                        "sine=frequency=880:duration=0.5", "-c:a", "pcm_s16le",
                        str(other)], capture_output=True)
        with self.assertRaises(TranscriptionCacheError):
            load_transcription(self.cache_path, audio_path=other)
        self.assertFalse(transcription_is_current(self.cache_path, other))

    def test_missing_cache_rejected(self):
        with self.assertRaises(TranscriptionCacheError):
            load_transcription(self.tmp / "nope.json", audio_path=self.audio)

    def test_cache_without_identity_rejected(self):
        save_transcription(self._result(), self.cache_path, audio_path=self.audio)
        data = json.loads(self.cache_path.read_text(encoding="utf-8"))
        data["audio_sha256"] = None
        self.cache_path.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaises(TranscriptionCacheError):
            load_transcription(self.cache_path, audio_path=self.audio)

    def test_save_without_audio_raises(self):
        with self.assertRaises(TranscriptionCacheError):
            save_transcription(self._result(), self.cache_path)


if __name__ == "__main__":
    unittest.main()
