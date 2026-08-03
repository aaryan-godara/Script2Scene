import time
import torch
import stable_whisper
from .provider_base import TranscriptionProvider, TranscriptionResult, TranscribedWord, TranscribedSegment

class StableWhisperProvider(TranscriptionProvider):
    def __init__(self, model_name: str = "base", no_speech_threshold: float = 0.9):
        self.model_name = model_name
        self.no_speech_threshold = no_speech_threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load the model
        self.model = stable_whisper.load_model(model_name, device=self.device)

    def transcribe(self, audio_path: str) -> TranscriptionResult:
        start_time = time.time()
        
        # stable-ts alignment transcription
        # word_timestamps=True is the default for stable-ts, but good to be explicit.
        # no_speech_threshold is raised above the stable-whisper default (0.6) so
        # real speech windows are not discarded on long chunked audio. The default
        # (0.6) silently drops genuine narration regions during chunk transcription
        # (reproduced: chunk_003 loses scenes 100-102). 0.9 keeps such windows.
        result = self.model.transcribe(audio_path, word_timestamps=True,
                                       no_speech_threshold=self.no_speech_threshold)
        
        end_time = time.time()
        processing_seconds = end_time - start_time
        
        words = []
        segments = []
        
        for seg in result.segments:
            seg_words = []
            for w in seg.words:
                # Drop zero-width words: stable-whisper occasionally emits words
                # where end == start (no duration). These are alignment artifacts,
                # not real speech. The real acceptance test showed they corrupt
                # scene alignment (duplicated "They're buying real estate, ..."
                # at 965.44 degraded scene 141 from HIGH to REVIEW), so they are
                # filtered out here.
                if w.end - w.start < 1e-6:
                    continue
                tw = TranscribedWord(
                    word=w.word,
                    start=w.start,
                    end=w.end,
                    confidence=w.probability
                )
                seg_words.append(tw)
                words.append(tw)
                
            segments.append(TranscribedSegment(
                text=seg.text,
                start=seg.start,
                end=seg.end,
                words=seg_words
            ))
            
        # Optional: We could use ffprobe to get exact duration, but stable-ts result might have info
        # usually last word end time is a good proxy, or we can just require audio_duration passed in.
        audio_duration = words[-1].end if words else 0.0

        return TranscriptionResult(
            provider="stable_whisper",
            model=self.model_name,
            device=self.device,
            language=result.language or "en",
            audio_duration=audio_duration,
            processing_seconds=processing_seconds,
            words=words,
            segments=segments
        )
