import time
import torch
import stable_whisper
from .provider_base import TranscriptionProvider, TranscriptionResult, TranscribedWord, TranscribedSegment

class StableWhisperProvider(TranscriptionProvider):
    def __init__(self, model_name: str = "base"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load the model
        self.model = stable_whisper.load_model(model_name, device=self.device)

    def transcribe(self, audio_path: str) -> TranscriptionResult:
        start_time = time.time()
        
        # stable-ts alignment transcription
        # word_timestamps=True is the default for stable-ts, but good to be explicit
        result = self.model.transcribe(audio_path, word_timestamps=True)
        
        end_time = time.time()
        processing_seconds = end_time - start_time
        
        words = []
        segments = []
        
        for seg in result.segments:
            seg_words = []
            for w in seg.words:
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
