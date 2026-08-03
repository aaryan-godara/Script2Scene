from abc import ABC, abstractmethod
from typing import List, Optional
from pydantic import BaseModel, Field

class TranscribedWord(BaseModel):
    word: str
    start: float
    end: float
    confidence: Optional[float] = None

class TranscribedSegment(BaseModel):
    text: str
    start: float
    end: float
    words: List[TranscribedWord] = Field(default_factory=list)

class TranscriptionResult(BaseModel):
    provider: str
    model: str
    device: str
    language: str
    audio_duration: float
    processing_seconds: float
    words: List[TranscribedWord] = Field(default_factory=list)
    segments: List[TranscribedSegment] = Field(default_factory=list)

class TranscriptionProvider(ABC):
    @abstractmethod
    def transcribe(self, audio_path: str) -> TranscriptionResult:
        """Transcribe audio and return standard word-level timestamps."""
        pass
