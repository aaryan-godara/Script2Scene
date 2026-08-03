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
    # Cache identity / chunking metadata (optional, filled by chunked pipeline).
    audio_sha256: Optional[str] = None
    chunking_enabled: Optional[bool] = None
    chunk_duration: Optional[float] = None
    overlap: Optional[float] = None
    created_at: Optional[str] = None
    chunk_count: Optional[int] = None
    chunk_boundaries: List[List[float]] = Field(default_factory=list)
    words_per_chunk: List[int] = Field(default_factory=list)
    duplicates_removed: Optional[int] = None

class TranscriptionProvider(ABC):
    @abstractmethod
    def transcribe(self, audio_path: str) -> TranscriptionResult:
        """Transcribe audio and return standard word-level timestamps."""
        pass
