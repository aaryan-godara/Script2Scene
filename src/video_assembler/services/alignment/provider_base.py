from abc import ABC, abstractmethod
from typing import Dict, List, Optional
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
    # EOF tail-recovery configuration used to build this transcription. These are
    # part of the cache identity: a cache built with a different tail-recovery
    # algorithm must not be silently reused.
    tail_recovery_enabled: Optional[bool] = None
    tail_gap_trigger_seconds: Optional[float] = None
    tail_recovery_context_seconds: Optional[float] = None
    tail_recovery: Optional[Dict] = None

class TranscriptionProvider(ABC):
    @abstractmethod
    def transcribe(self, audio_path: str) -> TranscriptionResult:
        """Transcribe audio and return standard word-level timestamps."""
        pass
