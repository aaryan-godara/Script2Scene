from pydantic import BaseModel, Field
from typing import List, Optional

class Scene(BaseModel):
    scene_id: int
    script_text: str
    images: List[str] = Field(default_factory=list)
    
    # Alignment timestamps (populated later)
    speech_start: Optional[float] = None
    speech_end: Optional[float] = None
    
    # Diagnostics (populated by acoustic refinement)
    raw_speech_end: Optional[float] = None
    
    # Timeline timestamps (populated later)
    visual_start: Optional[float] = None
    visual_end: Optional[float] = None
    
    match_confidence: Optional[float] = None
    warning: Optional[str] = None

class ProjectInput(BaseModel):
    project: str
    scenes: List[Scene]

class ValidationResult(BaseModel):
    valid: bool
    errors: List[dict] = Field(default_factory=list)
    warnings: List[dict] = Field(default_factory=list)


class TimelineImage(BaseModel):
    path: str
    visual_start: float
    visual_end: float


class TimelineScene(BaseModel):
    scene_id: int
    script_text: str
    speech_start: float
    speech_end: float
    visual_start: float
    visual_end: float
    images: List[TimelineImage] = Field(default_factory=list)


class Timeline(BaseModel):
    project: str
    audio_duration: float
    fps: int = 30
    scenes: List[TimelineScene] = Field(default_factory=list)
