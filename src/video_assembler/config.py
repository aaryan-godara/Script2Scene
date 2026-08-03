from pydantic import BaseModel, Field

class VideoConfig(BaseModel):
    width: int = 1920
    height: int = 1080
    fps: int = 30
    codec: str = "h264"
    crf: int = 20

class AlignmentConfig(BaseModel):
    engine: str = "whisperx"  # We'll determine default later, but keeping as placeholder
    device: str = "auto"
    review_threshold: float = 0.75

class VisualConfig(BaseModel):
    motion_enabled: bool = False
    motion_strength: float = 0.06
    transition: str = "cut"

class AppConfig(BaseModel):
    video: VideoConfig = Field(default_factory=VideoConfig)
    alignment: AlignmentConfig = Field(default_factory=AlignmentConfig)
    visual: VisualConfig = Field(default_factory=VisualConfig)
