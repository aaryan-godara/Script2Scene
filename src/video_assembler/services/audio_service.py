import subprocess
import json
from pathlib import Path
from typing import Dict, Any

class AudioService:
    def __init__(self, project_dir: Path):
        self.project_dir = Path(project_dir)

    def get_audio_metadata(self, audio_path: Path) -> Dict[str, Any]:
        """Uses ffprobe to extract audio metadata."""
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(audio_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            probe_data = json.loads(result.stdout)
            
            # Extract relevant metadata
            format_data = probe_data.get("format", {})
            streams = probe_data.get("streams", [])
            audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), {})
            
            return {
                "duration": float(format_data.get("duration", 0.0)),
                "codec": audio_stream.get("codec_name", "unknown"),
                "sample_rate": int(audio_stream.get("sample_rate", 0)),
                "channels": int(audio_stream.get("channels", 0)),
                "bitrate": int(format_data.get("bit_rate", 0))
            }
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to probe audio file: {e.stderr}")
            
    def normalize_audio(self, source_audio: Path) -> Path:
        """Converts audio to 16kHz mono PCM WAV for analysis."""
        if not source_audio.exists():
            raise FileNotFoundError(f"Source audio not found: {source_audio}")
            
        output_path = self.project_dir / "intermediate" / "narration_normalized.wav"
        
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-i", str(source_audio),
            "-ac", "1",           # Mono
            "-ar", "16000",       # 16 kHz
            "-c:a", "pcm_s16le",  # PCM 16-bit little-endian
            "-vn",                # No video
            str(output_path)
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            return output_path
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to normalize audio: {e.stderr}")
