import subprocess
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

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

    def extract_chunk(self, source_audio: Path, output_path: Path,
                      start: float, end: float) -> Path:
        """Extracts [start, end) seconds from source audio as 16kHz mono PCM WAV.

        Reads the canonical narration but never writes back to it; chunk files
        are transient analysis artifacts only.
        """
        source_audio = Path(source_audio)
        output_path = Path(output_path)
        if not source_audio.exists():
            raise FileNotFoundError(f"Source audio not found: {source_audio}")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg",
            "-y",
            "-ss", f"{start:.6f}",
            "-to", f"{end:.6f}",
            "-i", str(source_audio),
            "-ac", "1",
            "-ar", "16000",
            "-c:a", "pcm_s16le",
            "-vn",
            str(output_path),
        ]
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to extract audio chunk: {e.stderr}")
        return output_path

    def extract_chunks(self, source_audio: Path, chunk_dir: Path,
                       chunk_duration: float, overlap: float) -> List[Tuple[Path, float, float]]:
        """Slices the full narration into overlapping 16kHz mono PCM chunks.

        Chunk k covers [start_k, end_k) with a fixed step of
        (chunk_duration - overlap) seconds:

            Chunk 1: 0 - chunk_duration
            Chunk 2: (chunk_duration - overlap) - (2*chunk_duration - overlap)
            ...

        The final chunk is truncated at the audio end. Returns a list of
        (chunk_path, global_start, global_end).
        """
        source_audio = Path(source_audio)
        chunk_dir = Path(chunk_dir)
        metadata = self.get_audio_metadata(source_audio)
        duration = float(metadata["duration"])
        if duration <= 0:
            raise RuntimeError("Cannot chunk audio with zero duration.")

        step = chunk_duration - overlap
        if step <= 0:
            raise RuntimeError(f"Invalid chunk parameters: chunk_duration={chunk_duration}, "
                               f"overlap={overlap}")

        chunks: List[Tuple[Path, float, float]] = []
        start = 0.0
        idx = 0
        while start < duration - 1e-6:
            end = min(start + chunk_duration, duration)
            name = f"chunk_{idx:03d}_{start:07.3f}_{end:07.3f}.wav"
            out = chunk_dir / name
            self.extract_chunk(source_audio, out, start, end)
            chunks.append((out, start, end))
            idx += 1
            start += step
        return chunks
