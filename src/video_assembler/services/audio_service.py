import subprocess
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple


def tail_has_acoustic_energy(audio_path: Path, tail_start: float,
                             end_seconds: Optional[float] = None, *,
                             sample_rate: int = 16000, window_ms: int = 10,
                             silence_threshold_db: float = -40.0,
                             energy_floor: float = 1e-10) -> Tuple[bool, float]:
    """Whether the audio region [tail_start, end_seconds) has speech-like energy.

    Follows the same energy convention as AcousticBoundaryRefiner: the audio is
    decoded to 16 kHz mono PCM, a short-window RMS envelope is computed (10 ms
    frames), and the silence threshold is -40 dB relative to the global peak.
    Returns (has_energy, max_db_above_threshold). Used to decide whether a
    transcription that ends early really dropped narration as opposed to
    ending on legitimate trailing silence.
    """
    import numpy as np

    cmd = [
        "ffmpeg", "-i", str(audio_path),
        "-f", "s16le", "-acodec", "pcm_s16le",
        "-ar", str(sample_rate), "-ac", "1",
        "-loglevel", "error", "pipe:1",
    ]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode(errors="replace"))
    samples = np.frombuffer(r.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    n = int(sample_rate * window_ms / 1000)
    n_frames = len(samples) // n
    if n_frames == 0:
        return False, float("-inf")
    frames = samples[: n_frames * n].reshape(n_frames, n)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + energy_floor)
    peak = float(np.max(rms))
    if peak <= 0:
        return False, float("-inf")
    threshold = peak * (10.0 ** (silence_threshold_db / 20.0))
    frame_dt = window_ms / 1000.0
    frame_start = np.arange(n_frames) * frame_dt
    if end_seconds is None:
        mask = frame_start >= tail_start
    else:
        mask = (frame_start >= tail_start) & (frame_start < end_seconds)
    region = rms[mask]
    if region.size == 0:
        return False, float("-inf")
    max_db = float(20.0 * np.log10(np.max(region) / threshold)) if threshold > 0 else float("-inf")
    return bool(np.any(region >= threshold)), max_db


class AudioService:
    def __init__(self, project_dir: Path):
        self.project_dir = Path(project_dir)

    def tail_has_acoustic_energy(self, audio_path: Path, tail_start: float,
                                 end_seconds: Optional[float] = None) -> bool:
        return tail_has_acoustic_energy(audio_path, tail_start, end_seconds)[0]

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
