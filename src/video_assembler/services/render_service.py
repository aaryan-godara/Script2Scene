"""RenderService: renders a Timeline into an MP4 using FFmpeg.

Strategy (reliability-first, no premature optimization):
  1. For every image slot, encode an H.264 segment of exactly the intended
     number of frames (cumulative rounding on frame boundaries so the total
     frame count is exactly round(audio_duration * fps) -- no cumulative drift).
  2. Concatenate the segments with the concat demuxer (-c copy).
  3. Mux the ORIGINAL narration audio (AAC, 192 kbps).
  4. Clean up temp segments; validate the output with ffprobe.

Image normalization: scale to fit within WxH preserving aspect ratio, then pad
to WxH with black, centered. No stretching.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

from video_assembler.models import Timeline


class RenderError(RuntimeError):
    pass


@dataclass
class RenderConfig:
    width: int = 1920
    height: int = 1080
    fps: int = 30
    codec: str = "libx264"
    crf: int = 20
    preset: str = "medium"
    pixel_format: str = "yuv420p"
    audio_codec: str = "aac"
    audio_bitrate: str = "192k"
    pad_color: str = "black"


class RenderService:
    def __init__(self, config: Optional[RenderConfig] = None, temp_dir: Optional[Path] = None):
        self.config = config or RenderConfig()
        self.temp_dir = Path(temp_dir) if temp_dir else None

    # ------------------------------------------------------------------ ffmpeg
    def _run(self, cmd: List[str], label: str) -> subprocess.CompletedProcess:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            tail = stderr[-3000:] if len(stderr) > 3000 else stderr
            raise RenderError(
                f"{label} failed (ffmpeg exit {proc.returncode}).\nCommand: "
                f"{' '.join(cmd)}\nFFmpeg stderr (tail):\n{tail}"
            )
        return proc

    def _vf(self) -> str:
        c = self.config
        return (
            f"scale={c.width}:{c.height}:force_original_aspect_ratio=decrease,"
            f"pad={c.width}:{c.height}:(ow-iw)/2:(oh-ih)/2:color={c.pad_color},"
            f"format={c.pixel_format}"
        )

    # ------------------------------------------------------------------ render
    def render(self, timeline: Timeline, audio_path: Path, output_path: Path) -> Dict:
        c = self.config
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.temp_dir or (output_path.parent.parent / "intermediate" / "render_temp")
        temp.mkdir(parents=True, exist_ok=True)

        # Flatten image slots in timeline order.
        slots = [(slot.visual_start, slot.visual_end, slot.path) for sc in timeline.scenes for slot in sc.images]
        if not slots:
            raise RenderError("Timeline contains no image slots to render.")

        segments: List[Path] = []
        try:
            for idx, (vs, ve, img) in enumerate(slots):
                frames = round(ve * c.fps) - round(vs * c.fps)
                if frames <= 0:
                    raise RenderError(f"Image slot {idx + 1} ({vs:.3f}-{ve:.3f}s) resolves to {frames} frames.")
                seg = temp / f"seg_{idx:03d}.mp4"
                cmd = [
                    "ffmpeg", "-y",
                    "-loop", "1", "-framerate", str(c.fps), "-i", str(img),
                    "-frames:v", str(frames),
                    "-vf", self._vf(),
                    "-c:v", c.codec, "-crf", str(c.crf), "-preset", c.preset,
                    "-pix_fmt", c.pixel_format, "-r", str(c.fps),
                    str(seg),
                ]
                self._run(cmd, f"Segment {idx + 1}")
                segments.append(seg)

            # Concat segments (identical codec/params => copy is safe).
            concat_list = temp / "concat.txt"
            concat_list.write_text(
                "".join(f"file '{seg.name}'\n" for seg in segments), encoding="utf-8")
            video_only = temp / "video_only.mp4"
            self._run(
                ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list),
                 "-c", "copy", str(video_only)],
                "Concat")

            # Mux original narration audio.
            self._run(
                ["ffmpeg", "-y", "-i", str(video_only), "-i", str(audio_path),
                 "-map", "0:v:0", "-map", "1:a:0",
                 "-c:v", "copy", "-c:a", c.audio_codec, "-b:a", c.audio_bitrate,
                 str(output_path)],
                "Audio mux")

            return self.probe(output_path)
        finally:
            self._cleanup(temp)

    # ------------------------------------------------------------------ cleanup
    def _cleanup(self, temp: Path) -> None:
        try:
            for p in temp.iterdir():
                if p.is_file():
                    p.unlink(missing_ok=True)
        except OSError:
            pass

    # ------------------------------------------------------------------ probe
    def probe(self, path: Path) -> Dict:
        path = Path(path)
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", "-show_streams", str(path),
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True)
        except FileNotFoundError:
            raise RenderError("ffprobe not found on PATH.")
        if proc.returncode != 0:
            raise RenderError(f"ffprobe failed for {path}: {(proc.stderr or '')[-2000:]}")
        data = json.loads(proc.stdout)

        video = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
        audio = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None)
        return {
            "path": str(path),
            "container_duration": float(data.get("format", {}).get("duration", 0.0) or 0.0),
            "has_video": video is not None,
            "video_codec": video.get("codec_name") if video else None,
            "width": int(video.get("width", 0)) if video else 0,
            "height": int(video.get("height", 0)) if video else 0,
            "fps": _parse_fps(video.get("avg_frame_rate")) if video else 0.0,
            "video_duration": float(video.get("duration", 0.0) or 0.0) if video else 0.0,
            "has_audio": audio is not None,
            "audio_codec": audio.get("codec_name") if audio else None,
            "audio_duration": float(audio.get("duration", 0.0) or 0.0) if audio else 0.0,
        }


def _parse_fps(rate: Optional[str]) -> float:
    if not rate or "/" not in rate:
        try:
            return float(rate) if rate else 0.0
        except (TypeError, ValueError):
            return 0.0
    try:
        num, den = rate.split("/")
        d = float(den)
        return float(num) / d if d else 0.0
    except (ValueError, ZeroDivisionError):
        return 0.0
