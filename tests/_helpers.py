"""Shared fixtures for Phase-5 tests (no Whisper, no Gradio)."""

import json
import subprocess
from pathlib import Path

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def make_image(path: Path, color=(128, 0, 200), size=(1920, 1080)) -> Path:
    if not HAS_PIL:
        raise RuntimeError("PIL not available")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path


def synth_audio(path: Path, seconds: float = 1.0) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
         "-c:a", "pcm_s16le", str(path)],
        capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"audio synth failed: {(proc.stderr or '')[-2000:]}")
    return path


def write_project_json(project_dir: Path, scenes, name: str = "food_truck") -> Path:
    project_dir.mkdir(parents=True, exist_ok=True)
    path = project_dir / "project.json"
    path.write_text(json.dumps({"project": name, "scenes": scenes}), encoding="utf-8")
    return path
