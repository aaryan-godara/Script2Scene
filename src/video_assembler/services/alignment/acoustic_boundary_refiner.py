"""AcousticBoundaryRefiner: deterministic acoustic speech-end refinement.

Refines ONLY the speech_end of each aligned scene from the audio signal.
speech_start, scene matching, and scene ordering are never modified.

Algorithm (Phase-3 validated):
  - Decode audio deterministically to 16 kHz mono PCM (ffmpeg).
  - Compute a short-window RMS energy envelope (default 10 ms frames).
  - For each scene, search forward from raw_end - backtrack to
        min(next_scene_start - guard, raw_end + max_extension, audio_duration)
    and take the LAST frame still at or above the silence threshold
    (default -40 dB relative to the global audio peak).
  - Never shortens below the raw end; never crosses into the next scene;
    never exceeds the audio duration. If acoustic evidence is ambiguous the
    raw end is kept.

This is deliberately a DIFFERENT problem from AlignmentService (which script
words map to which audio words) and from TimelineService (how long an image
stays visible). It answers only: where does the acoustic speech tail end?

NOTE: this component reads only the audio signal, the existing alignment, and
the next-scene boundary. It never reads ground-truth values.
"""

from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

DEFAULT_ENERGY_FLOOR = 1e-10


@dataclass
class RefinerConfig:
    sample_rate: int = 16000
    window_ms: int = 10
    silence_threshold_db: float = -40.0
    search_backtrack_ms: float = 50.0
    search_extension_ms: float = 700.0
    next_scene_guard_ms: float = 80.0
    energy_floor: float = DEFAULT_ENERGY_FLOOR


@dataclass
class SceneRefinement:
    scene_id: int
    raw_speech_end: float
    refined_speech_end: float
    extension_ms: float
    detected_energy_db: float
    status: str  # REFINED | UNCHANGED | AMBIGUOUS | GUARD_LIMITED


@dataclass
class RefinementStats:
    total: int = 0
    refined: int = 0
    unchanged: int = 0
    ambiguous: int = 0
    guard_limited: int = 0
    overlaps: int = 0
    invalid: int = 0


class AcousticBoundaryRefiner:
    def __init__(self, audio_path: str, config: Optional[RefinerConfig] = None):
        self.config = config or RefinerConfig()
        self._samples, self._times, self._rms, self._threshold = self._analyze(audio_path)
        self.audio_duration = len(self._samples) / self.config.sample_rate

    # ---------------------------------------------------------------- analysis
    def _load_pcm(self, audio_path: str) -> np.ndarray:
        cmd = [
            "ffmpeg", "-i", str(audio_path),
            "-f", "s16le", "-acodec", "pcm_s16le",
            "-ar", str(self.config.sample_rate), "-ac", "1",
            "-loglevel", "error", "pipe:1",
        ]
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr.decode(errors="replace"))
        return np.frombuffer(r.stdout, dtype=np.int16).astype(np.float32) / 32768.0

    def _analyze(self, audio_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        samples = self._load_pcm(audio_path)
        sr = self.config.sample_rate
        n = int(sr * self.config.window_ms / 1000)
        n_frames = len(samples) // n
        times = np.zeros(n_frames, dtype=np.float64)
        rms = np.zeros(n_frames, dtype=np.float64)
        for i in range(n_frames):
            s, e = i * n, (i + 1) * n
            rms[i] = np.sqrt(np.mean(samples[s:e] ** 2) + self.config.energy_floor)
            times[i] = (s + n / 2) / sr
        peak = float(np.max(rms))
        threshold = peak * (10.0 ** (self.config.silence_threshold_db / 20.0))
        return samples, times, rms, threshold

    # -------------------------------------------------------------- refinement
    def _search_window(self, raw_end: float, next_start: Optional[float]) -> Tuple[float, float, float]:
        cfg = self.config
        search_start = raw_end - cfg.search_backtrack_ms / 1000.0
        guard_limit = next_start - cfg.next_scene_guard_ms / 1000.0 if next_start is not None else float("inf")
        extension_limit = raw_end + cfg.search_extension_ms / 1000.0
        search_end = min(guard_limit, extension_limit, self.audio_duration)
        return search_start, search_end, guard_limit

    def refine(self, alignments: List[Dict]) -> Tuple[List[SceneRefinement], RefinementStats]:
        stats = RefinementStats(total=len(alignments))
        results: List[SceneRefinement] = []

        for i, a in enumerate(alignments):
            sid = int(a["scene_id"])
            raw_end = float(a["speech_end"])
            speech_start = float(a["speech_start"])
            next_start = float(alignments[i + 1]["speech_start"]) if i + 1 < len(alignments) else None

            search_start, search_end, guard_limit = self._search_window(raw_end, next_start)

            refined_end = raw_end
            status = "UNCHANGED"
            detected_db = None

            if search_end <= search_start:
                status = "AMBIGUOUS"
            else:
                mask = (self._times >= search_start - 1e-9) & (self._times <= search_end + 1e-9)
                idx = np.where(mask & (self._rms >= self._threshold))[0]
                if len(idx) == 0:
                    status = "AMBIGUOUS"
                    region = self._rms[mask]
                    detected_db = float(20.0 * np.log10(np.max(region) / self._threshold)) if region.size else None
                else:
                    det = float(self._times[idx[-1]])
                    detected_db = float(20.0 * np.log10(self._rms[idx[-1]] / self._threshold))
                    if det <= raw_end + 1e-6:
                        status = "UNCHANGED"
                    elif guard_limit <= search_end + 1e-9 and det >= search_end - 1e-6 and search_end >= guard_limit - 1e-9:
                        refined_end = min(det, guard_limit)
                        status = "GUARD_LIMITED"
                    else:
                        refined_end = det
                        status = "REFINED"

            # Safety clamps: never shorten below raw end, never exceed duration,
            # never cross into the next scene.
            refined_end = max(refined_end, raw_end)
            refined_end = min(refined_end, self.audio_duration)

            if refined_end < speech_start:
                refined_end = raw_end
                status = "AMBIGUOUS"
            if next_start is not None and refined_end >= next_start:
                refined_end = next_start - self.config.next_scene_guard_ms / 1000.0
                status = "GUARD_LIMITED"

            if status == "REFINED":
                stats.refined += 1
            elif status == "UNCHANGED":
                stats.unchanged += 1
            elif status == "GUARD_LIMITED":
                stats.guard_limited += 1
            else:
                stats.ambiguous += 1

            if next_start is not None and refined_end >= next_start:
                stats.overlaps += 1
            if refined_end < speech_start or refined_end > self.audio_duration:
                stats.invalid += 1

            results.append(
                SceneRefinement(
                    scene_id=sid,
                    raw_speech_end=raw_end,
                    refined_speech_end=round(refined_end, 3),
                    extension_ms=round((refined_end - raw_end) * 1000.0, 1),
                    detected_energy_db=round(detected_db, 1) if detected_db is not None else None,
                    status=status,
                )
            )

        return results, stats

    def config_dict(self) -> Dict:
        return asdict(self.config)
