"""Phase 3 experiment: benchmark acoustic speech-end refinement.

Compares:
  A - Current Alignment (raw Whisper-based speech_end)
  B - Acoustic End Refined speech_end

against intermediate/ground_truth_verified.json (evaluation only).

Runs a small parameter grid (threshold x max extension x guard), reports
sensitivity, and writes intermediate/acoustic_refinement_benchmark.json.
No production file is modified.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(r"D:\RAHUL SIR\automation\VIDEO CREATOR")
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from video_assembler.experiments.acoustic_end_refinement.refiner import (
    AcousticBoundaryRefiner,
    RefinerConfig,
)

THRESHOLDS = [50, 100, 200, 300, 500]
GRID = [
    {"silence_threshold_db": t, "search_extension_ms": e, "next_scene_guard_ms": g}
    for t in (-35.0, -40.0, -45.0)
    for e in (500, 700)
    for g in (50, 100)
]
PRIMARY = {"silence_threshold_db": -40.0, "search_extension_ms": 700, "next_scene_guard_ms": 80}


def load(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def err_stats(errors: List[float]) -> Dict:
    errors = list(errors)
    n = len(errors)
    p95 = sorted(errors)[int(n * 0.95)] if errors else 0.0
    return {
        "mae": sum(abs(e) for e in errors) / n if n else 0.0,
        "median": statistics.median(errors) if errors else 0.0,
        "p95": p95,
        "max": max(abs(e) for e in errors) if errors else 0.0,
        "signed_mean": sum(errors) / n if n else 0.0,
    }


def boundary_accuracy(errors: List[float]) -> Dict:
    n = len(errors)
    return {t: (sum(1 for e in errors if abs(e) * 1000 <= t) / n * 100 if n else 0.0) for t in THRESHOLDS}


def scene_results(alignment, refinements, gt_map):
    start_errs, end_errs = [], []
    for r in refinements:
        g = gt_map[r.scene_id]
        start_errs.append(alignment[r.scene_id]["speech_start"] - g["speech_start"])
        end_errs.append(r.refined_speech_end - g["speech_end"])
    return {
        "start": err_stats(start_errs),
        "end": err_stats(end_errs),
        "start_boundary_accuracy": boundary_accuracy(start_errs),
        "end_boundary_accuracy": boundary_accuracy(end_errs),
    }


def main():
    alignment = {a["scene_id"]: a for a in load(ROOT / "intermediate" / "alignment.json")}
    gt = load(ROOT / "intermediate" / "ground_truth_verified.json")["annotations"]
    gt_map = {a["scene_id"]: a for a in gt}
    audio = ROOT / "input" / "narration.mp3"
    ordered = [alignment[sid] for sid in range(1, 12)]

    # Baseline (raw ends) vs GT
    base_end_errs = [alignment[sid]["speech_end"] - gt_map[sid]["speech_end"] for sid in range(1, 12)]
    base_start_errs = [alignment[sid]["speech_start"] - gt_map[sid]["speech_start"] for sid in range(1, 12)]
    baseline = {
        "start": err_stats(base_start_errs),
        "end": err_stats(base_end_errs),
        "start_boundary_accuracy": boundary_accuracy(base_start_errs),
        "end_boundary_accuracy": boundary_accuracy(base_end_errs),
    }

    configs = []
    for params in GRID:
        cfg = RefinerConfig(**params)
        refiner = AcousticBoundaryRefiner(str(audio), cfg)
        refinements, stats = refiner.refine(ordered)
        metrics = scene_results(alignment, refinements, gt_map)
        configs.append(
            {
                "params": params,
                "stats": {
                    "total": stats.total,
                    "refined": stats.refined,
                    "unchanged": stats.unchanged,
                    "ambiguous": stats.ambiguous,
                    "guard_limited": stats.guard_limited,
                    "overlaps": stats.overlaps,
                    "invalid": stats.invalid,
                },
                "metrics": metrics,
            }
        )

    # Primary (Phase-1 validated) config detail
    refiner = AcousticBoundaryRefiner(str(audio), RefinerConfig(**PRIMARY))
    refinements, primary_stats = refiner.refine(ordered)
    primary_metrics = scene_results(alignment, refinements, gt_map)
    primary_detail = {
        "params": PRIMARY,
        "stats": {
            "total": primary_stats.total,
            "refined": primary_stats.refined,
            "unchanged": primary_stats.unchanged,
            "ambiguous": primary_stats.ambiguous,
            "guard_limited": primary_stats.guard_limited,
            "overlaps": primary_stats.overlaps,
            "invalid": primary_stats.invalid,
        },
        "metrics": primary_metrics,
        "scene_level": [
            {
                "scene_id": r.scene_id,
                "raw_end": r.raw_speech_end,
                "refined_end": r.refined_speech_end,
                "extension_ms": r.extension_ms,
                "detected_energy_db": r.detected_energy_db,
                "status": r.status,
            }
            for r in refinements
        ],
    }

    out = {
        "experiment": "phase3_acoustic_end_refinement",
        "inputs": {
            "audio": "input/narration.mp3",
            "alignment": "intermediate/alignment.json",
            "ground_truth": "intermediate/ground_truth_verified.json",
        },
        "algorithm": "last frame >= threshold (10 ms RMS, threshold re global peak) in [raw_end - backtrack, min(next_start - guard, raw_end + ext, duration)]; never shortens; never crosses next scene",
        "gt_not_read_by_algorithm": True,
        "gt_used_for_evaluation_only": True,
        "baseline": baseline,
        "parameter_sensitivity_grid": configs,
        "primary_config": primary_detail,
    }

    dest = ROOT / "intermediate" / "acoustic_refinement_benchmark.json"
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote {dest}\n")
    print("PARAMETER GRID (End MAE / Start MAE ms, boundary acc ±50/±100/±200/±300/±500 %):")
    print(f"  {'threshold':>9} {'ext':>5} {'guard':>5} | {'EndMAE':>6} {'StMAE':>6} | {'StAcc':>14} {'EndAcc':>14} | R U A G O I")
    for c in configs:
        p = c["params"]
        m = c["metrics"]
        st = c["stats"]
        sa = "/".join(f"{m['start_boundary_accuracy'][t]:.0f}" for t in THRESHOLDS)
        ea = "/".join(f"{m['end_boundary_accuracy'][t]:.0f}" for t in THRESHOLDS)
        print(f"  {p['silence_threshold_db']:>8.0f} {p['search_extension_ms']:>5} {p['next_scene_guard_ms']:>5} | "
              f"{m['end']['mae']*1000:>6.0f} {m['start']['mae']*1000:>6.0f} | {sa:>14} {ea:>14} | "
              f"{st['refined']} {st['unchanged']} {st['ambiguous']} {st['guard_limited']} {st['overlaps']} {st['invalid']}")
    print("\nBASELINE (raw):", f"Start MAE {baseline['start']['mae']*1000:.0f}  End MAE {baseline['end']['mae']*1000:.0f}")
    print("PRIMARY config:", PRIMARY)
    print("PRIMARY:", f"Start MAE {primary_metrics['start']['mae']*1000:.0f}  End MAE {primary_metrics['end']['mae']*1000:.0f}",
          f"  End median {primary_metrics['end']['median']*1000:.0f}  P95 {primary_metrics['end']['p95']*1000:.0f}  Max {primary_metrics['end']['max']*1000:.0f}  signed {primary_metrics['end']['signed_mean']*1000:.0f}")
    print("PRIMARY End acc:", " ".join(f"{t}:{primary_metrics['end_boundary_accuracy'][t]:.0f}%" for t in THRESHOLDS))
    print("PRIMARY status:", primary_detail["stats"])


if __name__ == "__main__":
    main()
