"""Runs the pyVideoTrans-inspired candidate against the verified ground truth and
writes intermediate/pyvideotrans_comparison.json.

Fair-test design:
  * Same audio (narration.mp3), same canonical scenes (input/project.json),
    same transcription word timestamps (intermediate/transcription.json),
    same verified ground truth (intermediate/ground_truth_verified.json).
  * The candidate reproduces pyVideoTrans's mapping algorithm only; it does NOT
    retranscribe and does NOT add any timestamp correction.
  * Baseline alignment-stage runtime is measured with the current AlignmentService
    on the identical inputs (transcription is shared, so ASR runtime is excluded
    from both sides).
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(r"D:\RAHUL SIR\automation\VIDEO CREATOR")
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from video_assembler.models import Scene
from video_assembler.services.alignment.alignment_service import AlignmentService
from video_assembler.services.alignment.provider_base import TranscribedWord, TranscriptionResult

from video_assembler.experiments.pyvideotrans_alignment.candidate import align_scenes

THRESHOLDS = [50, 100, 200, 300, 500]


def load(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def stats(errors):
    errors = list(errors)
    if not errors:
        return {"mae": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0, "signed_mean": 0.0}
    signed = [e for e in errors]
    p95 = sorted(errors)[int(len(errors) * 0.95)] if errors else 0.0
    return {
        "mae": sum(abs(e) for e in errors) / len(errors),
        "median": statistics.median(errors),
        "p95": p95,
        "max": max(abs(e) for e in errors),
        "signed_mean": sum(signed) / len(signed),
    }


def boundary_accuracy(errors, thresholds=THRESHOLDS):
    n = len(errors)
    return {t: (sum(1 for e in errors if abs(e) * 1000 <= t) / n * 100 if n else 0.0) for t in thresholds}


def compute_metrics(name, preds, gt_map):
    starts = []
    ends = []
    for sid, v in preds.items():
        gt = gt_map[sid]
        starts.append(v["start"] - gt["speech_start"])
        ends.append(v["end"] - gt["speech_end"])
    return {
        "name": name,
        "start": stats(starts),
        "end": stats(ends),
        "start_boundary_accuracy": boundary_accuracy(starts),
        "end_boundary_accuracy": boundary_accuracy(ends),
    }


def main():
    transcription = load(ROOT / "intermediate" / "transcription.json")
    project = load(ROOT / "input" / "project.json")
    gt = load(ROOT / "intermediate" / "ground_truth_verified.json")["annotations"]
    gt_map = {a["scene_id"]: a for a in gt}

    scenes = project["scenes"]
    words = transcription["words"]
    n_scenes = len(scenes)

    from video_assembler.services.alignment.text_normalizer import TextNormalizer
    normalizer = TextNormalizer()

    def run_candidate(label, use_normalizer):
        t0 = time.perf_counter()
        b, meta = align_scenes(scenes, words, normalizer=normalizer if use_normalizer else None)
        dt = time.perf_counter() - t0
        preds = {x.scene_id: {"start": x.speech_start, "end": x.speech_end,
                              "matched_chars": x.matched_chars, "total_chars": x.total_chars} for x in b}
        questionable = [sid for sid, v in preds.items()
                        if v["total_chars"] and v["matched_chars"] / v["total_chars"] < 0.5]
        return {"label": label, "preds": preds, "meta": meta, "runtime": dt, "questionable": questionable}

    cand_a = run_candidate("pyvideotrans_literal", use_normalizer=False)
    cand_b = run_candidate("pyvideotrans_normalized", use_normalizer=True)

    # ---- Baseline (current AlignmentService) ----
    baseline_scenes = [Scene(**s) for s in scenes]
    alignment_service = AlignmentService()
    transcription_obj = TranscriptionResult(**transcription)
    t0 = time.perf_counter()
    aligned_scenes, diagnostics = alignment_service.align_scenes(baseline_scenes, transcription_obj)
    baseline_time = time.perf_counter() - t0

    baseline_preds = {}
    baseline_status = {}
    for s in aligned_scenes:
        diag = diagnostics.diagnostics.get(s.scene_id, {})
        baseline_preds[s.scene_id] = {"start": s.speech_start, "end": s.speech_end}
        baseline_status[s.scene_id] = diag.get("status", "FAILED")

    def pred_values(preds):
        return {sid: {"start": v["start"], "end": v["end"]} for sid, v in preds.items()}

    cand_a_metrics = compute_metrics(cand_a["label"], pred_values(cand_a["preds"]), gt_map)
    cand_b_metrics = compute_metrics(cand_b["label"], pred_values(cand_b["preds"]), gt_map)
    base_metrics = compute_metrics("current_alignment", pred_values(baseline_preds), gt_map)

    cand_a_failed = len(cand_a["questionable"])
    cand_b_failed = len(cand_b["questionable"])
    baseline_review = sum(1 for s in baseline_status.values() if s == "REVIEW")
    baseline_failed = sum(1 for s in baseline_status.values() if s == "FAILED")

    comparison = {
        "experiment": "phase2_pyvideotrans_comparison",
        "inputs": {
            "audio": "input/narration.mp3",
            "scenes": "input/project.json",
            "transcription": "intermediate/transcription.json",
            "ground_truth": "intermediate/ground_truth_verified.json",
        },
        "candidate_literal": {
            "name": "pyvideotrans_char_difflib (literal, as shipped)",
            "source_files": [
                "videotrans/component/textmatching.py (pyVideoTrans)",
                "src/video_assembler/experiments/pyvideotrans_alignment/candidate.py",
            ],
            "algorithm": "global char-level difflib.SequenceMatcher + linear interpolation, raw case-sensitive chars, no normalizer; scene boundary = first/last non-punct char time",
            "shared_transcription": True,
            "timestamp_correction_applied": False,
            "meta": cand_a["meta"],
            "boundaries_sec": {sid: {"speech_start": v["start"], "speech_end": v["end"]} for sid, v in cand_a["preds"].items()},
            "metrics": cand_a_metrics,
            "alignment_stage_runtime_sec": cand_a["runtime"],
            "questionable_scenes_low_char_coverage": cand_a["questionable"],
            "incorrect_or_questionable_count": cand_a_failed,
        },
        "candidate_normalized": {
            "name": "pyvideotrans_char_difflib (with baseline TextNormalizer preprocessing)",
            "source_files": [
                "videotrans/component/textmatching.py (pyVideoTrans)",
                "src/video_assembler/experiments/pyvideotrans_alignment/candidate.py",
                "src/video_assembler/services/alignment/text_normalizer.py (shared preprocessing only)",
            ],
            "algorithm": "same pyVideoTrans char-difflib mapping, but script and whisper words are pre-normalized with the baseline TextNormalizer so the mapping algorithm is isolated fairly",
            "shared_transcription": True,
            "timestamp_correction_applied": False,
            "meta": cand_b["meta"],
            "boundaries_sec": {sid: {"speech_start": v["start"], "speech_end": v["end"]} for sid, v in cand_b["preds"].items()},
            "metrics": cand_b_metrics,
            "alignment_stage_runtime_sec": cand_b["runtime"],
            "questionable_scenes_low_char_coverage": cand_b["questionable"],
            "incorrect_or_questionable_count": cand_b_failed,
        },
        "baseline": {
            "name": "current_stable_whisper_sequential",
            "source_files": [
                "src/video_assembler/services/alignment/alignment_service.py",
            ],
            "boundaries_sec": {sid: {"speech_start": v["start"], "speech_end": v["end"]} for sid, v in baseline_preds.items()},
            "metrics": base_metrics,
            "alignment_stage_runtime_sec": baseline_time,
            "status_counts": {
                "HIGH": sum(1 for s in baseline_status.values() if s == "HIGH"),
                "REVIEW": baseline_review,
                "FAILED": baseline_failed,
            },
            "wrong_match_count": baseline_failed,
        },
    }

    out = ROOT / "intermediate" / "pyvideotrans_comparison.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    print(f"Wrote {out}")
    print()
    for label, m in (("CANDIDATE A (literal)", cand_a_metrics),
                     ("CANDIDATE B (normalized)", cand_b_metrics),
                     ("BASELINE", base_metrics)):
        print(f"=== {label} ===")
        print(f"  Start MAE {m['start']['mae']*1000:.0f} ms  Median {m['start']['median']*1000:.0f}  P95 {m['start']['p95']*1000:.0f}  Max {m['start']['max']*1000:.0f}  Signed {m['start']['signed_mean']*1000:.0f}")
        print(f"  End   MAE {m['end']['mae']*1000:.0f} ms  Median {m['end']['median']*1000:.0f}  P95 {m['end']['p95']*1000:.0f}  Max {m['end']['max']*1000:.0f}  Signed {m['end']['signed_mean']*1000:.0f}")
        print(f"  Boundary acc START: " + " ".join(f"{t}:{m['start_boundary_accuracy'][t]:.0f}%" for t in THRESHOLDS))
        print(f"  Boundary acc END:   " + " ".join(f"{t}:{m['end_boundary_accuracy'][t]:.0f}%" for t in THRESHOLDS))
    print(f"=== Candidate A coverage = {cand_a['meta']['match_coverage_pct']}%  questionable={cand_a['questionable']}")
    print(f"=== Candidate B coverage = {cand_b['meta']['match_coverage_pct']}%  questionable={cand_b['questionable']}")
    print(f"=== Runtime (alignment stage only): A {cand_a['runtime']:.4f}s  B {cand_b['runtime']:.4f}s  baseline {baseline_time:.4f}s")
    print(f"=== Baseline status: {comparison['baseline']['status_counts']}")


if __name__ == "__main__":
    main()
