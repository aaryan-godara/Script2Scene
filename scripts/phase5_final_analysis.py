"""
Phase 5 — Final Acoustic Diagnostic Analysis
=============================================
Corrects the original acoustic_diagnostic.py search-window issue:
  - The original search extended to (next_scene_start + 0.1s), which allowed
    the energy detector to latch onto the NEXT scene's audio onset.
  - This script caps the search at (next_scene_start - 20ms) so the result
    cannot be contaminated by the following scene.

Produces:
  1. Per-scene reverified acoustic speech_end (capped at next_scene_start)
  2. Whisper error vs reverified acoustic end
  3. Original GT error vs reverified acoustic end
  4. Trailing silence (reverified acoustic end -> next scene start)
  5. Recalculated end-boundary metrics (MAE, Median, P95, Max)
  6. match_confidence vs timing_confidence distinction table
"""

import json
import numpy as np
import subprocess
from pathlib import Path

DIAGNOSTIC_SCENES = [1, 5, 6, 9, 10]
AUDIO_PATH = Path("input/narration.mp3")
OUTPUT_DIR = Path("intermediate/diagnostics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RMS_WINDOW_MS = 10
SILENCE_THRESHOLD_DB = -40
ENERGY_FLOOR = 1e-10
SEARCH_END_GUARD_MS = 20  # stop search 20ms before next scene — prevents contamination


def load_audio_pcm(path, sample_rate=16000):
    cmd = [
        "ffmpeg", "-i", str(path),
        "-f", "s16le", "-acodec", "pcm_s16le",
        "-ar", str(sample_rate), "-ac", "1",
        "-loglevel", "error",
        "pipe:1"
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()}")
    samples = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    return samples, sample_rate


def compute_rms_envelope(samples, sr, window_ms=10):
    window_samples = int(sr * window_ms / 1000)
    hop = window_samples
    n_frames = len(samples) // hop
    rms = np.zeros(n_frames)
    times = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop
        end = start + window_samples
        if end > len(samples):
            break
        frame = samples[start:end]
        rms[i] = np.sqrt(np.mean(frame ** 2) + ENERGY_FLOOR)
        times[i] = (start + window_samples / 2) / sr
    return times, rms


def find_acoustic_end_capped(times, rms, search_start, search_end, threshold_db=-40):
    """
    Find last frame above threshold between search_start and search_end.
    search_end MUST NOT extend into next scene territory.
    """
    peak_rms = np.max(rms)
    threshold_linear = peak_rms * (10 ** (threshold_db / 20))
    mask = (times >= search_start) & (times <= search_end)
    st = times[mask]
    sr_vals = rms[mask]
    if len(sr_vals) == 0:
        return float(search_start)
    above = sr_vals > threshold_linear
    if np.any(above):
        last_idx = int(np.max(np.where(above)))
        return float(st[last_idx])
    return float(search_start)


def main():
    print("Loading audio...")
    samples, sr = load_audio_pcm(AUDIO_PATH)
    duration = len(samples) / sr
    print(f"Audio: {duration:.3f}s | {sr} Hz | {len(samples):,} samples")

    print("Computing RMS envelope (10ms windows)...")
    times, rms = compute_rms_envelope(samples, sr, RMS_WINDOW_MS)

    with open("intermediate/transcription.json") as f:
        transcription = json.load(f)
    with open("intermediate/alignment.json") as f:
        alignment = json.load(f)
    with open("intermediate/ground_truth.json") as f:
        gt = json.load(f)

    gt_map    = {a["scene_id"]: a for a in gt["annotations"]}
    align_map = {a["scene_id"]: a for a in alignment}
    words     = transcription["words"]

    SEP = "=" * 90
    print(f"\n{SEP}")
    print("PHASE 5 ACOUSTIC END-BOUNDARY DIAGNOSTIC (CORRECTED SEARCH WINDOW)")
    print(SEP)

    results = []

    for scene_id in DIAGNOSTIC_SCENES:
        g = gt_map[scene_id]
        a = align_map[scene_id]
        whisper_end = a["speech_end"]
        gt_end      = g["speech_end"]
        match_conf  = a["confidence"]
        next_gt     = gt_map.get(scene_id + 1)
        next_start  = next_gt["speech_start"] if next_gt else duration

        last_word = None
        last_word_start = whisper_end
        for w in words:
            if abs(w["end"] - whisper_end) < 0.02:
                last_word = w["word"].strip()
                last_word_start = w["start"]
                break

        # KEY FIX: cap 20ms before next scene to prevent contamination
        search_start = max(0.0, whisper_end - 0.05)
        search_end   = next_start - (SEARCH_END_GUARD_MS / 1000.0)

        if search_end > search_start:
            acoustic_end = find_acoustic_end_capped(
                times, rms, search_start, search_end, SILENCE_THRESHOLD_DB
            )
        else:
            acoustic_end = whisper_end  # no gap to search

        whisper_error_ms       = (whisper_end - acoustic_end) * 1000
        gt_vs_acoustic_ms      = (gt_end      - acoustic_end) * 1000
        whisper_vs_gt_ms       = (whisper_end - gt_end)       * 1000
        trailing_silence_ms    = (next_start  - acoustic_end) * 1000
        gt_trailing_silence_ms = (next_start  - gt_end)       * 1000

        result = {
            "scene_id":                           scene_id,
            "last_word":                          last_word,
            "last_word_start":                    round(last_word_start, 3),
            "whisper_end":                        whisper_end,
            "original_gt_end":                    gt_end,
            "reverified_acoustic_end":            round(acoustic_end, 3),
            "next_scene_start":                   next_start,
            "search_end_used":                    round(search_end, 3),
            "match_confidence":                   round(match_conf, 4),
            "whisper_vs_acoustic_ms":             round(whisper_error_ms),
            "gt_vs_acoustic_ms":                  round(gt_vs_acoustic_ms),
            "whisper_vs_gt_ms":                   round(whisper_vs_gt_ms),
            "trailing_silence_after_acoustic_ms": round(trailing_silence_ms),
            "trailing_silence_after_gt_ms":       round(gt_trailing_silence_ms),
        }
        results.append(result)

        print(f"\n--- Scene {scene_id} | \"{last_word}\" ---")
        print(f"  last_word_start:      {last_word_start:.3f}")
        print(f"  whisper_end:          {whisper_end:.3f}")
        print(f"  search_end (capped):  {search_end:.3f}  (guard: -{SEARCH_END_GUARD_MS}ms from next)")
        print(f"  acoustic_end:         {acoustic_end:.3f}")
        print(f"  original_gt_end:      {gt_end:.3f}")
        print(f"  next_scene_start:     {next_start:.3f}")
        print(f"  match_confidence:     {match_conf:.4f}  <- TEXT MATCH ONLY")
        print(f"  whisper_vs_acoustic:  {whisper_error_ms:+.0f} ms  (neg=Whisper early)")
        print(f"  gt_vs_acoustic:       {gt_vs_acoustic_ms:+.0f} ms  (pos=GT includes silence)")
        print(f"  whisper_vs_gt:        {whisper_vs_gt_ms:+.0f} ms  (original benchmark error)")
        print(f"  trailing_sil(acou):   {trailing_silence_ms:.0f} ms")
        print(f"  trailing_sil(gt):     {gt_trailing_silence_ms:.0f} ms")

    print(f"\n{SEP}")
    print("SUMMARY METRICS")
    print(SEP)

    w_errors  = [abs(r["whisper_vs_acoustic_ms"]) for r in results]
    gt_errors = [abs(r["gt_vs_acoustic_ms"])       for r in results]
    orig_err  = [abs(r["whisper_vs_gt_ms"])         for r in results]
    trail_sil = [r["trailing_silence_after_acoustic_ms"] for r in results]
    gt_sil    = [r["trailing_silence_after_gt_ms"]       for r in results]

    print(f"\n  Whisper vs Reverified Acoustic (true provider error):")
    print(f"    MAE:    {np.mean(w_errors):.0f} ms")
    print(f"    Median: {np.median(w_errors):.0f} ms")
    print(f"    Max:    {max(w_errors):.0f} ms")

    print(f"\n  GT vs Reverified Acoustic (annotation definition error):")
    print(f"    MAE:    {np.mean(gt_errors):.0f} ms")
    print(f"    Median: {np.median(gt_errors):.0f} ms")
    print(f"    Max:    {max(gt_errors):.0f} ms")

    print(f"\n  Original benchmark (Whisper vs original GT) -- for reference only:")
    print(f"    MAE:    {np.mean(orig_err):.0f} ms")
    print(f"    Median: {np.median(orig_err):.0f} ms")
    print(f"    Max:    {max(orig_err):.0f} ms")

    print(f"\n  Trailing Silence (reverified acoustic -> next scene):")
    print(f"    Mean:   {np.mean(trail_sil):.0f} ms")
    print(f"    Median: {np.median(trail_sil):.0f} ms")
    print(f"    Min:    {min(trail_sil):.0f} ms")
    print(f"    Max:    {max(trail_sil):.0f} ms")

    print(f"\n  Trailing Silence (original GT -> next scene):")
    print(f"    Mean:   {np.mean(gt_sil):.0f} ms")
    print(f"    Median: {np.median(gt_sil):.0f} ms")

    print(f"\n{SEP}")
    print("CONFIDENCE DISTINCTION  (match_confidence != timing_confidence)")
    print(SEP)
    print(f"  {'Scene':<8} {'match_conf':<14} {'whisper_vs_gt_ms':<20} Interpretation")
    print(f"  {'-'*70}")
    for r in results:
        mc  = r["match_confidence"]
        we  = r["whisper_vs_gt_ms"]
        tag = "TEXT OK / TIMESTAMP SUSPECT" if mc >= 0.95 and abs(we) > 300 else "consistent"
        print(f"  {r['scene_id']:<8} {mc:<14.4f} {we:<20.0f} {tag}")

    out = {
        "classification":    "END_BOUNDARY_SYSTEMATIC_DISCREPANCY",
        "root_cause":        "UNDER_INVESTIGATION",
        "diagnostic_scenes": results,
        "summary": {
            "whisper_vs_acoustic_mae_ms":        round(np.mean(w_errors)),
            "gt_vs_acoustic_mae_ms":             round(np.mean(gt_errors)),
            "original_whisper_vs_gt_mae_ms":     round(np.mean(orig_err)),
            "mean_trailing_silence_ms":          round(float(np.mean(trail_sil))),
            "median_trailing_silence_ms":        round(float(np.median(trail_sil))),
            "mean_gt_trailing_silence_ms":       round(float(np.mean(gt_sil))),
        }
    }
    out_path = OUTPUT_DIR / "phase5_final_analysis.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results saved: {out_path}")


if __name__ == "__main__":
    main()
