"""
Phase 5 Step 1 — Re-Annotate All 11 Scenes
===========================================
Produces ground_truth_v2.json using the approved acoustic annotation convention:

  speech_start: first frame above -40dB threshold after the inter-scene silence
  speech_end:   last frame above -40dB threshold, search capped at
                (next_scene_start - GUARD_MS) to prevent contamination.

Both values use the SAME -40dB / peak convention for consistency.
These are described as `annotated_acoustic_end` / `annotated_acoustic_start`
in the metadata — NOT claimed to be physically perfect ground truth.

The existing GT is used only to seed the search window for each scene.
Stable Whisper timestamps are NOT used to drive annotation.

Annotation metadata is embedded per scene for benchmarking/debugging.
It does NOT need to become part of the production Scene model.
"""

import json
import numpy as np
import subprocess
from pathlib import Path

AUDIO_PATH  = Path("input/narration.mp3")
GT_IN       = Path("intermediate/ground_truth.json")
GT_OUT      = Path("intermediate/ground_truth_v2.json")
ALIGN_PATH  = Path("intermediate/alignment.json")

SAMPLE_RATE       = 16000
RMS_WINDOW_MS     = 10       # 10ms frames for good temporal resolution
SILENCE_DB        = -40      # dB below peak — consistent benchmark convention
ENERGY_FLOOR      = 1e-10
GUARD_MS          = 80       # stop end-search 80ms before next scene onset (guards against onset ramp)
SEARCH_PAD_S      = 1.0      # pad around expected start for onset search
MAX_END_EXT_S     = 0.700    # hard cap: acoustic_end <= whisper_end + 700ms


def load_audio_pcm(path, sr=SAMPLE_RATE):
    cmd = [
        "ffmpeg", "-i", str(path),
        "-f", "s16le", "-acodec", "pcm_s16le",
        "-ar", str(sr), "-ac", "1",
        "-loglevel", "error",
        "pipe:1"
    ]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg: {r.stderr.decode()}")
    return np.frombuffer(r.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def rms_envelope(samples, sr, win_ms=RMS_WINDOW_MS):
    n = int(sr * win_ms / 1000)
    frames = len(samples) // n
    times = np.zeros(frames)
    rms   = np.zeros(frames)
    for i in range(frames):
        s, e = i * n, (i + 1) * n
        if e > len(samples):
            break
        rms[i]   = np.sqrt(np.mean(samples[s:e] ** 2) + ENERGY_FLOOR)
        times[i] = (s + n / 2) / sr
    return times, rms


def threshold_linear(rms_full, db=SILENCE_DB):
    return np.max(rms_full) * (10 ** (db / 20))


def find_onset(times, rms, thr, t_min, t_max):
    """First frame >= thr in [t_min, t_max]."""
    mask = (times >= t_min) & (times <= t_max)
    t, r = times[mask], rms[mask]
    idx  = np.where(r >= thr)[0]
    return float(t[idx[0]]) if len(idx) else float(t_min)


def find_offset(times, rms, thr, t_min, t_max):
    """Last frame >= thr in [t_min, t_max]."""
    mask = (times >= t_min) & (times <= t_max)
    t, r = times[mask], rms[mask]
    idx  = np.where(r >= thr)[0]
    return float(t[idx[-1]]) if len(idx) else float(t_min)


def main():
    print("Loading audio …")
    samples = load_audio_pcm(AUDIO_PATH)
    sr      = SAMPLE_RATE
    dur     = len(samples) / sr
    print(f"  {dur:.3f}s  |  {sr} Hz  |  {len(samples):,} samples")

    print("Computing RMS envelope (10 ms windows) …")
    times, rms = rms_envelope(samples, sr)
    thr        = threshold_linear(rms, SILENCE_DB)
    thr_db_str = f"{SILENCE_DB} dB below peak ({thr:.6f} linear)"
    print(f"  Threshold: {thr_db_str}")

    with open(GT_IN)     as f: gt_old   = json.load(f)
    with open(ALIGN_PATH) as f: align   = json.load(f)
    with open("intermediate/transcription.json") as f: tx = json.load(f)

    old_map   = {a["scene_id"]: a for a in gt_old["annotations"]}
    align_map = {a["scene_id"]: a for a in align}
    n_scenes  = len(gt_old["annotations"])

    SEP = "=" * 80
    print(f"\n{SEP}")
    print("RE-ANNOTATION — All 11 Scenes")
    print(SEP)

    new_annotations = []

    for scene_id in range(1, n_scenes + 1):
        old  = old_map[scene_id]
        al   = align_map[scene_id]
        next_old = old_map.get(scene_id + 1)

        # ── Determine search window boundaries ─────────────────────────────
        # Start search: from 500ms before old GT start (or 0), up to
        #               500ms after old GT start — avoids pulling in previous scene.
        prev_end_guard = old_map[scene_id - 1]["speech_end"] + 0.05 if scene_id > 1 else 0.0
        start_search_lo = max(prev_end_guard, old["speech_start"] - SEARCH_PAD_S)
        start_search_hi = old["speech_start"] + SEARCH_PAD_S

        # End search: from whisper_end-50ms, capped GUARD_MS before next scene.
        next_start = next_old["speech_start"] if next_old else dur
        end_search_lo = max(0.0, al["speech_end"] - 0.05)
        end_search_hi = next_start - (GUARD_MS / 1000.0)

        # ── Find annotated acoustic start ───────────────────────────────────
        onset = find_onset(times, rms, thr, start_search_lo, start_search_hi)

        # ── Find annotated acoustic end ─────────────────────────────────────
        # Hard cap: never report acoustic_end more than 700ms after Whisper end.
        # This prevents the detector from leaping across a long silence to the
        # next scene's onset when the inter-scene gap is large (e.g. scene 4).
        hard_cap = al["speech_end"] + MAX_END_EXT_S
        effective_hi = min(end_search_hi, hard_cap)

        if effective_hi > end_search_lo:
            offset = find_offset(times, rms, thr, end_search_lo, effective_hi)
        else:
            # Extremely short gap — fall back to Whisper end
            offset = al["speech_end"]

        trailing_ms  = (next_start - offset) * 1000
        start_delta  = (onset  - old["speech_start"]) * 1000
        end_delta    = (offset - old["speech_end"])   * 1000

        print(f"\n  Scene {scene_id:>2}")
        print(f"    old start:  {old['speech_start']:.3f}  ->  acoustic: {onset:.3f}  (D: {start_delta:+.0f} ms)")
        print(f"    old end:    {old['speech_end']:.3f}  ->  acoustic: {offset:.3f}  (D: {end_delta:+.0f} ms)")
        print(f"    trailing silence to next scene: {trailing_ms:.0f} ms")

        new_annotations.append({
            "scene_id":     scene_id,
            "speech_start": round(onset,  3),
            "speech_end":   round(offset, 3),
            "notes": "",
            "annotation": {
                "method":           "acoustic_energy",
                "threshold_db":     SILENCE_DB,
                "manually_verified": (scene_id in {1, 5, 6, 9, 10}),
                "original_speech_start": old["speech_start"],
                "original_speech_end":   old["speech_end"],
                "start_delta_ms":   round(start_delta),
                "end_delta_ms":     round(end_delta),
                "trailing_silence_to_next_ms": round(trailing_ms),
            }
        })

    out = {
        "project": "golden_test",
        "version": "v2",
        "annotation_convention": (
            "speech_start and speech_end are annotated_acoustic_start / annotated_acoustic_end "
            f"using a consistent {SILENCE_DB} dB below peak RMS energy threshold "
            f"({RMS_WINDOW_MS} ms frames). "
            "This is a benchmark convention — not a claim of physical perfection. "
            "Trailing silence is excluded. End search is capped before next scene onset."
        ),
        "annotations": new_annotations
    }

    with open(GT_OUT, "w") as f:
        json.dump(out, f, indent=2)

    # Summary
    start_deltas = [a["annotation"]["start_delta_ms"] for a in new_annotations]
    end_deltas   = [a["annotation"]["end_delta_ms"]   for a in new_annotations]
    trailing     = [a["annotation"]["trailing_silence_to_next_ms"]
                    for a in new_annotations if a["annotation"]["trailing_silence_to_next_ms"] >= 0]

    print(f"\n{SEP}")
    print("ANNOTATION SUMMARY")
    print(SEP)
    print(f"  Start shift vs original GT:")
    print(f"    Mean:   {np.mean(start_deltas):+.0f} ms")
    print(f"    Median: {np.median(start_deltas):+.0f} ms")
    print(f"    Max:    {max(start_deltas):+.0f} ms  Min: {min(start_deltas):+.0f} ms")
    print(f"  End shift vs original GT:")
    print(f"    Mean:   {np.mean(end_deltas):+.0f} ms")
    print(f"    Median: {np.median(end_deltas):+.0f} ms")
    print(f"    Max:    {max(end_deltas):+.0f} ms  Min: {min(end_deltas):+.0f} ms")
    print(f"  Trailing silence (acoustic end -> next scene):")
    print(f"    Mean:   {np.mean(trailing):.0f} ms")
    print(f"    Median: {np.median(trailing):.0f} ms")
    print(f"\n  Written: {GT_OUT}")


if __name__ == "__main__":
    main()
