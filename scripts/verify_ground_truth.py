"""
Compute per-scene acoustic boundaries for manual verification.
Uses capped search (guard before next scene) WITHOUT MAX_END_EXT_S hard cap.
Scene-local peak for threshold when analyzing end boundary.
"""
import json
import numpy as np
import subprocess
from pathlib import Path

AUDIO_PATH = Path("input/narration.mp3")
SAMPLE_RATE = 16000
RMS_WINDOW_MS = 10
SILENCE_DB = -40
ENERGY_FLOOR = 1e-10
GUARD_MS = 80


def load_audio_pcm(path, sr=SAMPLE_RATE):
    cmd = [
        "ffmpeg", "-i", str(path),
        "-f", "s16le", "-acodec", "pcm_s16le",
        "-ar", str(sr), "-ac", "1",
        "-loglevel", "error", "pipe:1",
    ]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode())
    return np.frombuffer(r.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def rms_envelope(samples, sr, win_ms=RMS_WINDOW_MS):
    n = int(sr * win_ms / 1000)
    frames = len(samples) // n
    times = np.zeros(frames)
    rms = np.zeros(frames)
    for i in range(frames):
        s, e = i * n, (i + 1) * n
        if e > len(samples):
            break
        rms[i] = np.sqrt(np.mean(samples[s:e] ** 2) + ENERGY_FLOOR)
        times[i] = (s + n / 2) / sr
    return times, rms


def find_onset(times, rms, thr, t_min, t_max):
    mask = (times >= t_min) & (times <= t_max)
    t, r = times[mask], rms[mask]
    idx = np.where(r >= thr)[0]
    return float(t[idx[0]]) if len(idx) else float(t_min)


def find_offset(times, rms, thr, t_min, t_max):
    mask = (times >= t_min) & (times <= t_max)
    t, r = times[mask], rms[mask]
    idx = np.where(r >= thr)[0]
    return float(t[idx[-1]]) if len(idx) else float(t_min)


def scene_local_threshold(rms, t_min, t_max, times, db=SILENCE_DB):
    mask = (times >= t_min) & (times <= t_max)
    local_peak = np.max(rms[mask]) if np.any(mask) else np.max(rms)
    return local_peak * (10 ** (db / 20))


def main():
    samples = load_audio_pcm(AUDIO_PATH)
    duration = len(samples) / SAMPLE_RATE
    times, rms = rms_envelope(samples, SAMPLE_RATE)
    global_thr = np.max(rms) * (10 ** (SILENCE_DB / 20))

    with open("intermediate/ground_truth.json") as f:
        gt_old = {a["scene_id"]: a for a in json.load(f)["annotations"]}
    with open("intermediate/alignment.json") as f:
        align = {a["scene_id"]: a for a in json.load(f)}

    print(f"Duration: {duration:.3f}s  global_thr: {global_thr:.6f}\n")
    print(f"{'Sc':>3} {'onset':>8} {'end_loc':>8} {'end_glb':>8} {'wh_end':>8} {'next':>8} {'trail':>6}")
    print("-" * 60)

    results = []
    for sid in range(1, 12):
        old = gt_old[sid]
        al = align[sid]
        nxt = gt_old.get(sid + 1)
        next_start = nxt["speech_start"] if nxt else duration

        prev_end = gt_old[sid - 1]["speech_end"] + 0.05 if sid > 1 else 0.0
        start_lo = max(prev_end, old["speech_start"] - 1.0)
        start_hi = old["speech_start"] + 1.0

        end_lo = max(0.0, al["speech_end"] - 0.15)
        end_hi = next_start - (GUARD_MS / 1000.0)

        onset = find_onset(times, rms, global_thr, start_lo, start_hi)

        # Scene-local threshold for end (avoids global-peak sensitivity)
        local_thr = scene_local_threshold(rms, start_lo, end_hi, times)
        end_local = find_offset(times, rms, local_thr, end_lo, end_hi)
        end_global = find_offset(times, rms, global_thr, end_lo, end_hi)

        trailing = (next_start - end_local) * 1000

        print(
            f"{sid:3d} {onset:8.3f} {end_local:8.3f} {end_global:8.3f} "
            f"{al['speech_end']:8.3f} {next_start:8.3f} {trailing:6.0f}ms"
        )

        results.append({
            "scene_id": sid,
            "speech_start": round(onset, 3),
            "speech_end_local": round(end_local, 3),
            "speech_end_global": round(end_global, 3),
            "whisper_end": al["speech_end"],
            "next_scene_start": next_start,
            "trailing_ms": round(trailing),
        })

    # Scene 4 detail
    print("\n=== SCENE 4 DETAIL (14.0 - 16.5 s) ===")
    mask = (times >= 14.0) & (times <= 16.5)
    local_thr_4 = scene_local_threshold(rms, 14.0, 16.44, times)
    for ti, ri in zip(times[mask], rms[mask]):
        db_g = 20 * np.log10(ri / np.max(rms) + ENERGY_FLOOR)
        db_l = 20 * np.log10(ri / (np.max(rms[mask]) + ENERGY_FLOOR) + ENERGY_FLOOR)
        tag = ""
        if abs(ti - 15.44) < 0.015:
            tag = " <- whisper end"
        if abs(ti - 16.44) < 0.015:
            tag = " <- scene5 start"
        if ri >= local_thr_4 and ti > 15.3 and ti < 16.0:
            tag += " LOCAL_ABOVE"
        print(f"  t={ti:.3f}  global={db_g:6.1f}dB  local={db_l:6.1f}dB{tag}")

    with open("intermediate/diagnostics/verify_boundaries.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
