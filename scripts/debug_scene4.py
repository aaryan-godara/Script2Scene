"""Debug scene 4 end detection — check if the acoustic end bled into scene 5."""
import json
import numpy as np
import subprocess
from pathlib import Path

SAMPLE_RATE   = 16000
RMS_WINDOW_MS = 10
ENERGY_FLOOR  = 1e-10
SILENCE_DB    = -40


def load_audio(path, sr=SAMPLE_RATE):
    cmd = ["ffmpeg", "-i", str(path), "-f", "s16le", "-acodec", "pcm_s16le",
           "-ar", str(sr), "-ac", "1", "-loglevel", "error", "pipe:1"]
    r = subprocess.run(cmd, capture_output=True)
    return np.frombuffer(r.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def rms_env(samples, sr, wms=RMS_WINDOW_MS):
    n      = int(sr * wms / 1000)
    frames = len(samples) // n
    t, rv  = np.zeros(frames), np.zeros(frames)
    for i in range(frames):
        s, e = i * n, (i + 1) * n
        if e > len(samples):
            break
        rv[i] = np.sqrt(np.mean(samples[s:e] ** 2) + ENERGY_FLOOR)
        t[i]  = (s + n / 2) / sr
    return t, rv


samples      = load_audio("input/narration.mp3")
times, rms   = rms_env(samples, SAMPLE_RATE)
peak         = np.max(rms)
thr          = peak * (10 ** (SILENCE_DB / 20))

with open("intermediate/alignment.json")    as f: al_data = json.load(f)
with open("intermediate/ground_truth.json") as f: gt_data = json.load(f)

al_map = {a["scene_id"]: a for a in al_data}
gt_map = {a["scene_id"]: a for a in gt_data["annotations"]}

# Inspect scenes 3, 4, 5 to understand the scene-4 bleed
for sc in [3, 4, 5]:
    a   = al_map[sc]
    g   = gt_map[sc]
    nxt = gt_map.get(sc + 1)
    ns  = nxt["speech_start"] if nxt else 60.0

    lo  = max(0.0, a["speech_end"] - 0.05)
    hi  = ns - 0.020

    mask    = (times >= lo) & (times <= hi)
    t_win   = times[mask]
    r_win   = rms[mask]
    above   = r_win >= thr
    idxs    = np.where(above)[0]
    found   = float(t_win[idxs[-1]]) if len(idxs) else lo

    db_found = 20 * np.log10(found / (peak + 1e-10) + 1e-10)  # won't use this

    print(f"\n{'='*60}")
    print(f"Scene {sc}")
    print(f"  whisper_end  : {a['speech_end']:.3f}")
    print(f"  gt_end (old) : {g['speech_end']:.3f}")
    print(f"  next_start   : {ns:.3f}")
    print(f"  search_lo    : {lo:.3f}")
    print(f"  search_hi    : {hi:.3f}")
    print(f"  acoustic_end : {found:.3f}  (trailing={(ns - found)*1000:.0f} ms)")

    # Print energy envelope around the found point
    p1 = max(lo, found - 0.4)
    p2 = min(hi + 0.05, found + 0.2)
    mask2 = (times >= p1) & (times <= p2)
    print(f"  Energy near boundary:")
    for ti, ri in zip(times[mask2], rms[mask2]):
        db  = 20 * np.log10(ri / peak + 1e-10)
        tag = " <<<" if abs(ti - found) < 0.012 else ""
        ab  = "ABOVE" if ri >= thr else "below"
        print(f"    t={ti:.3f}  {db:6.1f} dB  {ab}{tag}")
