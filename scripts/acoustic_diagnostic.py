"""
Acoustic End-Boundary Diagnostic for Scenes 1, 5, 6, 9, 10.

For each scene:
  1. Loads the waveform region from (last_word_end - 1.0s) to (next_scene_start + 0.5s)
  2. Computes a short-window RMS energy envelope
  3. Finds where the energy drops below a silence threshold after the last word
  4. Generates a waveform plot with markers for:
       - Whisper final-word start
       - Whisper final-word end
       - Human GT speech_end
       - Next scene speech_start (from GT)
       - Detected acoustic end (energy-based)
  5. Reports the acoustic findings.

Definition of speech_end for this analysis:
  The perceptual/acoustic end of the final spoken word or vocalization.
  NOT the point before the next scene begins.
  NOT inclusive of trailing silence.
"""

import json
import numpy as np
import subprocess
import struct
import os
from pathlib import Path

# ---- Config ----
DIAGNOSTIC_SCENES = [1, 5, 6, 9, 10]
AUDIO_PATH = Path("input/narration.mp3")
OUTPUT_DIR = Path("intermediate/diagnostics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# RMS parameters
RMS_WINDOW_MS = 20       # 20ms windows for energy computation
SILENCE_THRESHOLD_DB = -40  # dB below peak to consider silence
ENERGY_FLOOR = 1e-10

# ---- Load audio as raw PCM via ffmpeg ----
def load_audio_pcm(path, sample_rate=16000):
    """Load audio file as mono float32 numpy array using ffmpeg."""
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

# ---- Compute RMS energy envelope ----
def compute_rms_envelope(samples, sr, window_ms=20):
    """Compute RMS energy in short windows. Returns (times, rms_values)."""
    window_samples = int(sr * window_ms / 1000)
    hop = window_samples  # non-overlapping
    
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

# ---- Find acoustic end of speech ----
def find_acoustic_end(times, rms, search_start_time, search_end_time, threshold_db=-40):
    """
    Find where the energy drops below threshold after search_start_time.
    Returns the time of the last frame above threshold before a sustained drop.
    """
    # Convert threshold from dB relative to peak
    peak_rms = np.max(rms)
    threshold_linear = peak_rms * (10 ** (threshold_db / 20))
    
    # Find frames in the search region
    mask = (times >= search_start_time) & (times <= search_end_time)
    search_times = times[mask]
    search_rms = rms[mask]
    
    if len(search_rms) == 0:
        return search_start_time
    
    # Find the last frame where energy is above threshold
    # Require at least 3 consecutive frames below threshold to confirm silence
    min_silent_frames = 3
    last_speech_idx = 0
    
    for i in range(len(search_rms)):
        if search_rms[i] > threshold_linear:
            last_speech_idx = i
    
    # Refine: look for the transition point
    # Walk forward from the Whisper end and find where energy stays below threshold
    above_mask = search_rms > threshold_linear
    
    # Find the last True in the above_mask
    if np.any(above_mask):
        last_above = np.max(np.where(above_mask))
        return search_times[last_above]
    else:
        return search_start_time

# ---- Main ----
def main():
    print("Loading audio...")
    samples, sr = load_audio_pcm(AUDIO_PATH)
    duration = len(samples) / sr
    print(f"Audio: {duration:.2f}s, {sr} Hz, {len(samples)} samples")
    
    print("Computing full energy envelope...")
    times, rms = compute_rms_envelope(samples, sr, RMS_WINDOW_MS)
    
    # Load data
    with open("intermediate/transcription.json") as f:
        transcription = json.load(f)
    with open("intermediate/alignment.json") as f:
        alignment = json.load(f)
    with open("intermediate/ground_truth.json") as f:
        gt = json.load(f)
    
    gt_map = {a["scene_id"]: a for a in gt["annotations"]}
    align_map = {a["scene_id"]: a for a in alignment}
    
    # Build word list from transcription
    words = transcription["words"]
    
    print("\n" + "=" * 90)
    print("ACOUSTIC END-BOUNDARY DIAGNOSTIC")
    print("=" * 90)
    
    results = []
    
    for scene_id in DIAGNOSTIC_SCENES:
        g = gt_map[scene_id]
        a = align_map[scene_id]
        
        whisper_end = a["speech_end"]
        gt_end = g["speech_end"]
        
        # Find next scene's GT start
        next_gt = gt_map.get(scene_id + 1)
        next_start = next_gt["speech_start"] if next_gt else duration
        
        # Find the Whisper last word for this scene
        # The last word is the one ending at whisper_end
        last_word = None
        last_word_start = whisper_end
        for w in words:
            if abs(w["end"] - whisper_end) < 0.01:
                last_word = w["word"]
                last_word_start = w["start"]
                break
        
        # Region to analyze: from (whisper_end - 0.5s) to (next_start + 0.2s)
        region_start = max(0, whisper_end - 0.5)
        region_end = min(duration, next_start + 0.2)
        
        # Find acoustic end: search from whisper_end to next_start
        acoustic_end = find_acoustic_end(times, rms, whisper_end - 0.1, next_start + 0.1, SILENCE_THRESHOLD_DB)
        
        # Compute errors
        whisper_error = (whisper_end - acoustic_end) * 1000
        gt_error = (gt_end - acoustic_end) * 1000
        silence_after_acoustic = (next_start - acoustic_end) * 1000
        silence_after_gt = (next_start - gt_end) * 1000
        
        result = {
            "scene_id": scene_id,
            "last_word": last_word,
            "last_word_start": last_word_start,
            "whisper_end": whisper_end,
            "original_gt_end": gt_end,
            "acoustic_end": round(acoustic_end, 3),
            "next_scene_start": next_start,
            "whisper_vs_acoustic_ms": round(whisper_error),
            "gt_vs_acoustic_ms": round(gt_error),
            "silence_after_acoustic_ms": round(silence_after_acoustic),
            "silence_after_gt_ms": round(silence_after_gt),
        }
        results.append(result)
        
        print(f"\n--- Scene {scene_id} ---")
        print(f"  Last word:            '{last_word}'")
        print(f"  Last word start:      {last_word_start:.3f}")
        print(f"  Whisper end:          {whisper_end:.3f}")
        print(f"  Acoustic end:         {acoustic_end:.3f}")
        print(f"  Original GT end:      {gt_end:.3f}")
        print(f"  Next scene start:     {next_start:.3f}")
        print(f"  ---")
        print(f"  Whisper vs Acoustic:  {whisper_error:+.0f} ms")
        print(f"  GT vs Acoustic:       {gt_error:+.0f} ms")
        print(f"  Silence (acoustic->next): {silence_after_acoustic:.0f} ms")
        print(f"  Silence (GT->next):       {silence_after_gt:.0f} ms")
        
        # Generate waveform plot
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
            
            # Extract waveform region
            start_sample = int(region_start * sr)
            end_sample = int(region_end * sr)
            region_samples = samples[start_sample:end_sample]
            region_times = np.linspace(region_start, region_end, len(region_samples))
            
            # Waveform
            ax1.plot(region_times, region_samples, color='#888888', linewidth=0.3)
            ax1.set_ylabel('Amplitude')
            ax1.set_title(f'Scene {scene_id} - End Boundary Diagnostic  |  Last word: "{last_word}"')
            
            # Energy envelope
            mask = (times >= region_start) & (times <= region_end)
            ax2.plot(times[mask], 20 * np.log10(rms[mask] / np.max(rms) + ENERGY_FLOOR), color='#2196F3', linewidth=1.2)
            ax2.set_ylabel('Energy (dB)')
            ax2.set_xlabel('Time (s)')
            ax2.axhline(y=SILENCE_THRESHOLD_DB, color='gray', linestyle=':', alpha=0.5, label=f'Silence threshold ({SILENCE_THRESHOLD_DB} dB)')
            
            # Markers on both axes
            for ax in [ax1, ax2]:
                ax.axvline(x=last_word_start, color='#FF9800', linestyle='--', alpha=0.7, label=f'Last word start ({last_word_start:.3f})')
                ax.axvline(x=whisper_end, color='#F44336', linestyle='-', linewidth=2, alpha=0.8, label=f'Whisper end ({whisper_end:.3f})')
                ax.axvline(x=acoustic_end, color='#4CAF50', linestyle='-', linewidth=2, alpha=0.8, label=f'Acoustic end ({acoustic_end:.3f})')
                ax.axvline(x=gt_end, color='#9C27B0', linestyle='-', linewidth=2, alpha=0.8, label=f'GT end ({gt_end:.3f})')
                ax.axvline(x=next_start, color='#00BCD4', linestyle='--', alpha=0.7, label=f'Next scene start ({next_start:.3f})')
            
            ax2.legend(loc='lower left', fontsize=7)
            
            plt.tight_layout()
            plot_path = OUTPUT_DIR / f"scene_{scene_id}_end_diagnostic.png"
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"  Plot saved: {plot_path}")
        except Exception as e:
            print(f"  Plot error: {e}")
    
    # Summary
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    
    whisper_errors = [abs(r["whisper_vs_acoustic_ms"]) for r in results]
    gt_errors = [abs(r["gt_vs_acoustic_ms"]) for r in results]
    
    print(f"\n  Whisper vs Acoustic End:")
    print(f"    MAE:    {np.mean(whisper_errors):.0f} ms")
    print(f"    Median: {np.median(whisper_errors):.0f} ms")
    print(f"    Max:    {np.max(whisper_errors):.0f} ms")
    
    print(f"\n  GT (original) vs Acoustic End:")
    print(f"    MAE:    {np.mean(gt_errors):.0f} ms")
    print(f"    Median: {np.median(gt_errors):.0f} ms")
    print(f"    Max:    {np.max(gt_errors):.0f} ms")
    
    # Save results JSON
    with open(OUTPUT_DIR / "diagnostic_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n  Results saved to: {OUTPUT_DIR / 'diagnostic_results.json'}")

if __name__ == "__main__":
    main()
