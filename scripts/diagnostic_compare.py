"""Detailed scene-by-scene diagnostic comparison for the sanity test."""
import json

with open("intermediate/alignment.json") as f:
    alignment = json.load(f)
with open("intermediate/ground_truth.json") as f:
    gt = json.load(f)

gt_map = {a["scene_id"]: a for a in gt["annotations"]}

print(f"{'Scene':>5} | {'GT Start':>9} {'Pred Start':>11} {'D Start':>8} | {'GT End':>9} {'Pred End':>11} {'D End':>8} | {'Dir':>4} | Conf")
print("-" * 100)

for scene in alignment:
    sid = scene["scene_id"]
    g = gt_map.get(sid)
    if not g:
        continue
    
    gs = g["speech_start"]
    ge = g["speech_end"]
    ps = scene["speech_start"]
    pe = scene["speech_end"]
    
    ds = (ps - gs) * 1000  # positive = predicted is LATER
    de = (pe - ge) * 1000  # negative = predicted is EARLIER
    
    direction = "EARLY" if de < -50 else ("LATE" if de > 50 else "OK")
    
    print(f"{sid:>5} | {gs:>9.3f} {ps:>11.3f} {ds:>+8.0f} | {ge:>9.3f} {pe:>11.3f} {de:>+8.0f} | {direction:>5} | {scene['confidence']:.3f}")

# Summary
end_errors = []
for scene in alignment:
    g = gt_map.get(scene["scene_id"])
    if g:
        end_errors.append((scene["speech_end"] - g["speech_end"]) * 1000)

print(f"\nEnd Error Direction Analysis:")
print(f"  All early (negative): {sum(1 for e in end_errors if e < -50)}")
print(f"  All late (positive):  {sum(1 for e in end_errors if e > 50)}")
print(f"  Within ±50ms:         {sum(1 for e in end_errors if abs(e) <= 50)}")
print(f"  Mean signed error:    {sum(end_errors)/len(end_errors):+.0f} ms")
