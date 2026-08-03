import json
import statistics
import argparse
from pathlib import Path

def classify_error(scene_id, start_err, end_err):
    if max(start_err, end_err) <= 0.100:
        return "UNKNOWN"
    return "PROVIDER_ERROR"

def calculate_metrics(alignment_data, ground_truth_data):
    start_errors = []
    end_errors = []
    start_signed = []
    end_signed = []
    
    if isinstance(ground_truth_data, dict) and "annotations" in ground_truth_data:
        ground_truth_data = ground_truth_data["annotations"]
        
    gt_map = {item['scene_id']: item for item in ground_truth_data}
    
    high_count = 0
    review_count = 0
    failed_count = 0
    
    worst_scenes = []
    
    for scene in alignment_data:
        scene_id = scene.get('scene_id')
        status = scene.get('status', 'FAILED')
        
        if status == 'HIGH':
            high_count += 1
        elif status == 'REVIEW':
            review_count += 1
        else:
            failed_count += 1
            
        gt = gt_map.get(scene_id)
        if gt and status != 'FAILED':
            pred_start = scene.get('speech_start')
            pred_end = scene.get('speech_end')
            gt_start = gt.get('speech_start')
            gt_end = gt.get('speech_end')
            
            if pred_start is not None and gt_start is not None:
                err_start = abs(pred_start - gt_start)
                start_errors.append(err_start)
                start_signed.append(pred_start - gt_start)
            else:
                err_start = None
                
            if pred_end is not None and gt_end is not None:
                err_end = abs(pred_end - gt_end)
                end_errors.append(err_end)
                end_signed.append(pred_end - gt_end)
            else:
                err_end = None
                
            if err_start is not None and err_end is not None:
                worst_scenes.append({
                    "scene_id": scene_id,
                    "max_err": max(err_start, err_end),
                    "start_err": err_start,
                    "end_err": err_end,
                    "classification": classify_error(scene_id, err_start, err_end)
                })

    def get_stats(errors):
        if not errors:
            return 0, 0, 0, 0
        return (
            sum(errors) / len(errors),
            statistics.median(errors),
            max(errors),
            statistics.quantiles(errors, n=100)[94] if len(errors) >= 100 else sorted(errors)[int(len(errors) * 0.95)] if errors else 0
        )

    start_mae, start_med, start_max, start_p95 = get_stats(start_errors)
    end_mae, end_med, end_max, end_p95 = get_stats(end_errors)
    
    all_errors = start_errors + end_errors
    total_boundaries = len(all_errors)
    
    thresholds = [50, 100, 200, 300, 500]
    def boundary_accuracy(errors):
        n = len(errors)
        return {t: (sum(1 for e in errors if e * 1000 <= t) / n * 100 if n else 0.0) for t in thresholds}
    start_accuracy = boundary_accuracy(start_errors)
    end_accuracy = boundary_accuracy(end_errors)

    start_signed_mean = sum(start_signed) / len(start_signed) if start_signed else 0.0
    end_signed_mean = sum(end_signed) / len(end_signed) if end_signed else 0.0

    return {
        "start_mae": start_mae, "start_med": start_med, "start_max": start_max, "start_p95": start_p95,
        "end_mae": end_mae, "end_med": end_med, "end_max": end_max, "end_p95": end_p95,
        "start_signed_mean": start_signed_mean,
        "end_signed_mean": end_signed_mean,
        "high_count": high_count, "review_count": review_count, "failed_count": failed_count,
        "start_accuracy": start_accuracy,
        "end_accuracy": end_accuracy,
        "worst_scenes": sorted(worst_scenes, key=lambda x: x["max_err"], reverse=True)[:3]
    }

def main():
    parser = argparse.ArgumentParser(description="Benchmark Alignment output against Ground Truth")
    parser.add_argument("--alignment", required=True, help="Path to alignment.json")
    parser.add_argument("--ground-truth", required=True, help="Path to ground_truth.json")
    parser.add_argument("--transcription", required=True, help="Path to transcription.json")
    parser.add_argument("--output", required=False, help="Optional path to write benchmark JSON")
    args = parser.parse_args()

    with open(args.alignment, "r") as f:
        alignment_data = json.load(f)
        
    with open(args.ground_truth, "r") as f:
        ground_truth_data = json.load(f)
        
    with open(args.transcription, "r") as f:
        transcription_data = json.load(f)

    metrics = calculate_metrics(alignment_data, ground_truth_data)
    
    duration = transcription_data.get("audio_duration", 0)
    processing = transcription_data.get("processing_seconds", 0)
    realtime_factor = processing / duration if duration > 0 else 0

    print("ALIGNMENT BENCHMARK")
    print("===================")
    print(f"Provider: {transcription_data.get('provider')}")
    print(f"Model: {transcription_data.get('model')}")
    print(f"Device: {transcription_data.get('device')}")
    print(f"Audio Duration: {duration:.2f} sec")
    print(f"Scenes: {len(alignment_data)}")
    print("\nACCURACY")
    print("--------")
    print(f"Start MAE:       {metrics['start_mae']*1000:.0f} ms")
    print(f"End MAE:         {metrics['end_mae']*1000:.0f} ms")
    print(f"Start Median:    {metrics['start_med']*1000:.0f} ms")
    print(f"End Median:      {metrics['end_med']*1000:.0f} ms")
    print(f"Start P95:       {metrics['start_p95']*1000:.0f} ms")
    print(f"End P95:         {metrics['end_p95']*1000:.0f} ms")
    print(f"Max Start:       {metrics['start_max']*1000:.0f} ms")
    print(f"Max End:         {metrics['end_max']*1000:.0f} ms")
    print(f"Mean Signed Start: {metrics['start_signed_mean']*1000:.0f} ms")
    print(f"Mean Signed End:   {metrics['end_signed_mean']*1000:.0f} ms")
    print("\nBOUNDARY ACCURACY")
    print("-----------------")
    print("Within +-t ms:  START    END")
    for t in [50, 100, 200, 300, 500]:
        print(f"   +-{t:>3} ms:   {metrics['start_accuracy'][t]:5.0f}%  {metrics['end_accuracy'][t]:5.0f}%")
    print("\nALIGNMENT STATUS")
    print("----------------")
    print(f"HIGH:    {metrics['high_count']}")
    print(f"REVIEW:  {metrics['review_count']}")
    print(f"FAILED:  {metrics['failed_count']}")
    print("\nPERFORMANCE")
    print("-----------")
    print(f"Processing: {processing:.1f} sec")
    print(f"Realtime Factor: {realtime_factor:.2f}x")
    print("\nWORST SCENES")
    print("------------")
    for scene in metrics['worst_scenes']:
        print(f"Scene {scene['scene_id']} ({scene['classification']})")
        print(f"Start error: {scene['start_err']*1000:.0f} ms")
        print(f"End error: {scene['end_err']*1000:.0f} ms\n")

    if args.output:
        result = {
            "alignment_path": str(args.alignment),
            "ground_truth_path": str(args.ground_truth),
            "provider": transcription_data.get('provider'),
            "model": transcription_data.get('model'),
            "device": transcription_data.get('device'),
            "audio_duration_sec": duration,
            "processing_sec": processing,
            "realtime_factor": realtime_factor,
            "metrics": metrics
        }
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Wrote benchmark JSON to {args.output}")

if __name__ == "__main__":
    main()
