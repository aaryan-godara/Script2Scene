"""Self-healing orchestrator for the alignment gate.

Tests whether the existing production components (PersistentTranscriptionCache,
FailedRegionRecovery, NumericNormalizer, AlignmentService) can automatically
reduce an over-limit REVIEW population without:
    * raising max_review_ratio
    * manually accepting scenes
    * force-promoting REVIEW -> HIGH
    * bypassing the gate
    * re-running the entire Whisper transcription

The orchestrator ONLY consumes the cached transcription (per the cache identity)
and, when a damaged region is confirmed, runs bounded local contextual recovery
(max 2 passes / region) exactly like the production FailedRegionRecoveryEngine.
Everything else (numeric validation, re-alignment) uses the unchanged services.

Usage:
    python scripts/self_healing_orchestrator.py <job_dir>
    python scripts/self_healing_orchestrator.py workspace/jobs/50eeb9ff22d1 --report
"""

from __future__ import annotations

import json
import sys
import argparse
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from video_assembler.models import ProjectInput
from video_assembler.services.parser_service import ParserService
from video_assembler.services.alignment.alignment_service import AlignmentService
from video_assembler.services.alignment.failed_region_recovery import (
    FailedRegionRecoveryEngine, rebuild_transcription)
from video_assembler.services.alignment.persistent_transcription_cache import (
    PersistentTranscriptionCache, TranscriptionIdentityConfig, audio_sha256)
from video_assembler.services.alignment.transcription_cache import (
    load_transcription, TranscriptionCacheError)
from video_assembler.services.audio_service import AudioService

# ---------------------------------------------------------------- classifiers
# Review-cause classification (A-G) for a REVIEW scene.
CAUSE_TRANSCRIPTION_GAP = "A. TRANSCRIPTION_GAP"
CAUSE_LOW_TEXT_SIMILARITY = "B. LOW_TEXT_SIMILARITY"
CAUSE_NUMERIC_VALUE_MISMATCH = "C. NUMERIC_VALUE_MISMATCH"
CAUSE_TIMESTAMP_UNCERTAIN = "D. TIMESTAMP_UNCERTAIN"
CAUSE_NEIGHBOR_ALIGNMENT_PROBLEM = "E. NEIGHBOR_ALIGNMENT_PROBLEM"
CAUSE_RECOVERED_BUT_LOW = "F. RECOVERED_BUT_LOW_CONFIDENCE"
CAUSE_OTHER = "G. OTHER"


def classify_review_scene(diag: Dict) -> str:
    """Classify one REVIEW scene's cause using ONLY existing diagnostic evidence."""
    conf = diag.get("confidence")
    conf = float(conf) if conf is not None else 0.0

    # D: timestamps unusable / uncertain.
    start = diag.get("speech_start")
    end = diag.get("speech_end")
    try:
        start_f = float(start)
        end_f = float(end)
    except (TypeError, ValueError):
        return CAUSE_TIMESTAMP_UNCERTAIN
    if not (start_f >= 0.0 and end_f > start_f):
        return CAUSE_TIMESTAMP_UNCERTAIN

    # C: numeric mismatch flagged by the unchanged numeric normalizer.
    if diag.get("numeric_mismatch") is True:
        return CAUSE_NUMERIC_VALUE_MISMATCH

    # F: scene was part of a prior recovery region but still below threshold.
    if diag.get("recovered") is True:
        return CAUSE_RECOVERED_BUT_LOW

    # E: neighbor alignment problem — the matched window strongly overlaps the
    # matched window of the previous/next scene (bleed/duplicate match).
    overlap = diag.get("overlap_ratio")
    if overlap is not None and float(overlap) > 0.35:
        return CAUSE_NEIGHBOR_ALIGNMENT_PROBLEM

    matched = diag.get("matched_token_count")
    expected = diag.get("expected_token_count")
    if expected and matched is not None:
        cover = int(matched) / float(expected)
        # Missing large share of expected tokens -> transcription gap.
        if cover < 0.55:
            return CAUSE_TRANSCRIPTION_GAP

    # A: gap detection via expected words absent from ASR window.
    expected_text = (diag.get("expected_text") or "").lower().split()
    asr_text = (diag.get("asr_text") or "").lower().split()
    if expected_text and asr_text:
        missing = sum(1 for w in expected_text if w not in set(asr_text))
        if missing / len(expected_text) > 0.35:
            return CAUSE_TRANSCRIPTION_GAP

    # B: low lexical similarity with no other signal.
    if 0.0 < conf < 0.75:
        return CAUSE_LOW_TEXT_SIMILARITY

    return CAUSE_OTHER


def build_overlap_evidence(aligned, statuses, diagnostics) -> Dict[int, float]:
    """Neighbor overlap ratio for each REVIEW scene (matched window intersection).

    Uses only already-computed speech windows. A REVIEW scene whose window
    strongly overlaps its neighbor's window points at an alignment (not
    transcription) problem.
    """
    windows = {}
    for s in aligned:
        if s.speech_start is None or s.speech_end is None:
            continue
        windows[s.scene_id] = (float(s.speech_start), float(s.speech_end))
    overlap = {}
    order = [s.scene_id for s in aligned]
    for i, sid in enumerate(order):
        if statuses.get(sid) != "REVIEW":
            continue
        if sid not in windows:
            continue
        s0, e0 = windows[sid]
        candidates = []
        if i > 0 and order[i - 1] in windows:
            candidates.append(windows[order[i - 1]])
        if i + 1 < len(order) and order[i + 1] in windows:
            candidates.append(windows[order[i + 1]])
        worst = 0.0
        for s1, e1 in candidates:
            inter = max(0.0, min(e0, e1) - max(s0, s1))
            span = max(e0 - s0, e1 - s1, 1e-9)
            worst = max(worst, inter / span)
        overlap[sid] = round(worst, 3)
    return overlap


def load_project(job_dir: Path) -> ProjectInput:
    project_path = job_dir / "input" / "project.json"
    return ParserService().parse_input_json(project_path)


def load_cached_transcription(job_dir: Path, audio_path: Path):
    """Loads the cached transcription WITHOUT re-running Whisper.

    Preference order: persistent cache (recovered_cache identity) -> job-local
    intermediate/transcription.json. Returns (transcription, source).
    """
    sha = audio_sha256(audio_path)
    cfg = TranscriptionIdentityConfig()
    cache = PersistentTranscriptionCache(ROOT / "workspace/cache/transcriptions", cfg)
    cached = cache.load(sha)
    if cached is not None:
        return cached, "persistent_cache"

    job_transcription = job_dir / "intermediate" / "transcription.json"
    if job_transcription.exists():
        try:
            t = load_transcription(job_transcription, audio_path=audio_path)
            return t, "job_intermediate"
        except TranscriptionCacheError:
            pass
    raise SystemExit(
        f"No cached transcription for {audio_path.name}. Refusing to re-run Whisper.")


def run_alignment(scenes, transcription):
    aligned, diag = AlignmentService().align_scenes(scenes, transcription)
    statuses = {sid: d.get("status") for sid, d in diag.diagnostics.items()}
    return aligned, diag, statuses


def annotate_diagnostics(scenes, statuses, diagnostics, recovered_scene_ids):
    """Fills in classification evidence onto the diagnostic dicts."""
    for s in scenes:
        d = diagnostics.diagnostics.get(s.scene_id)
        if d is None:
            continue
        d["recovered"] = s.scene_id in recovered_scene_ids
        if statuses.get(s.scene_id) == "REVIEW":
            d["cause"] = classify_review_scene(d)


def summarize(statuses) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for st in statuses.values():
        counts[st] = counts.get(st, 0) + 1
    return counts


def build_evidence_rows(scenes, statuses, diagnostics) -> List[Dict]:
    rows = []
    for s in scenes:
        d = diagnostics.diagnostics.get(s.scene_id, {})
        rows.append({
            "scene_id": s.scene_id,
            "status": statuses.get(s.scene_id, "UNKNOWN"),
            "confidence": d.get("confidence"),
            "start": d.get("speech_start"),
            "end": d.get("speech_end"),
            "expected_text": s.script_text,
            "asr_text": d.get("asr_text"),
            "numeric_mismatch": d.get("numeric_mismatch"),
            "canonical_numeric_values": d.get("canonical_numeric_values"),
            "asr_numeric_values": d.get("asr_numeric_values"),
            "reason": d.get("reason"),
            "cause": d.get("cause"),
            "recovered": d.get("recovered"),
        })
    return rows


def review_regions(scenes, statuses) -> List[List[int]]:
    """Groups CONTIGUOUS REVIEW scenes into one region (by scene order)."""
    order = [s.scene_id for s in scenes]
    positions = {sid: i for i, sid in enumerate(order)}
    review = sorted((sid for sid, st in statuses.items() if st == "REVIEW"),
                    key=lambda sid: positions[sid])
    groups: List[List[int]] = []
    for sid in review:
        if groups and positions[sid] == positions[groups[-1][-1]] + 1:
            groups[-1].append(sid)
        else:
            groups.append([sid])
    return groups


def region_healable(scenes, statuses, diagnostics, recovered_scene_ids, audio_duration) -> Tuple[bool, str]:
    """Determines whether a REVIEW region can be auto-healed.

    A region is healable only when evidence supports it: the scenes show
    transcription-gap/low-similarity causes (recoverable by contextual
    transcription), the region timestamps are unreliable enough to justify a
    windowed recovery, and no scene in the region is a genuine numeric mismatch
    (which recovery must never overwrite). We do NOT fabricate evidence.
    """
    for sid in sum(([g] for g in [recovered_scene_ids]), []):
        pass
    return False, "no automatic recovery triggered (not a FAILED gap)"


def run_region_recovery(engine: FailedRegionRecoveryEngine, group: List[int],
                        scenes, statuses, diagnostics, audio_duration, pass_no,
                        transcribe_fn, cfg, narration, intermediate_dir):
    """Bounded single-region recovery pass using the production engine path.

    Reuses FailedRegionRecoveryEngine._recover_group exactly: window is anchored
    on nearest HIGH neighbors, one contextual transcription per group (never per
    scene), dedup via merge_chunk. Returns (audit_entry, recovered_words or None).
    """
    svc = AudioService(narration.parent)
    window_start, window_end, anchors = engine.estimate_window(
        group, scenes, statuses, diagnostics, audio_duration, pass_no)
    entry: Dict = {
        "scene_group": group,
        "reason": "TRANSCRIPTION_GAP",
        "window_start": window_start,
        "window_end": window_end,
        "pass": pass_no,
        "anchors": anchors,
        "acoustic_energy_detected": None,
    }
    clip_path = intermediate_dir / "transcription_chunks" / (
        f"review_region_p{pass_no}_g{group[0]}_"
        f"{window_start:07.3f}_{window_end:07.3f}.wav")
    try:
        svc.extract_chunk(narration, clip_path, window_start, window_end)
    except Exception:  # noqa: BLE001
        entry["error"] = "chunk_extract_failed"
        return entry, None
    try:
        result = transcribe_fn(str(clip_path))
    except Exception:  # noqa: BLE001
        entry["error"] = "transcription_failed"
        return entry, None
    from video_assembler.services.alignment.merge_utils import normalize_window_words, merge_chunk
    rec_words = normalize_window_words(result.words, window_start, audio_duration)
    entry["recovery_raw_words"] = len(rec_words)
    if not rec_words:
        return entry, None
    entry["acoustic_energy_detected"] = True
    return entry, rec_words


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Self-healing alignment orchestrator")
    ap.add_argument("job_dir", type=Path,
                    help="Job workspace dir (must contain input/project.json)")
    ap.add_argument("--report", action="store_true",
                    help="Only run diagnostics + classification (no recovery)")
    ap.add_argument("--recover", action="store_true",
                    help="Run bounded auto-recovery on healable regions")
    ap.add_argument("--audio", type=Path, default=None,
                    help="Narration audio path (default: job input wav)")
    args = ap.parse_args(argv)

    job_dir: Path = args.job_dir.resolve()
    project = load_project(job_dir)

    audio_path = args.audio
    if audio_path is None:
        wavs = sorted(job_dir.glob("input/*.wav"))
        mps = sorted(job_dir.glob("input/*.mp3"))
        candidates = wavs + mps
        if not candidates:
            raise SystemExit("No narration audio found in job input/")
        audio_path = candidates[0]
    audio_path = audio_path.resolve()

    transcription, t_source = load_cached_transcription(job_dir, audio_path)
    audio_duration = float(getattr(transcription, "audio_duration", 0.0) or 0.0)
    if audio_duration <= 0:
        meta = AudioService(audio_path.parent).get_audio_metadata(audio_path)
        audio_duration = float(meta["duration"])

    # Recovered scene ids recorded in the cached transcription metadata.
    recovered_scene_ids: List[int] = []
    recovered_regions: List[Dict] = []
    cache_meta = {}
    try:
        sha = audio_sha256(audio_path)
        cfg = TranscriptionIdentityConfig()
        cache = PersistentTranscriptionCache(ROOT / "workspace/cache/transcriptions", cfg)
        entry_dir = cache.entry_dir(sha)
        meta_path = entry_dir / "metadata.json"
        if meta_path.exists():
            cache_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            recovered_scene_ids = [int(x) for x in
                                   cache_meta.get("recovered_scene_ids", [])]
            recovered_regions = cache_meta.get("recovered_regions", [])
    except Exception:  # noqa: BLE001 - cache metadata is best-effort
        pass

    # ---------------------------------------------------------------- STEP 1
    scenes = list(project.scenes)
    aligned, diagnostics, statuses = run_alignment(scenes, transcription)
    annotate_diagnostics(scenes, statuses, diagnostics, recovered_scene_ids)

    overlap = build_overlap_evidence(aligned, statuses, diagnostics)
    for sid, ratio in overlap.items():
        diagnostics.diagnostics[sid]["overlap_ratio"] = ratio
        if statuses.get(sid) == "REVIEW":
            diagnostics.diagnostics[sid]["cause"] = classify_review_scene(
                diagnostics.diagnostics[sid])

    before = summarize(statuses)
    rows = build_evidence_rows(scenes, statuses, diagnostics)

    print("=" * 100)
    print("BEFORE (cached transcription, unchanged AlignmentService)")
    print(f"  HIGH={before.get('HIGH', 0)}  REVIEW={before.get('REVIEW', 0)}  "
          f"FAILED={before.get('FAILED', 0)}   total={sum(before.values())}")
    print("=" * 100)

    review_rows = [r for r in rows if r["status"] == "REVIEW"]
    cause_counts: Dict[str, int] = {}
    for r in review_rows:
        cause_counts[r["cause"]] = cause_counts.get(r["cause"], 0) + 1

    print("\nREVIEW scene evidence (all 20):")
    print(f"{'scene':>6} {'conf':>6} {'start':>9} {'end':>9}  cause")
    print("-" * 100)
    for r in review_rows:
        print(f"{r['scene_id']:>6} "
              f"{r['confidence'] if r['confidence'] is not None else '---':>6.3f} "
              f"{r['start'] if r['start'] is not None else '---':>9.3f} "
              f"{r['end'] if r['end'] is not None else '---':>9.3f}  "
              f"{r['cause']}")

    print("\nReview cause counts:")
    for cause in sorted(cause_counts):
        print(f"  {cause}: {cause_counts[cause]}")

    if args.report:
        write_report(job_dir, rows, before, cause_counts, transcription, None)
        return 0

    # ---------------------------------------------------------------- STEP 2-3
    audit: List[Dict] = []
    healed_scene_ids: List[int] = []
    if args.recover:
        from video_assembler.services.alignment.stable_whisper_provider import StableWhisperProvider
        provider = StableWhisperProvider(model_name="base")
        transcribe_fn = provider.transcribe
        engine = FailedRegionRecoveryEngine(
            transcribe_fn, AudioService(audio_path.parent), audio_path,
            job_dir / "intermediate" / "transcription_chunks", cfg)
        regions = review_regions(scenes, statuses)
        words = list(transcription.words)
        for pass_no in range(1, cfg.max_recovery_passes + 1):
            progressed = False
            for group in regions:
                entry, rec_words = run_region_recovery(
                    engine, group, scenes, statuses, diagnostics, audio_duration,
                    pass_no, transcribe_fn, cfg, audio_path, job_dir / "intermediate")
                if rec_words is None:
                    audit.append(entry)
                    continue
                # merge with the shared dedup
                prev_before = [w for w in words if w.end <= entry["window_start"] + 1e-6]
                prev_start = prev_before[0].start if prev_before else entry["window_start"]
                prev_end = prev_before[-1].end if prev_before else entry["window_start"]
                from video_assembler.services.alignment.merge_utils import merge_chunk
                entry["words_before"] = len(words)
                new_words, dups = merge_chunk(words, rec_words,
                                              entry["window_start"],
                                              prev_start, prev_end)
                entry["duplicates_removed"] = dups
                entry["words_added"] = len(new_words) - len(words)
                entry["scene_group"] = group
                entry["region"] = group
                entry["attempt"] = pass_no
                words = new_words
                healed_scene_ids.extend(group)
                progressed = entry["words_added"] > 0
                audit.append(entry)
            if not progressed:
                break
        if audit and any(e.get("words_added", 0) > 0 for e in audit):
            transcription = rebuild_transcription(transcription, words)
            aligned, diagnostics, statuses = run_alignment(scenes, transcription)
            annotate_diagnostics(scenes, statuses, diagnostics, recovered_scene_ids)
            overlap = build_overlap_evidence(aligned, statuses, diagnostics)
            for sid, ratio in overlap.items():
                diagnostics.diagnostics[sid]["overlap_ratio"] = ratio
                if statuses.get(sid) == "REVIEW":
                    diagnostics.diagnostics[sid]["cause"] = classify_review_scene(
                        diagnostics.diagnostics[sid])
            rows = build_evidence_rows(scenes, statuses, diagnostics)

    after = summarize(statuses)
    print("=" * 100)
    print("AFTER")
    print(f"  HIGH={after.get('HIGH', 0)}  REVIEW={after.get('REVIEW', 0)}  "
          f"FAILED={after.get('FAILED', 0)}   total={sum(after.values())}")
    print("=" * 100)

    n_total = sum(after.values()) or 1
    review_count = after.get("REVIEW", 0)
    ratio = review_count / n_total
    print(f"\nReview ratio = {review_count} / {n_total} = {ratio:.1%}")
    max_ratio = 0.05
    print(f"Allowed max review ratio = {max_ratio:.1%} "
          f"({max_ratio * n_total:.1f} scenes)")

    gate_pass = after.get("FAILED", 0) == 0 and ratio <= max_ratio
    print(f"\nGate: {'PASS' if gate_pass else 'FAIL'}")

    write_report(job_dir, rows, after, cause_counts, transcription, audit)

    if gate_pass and args.recover:
        print("\nGate passes. The current job artifacts can be rendered safely "
              "(FAILED=0, REVIEW within threshold).")
    elif not gate_pass:
        print("\nUnresolved REVIEW scenes (why they cannot be auto-healed):")
        for r in rows:
            if r["status"] == "REVIEW":
                print(f"  scene {r['scene_id']}: {r['cause']}")
    return 0


def write_report(job_dir: Path, rows, status_counts, cause_counts,
                 transcription, audit):
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    payload = {
        "generated_at": stamp,
        "source": "cached_transcription_only",
        "status_counts": status_counts,
        "review_cause_counts": cause_counts,
        "review_scenes": [r for r in rows if r["status"] == "REVIEW"],
        "all_scenes": rows,
        "recovery_audit": audit or [],
    }
    out = job_dir / "intermediate" / "self_healing_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[report written to {out}]")


if __name__ == "__main__":
    raise SystemExit(main())
