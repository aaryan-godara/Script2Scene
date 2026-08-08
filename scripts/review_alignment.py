"""Review workflow CLI for alignment.

Inspects the informational alignment report a pipeline run wrote for a job and
records manual decisions. Manual actions NEVER promote a scene to HIGH and never
rewrite raw ASR data; overrides simply record explicit timestamps a reviewer
confirmed. The scene stays REVIEW until rerun/realignment.

Usage:
  python review_alignment.py list <intermediate_dir>
  python review_alignment.py accept <intermediate_dir> <scene_id>
  python review_alignment.py override <intermediate_dir> <scene_id> <start> <end>
  python review_alignment.py show <intermediate_dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_assembler.services.alignment.review_report import AlignmentReviewStore


def _load_report(intermediate_dir: Path):
    path = intermediate_dir / "alignment_review.json"
    if not path.exists():
        raise SystemExit(f"No alignment report at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def cmd_list(intermediate_dir: Path):
    data = _load_report(intermediate_dir)
    print("Summary:", json.dumps(data["summary"]))
    for b in data["blocks"]:
        status = b["status"]
        mark = "" if status in ("HIGH",) else (
            "  <-- needs review" if status == "REVIEW" else "  <-- HARD BLOCK")
        print(f"scene {b['scene_id']:<5} {status:<7} "
              f"confidence={b['confidence']} "
              f"[{b['speech_start']} .. {b['speech_end']}] "
              f"{b.get('reason') or ''}{mark}")


def cmd_show(intermediate_dir: Path, scene_id: str):
    data = _load_report(intermediate_dir)
    for b in data["blocks"]:
        if str(b["scene_id"]) == str(scene_id):
            print(json.dumps(b, indent=2, ensure_ascii=False))
            return
    raise SystemExit(f"scene {scene_id} not found in report")


def cmd_accept(intermediate_dir: Path, reviews_dir: Path, scene_id: str):
    data = _load_report(intermediate_dir)
    ids = {str(b["scene_id"]) for b in data["blocks"]}
    if str(scene_id) not in ids:
        raise SystemExit(f"scene {scene_id} not found in report")
    store = AlignmentReviewStore(reviews_dir)
    store.accept(scene_id)
    print(f"accepted scene {scene_id} (stays REVIEW; not auto-promoted).")


def cmd_override(intermediate_dir: Path, reviews_dir: Path, scene_id: str,
                 start: float, end: float):
    data = _load_report(intermediate_dir)
    ids = {str(b["scene_id"]) for b in data["blocks"]}
    if str(scene_id) not in ids:
        raise SystemExit(f"scene {scene_id} not found in report")
    if start >= end:
        raise SystemExit("start must be < end")
    store = AlignmentReviewStore(reviews_dir)
    store.override(scene_id, start, end)
    print(f"overrode scene {scene_id} timestamps [{start}, {end}] "
          "(stays REVIEW; never auto-promoted)")


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 1
    sub = argv[1]
    intermediate_dir = Path(argv[2])
    reviews_dir = intermediate_dir
    if sub == "list":
        cmd_list(intermediate_dir)
    elif sub == "show":
        if len(argv) < 4:
            print(__doc__)
            return 1
        show(intermediate_dir, argv[3])
    elif sub == "accept":
        if len(argv) < 4:
            print(__doc__)
            return 1
        cmd_accept(intermediate_dir, reviews_dir, argv[3])
    elif sub == "override":
        if len(argv) < 6:
            print(__doc__)
            return 1
        cmd_override(intermediate_dir, reviews_dir, argv[3],
                     float(argv[4]), float(argv[5]))
    else:
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))