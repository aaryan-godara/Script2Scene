"""Review-workflow report for manual alignment.

The pipeline writes an ``alignment report`` (``alignment_review.json``) whenever
a job has REVIEW or FAILED scenes so a human can inspect transcription evidence
and decide what to do. Reports are purely informational: they never change scene
scores and never write raw ASR data back into the pipeline. Manual overrides
recorded through ``AlignmentReviewStore`` keep the scene marked REVIEW and are
never auto-promoted, and stopping short of a committed render never touches the
frozen assets.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from video_assembler.services.alignment.provider_base import TranscriptionResult
from video_assembler.models import Scene


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class AlignmentReviewRecord:
    """A single manual alignment decision recorded for one scene."""

    scene_id: str
    status: str  # "accept" | "override"
    speech_start: Optional[float] = None
    speech_end: Optional[float] = None
    note: Optional[str] = None
    reviewed_at: str = _now()

    def to_dict(self) -> Dict[str, object]:
        return {
            "scene_id": self.scene_id,
            "status": self.status,
            "speech_start": self.speech_start,
            "speech_end": self.speech_end,
            "note": self.note,
            "reviewed_at": self.reviewed_at,
        }


class AlignmentReviewStore:
    """Persists manual review decisions for a job's scene set.

    Decisions are written to ``reviews_dir/alignment_review.json``. Recording a
    manual override with explicit timestamps keeps the scene marked REVIEW; it
    never auto-promotes a scene to HIGH. The store is independent of rendering
    so a reviewed, frozen job is never silently overwritten.
    """

    def __init__(self, reviews_dir: Path):
        self.reviews_dir = Path(reviews_dir)
        self.reviews_dir.mkdir(parents=True, exist_ok=True)
        self._records: Dict[str, AlignmentReviewRecord] = {}
        self.load()

    def _path(self) -> Path:
        return self.reviews_dir / "alignment_review_actions.json"

    def load(self) -> None:
        self._records = {}
        path = self._path()
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                return
            for item in data:
                if not isinstance(item, dict):
                    continue
                rec = AlignmentReviewRecord(
                    scene_id=str(item.get("scene_id")),
                    status=str(item.get("status")),
                    speech_start=item.get("speech_start"),
                    speech_end=item.get("speech_end"),
                    note=item.get("note"),
                    reviewed_at=item.get("reviewed_at") or _now(),
                )
                self._records[rec.scene_id] = rec
        except (json.JSONDecodeError, TypeError, ValueError, KeyError):
            self._records = {}

    def review(self, scene_id: str, status: str, speech_start: Optional[float],
               speech_end: Optional[float], note: Optional[str] = None) -> AlignmentReviewRecord:
        if status not in ("accept", "override"):
            raise ValueError("status must be 'accept' or 'override'")
        if (speech_start is None) != (speech_end is None):
            raise ValueError("speech_start and speech_end must both be set or both omitted")
        if status == "override" and speech_start is None:
            raise ValueError("override requires explicit speech_start/speech_end")
        if status == "override" and not (speech_start < speech_end):
            raise ValueError("override requires start < end")
        rec = AlignmentReviewRecord(
            scene_id=scene_id,
            status=status,
            speech_start=speech_start,
            speech_end=speech_end,
            note=note,
        )
        self._records[scene_id] = rec
        self._save()
        return rec

    def accept(self, scene_id: str, note: Optional[str] = None) -> AlignmentReviewRecord:
        return self.review(scene_id, "accept", None, None, note)

    def override(self, scene_id: str, speech_start: float, speech_end: float,
                 note: Optional[str] = None) -> AlignmentReviewRecord:
        return self.review(scene_id, "override", speech_start, speech_end, note)

    def _save(self) -> None:
        payload = json.dumps(
            [r.to_dict() for r in self._records.values()],
            indent=2, ensure_ascii=False)
        self._path().write_text(payload, encoding="utf-8")

    def get(self, scene_id: str) -> Optional[AlignmentReviewRecord]:
        return self._records.get(scene_id)

    def all(self) -> List[AlignmentReviewRecord]:
        return list(self._records.values())


def build_review_report(scenes: List[Scene], statuses: Dict[str, str],
                        diagnostics, transcription: Optional[TranscriptionResult] = None) -> Dict[str, object]:
    """Build a review payload covering every scene requiring attention.

    The payload only references existing evidence (asr text, timestamps,
    confidence, numeric consistency). It does not synthesize values.
    """
    summary: Dict[str, int] = {}
    blocks: List[Dict[str, object]] = []
    get = getattr(diagnostics, "get", None)
    for scene in scenes:
        status = statuses.get(scene.scene_id, "UNKNOWN")
        summary[status] = summary.get(status, 0) + 1
        diag = get(scene.scene_id) if get else None
        diag = diag if isinstance(diag, dict) else {}
        blocks.append({
            "scene_id": scene.scene_id,
            "status": status,
            "confidence": diag.get("confidence"),
            "speech_start": diag.get("speech_start"),
            "speech_end": diag.get("speech_end"),
            "expected_text": scene.script_text,
            "asr_text": diag.get("asr_text"),
            "expected_numeric_values": diag.get("canonical_numeric_values"),
            "asr_numeric_values": diag.get("asr_numeric_values"),
            "numeric_consistency": diag.get("numeric_consistency"),
            "reason": diag.get("reason"),
        })
    return {
        "summary": summary,
        "generated_at": _now(),
        "note": ("Manual review required before rendering. Manual overrides keep "
                 "the scene marked REVIEW and are never auto-promoted."),
        "blocks": blocks,
    }


def write_review_report(report_path: Path, scenes: List[Scene],
                        statuses: Dict[str, str], diagnostics,
                        transcription: Optional[TranscriptionResult] = None) -> Path:
    """Write an alignment review report to ``report_path``."""
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_review_report(scenes, statuses, diagnostics, transcription)
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    return report_path