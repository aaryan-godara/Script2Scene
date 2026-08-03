"""ProjectValidator: fast preflight validation that runs WITHOUT Whisper.

Checks every input the pipeline depends on so the user can be told *before*
transcription/rendering that the project is or is not usable. All error
messages are user-friendly (no Python tracebacks). The full mapping table is
also produced here so the UI can show Scene -> Script -> Image before any
expensive work.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from pydantic import ValidationError

from video_assembler.models import ProjectInput
from video_assembler.services.audio_service import AudioService
from video_assembler.services.image_matcher import ImageMatcher
from video_assembler.services.parser_service import ParserService

try:
    from PIL import Image as PILImage
    HAS_PIL = True
except ImportError:  # pragma: no cover
    HAS_PIL = False

ALLOWED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a"}
ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass
class ValidationOutcome:
    valid: bool
    message: str
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    project_name: Optional[str] = None
    scene_count: int = 0
    image_count: int = 0
    narration_name: Optional[str] = None
    narration_duration: Optional[float] = None
    rows: List[dict] = field(default_factory=list)


class ProjectValidator:
    def parse_and_validate(self, project_json_path: Path | str, narration: Path | str,
                           images_dir: Path | str) -> ValidationOutcome:
        """Parses the uploaded JSON and validates everything. Fast: no Whisper."""
        path = Path(project_json_path)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            project_input = ParserService().parse_input_json(path)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            return ValidationOutcome(valid=False, message="Project validation failed.",
                                     errors=[f"Invalid JSON: {e}"])
        except ValidationError as e:
            first = e.errors()[0] if e.errors() else {}
            loc = ".".join(str(x) for x in first.get("loc", []))
            msg = first.get("msg", "invalid structure")
            return ValidationOutcome(valid=False, message="Project validation failed.",
                                     errors=[f"Invalid project JSON format (missing/incorrect '{loc}'): {msg}"])
        except Exception as e:  # noqa: BLE001 - user-facing boundary
            return ValidationOutcome(valid=False, message="Project validation failed.",
                                     errors=[f"Could not read project file: {e}"])
        return self.validate(project_input, narration, images_dir)

    def validate(self, project_input: ProjectInput, narration: Path | str,
                 images_dir: Path | str) -> ValidationOutcome:
        narration = Path(narration)
        images_dir = Path(images_dir)
        errors: List[str] = []
        warnings: List[str] = []
        narration_duration = None

        # ---------------------------------------------------------------- scenes
        if not project_input.scenes:
            errors.append("Project has no scenes.")

        seen: set = set()
        for sc in project_input.scenes:
            if sc.scene_id is None or sc.scene_id <= 0:
                errors.append(f"Scene has invalid scene_id ({sc.scene_id!r}); scene IDs must be positive integers.")
            if sc.scene_id in seen:
                errors.append(f"Duplicate scene_id {sc.scene_id}.")
            seen.add(sc.scene_id)
            if not (sc.script_text or "").strip():
                errors.append(f"Scene {sc.scene_id} has no script_text.")

        # -------------------------------------------------------------- narration
        if not narration.exists():
            errors.append("Narration file not found.")
        else:
            if narration.stat().st_size == 0:
                errors.append("Narration file is empty (0 bytes).")
            if narration.suffix.lower() not in ALLOWED_AUDIO_EXTENSIONS:
                errors.append(
                    f"Unsupported narration format '{narration.suffix}'; use .mp3, .wav or .m4a.")
            if narration.exists() and narration.suffix.lower() in ALLOWED_AUDIO_EXTENSIONS \
                    and narration.stat().st_size > 0:
                try:
                    meta = AudioService(narration.parent).get_audio_metadata(narration)
                    narration_duration = float(meta.get("duration", 0.0))
                except Exception:  # noqa: BLE001
                    errors.append("Narration could not be decoded.")

        # ---------------------------------------------------------------- images
        matcher = ImageMatcher(images_dir)
        matched_scenes, matcher_errors, matcher_warnings = matcher.match_scenes(project_input.scenes)
        for me in matcher_errors:
            if me.get("type") == "MISSING_EXPLICIT_IMAGE":
                errors.append(
                    f"Scene {me['scene_id']} expects: {me['file']} but that image was not uploaded.")
            elif me.get("type") == "MISSING_IMAGE":
                errors.append(f"Scene {me['scene_id']} has no images.")
        for w in matcher_warnings:
            warnings.append(f"Image not used by any scene: {w['file']}")

        image_count = 0
        for sc in matched_scenes:
            for img_name in sc.images:
                image_count += 1
                img_path = images_dir / img_name
                if not img_path.exists():
                    continue
                if img_path.stat().st_size == 0:
                    errors.append(f"Scene {sc.scene_id} image '{img_name}' is empty (0 bytes).")
                if Path(img_name).suffix.lower() not in ALLOWED_IMAGE_EXTENSIONS:
                    errors.append(
                        f"Scene {sc.scene_id} image '{img_name}' has unsupported format; "
                        f"use .png, .jpg, .jpeg or .webp.")
                if HAS_PIL:
                    try:
                        with PILImage.open(img_path) as im:
                            im.verify()
                    except Exception:  # noqa: BLE001
                        errors.append(f"Scene {sc.scene_id} image '{img_name}' is not a valid image file.")

        valid = not errors
        rows = [
            {"scene_id": s.scene_id, "script_text": s.script_text, "images": ", ".join(s.images)}
            for s in matched_scenes
        ]
        return ValidationOutcome(
            valid=valid,
            message="Project valid" if valid else "Project validation failed.",
            errors=errors,
            warnings=warnings,
            project_name=project_input.project or None,
            scene_count=len(matched_scenes),
            image_count=image_count,
            narration_name=narration.name if narration.exists() else None,
            narration_duration=narration_duration,
            rows=rows,
        )
