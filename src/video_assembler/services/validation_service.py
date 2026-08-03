from pathlib import Path
from video_assembler.models import ProjectInput, ValidationResult

class ValidationService:
    def __init__(self, project_dir: Path):
        self.project_dir = Path(project_dir)

    def validate_project_assets(self, parsed_input: ProjectInput, matcher_errors: list, matcher_warnings: list) -> ValidationResult:
        errors = list(matcher_errors)
        warnings = list(matcher_warnings)
        
        # Check audio
        audio_path = self.project_dir / "input" / "narration.mp3"
        if not audio_path.exists():
            # Fallback to wav or m4a
            if (self.project_dir / "input" / "narration.wav").exists():
                audio_path = self.project_dir / "input" / "narration.wav"
            elif (self.project_dir / "input" / "narration.m4a").exists():
                audio_path = self.project_dir / "input" / "narration.m4a"
            else:
                errors.append({"type": "MISSING_AUDIO", "file": "narration.[mp3|wav|m4a]"})
        elif audio_path.stat().st_size == 0:
            errors.append({"type": "ZERO_BYTE_AUDIO", "file": audio_path.name})
            
        # Check scenes
        if not parsed_input.scenes:
            errors.append({"type": "NO_SCENES_FOUND"})
            
        # Check duplicate scene IDs
        seen_ids = set()
        for scene in parsed_input.scenes:
            if scene.scene_id in seen_ids:
                errors.append({"type": "DUPLICATE_SCENE_ID", "scene_id": scene.scene_id})
            seen_ids.add(scene.scene_id)
            
        is_valid = len(errors) == 0
        return ValidationResult(valid=is_valid, errors=errors, warnings=warnings)
