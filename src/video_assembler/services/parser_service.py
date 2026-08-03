import json
from pathlib import Path
from video_assembler.models import ProjectInput

class ParserService:
    def __init__(self):
        pass

    def parse_input_json(self, file_path: Path | str) -> ProjectInput:
        """Parses the strict JSON format for MVP into Canonical Scene models."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
            
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        return ProjectInput(**data)
