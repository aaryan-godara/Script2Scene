import os
import shutil
from pathlib import Path

class ProjectService:
    def __init__(self, base_storage_dir: str | Path):
        self.base_storage_dir = Path(base_storage_dir)

    def get_project_dir(self, project_id: str) -> Path:
        return self.base_storage_dir / "projects" / project_id

    def init_project(self, project_id: str) -> Path:
        """Initializes the standard directory structure for a project."""
        project_dir = self.get_project_dir(project_id)
        
        directories = [
            "input",
            "images",
            "intermediate",
            "output",
            "logs",
        ]
        
        for d in directories:
            (project_dir / d).mkdir(parents=True, exist_ok=True)
            
        return project_dir

    def get_dir(self, project_id: str, dir_name: str) -> Path:
        """Helper to get a specific subdirectory path for a project."""
        if dir_name not in ["input", "images", "intermediate", "output", "logs"]:
            raise ValueError(f"Invalid directory name: {dir_name}")
        return self.get_project_dir(project_id) / dir_name
