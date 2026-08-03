"""JobManager: creates isolated per-generation workspaces.

Every UI run gets its own unique directory so stale alignment.json,
timeline.json, wrong images, or a previous output.mp4 can never contaminate
another project.

Layout per job:

    <workspace>/jobs/<job_id>/
        input/
            project.json
            <narration original filename>
            images/
                <original image filenames>
        intermediate/
        output/
        logs/
"""

from __future__ import annotations

import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


class JobError(RuntimeError):
    pass


@dataclass
class Job:
    job_id: str
    root: Path
    input_dir: Path
    images_dir: Path
    intermediate_dir: Path
    output_dir: Path
    logs_dir: Path


class JobManager:
    def __init__(self, workspace_root: Path | str):
        self.workspace_root = Path(workspace_root)
        self.jobs_root = self.workspace_root / "jobs"

    # ------------------------------------------------------------------ create
    def create_job(self) -> Job:
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        job_id = uuid.uuid4().hex[:12]
        return self._materialize(job_id)

    def get_job(self, job_id: str) -> Job:
        root = self.jobs_root / job_id
        if not root.is_dir():
            raise JobError(f"Job not found: {job_id}")
        return self._materialize(job_id)

    def _materialize(self, job_id: str) -> Job:
        root = self.jobs_root / job_id
        input_dir = root / "input"
        dirs = {
            "input_dir": input_dir,
            "images_dir": input_dir / "images",
            "intermediate_dir": root / "intermediate",
            "output_dir": root / "output",
            "logs_dir": root / "logs",
        }
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        return Job(job_id=job_id, root=root, **dirs)

    # ------------------------------------------------------------------ writes
    @staticmethod
    def _safe_basename(name: str) -> str:
        base = Path(name).name.strip()
        if not base or base in (".", ".."):
            raise JobError(f"Invalid upload filename: {name!r}")
        return base

    def write_project_json(self, job: Job, source: Path) -> Path:
        source = Path(source)
        if not source.is_file():
            raise JobError(f"Project JSON file not found: {source}")
        dest = job.input_dir / "project.json"
        shutil.copyfile(source, dest)
        return dest

    def write_narration(self, job: Job, source: Path, orig_name: str) -> Path:
        source = Path(source)
        if not source.is_file():
            raise JobError(f"Narration file not found: {source}")
        name = self._safe_basename(orig_name)
        dest = job.input_dir / name
        shutil.copyfile(source, dest)
        return dest

    def write_images(self, job: Job, sources: List[Tuple[str, Path]]) -> List[Path]:
        """Copies uploaded images into the job images dir by ORIGINAL filename.

        The mapping is purely filename-based, so browser upload order never
        matters. ``sources`` is a list of ``(original_filename, local_path)``.
        """
        written: List[Path] = []
        for orig_name, local_path in sources:
            local_path = Path(local_path)
            name = self._safe_basename(orig_name)
            dest = job.images_dir / name
            if dest.exists():
                raise JobError(f"Duplicate image filename: {name}")
            if not local_path.is_file():
                raise JobError(f"Uploaded image not found: {local_path}")
            shutil.copyfile(local_path, dest)
            written.append(dest)
        return written

    # ------------------------------------------------------------------ cleanup
    def remove_job(self, job: Job) -> None:
        if job.root.is_dir():
            shutil.rmtree(job.root, ignore_errors=True)
