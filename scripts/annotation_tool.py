import os
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import argparse

app = FastAPI()

PROJECT_DIR = None

class GroundTruthAnnotation(BaseModel):
    scene_id: int
    speech_start: float
    speech_end: float
    notes: str = ""

class GroundTruth(BaseModel):
    project: str
    annotations: list[GroundTruthAnnotation] = []

@app.get("/")
def read_index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")

@app.get("/api/project")
def get_project():
    project_json_path = PROJECT_DIR / "input" / "project.json"
    if not project_json_path.exists():
        raise HTTPException(status_code=404, detail="project.json not found in input directory")
    
    with open(project_json_path, "r", encoding="utf-8") as f:
        return json.load(f)

@app.get("/api/audio")
def get_audio():
    audio_path = PROJECT_DIR / "input" / "narration.mp3"
    if not audio_path.exists():
        audio_path = PROJECT_DIR / "input" / "narration.wav"
        if not audio_path.exists():
            raise HTTPException(status_code=404, detail="Audio file (narration.mp3 or narration.wav) not found")
    
    return FileResponse(audio_path, media_type="audio/mpeg" if audio_path.suffix == ".mp3" else "audio/wav")

@app.get("/api/ground_truth")
def get_ground_truth():
    gt_path = PROJECT_DIR / "intermediate" / "ground_truth.json"
    if not gt_path.exists():
        return {"project": "test_project", "annotations": []}
    
    with open(gt_path, "r", encoding="utf-8") as f:
        return json.load(f)

@app.post("/api/save")
def save_ground_truth(gt: GroundTruth):
    gt_path = PROJECT_DIR / "intermediate" / "ground_truth.json"
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(gt.model_dump(), f, indent=2)
        
    return {"status": "success"}

def main():
    global PROJECT_DIR
    parser = argparse.ArgumentParser(description="Golden Test Annotation Tool")
    parser.add_argument("--project-dir", required=True, help="Path to the project directory")
    args = parser.parse_args()
    
    PROJECT_DIR = Path(args.project_dir)
    if not PROJECT_DIR.exists():
        print(f"Error: Project directory {PROJECT_DIR} does not exist.")
        return
        
    print(f"Starting annotation tool for project: {PROJECT_DIR}")
    uvicorn.run(app, host="127.0.0.1", port=8000)

if __name__ == "__main__":
    main()
