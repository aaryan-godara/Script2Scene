# Script2Scene

Automated narration-to-visual video assembly. Upload a narration audio file and a
JSON scene map, and Script2Scene generates a synchronized MP4 video in which each
scene's image is displayed during the corresponding spoken narration.

## Features

- **Scene alignment** – matches each scene's script against the transcribed narration
  using token-level similarity, producing word-accurate start/end timestamps.
- **Long-form chunked transcription** – long narrations are transcribed in overlapping
  chunks instead of one pass, then merged back into a single global word timeline
  (`ChunkedTranscriptionProvider`).
- **Acoustic boundary refinement** – refines scene end boundaries against actual
  speech/silence in the audio.
- **Confidence gating** – scenes align as `HIGH`, `REVIEW`, or `FAILED`:
  - `FAILED` scenes always stop generation.
  - `REVIEW` scenes are allowed only when their timestamps are usable and they stay
    within `max_review_ratio` (default 5%) of the project; they render with a warning.
- **Transcription cache safety** – cached `transcription.json` files are only reused
  when the audio SHA-256 matches the current narration.
- **Web UI and CLI** – both drive the same `PipelineRunner` core.

## Project layout

```
├── input/                     # Default project inputs (project.json, narration, images)
├── intermediate/              # Default cache/artifacts (transcription.json, timeline, ...)
├── output/                    # Rendered videos
├── scripts/run_pipeline.py    # CLI entry point
├── webui.py                   # Gradio web UI
├── src/video_assembler/
│   ├── services/
│   │   ├── pipeline_runner.py # Shared pipeline orchestration
│   │   ├── alignment/         # Transcription providers + alignment service
│   │   │   ├── stable_whisper_provider.py
│   │   │   ├── chunked_transcription_provider.py
│   │   │   ├── alignment_service.py
│   │   │   ├── transcription_cache.py
│   │   │   └── acoustic_boundary_refiner.py
│   │   ├── timeline_service.py
│   │   ├── render_service.py
│   │   └── ...
│   └── models.py              # Scene / timeline data models
└── tests/                     # Unit tests (Whisper mocked)
```

## Installation

Requires Python 3.10+, `ffmpeg`, and `ffprobe` on PATH.

```bash
python -m venv venv
venv\Scripts\activate            # Windows
source venv/bin/activate         # macOS / Linux

pip install -r requirements.txt
```

## Usage

### CLI

```bash
python scripts/run_pipeline.py \
    --project input/project.json \
    --audio input/narration.mp3 \
    --images input/images \
    --output output/final_video.mp4
```

- `--use-cached-alignment` reuses `intermediate/transcription.json` and skips Whisper
  (only if its audio SHA-256 matches the current narration).
- `--no-render` stops after writing `timeline.json`.

### Web UI

```bash
python webui.py
```

Open the printed Gradio URL, upload a project JSON, narration audio, and scene images,
then validate and generate.

## Project JSON format

```json
{
  "project": "my_project",
  "scenes": [
    {
      "scene_id": 1,
      "script_text": "Every day, millions of Americans wake up behind bars.",
      "images": ["scene_001.png"]
    }
  ]
}
```

## Tests

```bash
$env:PYTHONPATH="src"; python -m unittest discover -s tests -v   # Windows PowerShell
PYTHONPATH=src python -m unittest discover -s tests -v           # macOS / Linux
```

## Pipeline stages

1. Validate project assets
2. Analyze narration (duration, codec)
3. Transcribe narration (chunked for long audio)
4. Align scenes to the transcript
5. Refine speech boundaries acoustically
6. Build the timeline
7. Render the video (FFmpeg)
8. Validate the output (resolution, duration, codecs)

Artifacts are written to the job workspace (`workspace/jobs/<job_id>/`) when run
through the WebUI, or to `intermediate/` and `output/` when run through the CLI.
