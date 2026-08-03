# Phase 5 Report — Minimal Web UI for Real End-to-End Testing

## 1. UI Implemented

`webui.py` is a single-file Gradio app titled **"Automated Narration-to-Visual
Video Assembler"** with subtitle *"Upload your narration, scene mapping, and
generated images to create a synchronized video."*

Workflow:
1. **Section 1 — Project Input**: three upload controls — PROJECT/SCENE JSON
   (`.json`), NARRATION (`.mp3/.wav/.m4a`), and SCENE IMAGES (multi-file,
   `.png/.jpg/.jpeg/.webp`).
2. **[Validate Project]** — runs fast preflight checks with **no Whisper**, and
   shows either a green summary (`Project`, `Scenes`, `Images`, `Narration`,
   `Duration`) or a friendly bulleted list of failures (e.g. *"Scene 7 expects:
   scene_007.png but that image was not uploaded."*). No Python tracebacks.
3. **Scene Mapping Preview** — a table `Scene | Script | Image`, multi-image
   scenes listed as `scene_005_a.png, scene_005_b.png`. Shown *before*
   transcription so the user confirms the mapping first.
4. **[Generate Video]** — enabled only after successful validation. Runs the
   real pipeline with live progress labels (Preparing / Validating / Analyzing /
   Transcribing / Aligning / Refining / Creating timeline / Rendering /
   Validating output / Complete).
5. **Video Preview** — `gr.Video` plus a downloadable MP4 and a metadata block
   (Duration, Resolution, FPS, Scenes, Processing time).

The UI is local-only MVP: no auth, database, job queue, Docker, or effects.

## 2. Backend Integration

The WebUI does **not** duplicate the pipeline and does **not** shell out to the
CLI. A new shared service `PipelineRunner.run()` (`src/.../pipeline_runner.py`)
contains the one and only generation implementation:

    validate -> audio analysis -> transcription (StableWhisperProvider)
    -> AlignmentService -> AcousticBoundaryRefiner -> TimelineService
    -> RenderService -> output validation

- **CLI** (`scripts/run_pipeline.py`) calls `PipelineRunner.run()`.
- **WebUI** (`webui.py`) calls `PipelineRunner.run()`.
- `StableWhisperProvider` (torch/stable-whisper) is imported lazily so tests
  and the CLI run without it; `PipelineRunner(transcriber=...)` allows mocking.
- Every backend failure is converted to a user-friendly `PipelineError`; the
  full traceback/ffmpeg stderr goes to the job's `logs/pipeline.log`, never the
  UI.

## 3. Workspace Isolation

`JobManager` (`src/.../job_manager.py`) gives every UI run its own directory:

    workspace/jobs/<unique_job_id>/
        input/project.json
        input/<narration original name>
        input/images/<original image names>
        intermediate/   (transcription.json, alignment.json, timeline.json,
                         timeline_report.txt, render_temp/)
        output/final_video.mp4
        logs/pipeline.log

Unique id = `uuid4().hex[:12]`. Stale `alignment.json`/`timeline.json`/wrong
images/previous MP4 from another run can never contaminate a job. Repository
`input/`, `intermediate/`, `output/` are untouched (verified by SHA-256 in the
manual test). Upload filenames are basename-sanitized; duplicate image names are
rejected.

## 4. Validation

Preflight checks (fast, no Whisper) in `ProjectValidator`:
- JSON parses; expected structure (`project`, `scenes[]` with `scene_id`,
  `script_text`, `images[]`)
- at least one scene; scene IDs positive integers and unique; `script_text`
  non-empty
- narration present, non-zero-byte, extension in `.mp3/.wav/.m4a`, decodable by
  ffprobe
- every referenced image uploaded (missing image reported per scene+filename),
  non-zero-byte, valid image file, extension in `.png/.jpg/.jpeg/.webp`
- unused-image warnings

## 5. Mapping Preview

Matching reuses the Phase-2 `ImageMatcher` and is **purely filename-based**:
explicit `"images": ["scene_001.png"]` entries are matched against the uploaded
filenames. Browser upload order is irrelevant — the manual test uploaded
`scene_008, scene_004, scene_003, scene_009, ...` and the mapping was still
Scene 1→scene_001.png … Scene 11→scene_011.png. Scenes with `images: []`
fall back to normalized scene-ID matching (`scene_001.png`).

## 6. Generation Flow

After **[Generate Video]** on a validated job:
1. `JobManager.get_job(job_id)` locates the isolated workspace
2. `ParserService` re-parses the job's `project.json`
3. `PipelineRunner.run(transcribe=True)` executes: validation → audio analysis →
   **fresh Whisper transcription** (never cached) → alignment → FAILED/REVIEW
   policy gate → acoustic refinement → timeline → FFmpeg render (1920×1080,
   30 fps, H.264, AAC 192k, static images, hard cuts) → output validation
4. Status, `gr.Video` preview path, and download path returned

If any scene is `FAILED` (or `REVIEW` without `allow_review`), generation stops
with the offending scene(s) named — a potentially mis-synchronized video is
never silently rendered.

## 7. Tests

`python -m pytest` → **43 passed, 0 failed.**

- `tests/test_job_manager.py` (6): layout, distinct jobs, unordered
  filename-preserving writes, duplicate rejection, narration naming,
  filename sanitization
- `tests/test_project_validator.py` (14): valid project, unordered upload
  mapping, missing image, invalid JSON, missing script_text, empty scenes,
  duplicate IDs, unsupported/zero-byte/undecodable narration, zero-byte/corrupt/
  unsupported-format images, unused-image warning, UI-style write→validate
- `tests/test_pipeline_runner.py` (7): real render with mocked transcription,
  two-job artifact isolation, alignment-FAILED stop, REVIEW stop + allow_review,
  backend exception → friendly error (traceback in log), validation error
  message, `render=False` timeline-only
- Phase-1..4 suite (16 tests) still green

## 8. Manual Real-Asset Test

Executed the exact WebUI functions (`do_validate` → `do_generate`) with the
**real** Golden Test assets (real `project.json`, real 60 s `narration.mp3`,
real 11 images), images uploaded in **random order**
(`scene_008, scene_004, scene_003, scene_009, scene_006, scene_007,
scene_010, scene_005, scene_001, scene_002, scene_011`):

1. upload JSON ✅  2. upload narration ✅  3. upload images (random order) ✅
4. validate ✅ (fast, before Whisper)  5. inspect mapping ✅ (filename-based,
   correct)  6. generate ✅ (fresh Whisper transcription)  7. preview MP4 ✅
   (server + video produced)

Every step worked. `GOLDEN FILE INTEGRITY: NONE - repository untouched`.

## 9. Generated Output

- Filename: `workspace/jobs/086865c20e05/output/final_video.mp4`
- Duration: 60.200 s container (60.186 s audio)
- Resolution: 1920 × 1080
- FPS: 30
- Scene count: 11
- Processing time: 79.0 s (incl. 23 s Whisper transcription)
- Codecs: h264 video / aac 44.1 kHz audio; size 1,589,309 bytes
- All 11 scenes aligned HIGH

## 10. Files Created / Modified

Created:
- `src/video_assembler/services/job_manager.py`
- `src/video_assembler/services/project_validator.py`
- `src/video_assembler/services/pipeline_runner.py`
- `webui.py`
- `tests/_helpers.py`, `tests/test_job_manager.py`,
  `tests/test_project_validator.py`, `tests/test_pipeline_runner.py`
- `PHASE5_REPORT.md`

Modified:
- `scripts/run_pipeline.py` — refactored to delegate to `PipelineRunner.run()`
  (Golden Test timeline/timeline_report reproduced byte-identical)
- `venv` — installed `gradio` (6.22.0)

## 11. Remaining Problems

- `render_temp/` directory remains (empty) inside each job's `intermediate/`;
  files within it are cleaned but the directory itself is left behind. Cosmetic.
- Whisper model downloads on first use (transient network requirement).
- CPU rendering (~80 s per 60 s video) is acceptable for an MVP, not optimized.

## 12. Verdict

**PHASE 5 COMPLETE — UI GENERATED REAL SYNCHRONIZED VIDEO**
