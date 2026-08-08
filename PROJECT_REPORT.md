# Script2Scene — Project Deep-Analysis Report

> **Purpose**: Capture the full project state, decisions, and verified results so the
> next session can resume from exactly this point without re-deriving context.
>
> **Date**: 04-Aug-2026 · **Head commit**: `c1189bc` · **Branch**: `main`
> **Remote**: https://github.com/aaryan-godara/Script2Scene.git

---

## 1. TL;DR — where we are

- The pipeline **works end-to-end** and has produced a **real, validated 222-scene MP4**
  (24m11s narration) using **chunked long-form transcription**.
- 68 unit tests pass (`PYTHONPATH=src; python -m unittest discover -s tests -v`).
- Pushed to GitHub as 2 commits: initial skeleton (`b9d1961`) + chunked-transcription /
  review-tolerant-gate work (`c1189bc`).
- 1 known open item: `base` Whisper model drops a 3-word phrase inside 180s chunks;
  demonstrated fix is smaller chunks (~90s). **Not yet implemented.**

---

## 2. What the product is

Automated **narration-to-visual video assembly**. Given:

- a narration audio file (mp3/wav/m4a),
- a JSON scene map (`scene_id`, `script_text`, `images`),
- a folder of images,

it produces a synchronized MP4 where each scene's image is shown during its spoken
narration, with word-accurate start/end timestamps, acoustic boundary refinement,
confidence-based gating, and FFmpeg rendering + output validation.

Both a **Gradio Web UI** (`webui.py`) and a **CLI** (`scripts/run_pipeline.py`) drive
the same core: `PipelineRunner.run()`.

---

## 3. Architecture (as of this commit)

```
webui.py ─────────────┐
scripts/run_pipeline.py─┤  -> PipelineRunner.run()  -> stages:
                       │       1. Validate project assets
                       │       2. Analyze narration (ffprobe duration)
                       │       3. Transcribe narration (chunked for long audio)
                       │       4. Align scenes to transcript (HIGH/REVIEW/FAILED)
                       │       5. Refine speech boundaries (acoustic)
                       │       6. Build timeline (validated ordering)
                       │       7. Render video (FFmpeg)
                       │       8. Validate output (res, fps, duration, codecs)
                       └──────────┘
```

### Core modules (`src/video_assembler/`)

| Module | Responsibility |
|---|---|
| `services/pipeline_runner.py` | Orchestration, stage wrapping, log/metadata, alignment gate |
| `services/alignment/stable_whisper_provider.py` | Whisper word-level transcription (stable-ts) |
| `services/alignment/chunked_transcription_provider.py` | Long-audio chunking + deterministic merge |
| `services/alignment/transcription_cache.py` | Audio-SHA-256-stamped cache safety |
| `services/alignment/alignment_service.py` | Sequential token matcher → timestamps + status |
| `services/alignment/acoustic_boundary_refiner.py` | Silence-aware end-boundary refinement |
| `services/alignment/text_normalizer.py` | Token normalization (numbers, casing, punctuation) |
| `services/timeline_service.py` | Ordering/duration validation, image slots |
| `services/render_service.py` | FFmpeg render + probe |
| `services/audio_service.py` | ffprobe metadata + chunk extraction (16kHz mono PCM) |
| `services/job_manager.py` | WebUI job workspace management |
| `models.py` | `Scene`, `Timeline`, Pydantic models |

### Chunked transcription design

- `ChunkingConfig`: `chunk_duration=180s`, `overlap=10s`, `long_audio_threshold=300s`.
- Audio is sliced into overlapping 16kHz mono PCM WAVs via FFmpeg.
- Each chunk is transcribed independently, shifted to global timestamps.
- Overlap regions are reconciled by sequential normalized-text matching
  (`_merge_chunk`) — a spoken word seen by both chunks is kept once, preferring the
  copy farther from a chunk boundary.
- Result is **one global word timeline**; `AlignmentService` never knows chunks existed.
- Metadata stamped: `audio_sha256`, `chunking_enabled`, `chunk_count`,
  `chunk_boundaries`, `words_per_chunk`, `duplicates_removed`, `created_at`.

### Alignment gate (changed in `c1189bc`)

Statuses per scene: `HIGH`, `REVIEW`, `FAILED`.

| Condition | Behavior |
|---|---|
| Any `FAILED` | **Hard blocker** — always stops generation |
| `REVIEW` scenes, any invalid timestamps | Block (invalid/unsafe timestamps) |
| `REVIEW` share > `max_review_ratio` (default **0.05**) | Block (manual review required) |
| `REVIEW` share ≤ ratio + all timestamps usable | **Allow**, render with warning (status stays REVIEW) |
| All `HIGH` | Continue, no warnings |

- Warnings are surfaced in `result.metadata["alignment_warnings"]`,
  `metadata["alignment_statuses"]`, and as `Scene.warning` on each REVIEW scene.
- The old `allow_review` flag was **removed** (CLI/WebUI never used it).

### Transcription cache safety

- `transcription.json` is only reused when its `audio_sha256` matches the current
  narration's SHA-256 (computed streaming over the whole file).
- CLI `--use-cached-alignment` skips Whisper and validates the cache; a mismatch or
  missing identity raises `TranscriptionCacheError` and aborts.

---

## 4. What was fixed / added (commit `c1189bc`)

1. **`StableWhisperProvider`**
   - Added `no_speech_threshold: float = 0.9` (was Whisper default 0.6). The 0.6
     default silently discarded real narration windows on chunked long audio.
   - Filters zero-width words (`end - start < 1e-6`): stable-whisper artifacts that
     duplicated phrases and degraded alignment (repro: duplicated "They're buying real
     estate, ..." at 965.44 degraded Scene 141 HIGH→REVIEW).
2. **`ChunkedTranscriptionProvider`** (new) — overlapping chunk transcription with
   deterministic merge; produces one global timeline.
3. **`TranscriptionCache`** (new) — SHA-256 identity; `save/load_transcription`,
   `transcription_is_current`.
4. **`AudioService.extract_chunk(s)`** — FFmpeg slicing to 16kHz mono PCM chunks.
5. **`PipelineRunner`** — review-tolerant gate (section 3), `max_review_ratio`
   constructor param, `alignment_warnings` / `alignment_statuses` metadata,
   `Scene.warning`, removal of `allow_review`.
6. **CLI** — `--use-cached-alignment` now validates SHA-256 instead of blind JSON load.
7. **Tests** — `tests/test_chunked_transcription.py` (new),
   `tests/test_pipeline_runner.py` (gate scenarios + old-review-test replaced).
8. **README.md** (new).

---

## 5. Real acceptance result (the benchmark job)

- **Job**: `workspace/jobs/31c9033c3b6d`
- **Narration**: `input/final.mp3` — duration **1450.771s** (24m11s)
  - SHA-256: `9b0dd7e335867db24677af9072418f10d594067dcfb5bfb3d4a36b02649c60b9`
- **Project**: `input/project.json` — **222 scenes**
- **Transcription**: `chunked / base / cpu`, 9 chunks @180s, overlap 10s,
  3080 merged words, 155 duplicates removed.
- **Alignment**: **221 HIGH / 1 REVIEW / 0 FAILED** — no regressions.
  - Scene 86 (the one REVIEW): canonical `"It's Washington, D.C."`, conf 0.6667,
    timestamps 549.54–550.24s, verified genuine match → **allowed with warning**.
- **Render**: `output/final_video.mp4` — 1920×1080 @30fps, h264+aac,
  container duration 1450.770s == narration (FFprobe verified), **112 MB**,
  ~21 min processing time (CPU).
- Output validation (resolution / fps / video+audio+container duration) passed.

### Timeline safety (all 222 scenes)

- speech_start < speech_end and within audio for every scene.
- 5 backward jumps — all are `AlignmentService`'s own backtracking recovery
  (pre-existing behavior, not modified).
- 0 invalid timestamps, 0 gaps > 20s, first word at 0.0s, last word 1449.92→1450.26s.

---

## 6. Scene 100 diagnosis (important follow-up, NOT yet fixed)

During the last session, Scene 100 ("Add food services, mental health programs, ...")
was reported as `FAILED`. Full diagnosis performed (see previous session notes):

**Root cause — TRANSCRIPTION_MISSING (Whisper long-context omission at chunk level).**
The audio genuinely contains "Add food services" (proven by a 90s local clip at
654.86–655.84s), but the merged 180s chunk-3 transcript jumps from `business.`
(654.04s) straight to `mental` (656.38s) — a 2.34s gap. `add` appears **0 times** in
the entire merged transcript. With the current cache, the matcher still scores 0.889
by skipping the 3 missing words; a slightly different transcription pushes it below
the 0.5 FAILED threshold.

**Verified experiments (same provider/model/threshold):**
- 180s chunk 3 WAV [510, 690] → omits "add food services".
- 90s clip [630, 720] → contains it correctly.

**Recommended fix (choose 1):**
1. **Preferred**: `ChunkingConfig(chunk_duration_seconds=90.0)` — proven to recover
   the phrase; one-line config change.
2. Heavier: keep 180s chunks + gap-triggered local re-transcription/splice in the
   chunked provider.

Then re-run acceptance test + re-render. Do **not** modify `AlignmentService`,
thresholds, `project.json`, or narration.

> ⚠️ `stable_whisper_provider.py` comment currently says "chunk_003 loses scenes
> 100-102" as the no_speech_threshold rationale; scene 100's omission is actually a
> chunk-size (long-context) effect, not no_speech — the 0.9 threshold fix addressed a
> separate real issue (scenes 112–125, 161–164, 180). Worth clarifying the comment
> next session.

---

## 7. Known stale artifact (do not confuse)

`intermediate/transcription.json` at the **repo root** (39 KB, 134 words, 59.62s,
provider `stable_whisper`, no chunking metadata) is a **legacy stale cache** from the
original 11-scene test project (`input/`). It is **not** the 222-scene job's cache
(job cache lives in `workspace/jobs/31c9033c3b6d/intermediate/transcription.json`,
879 KB). The root cache will be rejected by SHA validation if the CLI is pointed at
it with a different audio.

---

## 8. Reproducing everything

### Tests
```powershell
$env:PYTHONPATH="src"; python -m unittest discover -s tests -v   # 68 tests, OK
```

### Render the 222-scene job (no Whisper re-run — cached transcription)
Driver: `C:\Users\HP\AppData\Local\Temp\opencode\render_job.py`
```powershell
python "C:\Users\HP\AppData\Local\Temp\opencode\render_job.py" --no-render  # gate check
python "C:\Users\HP\AppData\Local\Temp\opencode\render_job.py"              # full render
```
Uses `max_review_ratio=0.05`; result: 221 HIGH / 1 REVIEW (scene 86, warned) / 0 FAILED.

### Acceptance test (regenerates transcription — re-runs Whisper on the full file)
```powershell
python "C:\Users\HP\AppData\Local\Temp\opencode\acceptance_test.py"
```

### Other diagnostics (in `%TEMP%\opencode\`)
- `step1_scene86.py` — Scene 85/86/87 verification.
- `compare_statuses.py` — old vs new alignment statuses.
- `diag_scene100.py`, `diag_scene100_raw.py`, `diag_scene100_clip.py` — Scene 100
  diagnosis + local-clip re-transcription evidence.

---

## 9. Git state

```
c1189bc (HEAD, main) Add chunked long-form transcription and review-tolerant alignment gate
b9d1961             Script2Scene: narration-to-visual video assembly pipeline + web UI
```
Working tree clean after push. `workspace/`, `intermediate/`, `output/`, `input/images/`,
media files are gitignored.

---

## 10. Next session — start here (checklist)

1. Read this file. Working tree should be at `c1189bc`, clean.
2. Confirm tests: 68 pass.
3. **Decide Scene 100 chunk-size fix** (section 6): switch `ChunkingConfig` to
   `chunk_duration_seconds=90` and re-run the acceptance test + full render.
   Validate 222 scenes still 221 HIGH / 1 REVIEW (or better) and re-verify no
   regressions with `compare_statuses.py`.
4. Update the misleading `no_speech_threshold` comment in `stable_whisper_provider.py`.
5. Any follow-up (new features, WebUI polish, larger Whisper model, GPU) builds on top
   of the current chunked architecture — do not reintroduce single-pass transcription
   for long audio.

---

## 11. Constraints to respect going forward

- **Never** bypass FAILED scenes; FAILED is a hard blocker by design.
- **Never** retranscribe the entire 24-minute file unnecessarily — use the chunked
  cache (`--use-cached-alignment`) or transcribe only the affected chunk/clip.
- Do **not** modify `project.json` or `final.mp3` for the acceptance job.
- Do **not** change `AlignmentService` / thresholds to paper over transcription gaps.
- Both WebUI and CLI must continue to call the same `PipelineRunner.run()`.
