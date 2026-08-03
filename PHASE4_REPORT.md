# Phase 4 Report — First Synchronized Video Assembly

## 1. Executive Summary

Phase 4 wired the Phase-3 production acoustic refiner into an end-to-end assembly
pipeline and produced the first real synchronized 1920×1080 H.264 MP4 from the
11-scene Golden Test. The narration (60.186 s) is encoded as the video's AAC audio
stream, and the 11 static scene images play back with hard cuts exactly on the
refined speech boundaries. All 16 unit/integration tests pass (including FFmpeg
render round-trips) and the output passes ffprobe validation.

Delivered artifact: `output/final_video.mp4` (1.59 MB, 1920×1080, 30 fps,
h264/aac, 60.200 s container).

## 2. Scope Delivered

- `src/video_assembler/services/timeline_service.py` — TimelineService +
  TimelineValidationError; converts aligned scenes into an image-slot visual
  timeline (multi-image scenes split their visual duration equally).
- `src/video_assembler/services/render_service.py` — RenderService + RenderConfig;
  FFmpeg per-segment encode, concat, audio mux, temp cleanup, ffprobe probe.
- `src/video_assembler/models.py` — added Timeline / TimelineScene / TimelineImage
  models and `Scene.raw_speech_end` (pre-refinement end preserved for reporting).
- `scripts/run_pipeline.py` — rewritten as the full CLI entry point
  (parse → validate → audio analysis → transcribe/align → refine → timeline →
  render → validate), with `--use-cached-alignment`, `--no-render`, and
  relative-to-root path resolution (no hardcoded paths).
- `input/images/` — 11 seeded placeholder images (`scene_001.png` … `scene_011.png`)
  so the image-less Golden Test can be rendered.
- `tests/test_timeline_service.py` (9 cases) and `tests/test_render_service.py`
  (3 FFmpeg integration cases); `pytest.ini` so `src` resolves.
- Installed `num2words` into the test Python (optional dependency the locked
  Phase-1 `TextNormalizer` needs to pass its existing number test).

## 3. Architecture

```
project.json ─┐
narration.mp3 ─┤
images/       ─┴─> ParserService ─> ValidationService ─> AudioService (duration)
                    └> AlignmentService + StableWhisperProvider (or cached JSON)
                        └> AcousticBoundaryRefiner (Phase-3 production config)
                            └> TimelineService ─> intermediate/timeline.json, timeline_report.txt
                                └> RenderService ─> output/final_video.mp4
```

Locked Phase 1–3 components (`AlignmentService`, `TextNormalizer`,
`StableWhisperProvider`, the production refiner) are used as-is; Phase 4 adds the
timeline and render layers only.

## 4. Timeline Policy

- First scene: `visual_start = 0.0` (image leads the narration).
- Other scenes: `visual_start = speech_start`.
- Non-final scene: `visual_end = next scene's speech_start` — the image stays
  visible through inter-scene silence; **silence belongs to the visual duration,
  never to speech_end** (speech_end is untouched).
- Final scene: `visual_end = total audio duration` (60.186 s).
- Multi-image scenes split `[visual_start, visual_end]` equally across their images.
- Validation rejects zero/negative visual durations and any scene whose visual end
  would fall before its own speech end (broken ordering).

## 5. Render Pipeline (Reliability-First)

- One H.264 segment per image slot with exactly
  `round(visual_end·30) − round(visual_start·30)` frames (cumulative rounding ⇒
  total 1806 frames = round(60.186·30), no drift).
- `scale=1920:1080:force_original_aspect_ratio=decrease` + centered black
  `pad` — no stretching, no effects, hard cuts only.
- Concatenate segments via the concat demuxer (`-c copy`), then mux the original
  narration as AAC 192 kbps (`-map` from the source file). No MoviePy.
- Temp segments cleaned up on success; output validated with ffprobe.

## 6. Golden Test Result

Pipeline run: `python scripts/run_pipeline.py --use-cached-alignment`

- 11/11 scenes aligned HIGH, refiner refined all 11 (unchanged 0, ambiguous 0,
  guard-limited 0, overlaps 0, invalid 0).
- Refined speech ends reproduce the Phase-3 production values exactly:
  **4.465, 10.595, 13.595, 15.515, 23.085, 36.125, 37.825, 38.965, 42.595,
  56.215, 59.885**.
- Alignment starts match the cached `alignment.json` exactly
  (0.0, 5.1, 11.14, 14.26, 16.44, 23.82, 36.88, 38.46, 39.64, 43.18, 56.88).
- Visual timeline: full table in `intermediate/timeline_report.txt`; machine
  readable in `intermediate/timeline.json`.

## 7. Validation & Quality Checks

ffprobe of `output/final_video.mp4`:

| Property      | Value                          | Expected                     |
|---------------|--------------------------------|------------------------------|
| Container     | mp4, duration 60.200 s         | ≈ audio (60.186 s)           |
| Video codec   | h264, yuv420p                  | h264                         |
| Resolution    | 1920 × 1080                    | 1920 × 1080                  |
| Frame rate    | 30 fps                         | 30 fps                       |
| Video frames  | 1806                           | round(60.186·30) = 1806      |
| Audio codec   | aac 44100 Hz mono, ~168 kbps   | aac (192 k target)           |
| Audio duration| 60.186 s                       | narration.mp3                |

`intermediate/render_temp/` contains 0 files after success (cleanup verified).

## 8. Test Suite

`python -m pytest` → **16 passed**.

- `tests/test_timeline_service.py` (9): normal 3-scene, silence ownership,
  first-scene lead-in, final-scene extension, multi-image equal split, broken
  ordering, zero-duration, missing image, empty image list.
- `tests/test_render_service.py` (3): real FFmpeg encode round-trip with
  ffprobe assertions, temp-dir cleanup, missing-image error path.
- Existing `tests/test_text_normalizer.py` now passes after installing the
  optional `num2words` dependency (pre-existing Phase-1 failure, unrelated to
  Phase 4).

## 9. Reproducibility

- Use cached alignment to skip Whisper:
  `python scripts/run_pipeline.py --use-cached-alignment`
- Full run (re-transcribe):
  `python scripts/run_pipeline.py`
- Timeline-only (no FFmpeg): append `--no-render`.
- Paths default relative to the repo root and can be overridden with
  `--project/--audio/--images/--output`.
- Image filename matching uses the Phase-2 convention `scene_{id:03d}.png`.

## 10. Known Limitations & Next Steps

- Placeholder images are solid-color seeded placeholders; swap in real assets in
  `input/images/` and re-run (no code change required).
- Inter-scene silence is currently held on the previous image (policy choice);
  a future "black/slide transition" policy is a render-layer option.
- Rendering is sequential per segment; parallel segment encoding is a
  performance option once real assets grow long.
- The narration audio is re-encoded (AAC 192k) rather than stream-copied; the
  source mp3 remains the canonical audio master.
