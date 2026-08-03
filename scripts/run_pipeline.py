"""End-to-end pipeline CLI (thin wrapper over the shared PipelineRunner).

Usage:
    python scripts/run_pipeline.py \
        --project input/project.json \
        --audio input/narration.mp3 \
        --images input/images \
        --output output/final_video.mp4 \
        [--use-cached-alignment]

The WebUI and this CLI call the exact same PipelineRunner.run() function.

--use-cached-alignment reuses intermediate/transcription.json (and skips Whisper)
instead of re-transcribing. Cache use is explicit. Without it, the narration is
transcribed fresh on every run.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from video_assembler.services.alignment.provider_base import TranscriptionResult
from video_assembler.services.parser_service import ParserService
from video_assembler.services.pipeline_runner import PipelineRunner, PipelineError


def resolve(path_arg, default: Path) -> Path:
    p = Path(path_arg) if path_arg else default
    if not p.is_absolute():
        p = ROOT / p
    return p


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Narration-to-visual video assembly pipeline")
    parser.add_argument("--project", default="input/project.json", help="Canonical scene JSON")
    parser.add_argument("--audio", default="input/narration.mp3", help="Narration audio file")
    parser.add_argument("--images", default="input/images", help="Directory containing scene images")
    parser.add_argument("--output", default="output/final_video.mp4", help="Output MP4 path")
    parser.add_argument("--use-cached-alignment", action="store_true",
                        help="Reuse intermediate/transcription.json (skip Whisper)")
    parser.add_argument("--no-render", action="store_true",
                        help="Stop after writing timeline.json (skip FFmpeg rendering)")
    args = parser.parse_args(argv)

    project_path = resolve(args.project, ROOT / "input" / "project.json")
    audio_path = resolve(args.audio, ROOT / "input" / "narration.mp3")
    images_dir = resolve(args.images, ROOT / "input" / "images")
    output_path = resolve(args.output, ROOT / "output" / "final_video.mp4")
    intermediate = ROOT / "intermediate"

    print("=== parse ===")
    project_input = ParserService().parse_input_json(project_path)

    transcription = None
    if args.use_cached_alignment:
        print("using cached transcription.json (--use-cached-alignment); skipping Whisper")
        transcription = TranscriptionResult(
            **json.loads((intermediate / "transcription.json").read_text(encoding="utf-8")))

    runner = PipelineRunner(model_name="base")
    try:
        result = runner.run(
            project_input=project_input,
            narration=audio_path,
            images_dir=images_dir,
            intermediate_dir=intermediate,
            output_dir=output_path.parent,
            logs_dir=ROOT / "workspace" / "cli" / "logs",
            transcribe=not args.use_cached_alignment,
            transcription=transcription,
            render=not args.no_render,
            output_name=output_path.name,
            progress=lambda stage: print(f"  {stage}"),
        )
    except PipelineError as e:
        print(f"PIPELINE ERROR: {e}")
        return 1

    meta = result.metadata
    print("DONE")
    print(f"  project={result.project_name}")
    print(f"  scenes={meta['scene_count']}  duration={meta['duration_s']}s")
    if result.output_video is not None:
        print(f"  video={result.output_video}  resolution={meta['resolution']} "
              f"@ {meta['fps']} fps  processing={meta['processing_time_s']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
