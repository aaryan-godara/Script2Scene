"""Minimal browser UI for the narration-to-visual video assembler.

The UI is a thin layer over the existing pipeline services:
  - ProjectValidator  -> fast preflight validation + Scene/Script/Image mapping
  - JobManager        -> one isolated workspace per generation job
  - PipelineRunner    -> the SAME pipeline function the CLI calls

Run locally with:
    python webui.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import gradio as gr  # noqa: E402

from video_assembler.services.job_manager import JobManager, JobError  # noqa: E402
from video_assembler.services.parser_service import ParserService  # noqa: E402
from video_assembler.services.pipeline_runner import PipelineError, PipelineRunner  # noqa: E402
from video_assembler.services.project_validator import ProjectValidator  # noqa: E402

WORKSPACE = ROOT / "workspace"
AUDIO_EXTS = {".mp3", ".wav", ".m4a"}

job_manager = JobManager(WORKSPACE)
validator = ProjectValidator()
runner = PipelineRunner(model_name="base")
parser_service = ParserService()


def _file_parts(f) -> tuple:
    """Extracts (server_path, original_filename) from a Gradio FileData/dict."""
    if f is None:
        return None, None
    if isinstance(f, dict):
        return f.get("path"), f.get("orig_name") or Path(f.get("path") or "").name
    path = getattr(f, "path", None) or getattr(f, "name", None)
    orig = getattr(f, "orig_name", None) or Path(path or "").name
    return path, orig


def _fmt_duration(seconds) -> str:
    seconds = int(round(seconds or 0))
    if seconds >= 60:
        return f"{seconds // 60}m {seconds % 60}s"
    return f"{seconds}s"


def _narration_in_job(job) -> Path:
    for p in job.input_dir.iterdir():
        if p.suffix.lower() in AUDIO_EXTS:
            return p
    raise JobError("No narration file found in job input.")


# ------------------------------------------------------------------ validate
def do_validate(project_file, narration_file, image_files, _job_id):
    missing = []
    if project_file is None:
        missing.append("PROJECT / SCENE JSON")
    if narration_file is None:
        missing.append("NARRATION")
    if not image_files:
        missing.append("SCENE IMAGES")
    if missing:
        return (
            f"**Missing inputs:** {', '.join(missing)}",
            None,
            gr.update(interactive=False),
            "",
        )

    try:
        job = job_manager.create_job()
        project_json_path = job_manager.write_project_json(
            job, Path(_file_parts(project_file)[0]))
        narration_path, narration_name = _file_parts(narration_file)
        job_manager.write_narration(job, Path(narration_path), narration_name or "narration")
        image_sources = [
            (orig, Path(path))
            for path, orig in (_file_parts(f) for f in image_files)
            if path
        ]
        job_manager.write_images(job, image_sources)
    except (JobError, OSError) as e:
        return f"**Could not prepare job:** {e}", None, gr.update(interactive=False), ""

    outcome = validator.parse_and_validate(project_json_path, narration_path, job.images_dir)

    if not outcome.valid:
        lines = "\n".join(f"- {e}" for e in outcome.errors)
        warn = f"\n\nWarnings:\n" + "\n".join(f"- {w}" for w in outcome.warnings) \
            if outcome.warnings else ""
        return f"**Project validation failed.**\n\n{lines}{warn}", None, \
            gr.update(interactive=False), job.job_id

    summary = (
        f"**Project valid**\n\n"
        f"- Project: {outcome.project_name or '(unnamed)'}\n"
        f"- Scenes: {outcome.scene_count}\n"
        f"- Images: {outcome.image_count}\n"
        f"- Narration: {outcome.narration_name}\n"
        f"- Duration: {_fmt_duration(outcome.narration_duration)}"
    )
    if outcome.warnings:
        summary += "\n\n" + "\n".join(f"- {w}" for w in outcome.warnings)

    rows = [[r["scene_id"], r["script_text"], r["images"]] for r in outcome.rows]
    return summary, rows, gr.update(interactive=True), job.job_id


# ------------------------------------------------------------------ generate
def do_generate(job_id, progress=gr.Progress()):
    if not job_id:
        return "Please validate the project before generating.", None, None

    try:
        job = job_manager.get_job(job_id)
        project_input = parser_service.parse_input_json(job.input_dir / "project.json")
        narration = _narration_in_job(job)

        stage_index = {}

        def cb(stage):
            idx = stage_index.get(stage, 0)
            try:
                progress((idx + 1) / len(PipelineRunner.STAGES), desc=stage)
            except Exception:
                pass  # progress reporting is best-effort

        for i, stage in enumerate(PipelineRunner.STAGES):
            stage_index[stage] = i

        result = runner.run(
            project_input=project_input,
            narration=narration,
            images_dir=job.images_dir,
            intermediate_dir=job.intermediate_dir,
            output_dir=job.output_dir,
            logs_dir=job.logs_dir,
            transcribe=True,
            render=True,
            progress=cb,
        )
    except (PipelineError, JobError, OSError) as e:
        return f"**Generation failed.**\n\n{e}", None, None

    meta = result.metadata
    status = (
        f"**Video generated successfully**\n\n"
        f"- Duration: {meta['duration_s']}s\n"
        f"- Resolution: {meta['resolution']}\n"
        f"- FPS: {meta['fps']}\n"
        f"- Scenes: {meta['scene_count']}\n"
        f"- Processing time: {meta['processing_time_s']}s"
    )
    return status, str(result.output_video), str(result.output_video)


# ------------------------------------------------------------------- interface
with gr.Blocks(title="Automated Narration-to-Visual Video Assembler") as demo:
    gr.Markdown(
        "# Automated Narration-to-Visual Video Assembler\n"
        "\nUpload your narration, scene mapping, and generated images to create a "
        "synchronized video."
    )

    job_state = gr.State()

    with gr.Group():
        gr.Markdown("### Section 1 - Project Input")
        project_json = gr.File(
            label="PROJECT / SCENE JSON", file_types=[".json"], type="filepath")
        narration = gr.File(
            label="NARRATION", file_types=[".mp3", ".wav", ".m4a"], type="filepath")
        image_files = gr.File(
            label="SCENE IMAGES (multi-file)", file_count="multiple",
            file_types=[".png", ".jpg", ".jpeg", ".webp"], type="filepath")

    validate_btn = gr.Button("Validate Project", variant="primary")
    validation_box = gr.Markdown()

    gr.Markdown("### Scene Mapping Preview")
    mapping_table = gr.Dataframe(
        headers=["Scene", "Script", "Image"], interactive=False, row_count=(5, "dynamic"))

    generate_btn = gr.Button("Generate Video", variant="primary", interactive=False)
    status_box = gr.Markdown()
    video_preview = gr.Video(label="Video Preview", interactive=False)
    download = gr.File(label="Download MP4")

    validate_btn.click(
        do_validate,
        inputs=[project_json, narration, image_files, job_state],
        outputs=[validation_box, mapping_table, generate_btn, job_state],
    )
    generate_btn.click(
        do_generate,
        inputs=[job_state],
        outputs=[status_box, video_preview, download],
    )

    gr.Markdown(
        "\n---\n"
        "Each generation runs in its own isolated workspace under `workspace/jobs/`; "
        "repository golden-test files are never touched."
    )


if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", server_port=7860)
