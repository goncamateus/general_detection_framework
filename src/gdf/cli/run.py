from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

console = Console()


def run_cmd(
    weights: Optional[Path] = typer.Option(
        None, "--weights", "-w", "--model", "-m", help="Path to .onnx or .engine weights"
    ),
    webcam: Optional[int] = typer.Option(
        None, "--webcam", help="Webcam device index (e.g. --webcam 0)"
    ),
    video: Optional[Path] = typer.Option(None, "--video", help="Video file to run on"),
    task: str = typer.Option("segment", "--task", "-t", help="segment or detect"),
    backend: str = typer.Option("onnx", "--backend", "-b", help="onnx or tensorrt"),
    imgsz: int = typer.Option(640, "--imgsz", help="Model input size"),
    conf_threshold: float = typer.Option(0.25, "--conf-threshold", help="Confidence threshold"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save annotated video"),
    save_frames: Optional[Path] = typer.Option(
        None, "--save-frames", help="Directory for sample annotated frames"
    ),
    save_every: int = typer.Option(30, "--save-every", help="Save one frame every N frames"),
    class_names_file: Optional[Path] = typer.Option(
        None, "--class-names", help="File with class names, one per line"
    ),
    no_show: bool = typer.Option(False, "--no-show", help="Headless — don't open a window"),
    max_frames: Optional[int] = typer.Option(None, "--max-frames", help="Stop after N frames"),
) -> None:
    """Run a model live on a webcam or a video file, drawing masks or boxes."""
    if weights is None:
        raise typer.BadParameter("weights is required (--weights/--model)")
    if not weights.exists():
        raise typer.BadParameter(f"weights not found: {weights}")
    if (webcam is None) == (video is None):
        raise typer.BadParameter("pass exactly one of --webcam or --video")
    if task not in ("segment", "detect"):
        raise typer.BadParameter(f"--task must be segment or detect, got {task!r}")

    if video is not None and not video.exists():
        raise typer.BadParameter(f"video not found: {video}")
    source: int | str = webcam if webcam is not None else str(video)

    class_names: list[str] = []
    if class_names_file and class_names_file.exists():
        class_names = class_names_file.read_text().strip().splitlines()

    runner = _build_runner(weights, task, backend, imgsz)

    console.print(f"[cyan]Running[/cyan] {weights} [{task}/{backend}] on {source}")
    if not no_show:
        console.print("[dim]Press q or ESC to quit[/dim]")

    from gdf.inference.stream import run_stream

    frames = run_stream(
        runner=runner,
        source=source,
        task=task,
        conf_threshold=conf_threshold,
        class_names=class_names,
        output_path=output,
        save_frames_dir=save_frames,
        save_every=save_every,
        show=not no_show,
        max_frames=max_frames,
    )
    console.print(f"[green]Done.[/green] {frames} frames processed.")


def _build_runner(weights: Path, task: str, backend: str, imgsz: int) -> object:
    if backend == "onnx":
        if task == "segment":
            from gdf.inference.onnx_seg_runner import ONNXSegRunner

            return ONNXSegRunner(weights, imgsz)
        from gdf.inference.onnx_detect_runner import ONNXDetectRunner

        return ONNXDetectRunner(weights, imgsz)

    if backend == "tensorrt":
        if task == "segment":
            raise typer.BadParameter(
                "No TensorRT segmentation runner yet — use --backend onnx, or benchmark "
                "the engine directly with trtexec."
            )
        from gdf.inference.trt_detect_runner import TRTDetectRunner

        return TRTDetectRunner(weights, imgsz)

    raise typer.BadParameter(f"--backend must be onnx or tensorrt, got {backend!r}")
