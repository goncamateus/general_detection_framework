from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

console = Console()


def webcam_cmd(
    weights: Optional[Path] = typer.Option(None, "--weights", "-w", help="Detection model weights (.onnx or .engine)"),
    backend: Optional[str] = typer.Option(None, "--backend", "-b", help="Backend: onnx, tensorrt"),
    imgsz: Optional[int] = typer.Option(None, "--imgsz", help="Input image size"),
    conf_threshold: Optional[float] = typer.Option(None, "--conf-threshold", help="Detection confidence"),
    match_threshold: Optional[float] = typer.Option(None, "--match-threshold", help="ByteTrack IoU match threshold"),
    device: Optional[int] = typer.Option(None, "--device", help="Webcam device index"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save output video"),
    class_names_file: Optional[Path] = typer.Option(None, "--class-names", help="File with class names"),
    no_show: bool = typer.Option(False, "--no-show", help="Don't display window (headless)"),
    max_frames: Optional[int] = typer.Option(None, "--max-frames", help="Stop after N frames"),
) -> None:
    if weights is None:
        console.print("[red]--weights is required[/red]")
        raise typer.Exit(1)

    backend = backend or "onnx"
    imgsz = imgsz or 640
    conf_threshold = conf_threshold or 0.25
    match_threshold = match_threshold or 0.7
    device = device or 0

    class_names: list[str] = []
    if class_names_file and class_names_file.exists():
        class_names = class_names_file.read_text().strip().splitlines()

    console.print(f"[cyan]Webcam tracking[/cyan]")
    console.print(f"  Model: {weights}")
    console.print(f"  Backend: {backend}")
    console.print(f"  Device: {device}")
    console.print(f"  Press 'q' to quit")

    from gdf.inference.webcam_runner import WebcamRunner

    runner = WebcamRunner(
        weights=weights,
        backend=backend,
        imgsz=imgsz,
        conf_threshold=conf_threshold,
        match_threshold=match_threshold,
        class_names=class_names,
        device=device,
    )

    frames = runner.run(
        output_path=output,
        show=not no_show,
        max_frames=max_frames,
    )

    console.print(f"[green]Done. {frames} frames processed.[/green]")
