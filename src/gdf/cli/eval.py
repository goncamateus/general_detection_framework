from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from gdf.config.schema import EvalConfig

console = Console()


def eval_cmd(
    weights: Optional[Path] = typer.Option(
        None, "--weights", "-w", "--model", "-m", help="Path to .pt / .onnx / .engine weights"
    ),
    data: Optional[str] = typer.Option(
        None, "--data", "-d", help="data.yaml, or a dataset root containing one"
    ),
    split: Optional[str] = typer.Option(
        None, "--split", help="Split to score: train, val, test"
    ),
    task: Optional[str] = typer.Option(
        None, "--task", "-t", help="auto (default), cls, detect, segment"
    ),
    imgsz: Optional[int] = typer.Option(None, "--imgsz", help="Image size"),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", "-b", help="Batch size"),
    conf_threshold: Optional[float] = typer.Option(
        None, "--conf-threshold", help="Confidence threshold"
    ),
    iou_threshold: Optional[float] = typer.Option(None, "--iou-threshold", help="NMS IoU"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output CSV path"),
    device: Optional[str] = typer.Option(None, "--device", help="Device: auto, cpu, cuda:0"),
) -> None:
    """Score a trained model on a dataset split (mAP, precision/recall, latency)."""
    if weights is None:
        raise typer.BadParameter("weights is required (--weights/--model)")
    if data is None:
        raise typer.BadParameter("data is required (--data)")

    cfg = EvalConfig(
        **{
            k: v
            for k, v in {
                "weights": weights,
                "data": data,
                "split": split,
                "task": task,
                "imgsz": imgsz,
                "batch_size": batch_size,
                "conf_threshold": conf_threshold,
                "iou_threshold": iou_threshold,
                "output": output,
                "device": device,
            }.items()
            if v is not None
        }
    )

    if not cfg.weights.exists():
        raise typer.BadParameter(f"weights not found: {cfg.weights}")

    data_arg = _resolve_data(cfg.data)
    console.print(f"[cyan]Evaluating[/cyan] {cfg.weights} on {data_arg} [{cfg.split}]")

    from ultralytics import YOLO

    # Only .pt checkpoints carry their task; exported graphs must be told.
    if cfg.task == "auto" and cfg.weights.suffix != ".pt":
        raise typer.BadParameter(
            f"--task is required for {cfg.weights.suffix} weights (auto-detect needs a .pt)"
        )
    model = YOLO(str(cfg.weights)) if cfg.task == "auto" else YOLO(str(cfg.weights), task=cfg.task)

    results = model.val(
        data=data_arg,
        split=cfg.split,
        imgsz=cfg.imgsz,
        batch=cfg.batch_size,
        conf=cfg.conf_threshold,
        iou=cfg.iou_threshold,
        device=cfg.device if cfg.device != "auto" else None,
        plots=False,
        verbose=False,
    )

    row = {"model": str(cfg.weights), "data": str(data_arg), "split": cfg.split, "task": model.task}
    row.update({k: round(float(v), 4) for k, v in results.results_dict.items()})
    row.update(_speed_metrics(results.speed))

    _print_metrics(row, model.task)

    if cfg.output:
        import csv

        cfg.output.parent.mkdir(parents=True, exist_ok=True)
        with open(cfg.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        console.print(f"[green]Metrics saved:[/green] {cfg.output}")


def _resolve_data(data: str) -> str:
    """Accept a data.yaml, a dataset root holding one, or a cls ImageFolder root."""
    path = Path(data)
    if path.is_dir():
        yaml_path = path / "data.yaml"
        if yaml_path.exists():
            return str(yaml_path)
    elif not path.exists():
        raise typer.BadParameter(f"data not found: {data}")
    return str(path)


def _speed_metrics(speed: dict) -> dict:
    """Ultralytics reports per-image milliseconds per stage; report the end-to-end rate too."""
    total = sum(v for k, v in speed.items() if k != "loss")
    return {
        **{f"speed/{k}_ms": round(float(v), 3) for k, v in speed.items()},
        "speed/total_ms": round(total, 3),
        "fps": round(1000.0 / total, 1) if total > 0 else 0.0,
    }


def _print_metrics(row: dict, task: str) -> None:
    table = Table(title=f"Evaluation — {task}", show_lines=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")

    for key, value in row.items():
        if key in ("model", "data", "split", "task"):
            continue
        # Values are already rounded; :g drops the trailing zeros mAP and FPS don't share.
        table.add_row(_pretty(key), f"{value:g}" if isinstance(value, float) else str(value))

    console.print(table)
    if "fps" in row:
        console.print(
            "[dim]FPS is this machine's, not the Jetson's — benchmark the engine on device.[/dim]"
        )


def _pretty(key: str) -> str:
    # 'metrics/mAP50-95(M)' -> 'mAP50-95 (mask)'
    name = key.removeprefix("metrics/").removeprefix("speed/")
    name = name.replace("(B)", " (box)").replace("(M)", " (mask)")
    return f"speed: {name}" if key.startswith("speed/") else name
