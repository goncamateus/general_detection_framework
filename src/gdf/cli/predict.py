from __future__ import annotations

from pathlib import Path
from typing import Optional, Literal

import typer
from rich.console import Console
from rich.table import Table

from gdf.config.schema import PredictConfig

console = Console()


def predict_cmd(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="YAML config file"),
    weights: Optional[Path] = typer.Option(None, "--weights", "-w", help="Path to model weights"),
    source: Optional[str] = typer.Option(None, "--source", "-s", help="Image path or directory"),
    backend: Optional[str] = typer.Option(None, "--backend", "-b", help="Backend: pytorch, onnx, tensorrt"),
    task: Optional[str] = typer.Option(None, "--task", "-t", help="Task: cls or detect"),
    conf_threshold: Optional[float] = typer.Option(None, "--conf-threshold", help="Confidence threshold"),
    imgsz: Optional[int] = typer.Option(None, "--imgsz", help="Image size"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output CSV path"),
    device: Optional[str] = typer.Option(None, "--device", help="Device"),
    class_names_file: Optional[Path] = typer.Option(None, "--class-names", help="File with class names (one per line)"),
) -> None:
    overrides = {
        "weights": str(weights) if weights else None,
        "source": source,
        "backend": backend,
        "task": task,
        "conf_threshold": conf_threshold,
        "imgsz": imgsz,
        "output": str(output) if output else None,
        "device": device,
    }
    cfg = _load_and_merge(config, overrides)

    class_names: list[str] = []
    if class_names_file and class_names_file.exists():
        class_names = class_names_file.read_text().strip().splitlines()

    from gdf.inference.engine import UnifiedPredictor

    if cfg.backend == "pytorch" and cfg.weights.suffix == ".onnx":
        cfg.backend = "onnx"
    elif cfg.backend == "pytorch" and cfg.weights.suffix == ".engine":
        cfg.backend = "tensorrt"

    task = cfg.task if hasattr(cfg, "task") else "cls"

    predictor = UnifiedPredictor(
        weights=cfg.weights,
        backend=cfg.backend,
        class_names=class_names,
        imgsz=cfg.imgsz,
        conf_threshold=cfg.conf_threshold,
        task=task,
    )

    source_path = Path(cfg.source)

    if task == "detect":
        _predict_detect(predictor, source_path, cfg)
    else:
        _predict_cls(predictor, source_path, cfg)


def _predict_cls(predictor, source_path: Path, cfg) -> None:
    if source_path.is_dir():
        results = predictor.predict_dir(source_path)
        table = Table(title="Predictions", show_lines=True)
        table.add_column("Image", style="cyan")
        table.add_column("Class", style="green")
        table.add_column("Confidence", style="yellow")

        rows_for_csv = []
        for img_path, pred in results:
            table.add_row(img_path.name, pred.class_name, f"{pred.confidence:.4f}")
            rows_for_csv.append({"image": img_path.name, "class": pred.class_name, "confidence": pred.confidence})

        console.print(table)

        if cfg.output:
            import csv
            cfg.output.parent.mkdir(parents=True, exist_ok=True)
            with open(cfg.output, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["image", "class", "confidence"])
                writer.writeheader()
                writer.writerows(rows_for_csv)
            console.print(f"[green]Results saved:[/green] {cfg.output}")
    else:
        pred = predictor.predict(source_path)
        console.print(f"[cyan]{source_path.name}[/cyan] → {pred.class_name} ({pred.confidence:.4f})")


def _predict_detect(predictor, source_path: Path, cfg) -> None:
    if source_path.is_dir():
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        images = sorted([f for f in source_path.iterdir() if f.suffix.lower() in image_extensions])

        table = Table(title="Detection Results", show_lines=True)
        table.add_column("Image", style="cyan")
        table.add_column("Class", style="green")
        table.add_column("Confidence", style="yellow")
        table.add_column("BBox", style="dim")

        rows_for_csv = []
        for img_path in images:
            det = predictor.detect(img_path)
            for d in det.to_dicts():
                table.add_row(img_path.name, d["class_name"], f"{d['score']:.3f}", str(d["bbox"]))
                rows_for_csv.append({"image": img_path.name, **d})

        console.print(table)

        if cfg.output:
            import csv
            cfg.output.parent.mkdir(parents=True, exist_ok=True)
            with open(cfg.output, "w", newline="") as f:
                if rows_for_csv:
                    writer = csv.DictWriter(f, fieldnames=rows_for_csv[0].keys())
                    writer.writeheader()
                    writer.writerows(rows_for_csv)
            console.print(f"[green]Results saved:[/green] {cfg.output}")
    else:
        det = predictor.detect(source_path)
        console.print(f"[cyan]{source_path.name}[/cyan] → {len(det)} detections")
        for d in det.to_dicts()[:5]:
            console.print(f"  {d['class_name']} ({d['score']:.2f}) {d['bbox']}")


def _load_and_merge(config_path: Path | None, cli_overrides: dict) -> PredictConfig:
    import yaml

    base: dict = {}
    if config_path and config_path.exists():
        with open(config_path) as f:
            base = yaml.safe_load(f) or {}

    for k, v in cli_overrides.items():
        if v is not None:
            base[k] = v

    if "weights" not in base:
        raise typer.BadParameter("weights is required")
    if "source" not in base:
        raise typer.BadParameter("source is required")

    base["weights"] = Path(base["weights"])
    if "output" in base and base["output"] is not None:
        base["output"] = Path(base["output"])

    return PredictConfig(**base)
