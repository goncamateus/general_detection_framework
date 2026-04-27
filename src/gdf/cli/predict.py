from __future__ import annotations

from pathlib import Path
from typing import Optional

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
    conf_threshold: Optional[float] = typer.Option(None, "--conf-threshold", help="Confidence threshold"),
    imgsz: Optional[int] = typer.Option(None, "--imgsz", help="Image size"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output CSV path"),
    device: Optional[str] = typer.Option(None, "--device", help="Device"),
    class_names_file: Optional[Path] = typer.Option(None, "--class-names", help="File with class names (one per line)"),
) -> None:
    cfg = _load_and_merge(config, {
        "weights": str(weights) if weights else None,
        "source": source,
        "backend": backend,
        "conf_threshold": conf_threshold,
        "imgsz": imgsz,
        "output": str(output) if output else None,
        "device": device,
    })

    class_names: list[str] = []
    if class_names_file and class_names_file.exists():
        class_names = class_names_file.read_text().strip().splitlines()

    from gdf.inference.engine import UnifiedPredictor

    if cfg.backend == "pytorch" and cfg.weights.suffix == ".onnx":
        cfg.backend = "onnx"
    elif cfg.backend == "pytorch" and cfg.weights.suffix == ".engine":
        cfg.backend = "tensorrt"

    predictor = UnifiedPredictor(
        weights=cfg.weights,
        backend=cfg.backend,
        class_names=class_names,
        imgsz=cfg.imgsz,
        conf_threshold=cfg.conf_threshold,
    )

    source_path = Path(cfg.source)
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
