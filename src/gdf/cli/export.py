from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from gdf.config.schema import ExportConfig

console = Console()


def export_cmd(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="YAML config file"),
    weights: Optional[Path] = typer.Option(None, "--weights", "-w", help="Path to .pt weights"),
    format: Optional[str] = typer.Option(None, "--format", "-f", help="Export format: onnx, tensorrt, both"),
    half: Optional[bool] = typer.Option(None, "--half/--no-half", help="FP16 precision"),
    workspace: Optional[int] = typer.Option(None, "--workspace", help="TRT workspace MB"),
    imgsz: Optional[int] = typer.Option(None, "--imgsz", help="Image size"),
    device: Optional[str] = typer.Option(None, "--device", help="Device"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o", help="Output directory"),
) -> None:
    cfg = _load_and_merge(config, {
        "weights": str(weights) if weights else None,
        "format": format,
        "half": half,
        "workspace": workspace,
        "imgsz": imgsz,
        "device": device,
        "output_dir": str(output_dir) if output_dir else None,
    })

    console.print(f"[cyan]Exporting:[/cyan] {cfg.weights} → {cfg.format}")

    from gdf.export.onnx import export_onnx, verify_onnx
    from gdf.export.tensorrt import export_tensorrt

    onnx_path: Path | None = None
    engine_path: Path | None = None

    if cfg.format in ("onnx", "both"):
        onnx_path = export_onnx(
            weights=cfg.weights,
            imgsz=cfg.imgsz,
            half=cfg.half,
            output_dir=cfg.output_dir,
        )
        verify_onnx(onnx_path)
        console.print(f"[green]ONNX exported:[/green] {onnx_path}")

    if cfg.format in ("tensorrt", "both"):
        if onnx_path is None:
            onnx_path = cfg.weights.with_suffix(".onnx")
            if not onnx_path.exists():
                onnx_path = export_onnx(
                    weights=cfg.weights,
                    imgsz=cfg.imgsz,
                    half=cfg.half,
                    output_dir=cfg.output_dir,
                )

        engine_path = export_tensorrt(
            onnx_path=onnx_path,
            half=cfg.half,
            workspace=cfg.workspace,
            imgsz=cfg.imgsz,
        )
        console.print(f"[green]TensorRT engine:[/green] {engine_path}")

    console.print("[green]Export complete![/green]")


def _load_and_merge(config_path: Path | None, cli_overrides: dict) -> ExportConfig:
    import yaml

    base: dict = {}
    if config_path and config_path.exists():
        with open(config_path) as f:
            base = yaml.safe_load(f) or {}

    for k, v in cli_overrides.items():
        if v is not None:
            base[k] = v

    if "weights" not in base:
        raise typer.BadParameter("weights is required (via --weights or config file)")

    base["weights"] = Path(base["weights"])
    if "output_dir" in base:
        base["output_dir"] = Path(base["output_dir"])

    return ExportConfig(**base)
