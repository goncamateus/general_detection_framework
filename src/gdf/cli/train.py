from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from gdf.config.schema import TrainConfig

console = Console()


def train_cmd(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="YAML config file"),
    model_version: Optional[str] = typer.Option(None, "--model-version", "-mv", help="YOLO version: v8, v11, v26"),
    model_size: Optional[str] = typer.Option(None, "--model-size", "-ms", help="Model size: n, s, m, l, x"),
    source: Optional[str] = typer.Option(None, "--source", "-s", help="Dataset source: local, roboflow, http"),
    data_path: Optional[str] = typer.Option(None, "--data-path", "-d", help="Path or URL to dataset"),
    epochs: Optional[int] = typer.Option(None, "--epochs", "-e", help="Training epochs"),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", "-b", help="Batch size"),
    imgsz: Optional[int] = typer.Option(None, "--imgsz", help="Image size"),
    lr: Optional[float] = typer.Option(None, "--lr", help="Learning rate"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o", help="Output directory"),
    project_name: Optional[str] = typer.Option(None, "--project-name", "-p", help="Project name"),
    use_wandb: Optional[bool] = typer.Option(None, "--wandb/--no-wandb", help="Enable W&B logging"),
    use_tensorboard: Optional[bool] = typer.Option(None, "--tensorboard/--no-tensorboard", help="Enable TensorBoard"),
    patience: Optional[int] = typer.Option(None, "--patience", help="Early stopping patience"),
    workers: Optional[int] = typer.Option(None, "--workers", "-w", help="DataLoader workers"),
    device: Optional[str] = typer.Option(None, "--device", help="Device: auto, cpu, cuda:0"),
) -> None:
    cfg = _load_and_merge(config, {
        "model_version": model_version,
        "model_size": model_size,
        "source": source,
        "data_path": data_path,
        "epochs": epochs,
        "batch_size": batch_size,
        "imgsz": imgsz,
        "lr": lr,
        "output_dir": str(output_dir) if output_dir else None,
        "project_name": project_name,
        "use_wandb": use_wandb,
        "use_tensorboard": use_tensorboard,
        "patience": patience,
        "workers": workers,
        "device": device,
    })

    _print_config(cfg)

    from gdf.datasets.local import LocalDatasetSource
    from gdf.datasets.roboflow import RoboflowDatasetSource
    from gdf.datasets.http import HttpDatasetSource
    from gdf.models.yolo_cls import YOLOClsWrapper
    from gdf.training.loggers import TBLogger, WandbLogger, CompositeLogger
    from gdf.training.trainer import Trainer

    if cfg.source == "local":
        ds = LocalDatasetSource(cfg.data_path)
    elif cfg.source == "roboflow":
        parts = cfg.data_path.split("/")
        if len(parts) != 3:
            raise typer.BadParameter("roboflow data_path format: workspace/project/version")
        ds = RoboflowDatasetSource(parts[0], parts[1], int(parts[2]))
    elif cfg.source == "http":
        ds = HttpDatasetSource(cfg.data_path)
    else:
        raise typer.BadParameter(f"Unknown source: {cfg.source}")

    data_root = ds.resolve()
    if not ds.validate():
        console.print("[red]Dataset validation failed[/red]")
        raise typer.Exit(1)

    loggers = []
    if cfg.use_tensorboard:
        loggers.append(TBLogger(cfg.output_dir / cfg.project_name / "tb_logs"))
    if cfg.use_wandb:
        loggers.append(WandbLogger(project=cfg.project_name, config=cfg.model_dump()))

    model = YOLOClsWrapper(version=cfg.model_version, size=cfg.model_size)
    trainer = Trainer(
        model=model,
        data_path=data_root,
        output_dir=cfg.output_dir / cfg.project_name,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        imgsz=cfg.imgsz,
        lr=cfg.lr,
        patience=cfg.patience,
        workers=cfg.workers,
        device=cfg.device,
        loggers=loggers if loggers else None,
    )

    best_weights = trainer.train()
    console.print(f"\n[green]Training complete![/green] Best weights: {best_weights}")


def _load_and_merge(config_path: Path | None, cli_overrides: dict) -> TrainConfig:
    import yaml

    base: dict = {}
    if config_path and config_path.exists():
        with open(config_path) as f:
            base = yaml.safe_load(f) or {}

    for k, v in cli_overrides.items():
        if v is not None:
            base[k] = v

    if "data_path" not in base:
        raise typer.BadParameter("data_path is required (via --data-path or config file)")
    if "source" not in base:
        raise typer.BadParameter("source is required (via --source or config file)")

    return TrainConfig(**base)


def _print_config(cfg: TrainConfig) -> None:
    table = Table(title="Training Configuration", show_lines=True)
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    for k, v in cfg.model_dump().items():
        table.add_row(k, str(v))
    console.print(table)
