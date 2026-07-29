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
    model_version: Optional[str] = typer.Option(
        None, "--model-version", "-mv", help="YOLO version: v8, v11, v26"
    ),
    model_size: Optional[str] = typer.Option(
        None, "--model-size", "-ms", help="Model size: n, s, m, l, x"
    ),
    task: Optional[str] = typer.Option(
        None, "--task", "-t", help="Task: auto, cls, detect, segment"
    ),
    source: Optional[str] = typer.Option(
        None, "--source", "-s", help="Dataset source: local, roboflow, http"
    ),
    data_path: Optional[str] = typer.Option(
        None, "--data-path", "-d", help="Path or URL to dataset"
    ),
    epochs: Optional[int] = typer.Option(None, "--epochs", "-e", help="Training epochs"),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", "-b", help="Batch size"),
    imgsz: Optional[int] = typer.Option(None, "--imgsz", help="Image size"),
    lr: Optional[float] = typer.Option(None, "--lr", help="Learning rate"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o", help="Output directory"),
    project_name: Optional[str] = typer.Option(None, "--project-name", "-p", help="Project name"),
    use_wandb: Optional[bool] = typer.Option(None, "--wandb/--no-wandb", help="Enable W&B logging"),
    use_tensorboard: Optional[bool] = typer.Option(
        None, "--tensorboard/--no-tensorboard", help="Enable TensorBoard"
    ),
    patience: Optional[int] = typer.Option(None, "--patience", help="Early stopping patience"),
    workers: Optional[int] = typer.Option(None, "--workers", "-w", help="DataLoader workers"),
    device: Optional[str] = typer.Option(None, "--device", help="Device: auto, cpu, cuda:0"),
) -> None:
    cfg = _load_and_merge(
        config,
        {
            "model_version": model_version,
            "model_size": model_size,
            "task": task,
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
        },
    )

    _print_config(cfg)

    from gdf.datasets.http import HttpDatasetSource
    from gdf.datasets.local import LocalDatasetSource
    from gdf.datasets.roboflow import RoboflowDatasetSource
    from gdf.training.loggers import TBLogger, WandbLogger
    from gdf.training.trainer import Trainer

    if cfg.source == "local":
        ds = LocalDatasetSource(cfg.data_path)
    elif cfg.source == "roboflow":
        parts = cfg.data_path.split("/")
        if len(parts) != 3:
            raise typer.BadParameter("roboflow data_path format: workspace/project/version")
        yolo_format = cfg.model_version if cfg.model_version in ["v8", "v11"] else "26"
        ds = RoboflowDatasetSource(parts[0], parts[1], int(parts[2]), format=f"yolo{yolo_format}")
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

    task = _resolve_task(cfg.task, data_root)
    console.print(f"[cyan]Task: {task}[/cyan]")

    if task == "segment":
        from gdf.models.yolo_seg import YOLOSegWrapper

        model = YOLOSegWrapper(version=cfg.model_version, size=cfg.model_size)
    elif task == "detect":
        from gdf.models.yolo_detect import YOLODetectWrapper

        model = YOLODetectWrapper(version=cfg.model_version, size=cfg.model_size)
    else:
        from gdf.models.yolo_cls import YOLOClsWrapper

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


def _resolve_task(task: str, data_root: Path) -> str:
    """Pick the Ultralytics task for a dataset root.

    Detection and segmentation datasets share the same data.yaml; only the label rows
    differ (detect: `cls cx cy w h` = 5 fields, segment: `cls x1 y1 x2 y2 ...` = 7+).
    """
    if task != "auto":
        return task
    if not (data_root / "data.yaml").exists():
        return "cls"

    for labels_dir in (data_root / "train" / "labels", data_root / "labels"):
        if not labels_dir.is_dir():
            continue
        for label_file in sorted(labels_dir.glob("*.txt")):
            # Roboflow exports have no trailing newline, so readlines() undercounts.
            rows = [r for r in label_file.read_text().split("\n") if r.strip()]
            if not rows:
                continue
            return "segment" if len(rows[0].split()) > 5 else "detect"

    console.print("[yellow]No labels found to sniff; assuming detect[/yellow]")
    return "detect"


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
