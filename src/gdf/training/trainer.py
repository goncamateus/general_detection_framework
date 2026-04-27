from __future__ import annotations

from pathlib import Path
from typing import Any

from gdf.models.yolo_cls import YOLOClsWrapper
from gdf.training.callbacks import CheckpointCallback, EarlyStoppingCallback
from gdf.training.loggers import BaseLogger, CompositeLogger
from gdf.utils.logging import log


class Trainer:
    def __init__(
        self,
        model: YOLOClsWrapper,
        data_path: Path,
        output_dir: Path,
        epochs: int = 100,
        batch_size: int = 16,
        imgsz: int = 224,
        lr: float = 0.001,
        patience: int = 50,
        workers: int = 8,
        device: str = "auto",
        loggers: list[BaseLogger] | None = None,
    ) -> None:
        self.model = model
        self.data_path = data_path
        self.output_dir = output_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.imgsz = imgsz
        self.lr = lr
        self.patience = patience
        self.workers = workers
        self.device = device
        self.logger = CompositeLogger(loggers) if loggers else None

        self.ckpt_cb = CheckpointCallback(output_dir)
        self.early_stop_cb = EarlyStoppingCallback(patience=patience)

    def train(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        params = {
            "model_version": self.model.version,
            "model_size": self.model.size,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "imgsz": self.imgsz,
            "lr": self.lr,
            "data_path": str(self.data_path),
        }
        if self.logger:
            self.logger.log_params(params)

        log.info(f"Starting training: {self.model.model_name} for {self.epochs} epochs")
        log.info(f"Data: {self.data_path}, Output: {self.output_dir}")

        # Use data.yaml if available (detection datasets), else directory path (classification)
        data_yaml = self.data_path / "data.yaml"
        data_arg = str(data_yaml) if data_yaml.exists() else str(self.data_path)

        results = self.model.train(
            data=data_arg,
            epochs=self.epochs,
            batch=self.batch_size,
            imgsz=self.imgsz,
            lr0=self.lr,
            patience=self.patience,
            workers=self.workers,
            project=str(self.output_dir),
            device=self.device if self.device != "auto" else None,
            exist_ok=True,
        )

        best_weights = self.output_dir / "weights" / "best.pt"
        if not best_weights.exists():
            best_weights = self.output_dir / "best.pt"

        if self.logger:
            self.logger.finish()

        log.info(f"Training complete. Best weights: {best_weights}")
        return best_weights
