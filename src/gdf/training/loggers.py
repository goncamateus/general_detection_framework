from __future__ import annotations

import csv
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from gdf.utils.logging import log


class BaseLogger(ABC):
    @abstractmethod
    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        ...

    @abstractmethod
    def log_params(self, params: dict[str, Any]) -> None:
        ...

    @abstractmethod
    def finish(self) -> None:
        ...


class TBLogger(BaseLogger):
    def __init__(self, log_dir: Path) -> None:
        from torch.utils.tensorboard import SummaryWriter

        self._dir = log_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(str(log_dir))
        log.info(f"TensorBoard logging to: {log_dir}")

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        for k, v in metrics.items():
            self._writer.add_scalar(k, v, step)

    def log_params(self, params: dict[str, Any]) -> None:
        text = "\n".join(f"{k}: {v}" for k, v in params.items())
        self._writer.add_text("hyperparameters", text)

    def finish(self) -> None:
        self._writer.close()


class WandbLogger(BaseLogger):
    def __init__(self, project: str, config: dict[str, Any] | None = None) -> None:
        import wandb

        self._run = wandb.init(project=project, config=config)
        log.info(f"W&B run: {self._run.url}")

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        import wandb

        wandb.log(metrics, step=step)

    def log_params(self, params: dict[str, Any]) -> None:
        import wandb

        wandb.config.update(params)

    def finish(self) -> None:
        import wandb

        wandb.finish()


class CSVLogger(BaseLogger):
    def __init__(self, log_path: Path) -> None:
        self._path = log_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(log_path, "w", newline="")  # noqa: SIM115
        self._writer: csv.DictWriter | None = None
        log.info(f"CSV logging to: {log_path}")

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        row = {"step": step, **metrics}
        if self._writer is None:
            self._writer = csv.DictWriter(self._file, fieldnames=list(row.keys()))
            self._writer.writeheader()
        self._writer.writerow(row)
        self._file.flush()

    def log_params(self, params: dict[str, Any]) -> None:
        pass

    def finish(self) -> None:
        self._file.close()


class CompositeLogger(BaseLogger):
    def __init__(self, loggers: list[BaseLogger]) -> None:
        self._loggers = loggers

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        for logger in self._loggers:
            logger.log_metrics(metrics, step)

    def log_params(self, params: dict[str, Any]) -> None:
        for logger in self._loggers:
            logger.log_params(params)

    def finish(self) -> None:
        for logger in self._loggers:
            logger.finish()
