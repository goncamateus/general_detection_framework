from __future__ import annotations

from pathlib import Path
from typing import Any

from gdf.utils.logging import log


class CheckpointCallback:
    def __init__(self, output_dir: Path, save_best: bool = True) -> None:
        self._dir = output_dir / "checkpoints"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._save_best = save_best
        self.best_metric: float = 0.0

    def on_epoch_end(self, epoch: int, metrics: dict[str, float], model_path: Path) -> Path | None:
        if self._save_best:
            current = metrics.get("accuracy", metrics.get("f1_macro", 0.0))
            if current > self.best_metric:
                self.best_metric = current
                best_path = self._dir / "best.pt"
                if model_path.exists():
                    import shutil

                    shutil.copy2(model_path, best_path)
                    log.info(f"New best model (metric={current:.4f}): {best_path}")
                return best_path
        return None


class EarlyStoppingCallback:
    def __init__(self, patience: int = 50, min_delta: float = 0.001) -> None:
        self._patience = patience
        self._min_delta = min_delta
        self._counter = 0
        self.best_metric: float = 0.0
        self.should_stop = False

    def on_epoch_end(self, metrics: dict[str, float]) -> None:
        current = metrics.get("accuracy", metrics.get("f1_macro", 0.0))
        if current > self.best_metric + self._min_delta:
            self.best_metric = current
            self._counter = 0
        else:
            self._counter += 1
            if self._counter >= self._patience:
                self.should_stop = True
                log.info(f"Early stopping triggered after {self._patience} epochs without improvement")
