from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO

from gdf.models.registry import get_model_name
from gdf.utils.logging import log


class YOLOClsWrapper:
    def __init__(self, version: str = "v26", size: str = "n") -> None:
        self.version = version
        self.size = size
        self.model_name = get_model_name(version, size)
        self.model: YOLO | None = None

    def load(self) -> YOLO:
        log.info(f"Loading model: {self.model_name}")
        self.model = YOLO(self.model_name)
        return self.model

    def load_from_weights(self, weights_path: str | Path) -> YOLO:
        log.info(f"Loading weights: {weights_path}")
        self.model = YOLO(str(weights_path))
        return self.model

    def train(self, data_path: str | Path, **kwargs) -> object:  # type: ignore[no-untyped-def]
        if self.model is None:
            self.load()
        log.info(f"Training {self.model_name} on {data_path}")
        return self.model.train(data=str(data_path), **kwargs)  # type: ignore[union-attr]

    def export(self, format: str = "onnx", **kwargs) -> str:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() or load_from_weights() first.")
        log.info(f"Exporting model to {format}")
        return self.model.export(format=format, **kwargs)  # type: ignore[union-attr]

    def predict(self, source: str | Path, **kwargs) -> list:  # type: ignore[type-arg]
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() or load_from_weights() first.")
        results = self.model.predict(source=str(source), **kwargs)  # type: ignore[union-attr]
        return list(results)
