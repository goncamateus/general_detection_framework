from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO

from gdf.models.registry import get_model_name
from gdf.utils.logging import log


class YOLODetectWrapper:
    def __init__(self, version: str = "v26", size: str = "n") -> None:
        self.version = version
        self.size = size
        self.model_name = get_model_name(version, size, task="detect")
        self.model: YOLO | None = None

    def load(self) -> YOLO:
        log.info(f"Loading detection model: {self.model_name}")
        self.model = YOLO(self.model_name)
        return self.model

    def load_from_weights(self, weights_path: str | Path) -> YOLO:
        log.info(f"Loading detection weights: {weights_path}")
        self.model = YOLO(str(weights_path))
        return self.model

    def predict(self, source: str | Path, **kwargs) -> list:  # type: ignore[type-arg]
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() or load_from_weights() first.")
        results = self.model.predict(source=str(source), **kwargs)  # type: ignore[union-attr]
        return list(results)

    def track(self, source: str | Path, tracker: str = "bytetrack.yaml", **kwargs) -> list:  # type: ignore[type-arg]
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() or load_from_weights() first.")
        results = self.model.track(source=str(source), tracker=tracker, **kwargs)  # type: ignore[union-attr]
        return list(results)

    def export(self, format: str = "onnx", **kwargs) -> str:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() or load_from_weights() first.")
        log.info(f"Exporting detection model to {format}")
        return self.model.export(format=format, **kwargs)  # type: ignore[union-attr]
