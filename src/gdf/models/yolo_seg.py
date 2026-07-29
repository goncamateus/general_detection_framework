from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO

from gdf.models.registry import get_model_name
from gdf.utils.logging import log


class YOLOSegWrapper:
    # ponytail: v11 default (not v26 like detect/cls) — published -seg weights and TRT 8.2-safe
    # ops on the original Jetson Nano. Bump to v26 once its seg export is verified on device.
    def __init__(self, version: str = "v11", size: str = "n") -> None:
        self.version = version
        self.size = size
        self.model_name = get_model_name(version, size, task="segment")
        self.model: YOLO | None = None

    def load(self) -> YOLO:
        log.info(f"Loading segmentation model: {self.model_name}")
        self.model = YOLO(self.model_name)
        return self.model

    def load_from_weights(self, weights_path: str | Path) -> YOLO:
        log.info(f"Loading segmentation weights: {weights_path}")
        self.model = YOLO(str(weights_path))
        return self.model

    def train(self, data_path: str | Path | None = None, **kwargs) -> object:  # type: ignore[no-untyped-def]
        if self.model is None:
            self.load()
        path = data_path or kwargs.get("data")
        log.info(f"Training segmentation model {self.model_name} on {path}")
        if data_path is not None:
            kwargs["data"] = str(data_path)
        return self.model.train(**kwargs)  # type: ignore[union-attr]

    def predict(self, source: str | Path, **kwargs) -> list:  # type: ignore[type-arg]
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() or load_from_weights() first.")
        results = self.model.predict(source=str(source), **kwargs)  # type: ignore[union-attr]
        return list(results)

    def export(self, format: str = "onnx", **kwargs) -> str:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() or load_from_weights() first.")
        log.info(f"Exporting segmentation model to {format}")
        return self.model.export(format=format, **kwargs)  # type: ignore[union-attr]
