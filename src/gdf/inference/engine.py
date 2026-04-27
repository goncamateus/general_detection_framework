from __future__ import annotations

from pathlib import Path
from typing import Literal

from gdf.utils.logging import log


class PredictionResult:
    def __init__(self, class_id: int, class_name: str, confidence: float) -> None:
        self.class_id = class_id
        self.class_name = class_name
        self.confidence = confidence

    def __repr__(self) -> str:
        return f"PredictionResult(class={self.class_name}, conf={self.confidence:.4f})"


class UnifiedPredictor:
    def __init__(
        self,
        weights: Path,
        backend: Literal["pytorch", "onnx", "tensorrt"] = "pytorch",
        class_names: list[str] | None = None,
        imgsz: int = 224,
        conf_threshold: float = 0.5,
    ) -> None:
        self.weights = Path(weights)
        self.backend = backend
        self.class_names = class_names or []
        self.imgsz = imgsz
        self.conf_threshold = conf_threshold
        self._runner: object | None = None

    def _init_runner(self) -> None:
        if self.backend == "onnx":
            from gdf.inference.onnx_runner import ONNXRunner

            self._runner = ONNXRunner(self.weights, self.imgsz)
        elif self.backend == "tensorrt":
            from gdf.inference.trt_runner import TRTRunner

            self._runner = TRTRunner(self.weights, self.imgsz)
        elif self.backend == "pytorch":
            from gdf.models.yolo_cls import YOLOClsWrapper

            wrapper = YOLOClsWrapper()
            wrapper.load_from_weights(self.weights)
            self._runner = wrapper
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def predict(self, image: str | Path) -> PredictionResult:
        if self._runner is None:
            self._init_runner()

        if self.backend == "pytorch":
            results = self._runner.predict(image)  # type: ignore[union-attr]
            r = results[0]
            probs = r.probs
            class_id = int(probs.top1)
            confidence = float(probs.top1conf)
        else:
            class_id, confidence = self._runner.predict(image)  # type: ignore[union-attr]

        class_name = self.class_names[class_id] if class_id < len(self.class_names) else str(class_id)
        return PredictionResult(class_id=class_id, class_name=class_name, confidence=confidence)

    def predict_dir(self, directory: str | Path) -> list[tuple[Path, PredictionResult]]:
        dir_path = Path(directory)
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        results = []

        for img_path in sorted(dir_path.iterdir()):
            if img_path.suffix.lower() in image_extensions:
                pred = self.predict(img_path)
                results.append((img_path, pred))

        return results

    @staticmethod
    def auto_detect_backend(weights: Path) -> Literal["pytorch", "onnx", "tensorrt"]:
        suffix = weights.suffix.lower()
        if suffix == ".onnx":
            return "onnx"
        elif suffix == ".engine":
            return "tensorrt"
        elif suffix == ".pt":
            return "pytorch"
        else:
            raise ValueError(f"Cannot auto-detect backend for: {weights}")
