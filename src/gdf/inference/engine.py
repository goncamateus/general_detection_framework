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


class DetectionResult:
    def __init__(
        self,
        bboxes: "np.ndarray",
        class_ids: "np.ndarray",
        scores: "np.ndarray",
        class_names: list[str] | None = None,
    ) -> None:
        self.bboxes = bboxes
        self.class_ids = class_ids
        self.scores = scores
        self.class_names = class_names or []

    def __len__(self) -> int:
        return len(self.class_ids)

    def __repr__(self) -> str:
        return f"DetectionResult({len(self)} detections)"

    def to_dicts(self) -> list[dict]:
        results = []
        for i in range(len(self)):
            name = self.class_names[self.class_ids[i]] if self.class_ids[i] < len(self.class_names) else str(self.class_ids[i])
            results.append({
                "bbox": self.bboxes[i].tolist(),
                "class_id": int(self.class_ids[i]),
                "class_name": name,
                "score": float(self.scores[i]),
            })
        return results


class UnifiedPredictor:
    def __init__(
        self,
        weights: Path,
        backend: Literal["pytorch", "onnx", "tensorrt"] = "pytorch",
        class_names: list[str] | None = None,
        imgsz: int = 224,
        conf_threshold: float = 0.5,
        task: Literal["cls", "detect"] = "cls",
    ) -> None:
        self.weights = Path(weights)
        self.backend = backend
        self.class_names = class_names or []
        self.imgsz = imgsz
        self.conf_threshold = conf_threshold
        self.task = task
        self._runner: object | None = None

    def _init_runner(self) -> None:
        if self.task == "detect":
            self._init_detect_runner()
        else:
            self._init_cls_runner()

    def _init_cls_runner(self) -> None:
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

    def _init_detect_runner(self) -> None:
        if self.backend == "onnx":
            from gdf.inference.onnx_detect_runner import ONNXDetectRunner
            self._runner = ONNXDetectRunner(self.weights, self.imgsz)
        elif self.backend == "tensorrt":
            from gdf.inference.trt_detect_runner import TRTDetectRunner
            self._runner = TRTDetectRunner(self.weights, self.imgsz)
        elif self.backend == "pytorch":
            from gdf.models.yolo_detect import YOLODetectWrapper
            wrapper = YOLODetectWrapper()
            wrapper.load_from_weights(self.weights)
            self._runner = wrapper
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def predict(self, image: str | Path) -> PredictionResult:
        if self._runner is None:
            self._init_runner()

        if self.task == "detect":
            raise ValueError("Use detect() for detection models, not predict()")

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

    def detect(self, image: str | Path) -> DetectionResult:
        if self._runner is None:
            self._init_runner()

        if self.task != "detect":
            raise ValueError("Use predict() for classification models, not detect()")

        if self.backend == "pytorch":
            results = self._runner.predict(image, conf=self.conf_threshold)  # type: ignore[union-attr]
            r = results[0]
            bboxes = r.boxes.xyxy.cpu().numpy()
            class_ids = r.boxes.cls.cpu().numpy().astype(int)
            scores = r.boxes.conf.cpu().numpy()
        else:
            bboxes, class_ids, scores = self._runner.detect(image, conf_threshold=self.conf_threshold)  # type: ignore[union-attr]

        return DetectionResult(
            bboxes=bboxes,
            class_ids=class_ids,
            scores=scores,
            class_names=self.class_names,
        )

    def track(self, image: str | Path):
        """Run detection + tracking on a single frame. Returns ByteTrackResult."""
        if self._runner is None:
            self._init_runner()

        if self.task != "detect":
            raise ValueError("Tracking requires detection models")

        if self.backend == "pytorch":
            results = self._runner.track(image, conf=self.conf_threshold)  # type: ignore[union-attr]
            r = results[0]
            if r.boxes.id is None:
                import numpy as np
                from gdf.inference.tracker import ByteTrackResult
                return ByteTrackResult(
                    np.empty((0, 4), dtype=np.float32),
                    np.array([], dtype=np.int32),
                    np.array([], dtype=np.float32),
                    np.array([], dtype=np.int32),
                )
            bboxes = r.boxes.xyxy.cpu().numpy()
            class_ids = r.boxes.cls.cpu().numpy().astype(int)
            scores = r.boxes.conf.cpu().numpy()
            track_ids = r.boxes.id.cpu().numpy().astype(int)
            from gdf.inference.tracker import ByteTrackResult
            return ByteTrackResult(bboxes, class_ids, scores, track_ids)
        else:
            return self._runner.detect_and_track(image, conf_threshold=self.conf_threshold)  # type: ignore[union-attr]

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
