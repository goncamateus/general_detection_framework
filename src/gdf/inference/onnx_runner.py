from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from gdf.datasets.transforms import get_predict_transforms
from gdf.utils.logging import log


class ONNXRunner:
    def __init__(self, model_path: Path, imgsz: int = 224) -> None:
        import onnxruntime as ort

        providers = ort.get_available_providers()
        log.info(f"ONNX providers: {providers}")

        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.imgsz = imgsz
        self.transform = get_predict_transforms(imgsz)

    def predict(self, image: str | Path) -> tuple[int, float]:
        img = Image.open(image).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).numpy()

        outputs = self.session.run(None, {self.input_name: tensor})
        logits = outputs[0][0]

        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()
        class_id = int(np.argmax(probs))
        confidence = float(probs[class_id])

        return class_id, confidence

    def predict_batch(self, images: list[str | Path]) -> list[tuple[int, float]]:
        results = []
        for img_path in images:
            results.append(self.predict(img_path))
        return results
