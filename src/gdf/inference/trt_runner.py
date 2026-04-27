from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from gdf.datasets.transforms import get_predict_transforms
from gdf.utils.logging import log


class TRTRunner:
    def __init__(self, engine_path: Path, imgsz: int = 224) -> None:
        import tensorrt as trt

        self.trt = trt
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.imgsz = imgsz
        self.transform = get_predict_transforms(imgsz)

        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401

        self.cuda = cuda

    def predict(self, image: str | Path) -> tuple[int, float]:
        import torch

        img = Image.open(image).convert("RGB")
        tensor = self.transform(img).unsqueeze(0)
        input_data = tensor.numpy().astype(np.float32)

        output_shape = (1, 1000)
        output = np.empty(output_shape, dtype=np.float32)

        d_input = self.cuda.mem_alloc(input_data.nbytes)
        d_output = self.cuda.mem_alloc(output.nbytes)

        self.cuda.memcpy_htod(d_input, input_data)
        self.context.execute_v2(bindings=[int(d_input), int(d_output)])
        self.cuda.memcpy_dtoh(output, d_output)

        logits = output[0]
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()
        class_id = int(np.argmax(probs))
        confidence = float(probs[class_id])

        d_input.free()
        d_output.free()

        return class_id, confidence

    def predict_batch(self, images: list[str | Path]) -> list[tuple[int, float]]:
        return [self.predict(img) for img in images]
