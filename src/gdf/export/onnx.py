from __future__ import annotations

import subprocess
from pathlib import Path

from gdf.models.yolo_cls import YOLOClsWrapper
from gdf.utils.logging import log


def export_onnx(
    weights: Path,
    imgsz: int = 224,
    half: bool = False,
    output_dir: Path | None = None,
) -> Path:
    wrapper = YOLOClsWrapper()
    wrapper.load_from_weights(weights)

    log.info(f"Exporting to ONNX: imgsz={imgsz}, half={half}")
    export_path = wrapper.export(format="onnx", imgsz=imgsz, half=half)

    onnx_path = Path(export_path)
    if output_dir is not None:
        import shutil

        dest = output_dir / onnx_path.name
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(onnx_path, dest)
        onnx_path = dest

    log.info(f"ONNX exported: {onnx_path}")
    return onnx_path


def verify_onnx(onnx_path: Path) -> bool:
    try:
        import onnx

        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)
        log.info(f"ONNX model valid: {onnx_path}")
        return True
    except Exception as e:
        log.error(f"ONNX verification failed: {e}")
        return False
