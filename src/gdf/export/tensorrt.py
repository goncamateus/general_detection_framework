from __future__ import annotations

import subprocess
from pathlib import Path

from gdf.utils.logging import log


def export_tensorrt(
    onnx_path: Path,
    output_path: Path | None = None,
    half: bool = False,
    workspace: int = 4096,
    imgsz: int = 224,
) -> Path:
    if output_path is None:
        suffix = ".fp16.engine" if half else ".engine"
        output_path = onnx_path.with_suffix(suffix)

    log.info(f"Building TensorRT engine: {onnx_path} → {output_path}")

    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={output_path}",
        f"--workspace={workspace}",
        f"--shapes=input:1x3x{imgsz}x{imgsz}",
    ]
    if half:
        cmd.append("--fp16")

    log.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        log.error(f"trtexec failed:\n{result.stderr}")
        raise RuntimeError(f"TensorRT export failed: {result.stderr}")

    log.info(f"TensorRT engine built: {output_path}")
    return output_path


def export_tensorrt_python(
    onnx_path: Path,
    output_path: Path | None = None,
    half: bool = False,
    workspace: int = 4096,
) -> Path:
    import tensorrt as trt

    if output_path is None:
        suffix = ".fp16.engine" if half else ".engine"
        output_path = onnx_path.with_suffix(suffix)

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                log.error(f"ONNX parse error: {parser.get_error(i)}")
            raise RuntimeError("ONNX parsing failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace * 1024 * 1024)

    if half and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(serialized)

    log.info(f"TensorRT engine built (Python): {output_path}")
    return output_path
