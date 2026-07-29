# Export & Deployment

GDF supports exporting trained models to ONNX and TensorRT, plus remote deployment to NVIDIA Jetson.

## ONNX Export

```bash
gdf export --weights runs/train/gdf-exp/weights/best.pt --format onnx
```

Output: `model.onnx` in the same directory (or `--output-dir`).

ONNX export uses Ultralytics built-in exporter. The exported model is verified with `onnx.checker`.

### Don't pass `--half` when the target is TensorRT

`trtexec --fp16` builds an FP16 engine from an **FP32** ONNX, and does the conversion
properly. Exporting an FP16 ONNX first buys nothing and costs you two things:

- Ultralytics routes `half=True` through onnxconverter-common, which **appends** its
  `graph_input_cast*` node at the end of the node list even though the first Conv consumes
  it. That violates the ONNX topological-order requirement. onnxruntime sorts internally
  and does not care; TensorRT's ONNX parser walks the nodes as listed and can reject it.
- The graph still takes FP32 input, so nothing downstream gets simpler.

`verify_onnx()` repairs this automatically — it reorders the nodes and re-saves, leaving
output numerically identical — but the clean path is to skip `--half` on the ONNX step:

```bash
gdf export --weights best.pt --format onnx --imgsz 640      # FP32 ONNX
trtexec --onnx=best.onnx --fp16 --saveEngine=plume.engine   # FP16 happens here
```

### onnxruntime silently falling back to CPU

`gdf run` and `gdf predict` log the provider that actually loaded, and warn when GPU
providers were on offer but failed. The usual cause is that onnxruntime's wheel wants CUDA
libraries it cannot find: PyTorch ships them inside the venv, but onnxruntime only searches
the system loader path.

```
Failed to load library ...libonnxruntime_providers_cuda.so
  with error: libcublasLt.so.13: cannot open shared object file
```

Point the loader at the ones PyTorch already installed:

```bash
NV=$(python -c 'import nvidia, pathlib; print(pathlib.Path(nvidia.__file__).parent)')
export LD_LIBRARY_PATH="$NV/cu13/lib:$NV/cudnn/lib:$LD_LIBRARY_PATH"
```

`TensorrtExecutionProvider` needs a real TensorRT install (`libnvinfer.so`) on top of that;
the CUDA provider alone is enough for desktop testing.

## TensorRT Export

```bash
gdf export --weights best.pt --format tensorrt --half
```

**Two build methods:**

1. **trtexec CLI** (default) — calls `trtexec --onnx=model.onnx --saveEngine=model.engine`
2. **Python API** — uses `tensorrt` Python bindings (fallback if trtexec not found)

**Flags:**
- `--half` — FP16 precision (faster, slightly less accurate)
- `--workspace 4096` — TRT workspace in MB

## Both Formats

```bash
gdf export --weights best.pt --format both --half
```

Produces `model.onnx` + `model.engine`.

## Jetson Deployment

Deploy a TensorRT engine to a Jetson device over SSH:

```python
from gdf.export.jetson import JetsonDeployer

deployer = JetsonDeployer(host="192.168.1.50", user="nvidia")
deployer.deploy(
    engine_path=Path("model.engine"),
    class_names=["cat", "dog", "bird"],
)
```

**What it does:**
1. SSH into the Jetson
2. Create remote directory (`/home/nvidia/gdf_deploy/`)
3. Copy `.engine` + `class_names.txt` via SCP
4. Ready for inference

**Benchmark on Jetson:**

```python
result = deployer.benchmark(iterations=100)
print(result)
```

## Inference

### Unified Predictor

Auto-detects backend from file extension:

```python
from gdf.inference.engine import UnifiedPredictor

predictor = UnifiedPredictor(
    weights=Path("model.onnx"),  # auto → ONNX backend
    class_names=["cat", "dog"],
    imgsz=224,
)

result = predictor.predict("image.jpg")
print(result.class_name, result.confidence)
```

### Backend Selection

| Extension | Backend | Runner |
|-----------|---------|--------|
| `.pt` | PyTorch | Ultralytics YOLO |
| `.onnx` | ONNX Runtime | `onnx_runner.py` |
| `.engine` | TensorRT | `trt_runner.py` |

### Batch Prediction

```python
results = predictor.predict_dir("data/test_images/")
for img_path, pred in results:
    print(f"{img_path.name}: {pred.class_name} ({pred.confidence:.2%})")
```

## Dependencies

| Export Format | Extra Required |
|---------------|---------------|
| ONNX | `onnxruntime` (base deps) |
| TensorRT | `pip install -e ".[trt]"` |
| Jetson | `pip install -e ".[jetson]"` |
| ONNX GPU | `pip install -e ".[gpu]"` |
