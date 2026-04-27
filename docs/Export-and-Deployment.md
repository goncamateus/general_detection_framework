# Export & Deployment

GDF supports exporting trained models to ONNX and TensorRT, plus remote deployment to NVIDIA Jetson.

## ONNX Export

```bash
gdf export --weights runs/train/gdf-exp/weights/best.pt --format onnx
```

Output: `model.onnx` in the same directory (or `--output-dir`).

ONNX export uses Ultralytics built-in exporter. The exported model is verified with `onnx.checker`.

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
