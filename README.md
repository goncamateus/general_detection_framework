# GDF — General Detection Framework

YOLO classification training, export, and deployment pipeline. CLI-first, modular, production-ready.

## Features

- **Train** YOLO-cls models (v8, v11, v26) with a single command
- **Export** to ONNX and TensorRT (FP16/FP32)
- **Deploy** to NVIDIA Jetson via SSH + TensorRT
- **Predict** with unified backend (PyTorch, ONNX, TensorRT auto-detect)
- **Dataset sources**: local folders, Roboflow API, HTTP zip/tar downloads
- **Logging**: TensorBoard + Weights & Biases out of the box
- **CLI** built with Typer + Rich

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Check environment
gdf info

# Train
gdf train --source local --data-path data/my_dataset --model-version v26 --model-size n --epochs 50

# Export to ONNX + TensorRT
gdf export --weights runs/train/gdf-exp/weights/best.pt --format both

# Predict
gdf predict --weights runs/export/model.onnx --source data/test_images/ --backend onnx
```

## YAML Config

All commands accept `--config path/to/config.yaml`. CLI flags override YAML values.

```yaml
# configs/default.yaml
model_version: v26
model_size: n
source: local
data_path: data/my_dataset
epochs: 100
batch_size: 16
imgsz: 224
lr: 0.001
output_dir: runs/train
project_name: gdf-exp
use_wandb: true
use_tensorboard: true
```

```bash
gdf train --config configs/default.yaml --epochs 20 --model-size s
```

## Supported Models

| Version | Sizes | Nano model |
|---------|-------|------------|
| YOLOv8  | n/s/m/l/x | `yolov8n-cls.pt` |
| YOLO11  | n/s/m/l/x | `yolo11n-cls.pt` |
| YOLO26  | n/s/m/l/x | `yolo26n-cls.pt` |

## Dataset Format

ImageFolder layout with `train/` and `val/` splits:

```
data/my_dataset/
  train/
    cats/
      img001.jpg
    dogs/
      img002.jpg
  val/
    cats/
    dogs/
```

### Remote Sources

| Source | `--data-path` format | Env var |
|--------|---------------------|---------|
| Local  | `/path/to/dataset`  | —       |
| Roboflow | `workspace/project/version` | `ROBOFLOW_API_KEY` |
| HTTP   | `https://url/dataset.zip` | — |

## Project Structure

```
src/gdf/
├── cli/          # Typer CLI commands
├── config/       # Pydantic v2 schemas
├── datasets/     # Source resolution, transforms, DataLoader factory
├── models/       # YOLO registry + wrapper
├── training/     # Trainer, callbacks, loggers (TB/W&B/CSV), metrics
├── export/       # ONNX, TensorRT, Jetson deploy
├── inference/    # UnifiedPredictor (auto-detect backend)
└── utils/        # Logging, file I/O, device detection
```

## CLI Reference

| Command | Description |
|---------|-------------|
| `gdf info` | Show device, CUDA, TensorRT, available models |
| `gdf train` | Train a YOLO-cls model |
| `gdf export` | Export weights to ONNX / TensorRT |
| `gdf predict` | Run inference on image(s) |

Run `gdf <command> --help` for all flags.

## Development

```bash
pip install -e ".[dev]"

pytest                        # run tests
pytest tests/test_config.py   # single module
ruff check src/ tests/        # lint
ruff format src/ tests/       # format
mypy src/                     # typecheck
```

## Optional Extras

```bash
pip install -e ".[gpu]"       # onnxruntime-gpu
pip install -e ".[trt]"       # tensorrt
pip install -e ".[jetson]"    # tensorrt + jetson-stats
```

## License

MIT
