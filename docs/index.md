# General Detection Framework

GDF is a modular Python CLI for training YOLO image classification and detection models, exporting them to ONNX/TensorRT, deploying to NVIDIA Jetson, and tracking objects across video frames.

## Features

- **Training**: YOLOClsWrapper for classification, YOLODetectWrapper for detection
- **Export**: ONNX and TensorRT with FP16 support
- **Inference**: UnifiedPredictor with auto-backend detection
- **Tracking**: ByteTrack object tracking compatible with ONNX/TensorRT
- **CLI**: Typer-based commands with YAML + CLI flag merging

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Check environment
gdf info

# Train
gdf train --config configs/default.yaml

# Export to ONNX
gdf export --weights best.pt --format onnx

# Run webcam tracking
gdf webcam --weights best.onnx
```

## Documentation

- [Architecture](architecture.md) — code organization, module responsibilities
- [Configuration](configuration.md) — YAML config + CLI flag merge system
- [Datasets](datasets.md) — local, Roboflow, HTTP sources
- [Training](training.md) — train YOLO models, logging, callbacks
- [Export & Deployment](export-deployment.md) — ONNX, TensorRT, Jetson
- [Tracking](tracking.md) — ByteTrack with ONNX/TensorRT
- [CLI Reference](cli-reference.md) — all commands and flags