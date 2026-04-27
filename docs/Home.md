# GDF Wiki

Welcome to the **General Detection Framework** documentation.

GDF is a modular Python CLI for training YOLO image classification and detection models, exporting them to ONNX/TensorRT, deploying to NVIDIA Jetson, and tracking objects across video frames.

## Getting Started

- [Architecture](Architecture.md) — how the code is organized, module responsibilities
- [Configuration](Configuration.md) — YAML config + CLI flag merge system
- [Datasets](Datasets.md) — local, Roboflow, HTTP dataset sources

## Guides

- [Training](Training.md) — train YOLO-cls models, logging, callbacks
- [Export & Deployment](Export-and-Deployment.md) — ONNX, TensorRT, Jetson SSH deploy
- [Tracking](Tracking.md) — ByteTrack object tracking with ONNX/TensorRT
- [CLI Reference](CLI-Reference.md) — all commands and flags

## Quick Links

- **Install**: `pip install -e ".[dev]"`
- **Entry point**: `gdf.cli.app:app` (registered as `gdf` console script)
- **Config schema**: `src/gdf/config/schema.py`
- **Model registry**: `src/gdf/models/registry.py`
