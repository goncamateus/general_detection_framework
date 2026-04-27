# GDF Wiki

Welcome to the **General Detection Framework** documentation.

GDF is a modular Python CLI for training YOLO image classification models, exporting them to ONNX/TensorRT, and deploying to NVIDIA Jetson devices.

## Getting Started

- [Architecture](Architecture.md) — how the code is organized, module responsibilities
- [Configuration](Configuration.md) — YAML config + CLI flag merge system
- [Datasets](Datasets.md) — local, Roboflow, HTTP dataset sources

## Guides

- [Training](Training.md) — train YOLO-cls models, logging, callbacks
- Export & Deployment](Export-and-Deployment.md) — ONNX, TensorRT, Jetson SSH deploy
- [CLI Reference](CLI-Reference.md) — all commands and flags

## Quick Links

- **Install**: `pip install -e ".[dev]"`
- **Entry point**: `gdf.cli.app:app` (registered as `gdf` console script)
- **Config schema**: `src/gdf/config/schema.py`
- **Model registry**: `src/gdf/models/registry.py`
