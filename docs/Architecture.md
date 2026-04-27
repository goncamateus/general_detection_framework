# Architecture

GDF follows a src-layout Python package structure. Each submodule owns a single responsibility.

## Module Map

```
src/gdf/
├── cli/            ← Typer commands, YAML+CLI merge, Rich output
├── config/         ← Pydantic v2 schemas (TrainConfig, ExportConfig, PredictConfig)
├── datasets/       ← Source resolution (local/roboflow/http), transforms, DataLoader
├── models/         ← YOLO registry (version+size→.pt name), YOLOClsWrapper
├── training/       ← Trainer orchestrator, loggers (TB/W&B/CSV), callbacks, metrics
├── export/         ← ONNX export+verify, TensorRT build (trtexec or Python API), Jetson deploy
├── inference/      ← UnifiedPredictor (auto-detect backend), ONNX/TRT runners
└── utils/          ← Rich logging, file I/O, device detection
```

## Data Flow

```
CLI (train command)
  → load YAML → merge CLI flags → validate Pydantic
  → resolve dataset (download if remote)
  → build transforms + DataLoaders
  → init model from registry
  → init loggers (TensorBoard + W&B)
  → Trainer.train() → Ultralytics .train()
  → save best.pt + config snapshot
```

## Key Design Decisions

### Training delegates to Ultralytics

`YOLOClsWrapper` loads a `.pt` model and calls `model.train(data=..., epochs=...)`. GDF does **not** implement a custom training loop. Ultralytics handles the actual loop, LR scheduling, augmentation, and checkpointing internally.

Best weights land at `output_dir/project_name/weights/best.pt` (Ultralytics convention).

### Config merge order

Defaults → YAML file → CLI flags. Pydantic v2 validates the final merged dict. This means:

- YAML provides base values
- Any CLI flag overrides the YAML value
- `source` and `data_path` are always required

### Lazy imports in CLI

Heavy modules (`torch`, `ultralytics`) are imported **inside** command functions, not at module top. This keeps `gdf --version` fast and avoids import errors when only checking help.

### Backend-agnostic inference

`UnifiedPredictor` auto-detects backend from file extension:
- `.pt` → PyTorch (via Ultralytics)
- `.onnx` → ONNX Runtime
- `.engine` → TensorRT + pycuda

## Dependency Graph

```
cli/train.py
  ├── config/schema.py  (validation)
  ├── datasets/*        (data loading)
  ├── models/yolo_cls.py (model)
  ├── training/trainer.py (orchestration)
  │   ├── training/loggers.py
  │   └── training/callbacks.py
  └── utils/logging.py

cli/export.py
  ├── export/onnx.py
  └── export/tensorrt.py

cli/predict.py
  └── inference/engine.py
      ├── inference/onnx_runner.py
      └── inference/trt_runner.py
```
