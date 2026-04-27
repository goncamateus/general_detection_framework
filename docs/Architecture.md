# Architecture

GDF follows a src-layout Python package structure. Each submodule owns a single responsibility.

## Module Map

```
src/gdf/
├── cli/            ← Typer commands, YAML+CLI merge, Rich output
├── config/         ← Pydantic v2 schemas (TrainConfig, ExportConfig, PredictConfig, TrackConfig)
├── datasets/       ← Source resolution (local/roboflow/http), transforms, DataLoader
├── models/         ← YOLO registry (version+size→.pt name), YOLOClsWrapper, YOLODetectWrapper
├── training/       ← Trainer orchestrator, loggers (TB/W&B/CSV), callbacks, metrics
├── export/         ← ONNX export+verify, TensorRT build (trtexec or Python API), Jetson deploy
├── inference/
│   ├── engine.py           ← UnifiedPredictor (auto-detect backend, cls/detect task)
│   ├── onnx_runner.py      ← ONNX classification runner
│   ├── onnx_detect_runner.py ← ONNX detection + ByteTrack
│   ├── trt_runner.py       ← TensorRT classification runner
│   ├── trt_detect_runner.py  ← TensorRT detection + ByteTrack
│   └── tracker/            ← Custom ByteTrack implementation
│       ├── kalman.py       ← Kalman filter for bbox prediction
│       ├── track.py        ← Track state management
│       └── byte_tracker.py ← Hungarian matching + two-phase association
└── utils/          ← Rich logging, file I/O, device detection
```

## Data Flow

### Training

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

### Tracking

```
CLI (track command)
  → init UnifiedPredictor(task="detect")
  → load detection model (YOLO/ONNX/TRT)
  → for each frame:
      → detect: preprocess → inference → postprocess (NMS)
      → track: ByteTracker.update(bboxes, scores, class_ids)
          → Kalman predict
          → Phase 1: match high-conf detections to tracks (Hungarian)
          → Phase 2: match low-conf detections to remaining tracks
          → create new tracks, remove lost tracks
      → draw boxes + IDs on frame
  → write output video or CSV
```

## Key Design Decisions

### Training delegates to Ultralytics

`YOLOClsWrapper` loads a `.pt` model and calls `model.train(data=..., epochs=...)`. GDF does **not** implement a custom training loop. Ultralytics handles the actual loop, LR scheduling, augmentation, and checkpointing internally.

### Tracking: Ultralytics for PyTorch, custom for ONNX/TRT

- **PyTorch backend**: uses Ultralytics' built-in `model.track()` (BoT-SORT/ByteTrack)
- **ONNX/TRT backends**: uses GDF's custom ByteTrack implementation (Kalman + Hungarian)

This is because Ultralytics' tracker is tightly coupled to its inference pipeline. When running ONNX/TRT, we need a standalone tracker.

### ByteTrack is implemented from scratch

No external tracker library (like `supervision` or `boxmot`). Uses:
- `scipy.optimize.linear_sum_assignment` for Hungarian matching
- Custom Kalman filter for motion prediction
- Two-phase matching: high-confidence first, then low-confidence

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

Task is set explicitly: `task="cls"` for classification, `task="detect"` for detection/tracking.

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

cli/track.py
  └── inference/engine.py (task="detect")
      ├── models/yolo_detect.py (PyTorch tracking)
      ├── inference/onnx_detect_runner.py (ONNX + ByteTrack)
      ├── inference/trt_detect_runner.py (TRT + ByteTrack)
      └── inference/tracker/
          ├── kalman.py
          ├── track.py
          └── byte_tracker.py
```
