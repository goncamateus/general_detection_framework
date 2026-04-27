# AGENTS.md

## Project

GDF — YOLO classification + detection + tracking framework. Ultralytics models, PyTorch datasets, ONNX/TensorRT deploy, ByteTrack, Typer CLI.

## Package/Layout

- Package manager: `uv` (pip-compatible). Python ≥3.10
- Build: hatchling. Source in `src/gdf/` (src-layout, not flat `gdf/`)
- CLI entrypoint: `gdf.cli.app:app` registered as `gdf` console script
- Model names: Ultralytics convention. `yolo26n-cls.pt`, NOT `yolov26n-cls.pt`. Same for v8 (`yolov8n-cls.pt`) and v11 (`yolo11n-cls.pt`)
- Detection models: `yolo26n.pt`, `yolov8n.pt`, `yolo11n.pt` (no `-cls` suffix)

## Commands

```bash
uv pip install -e ".[dev]"          # install editable + dev deps
pytest                               # run tests
pytest tests/test_tracker.py         # tracker tests
pytest tests/test_config.py -k name  # single test / filter
ruff check src/ tests/               # lint
ruff format src/ tests/              # format
mypy src/                            # typecheck
gdf info                             # env/device check
gdf train --config configs/default.yaml --epochs 1
gdf track --weights model.onnx --source video.mp4 --backend onnx --output tracked.mp4
```

No CI, no pre-commit hooks configured yet.

## Config System

Merge order: defaults → YAML file → CLI flags. Pydantic v2 validates final config.
- `source` and `data_path` always required (YAML or CLI)
- Roboflow: `data_path` format is `workspace/project/version`, requires `ROBOFLOW_API_KEY` env var
- Secrets via env vars only (`ROBOFLOW_API_KEY`, `WANDB_API_KEY`)

## Training

Delegates to Ultralytics `.train()` — not a custom PyTorch loop. `YOLOClsWrapper` loads `.pt`, calls `model.train(data=..., epochs=..., ...)`. Ultralytics handles the actual training loop internally.

Best weights saved to `output_dir/project_name/weights/best.pt` (Ultralytics convention). Falls back to `output_dir/best.pt`.

## Tracking

Custom ByteTrack implementation for ONNX/TRT. Ultralytics built-in tracker for PyTorch.

- **ONNX/TRT**: `inference/tracker/` — Kalman filter + Hungarian matching, two-phase association
- **PyTorch**: delegates to `model.track()` from Ultralytics
- Detection runners: `innx_detect_runner.py`, `trt_detect_runner.py` — preprocess, postprocess (NMS), video I/O
- Track needs 3 hits to confirm, removed after 30 frames without match
- `UnifiedPredictor(task="detect")` — use `detect()` for single frame, `track()` for tracking

## Dataset Convention

ImageFolder layout expected:
```
data_root/
  train/
    class_a/
      img1.jpg
    class_b/
  val/
    class_a/
    class_b/
```

## Submodule Responsibilities

| Dir | Owns |
|-----|------|
| `config/` | Pydantic schemas only, no logic |
| `datasets/` | Source resolution (local/roboflow/http), transforms, DataLoader factory |
| `models/` | Registry (version+size→`.pt` name), YOLOClsWrapper, YOLODetectWrapper |
| `training/` | Trainer orchestrator, loggers (TB/W&B/CSV), callbacks, metrics |
| `export/` | ONNX export + verify, TensorRT build (trtexec CLI or Python API), Jetson deploy |
| `inference/` | UnifiedPredictor (auto-detect backend, cls/detect task), ONNX/TRT runners, ByteTrack |
| `inference/tracker/` | ByteTrack: Kalman filter, Track state, Hungarian matching |
| `cli/` | Typer commands, YAML+CLI merge logic |
| `utils/` | Logging (Rich), file I/O, device detection |

## Gotchas

- `onnxruntime` in deps (CPU). Use `uv pip install -e ".[gpu]"` for `onnxruntime-gpu`
- `tensorrt` not in base deps. Use `[trt]` or `[jetson]` extras
- TRT runner needs `pycuda` (not in deps, must be installed separately on CUDA machines)
- Roboflow `data_path` split expects exactly 3 parts: `workspace/project/version`
- CLI imports heavy modules (torch, ultralytics) lazily inside command functions, not at module top
- `--wandb/--no-wandb` and `--tensorboard/--no-tensorboard` are Typer flag pairs, not `--use-wandb`
- `gdf train` without `--config` requires both `--source` and `--data-path` or it errors
- Detection models use `task="detect"` in UnifiedPredictor, classification uses `task="cls"`
- ByteTrack output shape for YOLO detect ONNX: `[1, 84, 8400]` (4 box coords + 80 class scores)
