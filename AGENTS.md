# AGENTS.md

## Project

GDF — YOLO classification framework. Ultralytics models, PyTorch datasets, ONNX/TensorRT deploy, Typer CLI.

## Package/Layout

- Package manager: `uv` (pip-compatible). Python ≥3.10
- Build: hatchling. Source in `src/gdf/` (src-layout, not flat `gdf/`)
- CLI entrypoint: `gdf.cli.app:app` registered as `gdf` console script
- Model names: Ultralytics convention. `yolo26n-cls.pt`, NOT `yolov26n-cls.pt`. Same for v8 (`yolov8n-cls.pt`) and v11 (`yolo11n-cls.pt`)

## Commands

```bash
uv pip install -e ".[dev]"          # install editable + dev deps
pytest                               # run tests
pytest tests/test_config.py -k name  # single test / filter
ruff check src/ tests/               # lint
ruff format src/ tests/              # format
mypy src/                            # typecheck
gdf info                             # env/device check
gdf train --config configs/default.yaml --epochs 1
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
| `models/` | Registry (version+size→`.pt` name), YOLOClsWrapper |
| `training/` | Trainer orchestrator, loggers (TB/W&B/CSV), callbacks, metrics |
| `export/` | ONNX export + verify, TensorRT build (trtexec CLI or Python API), Jetson deploy |
| `inference/` | UnifiedPredictor (auto-detect backend), ONNX/TRT runners |
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
