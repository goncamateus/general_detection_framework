# Configuration

GDF uses a two-layer config system: YAML files provide base values, CLI flags override them.

## Merge Order

```
hardcoded defaults
  → YAML file (--config)
    → CLI flags (--epochs, --model-size, etc.)
      → Pydantic v2 validation
```

## YAML Config

Create a YAML file with training parameters:

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
workers: 8
patience: 50
device: auto
```

Override any value via CLI:

```bash
gdf train --config configs/default.yaml --epochs 20 --model-size s
```

## Pydantic Schemas

All config is validated by Pydantic v2 models in `src/gdf/config/schema.py`:

### TrainConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_version` | `v8\|v11\|v26` | `v26` | YOLO version |
| `model_size` | `n\|s\|m\|l\|x` | `n` | Model size |
| `source` | `local\|roboflow\|http` | required | Dataset source |
| `data_path` | `str` | required | Path or URL |
| `epochs` | `int ≥1` | `100` | Training epochs |
| `batch_size` | `int ≥1` | `16` | Batch size |
| `imgsz` | `int ≥32` | `224` | Image size |
| `lr` | `float >0` | `0.001` | Learning rate |
| `output_dir` | `Path` | `runs/train` | Output directory |
| `project_name` | `str` | `gdf-exp` | Experiment name |
| `use_wandb` | `bool` | `true` | Enable W&B logging |
| `use_tensorboard` | `bool` | `true` | Enable TensorBoard |
| `workers` | `int ≥0` | `8` | DataLoader workers |
| `patience` | `int ≥0` | `50` | Early stopping patience |
| `device` | `str` | `auto` | Device (auto/cpu/cuda:0) |

### ExportConfig

| Field | Type | Default |
|-------|------|---------|
| `weights` | `Path` | required |
| `format` | `onnx\|tensorrt\|both` | `onnx` |
| `half` | `bool` | `false` |
| `workspace` | `int ≥1024` | `4096` (MB) |
| `imgsz` | `int ≥32` | `224` |

### PredictConfig

| Field | Type | Default |
|-------|------|---------|
| `weights` | `Path` | required |
| `source` | `str` | required |
| `backend` | `pytorch\|onnx\|tensorrt` | `pytorch` |
| `conf_threshold` | `0-1` | `0.5` |
| `imgsz` | `int ≥32` | `224` |

## Secrets

Secrets are **never** in config files. Use environment variables:

```bash
export ROBOFLOW_API_KEY=your_key
export WANDB_API_KEY=your_key
```
