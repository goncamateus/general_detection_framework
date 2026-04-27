# CLI Reference

GDF CLI is built with [Typer](https://typer.tiangolo.com/) and [Rich](https://rich.readthedocs.io/).

## Global Flags

```bash
gdf --version          # show version
gdf --help             # show all commands
```

## gdf info

Show environment info: device, CUDA, TensorRT, available models.

```bash
gdf info
```

## gdf train

Train a YOLO-cls model.

```bash
gdf train [OPTIONS]
```

| Flag | Short | Type | Description |
|------|-------|------|-------------|
| `--config` | `-c` | Path | YAML config file |
| `--model-version` | `-mv` | str | `v8`, `v11`, `v26` |
| `--model-size` | `-ms` | str | `n`, `s`, `m`, `l`, `x` |
| `--source` | `-s` | str | `local`, `roboflow`, `http` |
| `--data-path` | `-d` | str | Path or URL to dataset |
| `--epochs` | `-e` | int | Training epochs |
| `--batch-size` | `-b` | int | Batch size |
| `--imgsz` | | int | Image size |
| `--lr` | | float | Learning rate |
| `--output-dir` | `-o` | Path | Output directory |
| `--project-name` | `-p` | str | Experiment name |
| `--wandb/--no-wandb` | | flag | Enable/disable W&B |
| `--tensorboard/--no-tensorboard` | | flag | Enable/disable TensorBoard |
| `--patience` | | int | Early stopping patience |
| `--workers` | `-w` | int | DataLoader workers |
| `--device` | | str | `auto`, `cpu`, `cuda:0` |

**Examples:**

```bash
# CLI only
gdf train --source local --data-path data/cats --epochs 50 --model-size s

# YAML + overrides
gdf train --config configs/default.yaml --epochs 10 --no-wandb

# Roboflow
gdf train --source roboflow --data-path my-workspace/project/1

# HTTP download
gdf train --source http --data-path https://example.com/data.zip
```

## gdf export

Export trained weights to ONNX / TensorRT.

```bash
gdf export [OPTIONS]
```

| Flag | Short | Type | Description |
|------|-------|------|-------------|
| `--config` | `-c` | Path | YAML config file |
| `--weights` | `-w` | Path | Path to `.pt` weights |
| `--format` | `-f` | str | `onnx`, `tensorrt`, `both` |
| `--half/--no-half` | | flag | FP16 precision |
| `--workspace` | | int | TRT workspace (MB) |
| `--imgsz` | | int | Image size |
| `--device` | | str | Device |
| `--output-dir` | `-o` | Path | Output directory |

**Examples:**

```bash
gdf export --weights best.pt --format onnx
gdf export --weights best.pt --format both --half
gdf export --config configs/export.yaml
```

## gdf predict

Run inference on images.

```bash
gdf predict [OPTIONS]
```

| Flag | Short | Type | Description |
|------|-------|------|-------------|
| `--config` | `-c` | Path | YAML config file |
| `--weights` | `-w` | Path | Model weights |
| `--source` | `-s` | str | Image path or directory |
| `--backend` | `-b` | str | `pytorch`, `onnx`, `tensorrt` |
| `--conf-threshold` | | float | Confidence threshold (0-1) |
| `--imgsz` | | int | Image size |
| `--output` | `-o` | Path | Output CSV path |
| `--device` | | str | Device |
| `--class-names` | | Path | File with class names |

**Examples:**

```bash
# Single image
gdf predict --weights model.onnx --source photo.jpg --backend onnx

# Directory with CSV output
gdf predict --weights model.engine --source data/test/ --backend tensorrt --output results.csv

# With class names file
gdf predict --weights model.onnx --source img.jpg --class-names classes.txt
```
