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

Train a YOLO classification, detection, or segmentation model.

```bash
gdf train [OPTIONS]
```

| Flag | Short | Type | Description |
|------|-------|------|-------------|
| `--config` | `-c` | Path | YAML config file |
| `--model-version` | `-mv` | str | `v8`, `v11`, `v26` |
| `--model-size` | `-ms` | str | `n`, `s`, `m`, `l`, `x` |
| `--task` | `-t` | str | `auto` (default), `cls`, `detect`, `segment` |
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

# Segmentation (plume dataset)
gdf train --config configs/plume_seg.yaml
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

## gdf eval

Score a trained model on a dataset split. Delegates to Ultralytics `.val()`, so the metrics
are whatever the task reports: mAP/precision/recall for detect and segment (box **and** mask
columns), top-1/top-5 for cls.

```bash
gdf eval --model best.pt --data data/plume
```

| Flag | Short | Type | Description |
|------|-------|------|-------------|
| `--weights` / `--model` | `-w` / `-m` | Path | `.pt`, `.onnx`, or `.engine` |
| `--data` | `-d` | str | `data.yaml`, or a dataset root containing one |
| `--split` | | str | `train`, `val` (default), `test` |
| `--task` | `-t` | str | `auto` (default), `cls`, `detect`, `segment` |
| `--imgsz` | | int | Image size (default 640) |
| `--batch-size` | `-b` | int | Batch size |
| `--conf-threshold` | | float | Confidence threshold (default 0.001, mAP convention) |
| `--iou-threshold` | | float | NMS IoU (default 0.6) |
| `--output` | `-o` | Path | Metrics CSV name/path (default `metrics.csv`) |
| `--plots/--no-plots` | | flag | Write prediction grids + PR curves (default on) |
| `--device` | | str | `auto`, `cpu`, `cuda:0` |

Artifacts land in the Ultralytics run directory (`runs/<task>/val*/`), not the CWD — a
relative `--output` is resolved against it, an absolute one wins. Alongside `metrics.csv`
you get `val_batch*_labels.jpg` (ground truth) and `val_batch*_pred.jpg` (model), which are
the quickest way to eyeball segmentation quality, plus PR/F1 curves and a confusion matrix.

**Examples:**

```bash
# Held-out test split, metrics to CSV
gdf eval -m runs/train/plume-seg/train/weights/best.pt -d data/plume --split test -o metrics.csv

# Exported graph — .onnx/.engine carry no task, so name it
gdf eval -m runs/export/plume-640/best.onnx -t segment -d data/plume --imgsz 640 -b 1

# Temporally honest holdout (see scripts/temporal_split.py)
python scripts/temporal_split.py data/plume
gdf eval -m best.pt -d data/plume_temporal
```

`--task` is required for `.onnx` and `.engine` weights: only a `.pt` checkpoint records
which head it was trained with.

The reported FPS is this machine's, and it excludes the training-only loss stage. It is a
sanity check, not a Jetson number — benchmark the engine on the device with `trtexec`.

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
| `--task` | `-t` | str | `cls` (default), `detect`, `segment` |
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

# Segmentation — reports per-instance mask area
gdf predict --weights best.onnx --source data/plume/test/images \
            --task segment --backend onnx --imgsz 640 --conf-threshold 0.25
```

`--task segment` has no `tensorrt` backend yet: export ONNX and benchmark/run the engine
with `trtexec`, or stay on `--backend onnx`.

## gdf run

Run a model live on a webcam or a video file, drawing masks (segment) or boxes (detect)
with a rolling FPS readout. Takes exported weights — `.onnx` or `.engine`, not `.pt`.

```bash
gdf run --model best.onnx --task segment --webcam 0
gdf run --model best.onnx --task segment --video flight.mp4 --output annotated.mp4
```

| Flag | Short | Type | Description |
|------|-------|------|-------------|
| `--weights` / `--model` | `-w` / `-m` | Path | `.onnx` or `.engine` |
| `--webcam` | | int | Camera index — mutually exclusive with `--video` |
| `--video` | | Path | Video file — mutually exclusive with `--webcam` |
| `--task` | `-t` | str | `segment` (default) or `detect` |
| `--backend` | `-b` | str | `onnx` (default) or `tensorrt` |
| `--imgsz` | | int | Model input size (default 640) |
| `--conf-threshold` | | float | Confidence threshold (default 0.25) |
| `--output` | `-o` | Path | Save the annotated video |
| `--save-frames` | | Path | Directory for sample annotated frames |
| `--save-every` | | int | Save one frame every N (default 30) |
| `--class-names` | | Path | Class names, one per line |
| `--no-show` | | flag | Headless — no window (for SSH / Jetson) |
| `--max-frames` | | int | Stop after N frames |

Press `q` or `ESC` to quit. Frames are only sampled to `--save-frames` when the model
actually found something, so you do not end up with a directory of empty background.

With no X/Wayland session (SSH, a headless Jetson) the run drops to headless automatically
and says so — pass `--output` or `--save-frames` or nothing is kept. Checking `DISPLAY`
up front matters: OpenCV's Qt backend aborts the whole process rather than raising, so
there would be nothing left to catch.

`--imgsz` cannot resize an exported graph — ONNX and TensorRT bake the input shape in. If
it disagrees with the model, the run logs the mismatch and uses the model's size. Export
at the size you want to run at.

`--backend tensorrt` works for `detect` only; there is no TensorRT segmentation runner yet.

**Grabbing example images from a video:**

```bash
gdf run -m best.onnx -t segment --video flight.mp4 --no-show \
        --save-frames examples/ --save-every 15 --class-names classes.txt
```

## gdf track

Multi-object tracking on video or image sequences. Uses ByteTrack with detection models.

```bash
gdf track [OPTIONS]
```

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--config` | `-c` | Path | — | YAML config file |
| `--weights` | `-w` | Path | required | Detection model weights |
| `--source` | `-s` | str | required | Video file or image directory |
| `--backend` | `-b` | str | `pytorch` | `pytorch`, `onnx`, `tensorrt` |
| `--conf-threshold` | | float | `0.3` | Detection confidence |
| `--match-threshold` | | float | `0.7` | IoU match threshold |
| `--max-time-lost` | | int | `30` | Frames before track removal |
| `--imgsz` | | int | `640` | Input image size |
| `--output` | `-o` | Path | auto | Output video or CSV path |
| `--device` | | str | `auto` | Device |
| `--class-names` | | Path | — | File with class names |

**Examples:**

```bash
# Track objects in video (ONNX)
gdf track --weights model.onnx --source video.mp4 --backend onnx --output tracked.mp4

# Track objects in video (TensorRT)
gdf track --weights model.engine --source video.mp4 --backend tensorrt --half

# Track frames in directory
gdf track --weights model.onnx --source frames/ --backend onnx --output results.csv

# PyTorch (uses Ultralytics built-in tracker)
gdf track --weights yolov8n.pt --source video.mp4 --backend pytorch
```
