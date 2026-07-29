# Training

GDF trains YOLO models by delegating to Ultralytics `.train()`. No custom PyTorch loop.

## Basic Usage

```bash
gdf train --source local --data-path data/my_dataset --epochs 50
```

## Tasks

`--task` picks the head: `cls`, `detect`, `segment`, or `auto` (default).

`auto` resolves as: no `data.yaml` → `cls`; otherwise it reads the first label file and
picks `segment` when rows carry more than 5 fields (polygons), `detect` otherwise.

```bash
gdf train --config configs/plume_seg.yaml          # segmentation, task pinned in the YAML
gdf train --data-path data/plume --task segment    # or pin it on the CLI
```

Segmentation defaults to `v11` (`yolo11n-seg.pt`): published `-seg` weights and TensorRT
8.2-safe ops for the original Jetson Nano. Classification and detection still default to `v26`.

## With YAML Config

```bash
gdf train --config configs/default.yaml --epochs 20
```

## Model Selection

Choose YOLO version and size:

```bash
gdf train --model-version v26 --model-size n   # YOLO26 Nano (default)
gdf train --model-version v26 --model-size s   # YOLO26 Small
gdf train --model-version v8 --model-size m    # YOLOv8 Medium
gdf train --model-version v11 --model-size l   # YOLO11 Large
```

See [Model Registry](../src/gdf/models/registry.py) for all version+size combinations.

## What Happens Internally

1. CLI loads YAML → merges CLI flags → validates with Pydantic
2. Dataset source resolved (download if remote)
3. Transforms + DataLoaders built
4. The task wrapper (`YOLOClsWrapper` / `YOLODetectWrapper` / `YOLOSegWrapper`) loads the
   pretrained `.pt` from the Ultralytics hub
5. `Trainer` calls `model.train(data=..., epochs=..., ...)`
6. Ultralytics runs the training loop internally
7. Best weights read back from the Ultralytics `save_dir` (it appends its own run name
   under `project`, so the path is not predictable from `output_dir` alone)

## Logging

### TensorBoard (default: on)

Logs to `output_dir/project_name/tb_logs/`. View with:

```bash
tensorboard --logdir runs/train/gdf-exp/tb_logs
```

### Weights & Biases (default: on)

Requires `WANDB_API_KEY` env var. Disable with `--no-wandb`.

### CSV Logger

Available programmatically via `training/loggers.py`. Always logs to CSV as well.

## Callbacks

- **CheckpointCallback**: saves `best.pt` when accuracy improves
- **EarlyStoppingCallback**: stops training after `patience` epochs without improvement

## Device Selection

```bash
gdf train --device auto    # auto-detect (default)
gdf train --device cpu     # force CPU
gdf train --device cuda:0  # specific GPU
```

## Output Structure

```
runs/train/
  gdf-exp/
    weights/
      best.pt
      last.pt
    tb_logs/
    args.yaml
    results.csv
```
