# Datasets

GDF supports three dataset sources. All resolve to an ImageFolder layout before training.

## ImageFolder Convention

```
data_root/
  train/
    class_a/
      img1.jpg
      img2.jpg
    class_b/
      img3.jpg
  val/
    class_a/
      img4.jpg
    class_b/
      img5.jpg
```

Class names are inferred from folder names. `train/` is required; `val/` is recommended.

## Detection / Segmentation Convention

Detection and segmentation datasets use the Ultralytics layout with a `data.yaml` at the root:

```
data_root/
  data.yaml
  train/
    images/frame.jpg
    labels/frame.txt
  valid/
    images/
    labels/
```

```yaml
# data.yaml — omit `path:`, Ultralytics then resolves relative to this file's directory.
# A relative `path:` is resolved against the global datasets_dir instead, which is rarely
# what you want.
train: train/images
val: valid/images
test: test/images

nc: 1
names: ["plume"]
```

Both tasks share this layout — **only the label rows differ**:

| Task | Label row | Fields |
|------|-----------|--------|
| `detect` | `cls cx cy w h` | 5 |
| `segment` | `cls x1 y1 x2 y2 x3 y3 ...` | 7+ (odd) |

`gdf train` sniffs the first label file to pick the task. Override with `--task segment`
when the sniff guesses wrong.

**A dataset must not mix the two.** If any file contains only 5-field rows, Ultralytics
warns `Box and segment counts should be equal` and then discards *every* mask in the
dataset, which crashes training. Find the offenders with:

```bash
for f in data/plume/train/labels/*.txt; do
  [ "$(awk '{if(NF>m)m=NF}END{print m+0}' "$f")" -le 6 ] && echo "$f"
done
```

## Local Datasets

Point `--data-path` to a directory with `train/` and `val/` subdirs:

```bash
gdf train --source local --data-path /data/cats_vs_dogs
```

## Roboflow

Download datasets from [Roboflow](https://roboflow.com) using their API.

**Requirements:**
- `ROBOFLOW_API_KEY` environment variable
- `data_path` format: `workspace/project/version`

```bash
export ROBOFLOW_API_KEY=your_key
gdf train --source roboflow --data-path my-workspace/cats-dogs/3
```

The dataset is downloaded to `data/roboflow_cache/` and reused on subsequent runs.

## HTTP Downloads

Download a zip or tar archive from a URL:

```bash
gdf train --source http --data-path https://example.com/dataset.zip
```

The archive is downloaded to `data/http_cache/`, extracted, and the folder structure is auto-detected.

## Transforms

Transforms are applied automatically by `datasets/transforms.py`:

**Training:**
- RandomResizedCrop
- RandomHorizontalFlip
- ColorJitter
- RandomRotation (15°)
- Normalize (ImageNet stats)
- RandomErasing

**Validation/Prediction:**
- Resize (1.14×)
- CenterCrop
- Normalize (ImageNet stats)

## DataLoader Factory

`datasets/loader.py` builds PyTorch DataLoaders:

```python
from gdf.datasets.loader import build_dataloaders

train_loader, val_loader, class_names = build_dataloaders(
    data_root=Path("data/my_dataset"),
    imgsz=224,
    batch_size=16,
    workers=8,
)
```
