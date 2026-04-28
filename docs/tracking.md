# Tracking

GDF includes a custom ByteTrack implementation for multi-object tracking. Works with ONNX and TensorRT detection models, plus Ultralytics' built-in tracker for PyTorch.

## How It Works

1. YOLO detection model runs on each frame → bounding boxes + classes + scores
2. ByteTracker associates detections across frames using IoU + Kalman filter
3. Each tracked object gets a persistent ID

### ByteTrack Algorithm

- **Phase 1**: Match high-confidence detections to existing tracks (Hungarian algorithm on IoU matrix)
- **Phase 2**: Match low-confidence detections to remaining unmatched tracks
- **New tracks**: Created from unmatched high-confidence detections
- **Lost tracks**: Removed after `max_time_lost` frames without a match
- **Confirmed**: Track needs 3 consecutive hits before it appears in output

## CLI Usage

### Track a video

```bash
gdf track --weights model.onnx --source video.mp4 --backend onnx --output tracked.mp4
```

### Track a directory of images

```bash
gdf track --weights model.engine --source frames/ --backend tensorrt --output results.csv
```

### PyTorch (Ultralytics built-in tracker)

```bash
gdf track --weights yolov8n.pt --source video.mp4 --backend pytorch
```

## CLI Flags

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--weights` | `-w` | required | Detection model weights |
| `--source` | `-s` | required | Video file or image directory |
| `--backend` | `-b` | `pytorch` | `pytorch`, `onnx`, `tensorrt` |
| `--conf-threshold` | | `0.3` | Detection confidence threshold |
| `--match-threshold` | | `0.7` | IoU threshold for matching |
| `--max-time-lost` | | `30` | Frames before track removal |
| `--imgsz` | | `640` | Input image size |
| `--output` | `-o` | auto | Output video or CSV path |
| `--class-names` | | — | File with class names |

## Python API

### PyTorch (Ultralytics)

```python
from gdf.models.yolo_detect import YOLODetectWrapper

model = YOLODetectWrapper(version="v26", size="n")
model.load()
results = model.track("video.mp4", tracker="bytetrack.yaml")
```

### ONNX + Custom ByteTrack

```python
from gdf.inference.onnx_detect_runner import ONNXDetectRunner

runner = ONNXDetectRunner(model_path=Path("model.onnx"), imgsz=640)
runner.enable_tracking(conf_threshold=0.3, match_threshold=0.7)

# Single frame
tracks = runner.detect_and_track("frame.jpg")
print(tracks.track_ids, tracks.bboxes, tracks.scores)

# Full video
results = runner.detect_video("video.mp4", output_path="out.mp4")
```

### TensorRT + Custom ByteTrack

```python
from gdf.inference.trt_detect_runner import TRTDetectRunner

runner = TRTDetectRunner(engine_path=Path("model.engine"), imgsz=640)
runner.enable_tracking()
results = runner.detect_video("video.mp4", output_path="out.mp4")
```

### UnifiedPredictor

```python
from gdf.inference.engine import UnifiedPredictor

predictor = UnifiedPredictor(
    weights=Path("model.onnx"),
    backend="onnx",
    task="detect",
    imgsz=640,
    conf_threshold=0.3,
)

# Single frame tracking
tracks = predictor.track("frame.jpg")
for i in range(len(tracks)):
    print(f"ID:{tracks.track_ids[i]} class={tracks.class_ids[i]} score={tracks.scores[i]:.2f}")
```

## ByteTrack Internals

### Kalman Filter

State vector: `[cx, cy, aspect_ratio, height, vcx, vcy, va, vh]`

Predicts object motion between frames using constant velocity model.

### Track States

| State | Condition | Behavior |
|-------|-----------|----------|
| Tentative | `hits < 3` | Not in output, still matching |
| Confirmed | `hits >= 3` | In output, actively tracked |
| Lost | `time_since_update > 30` | Removed from tracker |

### Matching

Uses `scipy.optimize.linear_sum_assignment` (Hungarian algorithm) on cost matrix `1 - IoU`.

## Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | Array operations |
| `scipy` | Hungarian matching |
| `opencv-python` | Video I/O, frame processing |

No external tracker library needed — ByteTrack is implemented from scratch.
