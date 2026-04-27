from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from gdf.config.schema import TrackConfig

console = Console()


def track_cmd(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="YAML config file"),
    weights: Optional[Path] = typer.Option(None, "--weights", "-w", help="Detection model weights"),
    source: Optional[str] = typer.Option(None, "--source", "-s", help="Video file or image directory"),
    backend: Optional[str] = typer.Option(None, "--backend", "-b", help="Backend: pytorch, onnx, tensorrt"),
    conf_threshold: Optional[float] = typer.Option(None, "--conf-threshold", help="Detection confidence threshold"),
    match_threshold: Optional[float] = typer.Option(None, "--match-threshold", help="IoU match threshold"),
    max_time_lost: Optional[int] = typer.Option(None, "--max-time-lost", help="Max frames before track removal"),
    imgsz: Optional[int] = typer.Option(None, "--imgsz", help="Image size"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output video path"),
    device: Optional[str] = typer.Option(None, "--device", help="Device"),
    class_names_file: Optional[Path] = typer.Option(None, "--class-names", help="File with class names"),
) -> None:
    cfg = _load_and_merge(config, {
        "weights": str(weights) if weights else None,
        "source": source,
        "backend": backend,
        "conf_threshold": conf_threshold,
        "match_threshold": match_threshold,
        "max_time_lost": max_time_lost,
        "imgsz": imgsz,
        "output": str(output) if output else None,
        "device": device,
    })

    class_names: list[str] = []
    if class_names_file and class_names_file.exists():
        class_names = class_names_file.read_text().strip().splitlines()

    source_path = Path(cfg.source)
    is_video = source_path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    from gdf.inference.engine import UnifiedPredictor

    predictor = UnifiedPredictor(
        weights=cfg.weights,
        backend=cfg.backend,
        class_names=class_names,
        imgsz=cfg.imgsz,
        conf_threshold=cfg.conf_threshold,
        task="detect",
    )

    if is_video:
        _track_video(predictor, cfg)
    else:
        _track_dir(predictor, cfg)


def _track_video(predictor, cfg: TrackConfig) -> None:
    import cv2
    import numpy as np

    source_path = Path(cfg.source)
    output_path = cfg.output or source_path.with_name(source_path.stem + "_tracked.mp4")

    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        console.print(f"[red]Cannot open video: {source_path}[/red]")
        raise typer.Exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    from gdf.inference.onnx_detect_runner import ONNXDetectRunner
    from gdf.inference.trt_detect_runner import TRTDetectRunner

    if predictor.backend == "pytorch":
        console.print("[yellow]PyTorch tracking: using Ultralytics built-in tracker[/yellow]")
        _track_video_pytorch(predictor, cfg, cap, writer, total_frames)
    else:
        runner = predictor._runner
        if hasattr(runner, "enable_tracking"):
            runner.enable_tracking(
                conf_threshold=cfg.conf_threshold,
                match_threshold=cfg.match_threshold,
                max_time_lost=cfg.max_time_lost,
            )
        _track_video_custom(predictor, cfg, cap, writer, total_frames)

    cap.release()
    writer.release()

    console.print(f"[green]Tracking complete. Output: {output_path}[/green]")


def _track_video_pytorch(predictor, cfg, cap, writer, total_frames):
    import cv2

    from gdf.models.yolo_detect import YOLODetectWrapper
    wrapper: YOLODetectWrapper = predictor._runner

    frame_idx = 0
    colors = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        import tempfile
        import os
        tmp = os.path.join(tempfile.gettempdir(), f"_gdf_frame_{frame_idx}.jpg")
        cv2.imwrite(tmp, frame)
        results = wrapper.track(tmp, conf=cfg.conf_threshold)
        os.remove(tmp)

        r = results[0]
        if r.boxes.id is not None:
            bboxes = r.boxes.xyxy.cpu().numpy().astype(int)
            track_ids = r.boxes.id.cpu().numpy().astype(int)
            class_ids = r.boxes.cls.cpu().numpy().astype(int)
            scores = r.boxes.conf.cpu().numpy()

            for i in range(len(track_ids)):
                tid = int(track_ids[i])
                if tid not in colors:
                    colors[tid] = (int(np.random.randint(0, 255)), int(np.random.randint(0, 255)), int(np.random.randint(0, 255)))
                color = colors[tid]
                x1, y1, x2, y2 = bboxes[i]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"ID:{tid} {class_ids[i]} {scores[i]:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        writer.write(frame)
        frame_idx += 1
        if frame_idx % 100 == 0:
            console.print(f"Frame {frame_idx}/{total_frames}")


def _track_video_custom(predictor, cfg, cap, writer, total_frames):
    import cv2
    import numpy as np

    runner = predictor._runner
    frame_idx = 0
    colors = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        import tempfile
        import os
        tmp = os.path.join(tempfile.gettempdir(), f"_gdf_frame_{frame_idx}.jpg")
        cv2.imwrite(tmp, frame)
        tracks = runner.detect_and_track(tmp, conf_threshold=cfg.conf_threshold)
        os.remove(tmp)

        for i in range(len(tracks)):
            tid = int(tracks.track_ids[i])
            if tid not in colors:
                colors[tid] = (int(np.random.randint(0, 255)), int(np.random.randint(0, 255)), int(np.random.randint(0, 255)))
            color = colors[tid]
            x1, y1, x2, y2 = tracks.bboxes[i].astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{tid} {int(tracks.class_ids[i])} {tracks.scores[i]:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        writer.write(frame)
        frame_idx += 1
        if frame_idx % 100 == 0:
            console.print(f"Frame {frame_idx}/{total_frames}")


def _track_dir(predictor, cfg: TrackConfig) -> None:
    from gdf.inference.engine import UnifiedPredictor
    from rich.table import Table

    dir_path = Path(cfg.source)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    images = sorted([f for f in dir_path.iterdir() if f.suffix.lower() in image_extensions])

    if not images:
        console.print(f"[red]No images found in {dir_path}[/red]")
        raise typer.Exit(1)

    table = Table(title="Tracking Results", show_lines=True)
    table.add_column("Frame", style="cyan")
    table.add_column("Track ID", style="green")
    table.add_column("Class", style="yellow")
    table.add_column("Score", style="white")
    table.add_column("BBox", style="dim")

    all_results = []
    for img_path in images:
        tracks = predictor.track(img_path)
        for i in range(len(tracks)):
            table.add_row(
                img_path.name,
                str(tracks.track_ids[i]),
                str(tracks.class_ids[i]),
                f"{tracks.scores[i]:.3f}",
                str(tracks.bboxes[i].astype(int).tolist()),
            )
        all_results.append((img_path, tracks))

    console.print(table)

    if cfg.output:
        import csv
        cfg.output.parent.mkdir(parents=True, exist_ok=True)
        with open(cfg.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["frame", "track_id", "class_id", "score", "bbox"])
            writer.writeheader()
            for img_path, tracks in all_results:
                for i in range(len(tracks)):
                    writer.writerow({
                        "frame": img_path.name,
                        "track_id": int(tracks.track_ids[i]),
                        "class_id": int(tracks.class_ids[i]),
                        "score": float(tracks.scores[i]),
                        "bbox": tracks.bboxes[i].tolist(),
                    })
        console.print(f"[green]Results saved: {cfg.output}[/green]")


def _load_and_merge(config_path: Path | None, cli_overrides: dict) -> TrackConfig:
    import yaml

    base: dict = {}
    if config_path and config_path.exists():
        with open(config_path) as f:
            base = yaml.safe_load(f) or {}

    for k, v in cli_overrides.items():
        if v is not None:
            base[k] = v

    if "weights" not in base:
        raise typer.BadParameter("weights is required")
    if "source" not in base:
        raise typer.BadParameter("source is required")

    base["weights"] = Path(base["weights"])
    if "output" in base and base["output"] is not None:
        base["output"] = Path(base["output"])

    return TrackConfig(**base)
