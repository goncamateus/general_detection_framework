from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class TrainConfig(BaseModel):
    model_version: Literal["v8", "v11", "v26"] = "v26"
    model_size: Literal["n", "s", "m", "l", "x"] = "n"
    # detect and segment datasets share the same data.yaml — only the labels differ, so
    # "auto" sniffs the label files. Set explicitly to skip the sniff.
    task: Literal["auto", "cls", "detect", "segment"] = "auto"
    source: Literal["local", "roboflow", "http"]
    data_path: str
    epochs: int = Field(default=100, ge=1)
    batch_size: int = Field(default=16, ge=1)
    imgsz: int = Field(default=224, ge=32)
    lr: float = Field(default=0.001, gt=0)
    output_dir: Path = Path("runs/train")
    project_name: str = "gdf-exp"
    use_wandb: bool = True
    use_tensorboard: bool = True
    workers: int = Field(default=8, ge=0)
    patience: int = Field(default=50, ge=0)
    device: str = "auto"


class ExportConfig(BaseModel):
    weights: Path
    format: Literal["onnx", "tensorrt", "both"] = "onnx"
    half: bool = False
    workspace: int = Field(default=4096, ge=1024)
    imgsz: int = Field(default=224, ge=32)
    device: str = "auto"
    output_dir: Path = Path("runs/export")


class PredictConfig(BaseModel):
    weights: Path
    source: str
    backend: Literal["pytorch", "onnx", "tensorrt"] = "pytorch"
    task: Literal["cls", "detect", "segment"] = "cls"
    conf_threshold: float = Field(default=0.5, ge=0, le=1)
    imgsz: int = Field(default=224, ge=32)
    output: Path | None = None
    device: str = "auto"


class EvalConfig(BaseModel):
    weights: Path
    data: str
    split: Literal["train", "val", "test"] = "val"
    task: Literal["auto", "cls", "detect", "segment"] = "auto"
    imgsz: int = Field(default=640, ge=32)
    batch_size: int = Field(default=16, ge=1)
    conf_threshold: float = Field(default=0.001, ge=0, le=1)
    iou_threshold: float = Field(default=0.6, ge=0, le=1)
    output: Path | None = None
    device: str = "auto"


class TrackConfig(BaseModel):
    weights: Path
    source: str
    backend: Literal["pytorch", "onnx", "tensorrt"] = "pytorch"
    conf_threshold: float = Field(default=0.3, ge=0, le=1)
    match_threshold: float = Field(default=0.7, ge=0, le=1)
    max_time_lost: int = Field(default=30, ge=1)
    imgsz: int = Field(default=640, ge=32)
    output: Path | None = None
    device: str = "auto"
