from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class TrainConfig(BaseModel):
    model_version: Literal["v8", "v11", "v26"] = "v26"
    model_size: Literal["n", "s", "m", "l", "x"] = "n"
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
    conf_threshold: float = Field(default=0.5, ge=0, le=1)
    imgsz: int = Field(default=224, ge=32)
    output: Path | None = None
    device: str = "auto"
