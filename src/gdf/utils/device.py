from __future__ import annotations

import os
from dataclasses import dataclass

import torch


@dataclass
class DeviceInfo:
    name: str
    cuda_available: bool
    cuda_version: str
    device_count: int
    trt_available: bool
    trt_version: str


def detect_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def get_device_info() -> DeviceInfo:
    cuda = torch.cuda.is_available()
    cuda_ver = torch.version.cuda or "N/A"
    count = torch.cuda.device_count() if cuda else 0

    trt_available = False
    trt_ver = "N/A"
    try:
        import tensorrt as trt

        trt_available = True
        trt_ver = trt.__version__
    except ImportError:
        pass

    name = "cpu"
    if cuda:
        name = torch.cuda.get_device_name(0)

    return DeviceInfo(
        name=name,
        cuda_available=cuda,
        cuda_version=cuda_ver,
        device_count=count,
        trt_available=trt_available,
        trt_version=trt_ver,
    )


def resolve_device(device_str: str) -> torch.device:
    return detect_device(device_str)
