from __future__ import annotations

YOLO_MODELS: dict[str, dict[str, str]] = {
    "v8": {
        "n": "yolov8n-cls.pt",
        "s": "yolov8s-cls.pt",
        "m": "yolov8m-cls.pt",
        "l": "yolov8l-cls.pt",
        "x": "yolov8x-cls.pt",
    },
    "v11": {
        "n": "yolo11n-cls.pt",
        "s": "yolo11s-cls.pt",
        "m": "yolo11m-cls.pt",
        "l": "yolo11l-cls.pt",
        "x": "yolo11x-cls.pt",
    },
    "v26": {
        "n": "yolo26n-cls.pt",
        "s": "yolo26s-cls.pt",
        "m": "yolo26m-cls.pt",
        "l": "yolo26l-cls.pt",
        "x": "yolo26x-cls.pt",
    },
}

YOLO_DETECT_MODELS: dict[str, dict[str, str]] = {
    "v8": {
        "n": "yolov8n.pt",
        "s": "yolov8s.pt",
        "m": "yolov8m.pt",
        "l": "yolov8l.pt",
        "x": "yolov8x.pt",
    },
    "v11": {
        "n": "yolo11n.pt",
        "s": "yolo11s.pt",
        "m": "yolo11m.pt",
        "l": "yolo11l.pt",
        "x": "yolo11x.pt",
    },
    "v26": {
        "n": "yolo26n.pt",
        "s": "yolo26s.pt",
        "m": "yolo26m.pt",
        "l": "yolo26l.pt",
        "x": "yolo26x.pt",
    },
}


def get_model_name(version: str, size: str, task: str = "cls") -> str:
    registry = YOLO_DETECT_MODELS if task == "detect" else YOLO_MODELS
    if version not in registry:
        raise ValueError(f"Unknown model version: {version}. Available: {list(registry)}")
    sizes = registry[version]
    if size not in sizes:
        raise ValueError(f"Unknown model size: {size}. Available: {list(sizes)}")
    return sizes[size]


def list_available_models(task: str = "cls") -> dict[str, list[str]]:
    registry = YOLO_DETECT_MODELS if task == "detect" else YOLO_MODELS
    return {v: list(sizes.keys()) for v, sizes in registry.items()}
