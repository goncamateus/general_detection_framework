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
YOLO_SEG_MODELS: dict[str, dict[str, str]] = {
    "v8": {
        "n": "yolov8n-seg.pt",
        "s": "yolov8s-seg.pt",
        "m": "yolov8m-seg.pt",
        "l": "yolov8l-seg.pt",
        "x": "yolov8x-seg.pt",
    },
    "v11": {
        "n": "yolo11n-seg.pt",
        "s": "yolo11s-seg.pt",
        "m": "yolo11m-seg.pt",
        "l": "yolo11l-seg.pt",
        "x": "yolo11x-seg.pt",
    },
    "v26": {
        "n": "yolo26n-seg.pt",
        "s": "yolo26s-seg.pt",
        "m": "yolo26m-seg.pt",
        "l": "yolo26l-seg.pt",
        "x": "yolo26x-seg.pt",
    },
}

TASK_REGISTRIES: dict[str, dict[str, dict[str, str]]] = {
    "cls": YOLO_MODELS,
    "detect": YOLO_DETECT_MODELS,
    "segment": YOLO_SEG_MODELS,
}


def _registry_for(task: str) -> dict[str, dict[str, str]]:
    # ponytail: unknown task falls back to cls, matching the previous if/else behaviour
    return TASK_REGISTRIES.get(task, YOLO_MODELS)


def get_model_name(version: str, size: str, task: str = "cls") -> str:
    registry = _registry_for(task)
    if version not in registry:
        raise ValueError(f"Unknown model version: {version}. Available: {list(registry)}")
    sizes = registry[version]
    if size not in sizes:
        raise ValueError(f"Unknown model size: {size}. Available: {list(sizes)}")
    return sizes[size]


def list_available_models(task: str = "cls") -> dict[str, list[str]]:
    return {v: list(sizes.keys()) for v, sizes in _registry_for(task).items()}
