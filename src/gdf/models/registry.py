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


def get_model_name(version: str, size: str) -> str:
    if version not in YOLO_MODELS:
        raise ValueError(f"Unknown model version: {version}. Available: {list(YOLO_MODELS)}")
    sizes = YOLO_MODELS[version]
    if size not in sizes:
        raise ValueError(f"Unknown model size: {size}. Available: {list(sizes)}")
    return sizes[size]


def list_available_models() -> dict[str, list[str]]:
    return {v: list(sizes.keys()) for v, sizes in YOLO_MODELS.items()}
