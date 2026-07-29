from pathlib import Path

from gdf.models.registry import (
    YOLO_MODELS,
    YOLO_SEG_MODELS,
    get_model_name,
    list_available_models,
)


def test_registry_has_all_versions():
    assert "v8" in YOLO_MODELS
    assert "v11" in YOLO_MODELS
    assert "v26" in YOLO_MODELS


def test_registry_all_sizes():
    for version, sizes in YOLO_MODELS.items():
        for size in ("n", "s", "m", "l", "x"):
            assert size in sizes, f"Missing size {size} in version {version}"


def test_get_model_name():
    assert get_model_name("v8", "n") == "yolov8n-cls.pt"
    assert get_model_name("v11", "s") == "yolo11s-cls.pt"
    assert get_model_name("v26", "n") == "yolo26n-cls.pt"
    assert get_model_name("v26", "x") == "yolo26x-cls.pt"


def test_get_model_name_invalid():
    import pytest
    with pytest.raises(ValueError, match="Unknown model version"):
        get_model_name("v99", "n")
    with pytest.raises(ValueError, match="Unknown model size"):
        get_model_name("v26", "z")


def test_list_available_models():
    models = list_available_models()
    assert isinstance(models, dict)
    assert len(models) == 3


def test_seg_registry_all_sizes():
    for version, sizes in YOLO_SEG_MODELS.items():
        for size in ("n", "s", "m", "l", "x"):
            assert size in sizes, f"Missing size {size} in seg version {version}"


def test_get_seg_model_name():
    # Ultralytics naming: yolo11n-seg.pt, not yolov11n-seg.pt
    assert get_model_name("v11", "n", task="segment") == "yolo11n-seg.pt"
    assert get_model_name("v8", "s", task="segment") == "yolov8s-seg.pt"
    assert get_model_name("v26", "x", task="segment") == "yolo26x-seg.pt"


def test_list_available_seg_models():
    assert list_available_models("segment") == {
        "v8": ["n", "s", "m", "l", "x"],
        "v11": ["n", "s", "m", "l", "x"],
        "v26": ["n", "s", "m", "l", "x"],
    }
