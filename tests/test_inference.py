from pathlib import Path


def test_engine_auto_detect():
    from gdf.inference.engine import UnifiedPredictor

    assert UnifiedPredictor.auto_detect_backend(Path("model.onnx")) == "onnx"
    assert UnifiedPredictor.auto_detect_backend(Path("model.engine")) == "tensorrt"
    assert UnifiedPredictor.auto_detect_backend(Path("model.pt")) == "pytorch"


def test_engine_auto_detect_unknown():
    import pytest

    from gdf.inference.engine import UnifiedPredictor

    with pytest.raises(ValueError, match="Cannot auto-detect"):
        UnifiedPredictor.auto_detect_backend(Path("model.xyz"))
