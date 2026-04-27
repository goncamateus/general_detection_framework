from pathlib import Path

from gdf.export.onnx import verify_onnx


def test_verify_onnx_nonexistent():
    result = verify_onnx(Path("/nonexistent/model.onnx"))
    assert result is False
