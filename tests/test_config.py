from pathlib import Path

from gdf.config.schema import ExportConfig, PredictConfig, TrainConfig


def test_train_config_defaults():
    cfg = TrainConfig(source="local", data_path="data/test")
    assert cfg.model_version == "v26"
    assert cfg.model_size == "n"
    assert cfg.epochs == 100
    assert cfg.batch_size == 16
    assert cfg.imgsz == 224
    assert cfg.use_wandb is True
    assert cfg.use_tensorboard is True


def test_train_config_validation():
    cfg = TrainConfig(source="local", data_path="data/test", epochs=50, batch_size=8)
    assert cfg.epochs == 50
    assert cfg.batch_size == 8


def test_export_config_defaults():
    cfg = ExportConfig(weights=Path("model.pt"))
    assert cfg.format == "onnx"
    assert cfg.half is False
    assert cfg.workspace == 4096


def test_predict_config_defaults():
    cfg = PredictConfig(weights=Path("model.onnx"), source="img.jpg")
    assert cfg.backend == "pytorch"
    assert cfg.conf_threshold == 0.5


def test_train_config_invalid_version():
    import pytest
    with pytest.raises(Exception):
        TrainConfig(source="local", data_path="data/test", model_version="v99")
