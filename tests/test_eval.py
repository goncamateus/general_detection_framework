from pathlib import Path

import pytest
import typer

from gdf.cli.eval import _pretty, _resolve_data, _speed_metrics
from gdf.config.schema import EvalConfig


def test_eval_config_defaults():
    cfg = EvalConfig(weights=Path("best.pt"), data="data/plume")
    assert cfg.split == "val"
    assert cfg.task == "auto"
    assert cfg.imgsz == 640


def test_eval_config_rejects_unknown_split():
    with pytest.raises(Exception):
        EvalConfig(weights=Path("best.pt"), data="data/plume", split="holdout")


def test_resolve_data_finds_yaml_in_dataset_root(tmp_path: Path):
    (tmp_path / "data.yaml").write_text("nc: 1\n")
    assert _resolve_data(str(tmp_path)) == str(tmp_path / "data.yaml")


def test_resolve_data_passes_through_a_yaml_path(tmp_path: Path):
    yaml_path = tmp_path / "custom.yaml"
    yaml_path.write_text("nc: 1\n")
    assert _resolve_data(str(yaml_path)) == str(yaml_path)


def test_resolve_data_passes_through_cls_imagefolder_root(tmp_path: Path):
    # No data.yaml: a classification root is handed to Ultralytics as-is.
    (tmp_path / "train").mkdir()
    assert _resolve_data(str(tmp_path)) == str(tmp_path)


def test_resolve_data_rejects_missing_path():
    with pytest.raises(typer.BadParameter):
        _resolve_data("/nonexistent/dataset")


def test_speed_metrics_excludes_loss_from_the_rate():
    # 2 ms of real work per image -> 500 FPS; the loss stage is training-only overhead.
    speed = {"preprocess": 0.5, "inference": 1.0, "loss": 99.0, "postprocess": 0.5}
    out = _speed_metrics(speed)
    assert out["speed/total_ms"] == 2.0
    assert out["fps"] == 500.0
    assert out["speed/loss_ms"] == 99.0


def test_speed_metrics_survives_zero_total():
    assert _speed_metrics({"inference": 0.0})["fps"] == 0.0


def test_pretty_labels():
    assert _pretty("metrics/mAP50-95(M)") == "mAP50-95 (mask)"
    assert _pretty("metrics/precision(B)") == "precision (box)"
    assert _pretty("metrics/accuracy_top1") == "accuracy_top1"
    assert _pretty("speed/inference_ms") == "speed: inference_ms"
