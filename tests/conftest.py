from pathlib import Path

import pytest


@pytest.fixture
def tmp_dataset(tmp_path: Path) -> Path:
    classes = ["cat", "dog", "bird"]
    for split in ("train", "val"):
        for cls in classes:
            d = tmp_path / split / cls
            d.mkdir(parents=True)
            (d / "img1.jpg").write_bytes(b"fake_image_data")
            (d / "img2.jpg").write_bytes(b"fake_image_data")
    return tmp_path


@pytest.fixture
def sample_config_dict() -> dict:
    return {
        "model_version": "v26",
        "model_size": "n",
        "source": "local",
        "data_path": "data/test",
        "epochs": 1,
        "batch_size": 2,
        "imgsz": 64,
        "lr": 0.001,
    }
