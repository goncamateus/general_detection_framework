from __future__ import annotations

from pathlib import Path

from gdf.datasets.base import BaseDatasetSource
from gdf.utils.logging import log


class LocalDatasetSource(BaseDatasetSource):
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def resolve(self) -> Path:
        if not self._path.exists():
            raise FileNotFoundError(f"Dataset path not found: {self._path}")
        log.info(f"Using local dataset: {self._path}")
        return self._path

    def validate(self) -> bool:
        if not self._path.exists():
            return False
        train = self._path / "train"
        val = self._path / "val"
        if not val.is_dir():
            val = self._path / "valid"
        if not train.is_dir() or not val.is_dir():
            log.warning(f"Expected train/ and val/ (or valid/) dirs in {self._path}")
            return False
        # Check for class folders (cls) or images/ subdir (detect)
        train_classes = [d for d in train.iterdir() if d.is_dir()]
        has_images_subdir = (train / "images").is_dir()
        if not train_classes and not has_images_subdir:
            log.warning(f"No class folders or images/ found in {train}")
            return False
        log.info(f"Dataset validated: {self._path}")
        return True
