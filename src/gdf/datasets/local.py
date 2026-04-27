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
        if not train.is_dir() or not val.is_dir():
            log.warning(f"Expected train/ and val/ dirs in {self._path}")
            return False
        train_classes = [d for d in train.iterdir() if d.is_dir()]
        if not train_classes:
            log.warning(f"No class folders found in {train}")
            return False
        log.info(f"Found {len(train_classes)} classes in train/")
        return True
