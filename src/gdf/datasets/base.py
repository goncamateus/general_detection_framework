from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class BaseDatasetSource(ABC):
    @abstractmethod
    def resolve(self) -> Path:
        """Return path to resolved dataset root (contains train/val dirs)."""
        ...

    @abstractmethod
    def validate(self) -> bool:
        """Return True if dataset structure is valid."""
        ...

    def get_class_names(self) -> list[str]:
        resolved = self.resolve()
        train_dir = resolved / "train"
        if not train_dir.exists():
            return []
        return sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
