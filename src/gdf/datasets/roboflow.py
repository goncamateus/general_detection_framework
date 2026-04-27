from __future__ import annotations

import os
from pathlib import Path

from gdf.datasets.base import BaseDatasetSource
from gdf.utils.logging import log


class RoboflowDatasetSource(BaseDatasetSource):
    def __init__(
        self,
        workspace: str,
        project: str,
        version: int,
        format: str = "folder",
        cache_dir: Path | None = None,
    ) -> None:
        self.workspace = workspace
        self.project = project
        self.version = version
        self.format = format
        self._cache_dir = cache_dir or Path("data/roboflow_cache")
        self._resolved: Path | None = None

    def resolve(self) -> Path:
        if self._resolved is not None and self._resolved.exists():
            return self._resolved

        api_key = os.environ.get("ROBOFLOW_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ROBOFLOW_API_KEY env var not set. "
                "Export it: export ROBOFLOW_API_KEY=your_key"
            )

        from roboflow import Roboflow

        log.info(
            f"Downloading Roboflow dataset: {self.workspace}/{self.project} v{self.version}"
        )
        rf = Roboflow(api_key=api_key)
        project = rf.workspace(self.workspace).project(self.project)
        version = project.version(self.version)
        dataset = version.download(self.format, location=str(self._cache_dir / self.project))

        self._resolved = Path(dataset.location)
        log.info(f"Roboflow dataset resolved to: {self._resolved}")
        return self._resolved

    def validate(self) -> bool:
        path = self.resolve()
        train = path / "train"
        if not train.is_dir():
            log.warning(f"No train/ dir in resolved Roboflow dataset: {path}")
            return False
        return True
