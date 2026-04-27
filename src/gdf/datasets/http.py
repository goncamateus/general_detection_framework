from __future__ import annotations

from pathlib import Path

from gdf.datasets.base import BaseDatasetSource
from gdf.utils.io import download_file, extract_archive
from gdf.utils.logging import log


class HttpDatasetSource(BaseDatasetSource):
    def __init__(self, url: str, cache_dir: Path | None = None) -> None:
        self._url = url
        self._cache_dir = cache_dir or Path("data/http_cache")
        self._resolved: Path | None = None

    def resolve(self) -> Path:
        if self._resolved is not None and self._resolved.exists():
            return self._resolved

        filename = self._url.split("/")[-1].split("?")[0]
        archive_path = self._cache_dir / filename

        download_file(self._url, archive_path)

        extract_dir = self._cache_dir / "extracted"
        extract_archive(archive_path, extract_dir)

        subdirs = [d for d in extract_dir.iterdir() if d.is_dir()]
        if len(subdirs) == 1 and (subdirs[0] / "train").is_dir():
            self._resolved = subdirs[0]
        elif (extract_dir / "train").is_dir():
            self._resolved = extract_dir
        else:
            self._resolved = extract_dir

        log.info(f"HTTP dataset resolved to: {self._resolved}")
        return self._resolved

    def validate(self) -> bool:
        path = self.resolve()
        train = path / "train"
        val = path / "val"
        if not train.is_dir():
            log.warning(f"No train/ dir found in {path}")
            return False
        if not val.is_dir():
            log.warning(f"No val/ dir found in {path}")
            return False
        return True
