from __future__ import annotations

import hashlib
import shutil
import tarfile
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from gdf.utils.logging import log


def download_file(url: str, dest: Path, chunk_size: int = 8192) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        log.info(f"File already exists: {dest}")
        return dest

    log.info(f"Downloading {url} → {dest}")
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))

    with (
        open(dest, "wb") as f,
        tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar,
    ):
        for chunk in resp.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            bar.update(len(chunk))

    return dest


def extract_archive(archive_path: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)

    if zipfile.is_zipfile(archive_path):
        log.info(f"Extracting zip: {archive_path}")
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(dest)
    elif tarfile.is_tarfile(archive_path):
        log.info(f"Extracting tar: {archive_path}")
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(dest)
    else:
        raise ValueError(f"Not a recognized archive: {archive_path}")

    return dest


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_rmdir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
        log.info(f"Removed: {path}")
