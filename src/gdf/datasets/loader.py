from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from gdf.datasets.transforms import get_train_transforms, get_val_transforms
from gdf.utils.logging import log


def build_dataloaders(
    data_root: Path,
    imgsz: int = 224,
    batch_size: int = 16,
    workers: int = 8,
) -> tuple[DataLoader, DataLoader, list[str]]:
    train_dir = data_root / "train"
    val_dir = data_root / "val"

    if not train_dir.is_dir():
        raise FileNotFoundError(f"Train dir not found: {train_dir}")

    train_ds = ImageFolder(str(train_dir), transform=get_train_transforms(imgsz))
    class_names = train_ds.classes

    val_ds = ImageFolder(str(val_dir), transform=get_val_transforms(imgsz))

    log.info(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}, Classes: {len(class_names)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )

    return train_loader, val_loader, class_names
