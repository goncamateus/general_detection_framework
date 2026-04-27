from __future__ import annotations

from torchvision import transforms


def get_train_transforms(imgsz: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(imgsz, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2),
        ]
    )


def get_val_transforms(imgsz: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(int(imgsz * 1.14)),
            transforms.CenterCrop(imgsz),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_predict_transforms(imgsz: int = 224) -> transforms.Compose:
    return get_val_transforms(imgsz)
