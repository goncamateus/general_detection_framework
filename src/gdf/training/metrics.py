from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str] | None = None,
) -> dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    metrics: dict[str, float] = {
        "accuracy": float(acc),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
    }

    return metrics


def get_classification_report(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str] | None = None,
) -> str:
    return classification_report(y_true, y_pred, target_names=class_names, zero_division=0)


def get_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
) -> np.ndarray:
    return confusion_matrix(y_true, y_pred)
