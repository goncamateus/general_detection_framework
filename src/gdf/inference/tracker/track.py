from __future__ import annotations

import numpy as np

from gdf.inference.tracker.kalman import KalmanFilter

_TRACK_ID = 0


def _next_id() -> int:
    global _TRACK_ID
    _TRACK_ID += 1
    return _TRACK_ID


def reset_id_counter() -> None:
    global _TRACK_ID
    _TRACK_ID = 0


class Track:
    """Single object track with Kalman filter state."""

    def __init__(
        self,
        bbox: np.ndarray,
        class_id: int,
        score: float,
    ) -> None:
        self.id = _next_id()
        self.bbox = bbox.copy()
        self.class_id = class_id
        self.score = score
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        self._kf = KalmanFilter()
        measurement = self._xyxy_to_cxayah(bbox)
        self.mean, self.covariance = self._kf.initiate(measurement)

    @staticmethod
    def _xyxy_to_cxayah(bbox: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = bbox[:4]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        a = w / max(h, 1e-6)
        return np.array([cx, cy, a, h])

    @staticmethod
    def _cxayah_to_xyxy(cxayah: np.ndarray) -> np.ndarray:
        cx, cy, a, h = cxayah[:4]
        w = a * h
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return np.array([x1, y1, x2, y2])

    def predict(self) -> None:
        self.mean, self.covariance = self._kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1
        self.bbox = self._cxayah_to_xyxy(self.mean)

    def update(self, bbox: np.ndarray, class_id: int, score: float) -> None:
        measurement = self._xyxy_to_cxayah(bbox)
        self.mean, self.covariance = self._kf.update(self.mean, self.covariance, measurement)
        self.bbox = self._cxayah_to_xyxy(self.mean)
        self.class_id = class_id
        self.score = score
        self.hits += 1
        self.time_since_update = 0

    @property
    def is_confirmed(self) -> bool:
        return self.hits >= 3

    @property
    def is_lost(self) -> bool:
        return self.time_since_update > 30

    def to_tlwh(self) -> np.ndarray:
        x1, y1, x2, y2 = self.bbox
        return np.array([x1, y1, x2 - x1, y2 - y1])

    def to_xyxy(self) -> np.ndarray:
        return self.bbox.copy()
