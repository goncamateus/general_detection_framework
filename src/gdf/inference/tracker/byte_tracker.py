from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from gdf.inference.tracker.track import Track, reset_id_counter


class ByteTrackResult:
    """Result from ByteTracker for a single frame."""

    def __init__(
        self,
        bboxes: np.ndarray,
        class_ids: np.ndarray,
        scores: np.ndarray,
        track_ids: np.ndarray,
    ) -> None:
        self.bboxes = bboxes
        self.class_ids = class_ids
        self.scores = scores
        self.track_ids = track_ids

    def __len__(self) -> int:
        return len(self.track_ids)

    def __repr__(self) -> str:
        return f"ByteTrackResult({len(self)} tracks)"


class ByteTracker:
    """ByteTrack tracker implementation.

    Uses Kalman filter for motion prediction and IoU-based association.
    Two-phase matching: high-confidence first, then low-confidence.

    Args:
        conf_threshold: Minimum confidence to consider a detection.
        match_threshold: Minimum IoU for matching detection to track.
        max_time_lost: Remove track after this many frames without match.
    """

    def __init__(
        self,
        conf_threshold: float = 0.3,
        match_threshold: float = 0.7,
        max_time_lost: int = 30,
    ) -> None:
        self.conf_threshold = conf_threshold
        self.match_threshold = match_threshold
        self.max_time_lost = max_time_lost
        self.tracks: list[Track] = []
        self._frame_count = 0

    def reset(self) -> None:
        self.tracks = []
        self._frame_count = 0
        reset_id_counter()

    def update(
        self,
        bboxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
    ) -> ByteTrackResult:
        """Update tracker with new detections.

        Args:
            bboxes: (N, 4) array in xyxy format.
            scores: (N,) confidence scores.
            class_ids: (N,) class IDs.

        Returns:
            ByteTrackResult with tracked objects.
        """
        self._frame_count += 1

        if len(bboxes) == 0:
            self._predict_tracks()
            self._remove_lost()
            return self._get_output()

        bboxes = np.asarray(bboxes, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)
        class_ids = np.asarray(class_ids, dtype=np.int32)

        for t in self.tracks:
            t.predict()

        # Phase 1: high-confidence detections
        high_mask = scores >= self.conf_threshold
        high_bboxes = bboxes[high_mask]
        high_scores = scores[high_mask]
        high_class_ids = class_ids[high_mask]

        unmatched_tracks_high = list(range(len(self.tracks)))
        unmatched_dets_high = list(range(len(high_bboxes)))

        if len(high_bboxes) > 0 and len(self.tracks) > 0:
            iou_matrix = self._iou_batch(high_bboxes, np.array([t.bbox for t in self.tracks]))
            matched, ut, ud = self._match(iou_matrix, self.match_threshold)
            self._apply_matches(matched, high_bboxes, high_scores, high_class_ids)
            unmatched_tracks_high = ut
            unmatched_dets_high = ud

        # Phase 2: low-confidence detections matched to unmatched tracks
        low_mask = ~high_mask
        low_bboxes = bboxes[low_mask]
        low_scores = scores[low_mask]
        low_class_ids = class_ids[low_mask]

        if len(low_bboxes) > 0 and len(unmatched_tracks_high) > 0:
            remaining_tracks = [self.tracks[i] for i in unmatched_tracks_high]
            iou_matrix = self._iou_batch(low_bboxes, np.array([t.bbox for t in remaining_tracks]))
            matched_low, ut2, ud2 = self._match(iou_matrix, 0.5)

            for det_idx, trk_idx in matched_low:
                orig_trk_idx = unmatched_tracks_high[trk_idx]
                self.tracks[orig_trk_idx].update(
                    low_bboxes[det_idx], int(low_class_ids[det_idx]), float(low_scores[det_idx])
                )
            unmatched_tracks_high = [unmatched_tracks_high[i] for i in ut2]

        # Create new tracks from unmatched high-confidence detections
        for det_idx in unmatched_dets_high:
            self.tracks.append(Track(high_bboxes[det_idx], int(high_class_ids[det_idx]), float(high_scores[det_idx])))

        self._remove_lost()

        return self._get_output()

    def _predict_tracks(self) -> None:
        for t in self.tracks:
            t.predict()

    def _remove_lost(self) -> None:
        self.tracks = [t for t in self.tracks if not t.is_lost]

    def _apply_matches(
        self,
        matched: list[tuple[int, int]],
        bboxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
    ) -> None:
        for det_idx, trk_idx in matched:
            self.tracks[trk_idx].update(bboxes[det_idx], int(class_ids[det_idx]), float(scores[det_idx]))

    def _get_output(self) -> ByteTrackResult:
        confirmed = [t for t in self.tracks if t.is_confirmed and t.time_since_update == 0]
        if not confirmed:
            empty = np.empty((0, 4), dtype=np.float32)
            return ByteTrackResult(empty, np.array([], dtype=np.int32), np.array([], dtype=np.float32), np.array([], dtype=np.int32))

        bboxes = np.array([t.to_xyxy() for t in confirmed], dtype=np.float32)
        class_ids = np.array([t.class_id for t in confirmed], dtype=np.int32)
        scores = np.array([t.score for t in confirmed], dtype=np.float32)
        track_ids = np.array([t.id for t in confirmed], dtype=np.int32)

        return ByteTrackResult(bboxes, class_ids, scores, track_ids)

    @staticmethod
    def _iou_batch(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
        """Compute IoU matrix between two sets of boxes (xyxy format)."""
        n = len(boxes_a)
        m = len(boxes_b)
        iou = np.zeros((n, m), dtype=np.float32)

        for i in range(n):
            x1 = np.maximum(boxes_a[i, 0], boxes_b[:, 0])
            y1 = np.maximum(boxes_a[i, 1], boxes_b[:, 1])
            x2 = np.minimum(boxes_a[i, 2], boxes_b[:, 2])
            y2 = np.minimum(boxes_a[i, 3], boxes_b[:, 3])

            inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
            area_a = (boxes_a[i, 2] - boxes_a[i, 0]) * (boxes_a[i, 3] - boxes_a[i, 1])
            area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
            union = area_a + area_b - inter

            iou[i] = inter / np.maximum(union, 1e-6)

        return iou

    @staticmethod
    def _match(
        iou_matrix: np.ndarray,
        threshold: float,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Hungarian matching on IoU matrix.

        Returns:
            matched: list of (det_idx, trk_idx) pairs
            unmatched_tracks: indices of unmatched tracks
            unmatched_dets: indices of unmatched detections
        """
        if iou_matrix.size == 0:
            return [], list(range(iou_matrix.shape[1])), list(range(iou_matrix.shape[0]))

        cost = 1 - iou_matrix
        row_indices, col_indices = linear_sum_assignment(cost)

        matched = []
        matched_rows = set()
        matched_cols = set()
        for r, c in zip(row_indices, col_indices):
            if iou_matrix[r, c] >= threshold:
                matched.append((r, c))
                matched_rows.add(r)
                matched_cols.add(c)

        unmatched_tracks = [c for c in range(iou_matrix.shape[1]) if c not in matched_cols]
        unmatched_dets = [r for r in range(iou_matrix.shape[0]) if r not in matched_rows]

        return matched, unmatched_tracks, unmatched_dets
