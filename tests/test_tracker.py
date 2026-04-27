import numpy as np

from gdf.inference.tracker import ByteTracker, ByteTrackResult


def test_tracker_init():
    tracker = ByteTracker()
    assert tracker.conf_threshold == 0.3
    assert tracker.match_threshold == 0.7
    assert tracker.max_time_lost == 30
    assert len(tracker.tracks) == 0


def test_tracker_empty_frame():
    tracker = ByteTracker()
    result = tracker.update(
        np.empty((0, 4), dtype=np.float32),
        np.array([], dtype=np.float32),
        np.array([], dtype=np.int32),
    )
    assert len(result) == 0


def test_tracker_single_detection():
    tracker = ByteTracker()
    bboxes = np.array([[100, 100, 200, 200]], dtype=np.float32)
    scores = np.array([0.9], dtype=np.float32)
    class_ids = np.array([0], dtype=np.int32)

    # First frame: track created but not confirmed yet
    result = tracker.update(bboxes, scores, class_ids)
    assert len(result) == 0  # needs 3 hits to confirm

    # Second frame
    result = tracker.update(bboxes, scores, class_ids)
    assert len(result) == 0

    # Third frame: track should be confirmed
    result = tracker.update(bboxes, scores, class_ids)
    assert len(result) == 1
    assert result.track_ids[0] > 0
    assert result.class_ids[0] == 0
    assert result.scores[0] > 0


def test_tracker_multiple_objects():
    tracker = ByteTracker()
    bboxes = np.array([
        [100, 100, 200, 200],
        [300, 300, 400, 400],
    ], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    class_ids = np.array([0, 1], dtype=np.int32)

    for _ in range(3):
        result = tracker.update(bboxes, scores, class_ids)

    assert len(result) == 2
    assert len(set(result.track_ids)) == 2


def test_tracker_id_consistency():
    tracker = ByteTracker()
    bboxes = np.array([[100, 100, 200, 200]], dtype=np.float32)
    scores = np.array([0.9], dtype=np.float32)
    class_ids = np.array([0], dtype=np.int32)

    for _ in range(3):
        result = tracker.update(bboxes, scores, class_ids)

    first_id = result.track_ids[0]

    # Same position should maintain ID
    result2 = tracker.update(bboxes, scores, class_ids)
    assert result2.track_ids[0] == first_id


def test_tracker_reset():
    tracker = ByteTracker()
    bboxes = np.array([[100, 100, 200, 200]], dtype=np.float32)
    scores = np.array([0.9], dtype=np.float32)
    class_ids = np.array([0], dtype=np.int32)

    for _ in range(3):
        tracker.update(bboxes, scores, class_ids)

    tracker.reset()
    assert len(tracker.tracks) == 0


def test_track_result():
    result = ByteTrackResult(
        bboxes=np.array([[100, 100, 200, 200]], dtype=np.float32),
        class_ids=np.array([0], dtype=np.int32),
        scores=np.array([0.9], dtype=np.float32),
        track_ids=np.array([1], dtype=np.int32),
    )
    assert len(result) == 1
    assert "ByteTrackResult" in repr(result)
