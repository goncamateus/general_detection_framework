"""Post-processing tests for ONNXSegRunner — no ONNX session, synthetic tensors only.

The letterbox round-trip (proto -> imgsz -> crop padding -> original frame) is where this
code breaks, so that is what these assert.
"""

import numpy as np
import pytest

from gdf.inference.onnx_seg_runner import ONNXSegRunner

IMGSZ = 64
ORIG_W, ORIG_H = 200, 100
NM = 32  # ultralytics always emits 32 mask prototypes
NUM_BOXES = 100

# _preprocess for a 200x100 frame at imgsz=64:
#   scale = min(64/100, 64/200) = 0.32 -> new_w=64, new_h=32 -> pad_h=16, pad_w=0
PAD = (16, 0, 64, 32, ORIG_W, ORIG_H)

# Target instance in original-frame coordinates.
BOX_ORIG = (50.0, 25.0, 150.0, 75.0)


def _runner() -> ONNXSegRunner:
    # Bypass __init__: it would open an onnxruntime session.
    runner = object.__new__(ONNXSegRunner)
    runner.imgsz = IMGSZ
    runner.tracker = None
    return runner


def _make_inputs(score: float = 0.9):
    """preds [1, 4+nc+nm, NUM_BOXES] and protos [1, nm, imgsz/4, imgsz/4]."""
    pad_h, pad_w, new_w, new_h, _, _ = PAD
    x1, y1, x2, y2 = BOX_ORIG

    # original -> letterbox space
    lx1, lx2 = x1 * new_w / ORIG_W + pad_w, x2 * new_w / ORIG_W + pad_w
    ly1, ly2 = y1 * new_h / ORIG_H + pad_h, y2 * new_h / ORIG_H + pad_h

    preds = np.zeros((1, 4 + 1 + NM, NUM_BOXES), dtype=np.float32)
    for slot, conf in ((0, score), (1, score * 0.5)):  # slot 1 duplicates slot 0 -> NMS food
        preds[0, 0, slot] = (lx1 + lx2) / 2
        preds[0, 1, slot] = (ly1 + ly2) / 2
        preds[0, 2, slot] = lx2 - lx1
        preds[0, 3, slot] = ly2 - ly1
        preds[0, 4, slot] = conf
        preds[0, 5, slot] = 1.0  # mask coefficient 0 -> selects prototype 0

    # Prototype 0 is hot exactly over the instance; proto space is letterbox / 4.
    mh = mw = IMGSZ // 4
    protos = np.zeros((1, NM, mh, mw), dtype=np.float32)
    protos[0, 0] = -10.0
    protos[0, 0, int(ly1) // 4 : int(ly2) // 4, int(lx1) // 4 : int(lx2) // 4] = 10.0
    return preds, protos


def test_mask_shape_matches_original_frame():
    masks, boxes, class_ids, scores = _runner()._postprocess_seg(*_make_inputs(), PAD, 0.25)

    assert masks.shape == (1, ORIG_H, ORIG_W)
    assert masks.dtype == np.bool_
    assert boxes.shape == (1, 4)
    assert class_ids.tolist() == [0]
    assert scores[0] == pytest.approx(0.9, abs=1e-5)


def test_nms_drops_the_duplicate_box():
    masks, _, _, _ = _runner()._postprocess_seg(*_make_inputs(), PAD, 0.25)
    assert len(masks) == 1


def test_box_survives_the_letterbox_round_trip():
    _, boxes, _, _ = _runner()._postprocess_seg(*_make_inputs(), PAD, 0.25)
    assert boxes[0] == pytest.approx(np.array(BOX_ORIG), abs=2.0)


def test_mask_is_inside_the_box_and_not_outside():
    masks, _, _, _ = _runner()._postprocess_seg(*_make_inputs(), PAD, 0.25)
    mask = masks[0]

    cx, cy = (ORIG_W // 2, ORIG_H // 2)
    assert mask[cy, cx], "instance centre should be masked"

    # Corners are far outside the box; padding leaking through would light these up.
    for y, x in ((5, 5), (5, ORIG_W - 5), (ORIG_H - 5, 5), (ORIG_H - 5, ORIG_W - 5)):
        assert not mask[y, x], f"pixel ({y},{x}) outside the box must stay unmasked"

    # Mask area should land near the box area, not the whole frame.
    frac = mask.sum() / mask.size
    assert 0.15 < frac < 0.45, f"mask covers {frac:.2%} of the frame"


def test_no_detections_returns_empty_with_frame_shape():
    masks, boxes, class_ids, scores = _runner()._postprocess_seg(
        *_make_inputs(score=0.1), PAD, 0.25
    )
    assert masks.shape == (0, ORIG_H, ORIG_W)
    assert boxes.shape == (0, 4)
    assert len(class_ids) == 0
    assert len(scores) == 0
