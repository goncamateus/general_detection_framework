"""Stream loop and overlay drawing — a stub runner stands in for ONNX."""

from pathlib import Path

import cv2
import numpy as np
import pytest
import typer

from gdf.cli.run import _build_runner, run_cmd
from gdf.inference.stream import draw_boxes, draw_masks, run_stream

W, H = 64, 48


class StubSegRunner:
    """Returns one mask covering the top-left quadrant of every frame."""

    def segment(self, frame, conf_threshold=0.25):
        mask = np.zeros(frame.shape[:2], dtype=bool)
        mask[: H // 2, : W // 2] = True
        return (
            mask[None],
            np.array([[0.0, 0.0, W / 2, H / 2]], dtype=np.float32),
            np.array([0], dtype=np.int32),
            np.array([0.9], dtype=np.float32),
        )


class StubEmptyRunner:
    def detect(self, frame, conf_threshold=0.25):
        return (
            np.empty((0, 4), dtype=np.float32),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.float32),
        )


@pytest.fixture
def video(tmp_path: Path) -> Path:
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (W, H))
    for _ in range(10):
        writer.write(np.zeros((H, W, 3), dtype=np.uint8))
    writer.release()
    return path


def test_draw_masks_tints_inside_only():
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    mask = np.zeros((H, W), dtype=bool)
    mask[10:20, 10:20] = True
    out = draw_masks(
        frame,
        mask[None],
        np.array([[10.0, 10.0, 20.0, 20.0]]),
        np.array([0]),
        np.array([0.9]),
        ["plume"],
    )
    assert out[15, 15].any(), "masked pixel should be tinted"
    assert not out[40, 40].any(), "pixel far outside the mask must stay untouched"
    assert frame[15, 15].sum() == 0, "the input frame must not be mutated"


def test_draw_boxes_leaves_the_interior_alone():
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    boxes = np.array([[10.0, 10.0, 30.0, 30.0]])
    out = draw_boxes(frame, boxes, np.array([0]), np.array([0.5]), [])
    assert out[10, 20].any(), "box edge should be drawn"
    # Row 25 clears both the label bar (rows 0-20) and the bottom edge (row 30).
    assert not out[25, 20].any(), "box interior is not filled"


def test_run_stream_processes_every_frame_and_writes_video(video: Path, tmp_path: Path):
    out_video = tmp_path / "out" / "annotated.mp4"
    frames_dir = tmp_path / "frames"

    n = run_stream(
        runner=StubSegRunner(),
        source=str(video),
        task="segment",
        class_names=["plume"],
        output_path=out_video,
        save_frames_dir=frames_dir,
        save_every=5,
        show=False,
    )

    assert n == 10
    assert out_video.exists(), "writer should create the parent directory"
    assert sorted(p.name for p in frames_dir.glob("*.jpg")) == [
        "frame_000000.jpg",
        "frame_000005.jpg",
    ]


def test_run_stream_max_frames_stops_early(video: Path):
    assert run_stream(StubSegRunner(), str(video), "segment", show=False, max_frames=3) == 3


def test_run_stream_skips_saving_frames_with_no_detections(video: Path, tmp_path: Path):
    frames_dir = tmp_path / "frames"
    run_stream(
        StubEmptyRunner(),
        str(video),
        "detect",
        save_frames_dir=frames_dir,
        save_every=1,
        show=False,
    )
    assert list(frames_dir.glob("*.jpg")) == []


def test_run_stream_rejects_an_unopenable_source():
    with pytest.raises(RuntimeError, match="Cannot open source"):
        run_stream(StubSegRunner(), "/nonexistent/clip.mp4", "segment", show=False)


def test_run_requires_exactly_one_source(tmp_path: Path):
    weights = tmp_path / "m.onnx"
    weights.write_bytes(b"")
    for kwargs in ({}, {"webcam": 0, "video": tmp_path}):
        with pytest.raises(typer.BadParameter, match="exactly one"):
            run_cmd(weights=weights, **kwargs)


def test_build_runner_rejects_trt_segmentation(tmp_path: Path):
    with pytest.raises(typer.BadParameter, match="No TensorRT segmentation runner"):
        _build_runner(tmp_path / "m.engine", "segment", "tensorrt", 640)


def test_build_runner_rejects_unknown_backend(tmp_path: Path):
    with pytest.raises(typer.BadParameter, match="--backend must be"):
        _build_runner(tmp_path / "m.onnx", "segment", "openvino", 640)
