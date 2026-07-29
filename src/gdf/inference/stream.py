from __future__ import annotations

import os
import platform
from pathlib import Path

import cv2
import numpy as np

from gdf.utils.logging import log

IS_WINDOWS = platform.system() == "Windows"

# Distinct, colourblind-safe-ish BGR palette, cycled by class id.
PALETTE = [
    (80, 175, 76),
    (60, 76, 231),
    (219, 152, 52),
    (34, 126, 230),
    (182, 89, 155),
    (15, 196, 241),
]


def has_display() -> bool:
    """Whether a GUI window can be opened.

    Checked before touching cv2's GUI: with no X/Wayland session the Qt plugin fails to
    initialize and aborts the process outright, so there is no exception left to catch.
    """
    if platform.system() != "Linux":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def open_capture(source: int | str) -> cv2.VideoCapture:
    """Webcam index or video path. Windows needs DSHOW for a live camera."""
    if isinstance(source, int) and IS_WINDOWS:
        return cv2.VideoCapture(source, cv2.CAP_DSHOW)
    return cv2.VideoCapture(source)


def draw_masks(
    frame: np.ndarray,
    masks: np.ndarray,
    bboxes: np.ndarray,
    class_ids: np.ndarray,
    scores: np.ndarray,
    class_names: list[str],
    alpha: float = 0.45,
) -> np.ndarray:
    """Tint each mask, outline it, and label its box."""
    out = frame.copy()

    for i in range(len(scores)):
        color = PALETTE[int(class_ids[i]) % len(PALETTE)]
        mask = masks[i]

        # Blend only inside the mask — cheaper and sharper than compositing a full overlay.
        out[mask] = (out[mask] * (1 - alpha) + np.array(color, dtype=np.float32) * alpha).astype(
            np.uint8
        )

        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(out, contours, -1, color, 2)

        x1, y1 = bboxes[i][:2].astype(int)
        name = class_names[int(class_ids[i])] if int(class_ids[i]) < len(class_names) else "obj"
        _label(out, f"{name} {scores[i]:.2f} | {int(mask.sum())}px", x1, y1, color)

    return out


def draw_boxes(
    frame: np.ndarray,
    bboxes: np.ndarray,
    class_ids: np.ndarray,
    scores: np.ndarray,
    class_names: list[str],
) -> np.ndarray:
    out = frame.copy()
    for i in range(len(scores)):
        color = PALETTE[int(class_ids[i]) % len(PALETTE)]
        x1, y1, x2, y2 = bboxes[i].astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        name = class_names[int(class_ids[i])] if int(class_ids[i]) < len(class_names) else "obj"
        _label(out, f"{name} {scores[i]:.2f}", x1, y1, color)
    return out


def _label(img: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    y = max(y, th + 8)
    cv2.rectangle(img, (x, y - th - 8), (x + tw + 6, y), color, -1)
    cv2.putText(img, text, (x + 3, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)


def run_stream(
    runner: object,
    source: int | str,
    task: str,
    conf_threshold: float = 0.25,
    class_names: list[str] | None = None,
    output_path: Path | None = None,
    save_frames_dir: Path | None = None,
    save_every: int = 30,
    show: bool = True,
    max_frames: int | None = None,
) -> int:
    """Run a runner over a webcam or video, drawing results and reporting throughput.

    Returns the number of frames processed.
    """
    class_names = class_names or []

    if show and not has_display():
        log.warning("No display detected (DISPLAY/WAYLAND_DISPLAY unset) — running headless")
        if output_path is None and save_frames_dir is None:
            log.warning("Nothing will be saved: pass --output and/or --save-frames")
        show = False

    cap = open_capture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    log.info(f"Source opened: {w}x{h} @ {src_fps:.0f}fps")

    writer = None
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (w, h)
        )
    if save_frames_dir:
        save_frames_dir.mkdir(parents=True, exist_ok=True)

    win = "GDF"
    if show:
        try:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win, w, h)
        except cv2.error as e:  # headless opencv build, or a broken GUI backend
            log.warning(f"Cannot open a window ({e.err.strip() if e.err else e}) — headless")
            show = False

    frame_idx = 0
    saved = 0
    recent_ms: list[float] = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t0 = cv2.getTickCount()
            if task == "segment":
                masks, bboxes, class_ids, scores = runner.segment(frame, conf_threshold)  # type: ignore[attr-defined]
            else:
                masks = None
                bboxes, class_ids, scores = runner.detect(frame, conf_threshold)  # type: ignore[attr-defined]
            infer_ms = (cv2.getTickCount() - t0) / cv2.getTickFrequency() * 1000

            recent_ms.append(infer_ms)
            if len(recent_ms) > 30:
                recent_ms.pop(0)

            display = (
                draw_masks(frame, masks, bboxes, class_ids, scores, class_names)
                if masks is not None
                else draw_boxes(frame, bboxes, class_ids, scores, class_names)
            )

            fps = 1000.0 / max(sum(recent_ms) / len(recent_ms), 1e-6)
            cv2.putText(
                display,
                f"FPS: {fps:.1f} | {infer_ms:.0f}ms | {len(scores)} obj",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            if writer:
                writer.write(display)
            if save_frames_dir and frame_idx % save_every == 0 and len(scores):
                cv2.imwrite(str(save_frames_dir / f"frame_{frame_idx:06d}.jpg"), display)
                saved += 1

            if show:
                cv2.imshow(win, display)
                # waitKeyEx: plain waitKey misses keys on Windows.
                if cv2.waitKeyEx(1) in (ord("q"), ord("Q"), 0x1B):
                    log.info("User quit")
                    break

            frame_idx += 1
            if max_frames and frame_idx >= max_frames:
                break
    finally:
        cap.release()
        if writer:
            writer.release()
            log.info(f"Video saved: {output_path}")
        if save_frames_dir:
            log.info(f"Saved {saved} sample frames to {save_frames_dir}")
        if show:
            cv2.destroyAllWindows()
            cv2.waitKey(1)

    mean_ms = sum(recent_ms) / len(recent_ms) if recent_ms else 0.0
    log.info(f"Processed {frame_idx} frames, {1000 / max(mean_ms, 1e-6):.1f} FPS avg")
    return frame_idx
