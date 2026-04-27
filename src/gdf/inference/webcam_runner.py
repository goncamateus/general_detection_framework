from __future__ import annotations

from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from gdf.inference.tracker import ByteTracker
from gdf.utils.logging import log


class WebcamRunner:
    """Real-time webcam detection + tracking.

    Supports ONNX and TensorRT backends. Displays results in an OpenCV window.

    Args:
        weights: Path to model weights (.onnx or .engine)
        backend: "onnx" or "tensorrt"
        imgsz: Input image size
        conf_threshold: Detection confidence threshold
        match_threshold: ByteTrack IoU match threshold
        class_names: Optional list of class names for display
        device: Webcam device index (default 0)
    """

    def __init__(
        self,
        weights: Path,
        backend: Literal["onnx", "tensorrt"] = "onnx",
        imgsz: int = 640,
        conf_threshold: float = 0.25,
        match_threshold: float = 0.7,
        class_names: list[str] | None = None,
        device: int = 0,
    ) -> None:
        self.weights = Path(weights)
        self.backend = backend
        self.imgsz = imgsz
        self.conf_threshold = conf_threshold
        self.match_threshold = match_threshold
        self.class_names = class_names or []
        self.device = device
        self._runner = None

    def _init_runner(self) -> None:
        if self.backend == "onnx":
            from gdf.inference.onnx_detect_runner import ONNXDetectRunner
            self._runner = ONNXDetectRunner(self.weights, self.imgsz)
        elif self.backend == "tensorrt":
            from gdf.inference.trt_detect_runner import TRTDetectRunner
            self._runner = TRTDetectRunner(self.weights, self.imgsz)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

        self._runner.enable_tracking(
            conf_threshold=self.conf_threshold,
            match_threshold=self.match_threshold,
        )

    def run(
        self,
        output_path: str | Path | None = None,
        show: bool = True,
        max_frames: int | None = None,
    ) -> int:
        """Run webcam detection + tracking.

        Args:
            output_path: Optional path to save output video
            show: Display OpenCV window (default True)
            max_frames: Stop after N frames (None = infinite until 'q')

        Returns:
            Total frames processed
        """
        if self._runner is None:
            self._init_runner()

        cap = cv2.VideoCapture(self.device)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open webcam device {self.device}")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30

        log.info(f"Webcam opened: {w}x{h} @ {fps:.0f}fps")

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

        frame_idx = 0
        avg_fps = 0
        colors: dict[int, tuple[int, int, int]] = {}
        fps_history: list[float] = []

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    log.warning("Failed to read frame from webcam")
                    break

                t_start = cv2.getTickCount()

                # Run detection + tracking
                blob, scale, pad = self._runner._preprocess(frame)

                if self.backend == "onnx":
                    outputs = self._runner.session.run(None, {self._runner.input_name: blob})
                    raw_output = outputs[0]
                else:
                    raw_output = self._runner._run_inference(blob)

                bboxes, class_ids, scores = self._runner._postprocess(raw_output, scale, pad, self.conf_threshold)
                tracks = self._runner.tracker.update(bboxes, scores, class_ids)

                t_end = cv2.getTickCount()
                infer_ms = (t_end - t_start) / cv2.getTickFrequency() * 1000
                fps_history.append(1000 / max(infer_ms, 1))
                if len(fps_history) > 30:
                    fps_history.pop(0)

                # Draw results
                display = frame.copy()

                for i in range(len(tracks)):
                    tid = int(tracks.track_ids[i])
                    cls_id = int(tracks.class_ids[i])
                    conf = float(tracks.scores[i])

                    if tid not in colors:
                        colors[tid] = (
                            int(np.random.randint(50, 255)),
                            int(np.random.randint(50, 255)),
                            int(np.random.randint(50, 255)),
                        )
                    color = colors[tid]

                    x1, y1, x2, y2 = tracks.bboxes[i].astype(int)
                    cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

                    cls_name = self.class_names[cls_id] if cls_id < len(self.class_names) else str(cls_id)
                    label = f"ID:{tid} {cls_name} {conf:.2f}"

                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    cv2.rectangle(display, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
                    cv2.putText(display, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                # FPS overlay
                avg_fps = sum(fps_history) / len(fps_history) if fps_history else 0
                fps_text = f"FPS: {avg_fps:.0f} | Tracks: {len(tracks)} | Infer: {infer_ms:.0f}ms"
                cv2.putText(display, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                if writer:
                    writer.write(display)

                if show:
                    cv2.imshow("GDF Tracking", display)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        log.info("User pressed 'q', stopping")
                        break

                frame_idx += 1
                if max_frames and frame_idx >= max_frames:
                    break

        finally:
            cap.release()
            if writer:
                writer.release()
                log.info(f"Output saved: {output_path}")
            if show:
                cv2.destroyAllWindows()

        log.info(f"Processed {frame_idx} frames, avg FPS: {avg_fps:.0f}")
        return frame_idx
