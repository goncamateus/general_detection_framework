from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from gdf.inference.tracker import ByteTracker, ByteTrackResult
from gdf.utils.logging import log


def as_frame(image: "str | Path | np.ndarray") -> np.ndarray:
    """Accept either a path to read or an already-decoded BGR frame."""
    if isinstance(image, np.ndarray):
        return image
    img = cv2.imread(str(image))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image}")
    return img


class ONNXDetectRunner:
    """ONNX Runtime runner for YOLO detection models with tracking."""

    def __init__(self, model_path: Path, imgsz: int = 640) -> None:
        import onnxruntime as ort

        providers = ort.get_available_providers()
        log.info(f"ONNX detect providers: {providers}")

        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        if self.session.get_inputs()[0].shape[2] != imgsz:
            real_size = self.session.get_inputs()[0].shape[2]
            log.warning(f"Model input size {real_size} does not match imgsz {imgsz}")
            log.warning("Transforms will be applied to resize input.")
            imgsz = self.session.get_inputs()[0].shape[2]
        self.imgsz = imgsz
        self.tracker: ByteTracker | None = None

    def enable_tracking(
        self,
        conf_threshold: float = 0.3,
        match_threshold: float = 0.7,
        max_time_lost: int = 30,
    ) -> None:
        self.tracker = ByteTracker(
            conf_threshold=conf_threshold,
            match_threshold=match_threshold,
            max_time_lost=max_time_lost,
        )

    def _preprocess(
        self, img: np.ndarray
    ) -> tuple[np.ndarray, float, tuple[int, int, int, int, int, int]]:
        h, w = img.shape[:2]
        # Letterbox: scale to fit within imgsz while preserving aspect ratio
        scale = min(self.imgsz / h, self.imgsz / w)
        new_h, new_w = int(h * scale), int(w * scale)

        resized = cv2.resize(img, (new_w, new_h))

        # Pad to square centered
        padded = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        pad_h, pad_w = (self.imgsz - new_h) // 2, (self.imgsz - new_w) // 2
        padded[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized

        blob = padded.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis]

        # Return: (pad_h, pad_w, new_w, new_h, orig_w, orig_h)
        return blob, scale, (pad_h, pad_w, new_w, new_h, w, h)

    def _postprocess(
        self,
        output: np.ndarray,
        scale: float,
        pad: tuple[int, int, int, int, int, int],
        conf_threshold: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Post-process YOLO detection output.

        Args:
            output: Raw model output
            scale: Scale factor from letterbox resize
            pad: (pad_h, pad_w, new_w, new_h, orig_w, orig_h)
            conf_threshold: Confidence threshold
            debug: Print raw output for debugging

        YOLO output shape: [1, num_classes+4, num_boxes]
        Format: [cx, cy, w, h, class_scores...] OR [x1, y1, x2, y2, score, class_id]
        """

        pred = output[0]

        if pred.shape[0] < pred.shape[1]:
            pred = pred.T

        boxes_xywh = pred[:, :4]
        class_scores = pred[:, 4:]

        max_scores = class_scores.max(axis=1)
        class_ids = class_scores.argmax(axis=1)

        mask = max_scores > conf_threshold
        boxes_xywh = boxes_xywh[mask]
        max_scores = max_scores[mask]
        class_ids = class_ids[mask]

        if len(boxes_xywh) == 0:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.array([], dtype=np.int32),
                np.array([], dtype=np.float32),
            )

        # Unpack pad info
        pad_h, pad_w, new_w, new_h, orig_w, orig_h = pad
        boxes_xyxy = np.zeros_like(boxes_xywh)

        # Convert from center xywh to corner xyxy
        boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
        boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
        boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
        boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2

        # Remove padding offset -> coords in resized image space
        boxes_xyxy[:, [0, 2]] -= pad_w
        boxes_xyxy[:, [1, 3]] -= pad_h

        # Clip to resized image bounds
        boxes_xyxy[:, 0] = np.clip(boxes_xyxy[:, 0], 0, new_w)
        boxes_xyxy[:, 1] = np.clip(boxes_xyxy[:, 1], 0, new_h)
        boxes_xyxy[:, 2] = np.clip(boxes_xyxy[:, 2], 0, new_w)
        boxes_xyxy[:, 3] = np.clip(boxes_xyxy[:, 3], 0, new_h)

        # Scale back to original image size
        boxes_xyxy /= scale

        # Clip to original image bounds
        boxes_xyxy[:, 0] = np.clip(boxes_xyxy[:, 0], 0, orig_w)
        boxes_xyxy[:, 1] = np.clip(boxes_xyxy[:, 1], 0, orig_h)
        boxes_xyxy[:, 2] = np.clip(boxes_xyxy[:, 2], 0, orig_w)
        boxes_xyxy[:, 3] = np.clip(boxes_xyxy[:, 3], 0, orig_h)

        # NMS
        keep = self._nms(boxes_xyxy, max_scores, iou_threshold=0.5)
        return boxes_xyxy[keep], class_ids[keep], max_scores[keep]

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        order = scores.argsort()[::-1]
        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / np.maximum(areas[i] + areas[order[1:]] - inter, 1e-6)

            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return np.array(keep)

    def detect(
        self,
        image: str | Path | np.ndarray,
        conf_threshold: float = 0.25,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run detection on a single image path or BGR frame.

        Returns:
            bboxes (N, 4) xyxy, class_ids (N,), scores (N,)
        """
        img = as_frame(image)

        blob, scale, pad = self._preprocess(img)
        outputs = self.session.run(None, {self.input_name: blob})

        return self._postprocess(outputs[0], scale, pad, conf_threshold)

    def detect_and_track(
        self,
        image: str | Path,
        conf_threshold: float = 0.25,
    ) -> ByteTrackResult:
        """Run detection + tracking on a single frame."""
        if self.tracker is None:
            self.enable_tracking(conf_threshold=conf_threshold)

        bboxes, class_ids, scores = self.detect(image, conf_threshold)
        return self.tracker.update(bboxes, scores, class_ids)

    def detect_video(
        self,
        video_path: str | Path,
        conf_threshold: float = 0.25,
        output_path: str | Path | None = None,
    ) -> list[ByteTrackResult]:
        """Process video with detection + tracking."""
        if self.tracker is None:
            self.enable_tracking(conf_threshold=conf_threshold)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        writer = None
        if output_path:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

        results = []
        frame_idx = 0
        colors = {}

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            blob, scale, pad = self._preprocess(frame)
            outputs = self.session.run(None, {self.input_name: blob})
            bboxes, class_ids, scores = self._postprocess(outputs[0], scale, pad, conf_threshold)
            tracks = self.tracker.update(bboxes, scores, class_ids)
            results.append(tracks)

            if writer:
                for i in range(len(tracks)):
                    tid = int(tracks.track_ids[i])
                    if tid not in colors:
                        colors[tid] = (
                            int(np.random.randint(0, 255)),
                            int(np.random.randint(0, 255)),
                            int(np.random.randint(0, 255)),
                        )
                    color = colors[tid]
                    x1, y1, x2, y2 = tracks.bboxes[i].astype(int)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"ID:{tid} {int(tracks.class_ids[i])} {tracks.scores[i]:.2f}"
                    cv2.putText(
                        frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                    )

                writer.write(frame)

            frame_idx += 1
            if frame_idx % 100 == 0:
                log.info(f"Processed {frame_idx} frames, {len(tracks)} tracks")

        cap.release()
        if writer:
            writer.release()
            log.info(f"Output saved: {output_path}")

        return results
