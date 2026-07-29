from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from gdf.inference.onnx_detect_runner import ONNXDetectRunner


class ONNXSegRunner(ONNXDetectRunner):
    """ONNX Runtime runner for YOLO instance-segmentation models.

    Reuses the letterbox `_preprocess` and `_nms` from ONNXDetectRunner but replaces the
    post-processing: a segmentation head emits two tensors instead of one.

        output0: [1, 4 + nc + nm, num_boxes]   e.g. [1, 37, 8400] for nc=1, nm=32
        output1: [1, nm, imgsz/4, imgsz/4]     mask prototypes

    Per-instance mask = sigmoid(mask_coeffs @ protos), cropped to the letterbox content
    region and resized back to the original frame.
    """

    def segment(
        self,
        image: str | Path,
        conf_threshold: float = 0.25,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run segmentation on a single image.

        Returns:
            masks (N, orig_h, orig_w) bool, bboxes (N, 4) xyxy, class_ids (N,), scores (N,)
        """
        img = cv2.imread(str(image))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image}")

        blob, _scale, pad = self._preprocess(img)
        outputs = self.session.run(None, {self.input_name: blob})

        # Identify by rank rather than output order: protos are 4-D, predictions 3-D.
        preds = next(o for o in outputs if o.ndim == 3)
        protos = next(o for o in outputs if o.ndim == 4)

        return self._postprocess_seg(preds, protos, pad, conf_threshold)

    def _postprocess_seg(
        self,
        preds: np.ndarray,
        protos: np.ndarray,
        pad: tuple[int, int, int, int, int, int],
        conf_threshold: float,
        iou_threshold: float = 0.5,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pad_h, pad_w, new_w, new_h, orig_w, orig_h = pad

        pred = preds[0]
        if pred.shape[0] < pred.shape[1]:
            pred = pred.T  # -> [num_boxes, 4 + nc + nm]

        proto = protos[0]
        nm, mh, mw = proto.shape
        nc = pred.shape[1] - 4 - nm
        if nc < 1:
            raise ValueError(
                f"Unexpected segmentation output: preds {pred.shape}, protos {proto.shape}"
            )

        boxes_xywh = pred[:, :4]
        class_scores = pred[:, 4 : 4 + nc]
        coeffs = pred[:, 4 + nc :]

        max_scores = class_scores.max(axis=1)
        class_ids = class_scores.argmax(axis=1)

        keep_conf = max_scores > conf_threshold
        boxes_xywh = boxes_xywh[keep_conf]
        max_scores = max_scores[keep_conf]
        class_ids = class_ids[keep_conf]
        coeffs = coeffs[keep_conf]

        if len(boxes_xywh) == 0:
            return self._empty(orig_h, orig_w)

        # Boxes in letterboxed (model input) space, xyxy.
        boxes_lb = np.empty_like(boxes_xywh)
        boxes_lb[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
        boxes_lb[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
        boxes_lb[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
        boxes_lb[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2

        # IoU is invariant to the uniform scale + translation of the letterbox, so NMS in
        # letterbox space gives the same survivors as NMS in original space.
        keep = self._nms(boxes_lb, max_scores, iou_threshold=iou_threshold)
        if len(keep) == 0:
            return self._empty(orig_h, orig_w)

        boxes_lb = boxes_lb[keep]
        max_scores = max_scores[keep]
        class_ids = class_ids[keep]
        coeffs = coeffs[keep]

        # Undo the letterbox for the boxes: strip padding, clip, rescale to the original.
        boxes_xyxy = boxes_lb.copy()
        boxes_xyxy[:, [0, 2]] -= pad_w
        boxes_xyxy[:, [1, 3]] -= pad_h
        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, new_w)
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, new_h)
        boxes_xyxy[:, [0, 2]] *= orig_w / new_w
        boxes_xyxy[:, [1, 3]] *= orig_h / new_h

        # ponytail: upsample the full proto to imgsz, then crop — exact w.r.t. the letterbox
        # params and easy to verify. O(N * imgsz^2); crop in proto space first if N grows.
        mask_maps = coeffs @ proto.reshape(nm, mh * mw)
        mask_maps = 1.0 / (1.0 + np.exp(-np.clip(mask_maps, -30.0, 30.0)))
        mask_maps = mask_maps.reshape(-1, mh, mw)

        masks = np.zeros((len(mask_maps), orig_h, orig_w), dtype=bool)
        for i, m in enumerate(mask_maps):
            full = cv2.resize(m, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
            content = full[pad_h : pad_h + new_h, pad_w : pad_w + new_w]
            resized = cv2.resize(content, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

            binary = resized > 0.5
            # Crop to the box — proto masks leak outside the instance otherwise.
            x1, y1, x2, y2 = boxes_xyxy[i]
            box_only = np.zeros_like(binary)
            xs, xe = int(np.floor(x1)), int(np.ceil(x2))
            ys, ye = int(np.floor(y1)), int(np.ceil(y2))
            box_only[ys:ye, xs:xe] = binary[ys:ye, xs:xe]
            masks[i] = box_only

        return masks, boxes_xyxy, class_ids.astype(np.int32), max_scores.astype(np.float32)

    @staticmethod
    def _empty(
        orig_h: int, orig_w: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.empty((0, orig_h, orig_w), dtype=bool),
            np.empty((0, 4), dtype=np.float32),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.float32),
        )
