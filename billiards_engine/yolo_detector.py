"""
yolo_detector.py — YOLO-based ball detector, drop-in for opencv_detector.py.

Falls back to OpenCVBallDetector automatically if the model file is missing
or ultralytics is not installed.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .detection_loader import Detection
from .detector_base import BallDetector
from .opencv_detector import OpenCVBallDetector, estimate_table_bbox

# Billiards-specific class names from a trained model
# plus COCO "sports ball" so a stock YOLOv8n can be used for testing
_NAME_TO_CAT = {
    "cue": 1, "white": 1,
    "black": 2, "8ball": 2, "8-ball": 2,
    "solid": 3,
    "striped": 4, "stripe": 4,
    "sports ball": 0,   # COCO pretrained fallback — useful before custom model exists
}


class YOLOBallDetector(BallDetector):
    """
    YOLO-based ball detector using Ultralytics.

    Parameters
    ----------
    model_path      : path to a .pt weights file
    confidence      : minimum detection confidence (0–1)
    device          : "" = auto, "cpu", "cuda", "cuda:0", etc.
    table_bbox      : (x, y, w, h) pre-computed table region; estimated on first
                      frame if None
    felt            : felt colour used by estimate_table_bbox ("blue"/"red"/"green")
    fallback_on_error: if True, silently fall back to OpenCVBallDetector instead
                      of raising when the model cannot be loaded
    """

    def __init__(
        self,
        model_path: str,
        confidence: float = 0.35,
        device: str = "",
        table_bbox: Optional[Tuple[int, int, int, int]] = None,
        felt: str = "red",
        fallback_on_error: bool = True,
    ):
        self._conf            = confidence
        self._table_bbox      = table_bbox
        self._table_estimated = table_bbox is not None
        self._felt            = felt
        self._model           = None
        self._fallback: Optional[OpenCVBallDetector] = None
        self._device          = device

        try:
            from ultralytics import YOLO  # noqa: PLC0415
            mp = Path(model_path)
            if not mp.exists():
                raise FileNotFoundError(f"YOLO model not found: {model_path}")
            self._model = YOLO(str(mp))
            # Warm-up pass so the first real frame isn't slow
            dummy = np.zeros((64, 64, 3), dtype=np.uint8)
            self._model(dummy, verbose=False)
            print(f"    [YOLOBallDetector] Loaded {mp.name}  conf={confidence}")
        except Exception as exc:
            msg = f"[YOLOBallDetector] Failed to load YOLO: {exc}"
            if fallback_on_error:
                warnings.warn(f"{msg}\n  → Falling back to OpenCVBallDetector")
                self._fallback = OpenCVBallDetector(table_bbox=table_bbox, felt=felt)
            else:
                raise

    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray, frame_id: int) -> List[Detection]:
        if self._fallback is not None:
            return self._fallback.detect(frame, frame_id)

        self._ensure_table(frame)

        results = self._model(
            frame, conf=self._conf, device=self._device, verbose=False
        )

        detections: List[Detection] = []
        for r in results:
            if r.boxes is None:
                continue
            for (x1, y1, x2, y2), conf, cls_id in zip(
                r.boxes.xyxy.cpu().numpy(),
                r.boxes.conf.cpu().numpy(),
                r.boxes.cls.cpu().numpy(),
            ):
                if conf < self._conf:
                    continue

                # Filter detections to the table region
                if self._table_bbox is not None:
                    tx, ty, tw, th = self._table_bbox
                    cx_det = (x1 + x2) / 2
                    cy_det = (y1 + y2) / 2
                    if not (tx <= cx_det <= tx + tw and ty <= cy_det <= ty + th):
                        continue

                cls_name = r.names.get(int(cls_id), "").lower()
                detections.append(Detection(
                    frame_id=frame_id,
                    ball_id=-1,
                    x=float(x1),
                    y=float(y1),
                    w=float(x2 - x1),
                    h=float(y2 - y1),
                    category=_NAME_TO_CAT.get(cls_name, 0),
                ))

        return detections

    # ------------------------------------------------------------------
    def _ensure_table(self, frame: np.ndarray) -> None:
        if not self._table_estimated:
            self._table_bbox      = estimate_table_bbox(frame, self._felt)
            self._table_estimated = True
