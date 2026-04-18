"""
YOLO-backed billiards ball detector.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .detection_loader import Detection
from .detector_base import BallDetector


class YOLOBallDetector(BallDetector):
    """
    Ball detector backed by Ultralytics YOLO.

    The detector assumes all predicted boxes correspond to billiards balls.
    Class IDs are not currently mapped to cue/8-ball/solid/striped categories;
    all outputs use category=0.
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        image_size: int = 1024,
        table_bbox: Optional[Tuple[int, int, int, int]] = None,
        device: Optional[str] = None,
    ):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is required for YOLO detection. "
                "Install it with `python3 -m pip install ultralytics`."
            ) from exc

        model_file = Path(model_path)
        if not model_file.is_file():
            raise FileNotFoundError(f"YOLO model not found: {model_path}")

        self._model = YOLO(str(model_file))
        self._conf_threshold = conf_threshold
        self._iou_threshold = iou_threshold
        self._image_size = image_size
        self._table_bbox = table_bbox
        self._device = device
        self.model_path = str(model_file)

    def predict_raw(self, frame: np.ndarray) -> tuple[list[dict], np.ndarray]:
        tx, ty, tw, th = self._table_bbox if self._table_bbox else (0, 0, frame.shape[1], frame.shape[0])
        roi = frame[ty: ty + th, tx: tx + tw]

        predict_kwargs = {
            "source": roi,
            "conf": self._conf_threshold,
            "iou": self._iou_threshold,
            "imgsz": self._image_size,
            "verbose": False,
        }
        if self._device:
            predict_kwargs["device"] = self._device

        results = self._model.predict(**predict_kwargs)
        if not results:
            return [], roi

        boxes = results[0].boxes
        if boxes is None or boxes.xyxy is None:
            return [], roi

        xyxy = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy), dtype=float)
        class_ids = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.zeros(len(xyxy), dtype=int)
        names = results[0].names if hasattr(results[0], "names") else {}

        raw = []
        for coords, confidence, class_id in zip(xyxy, confidences, class_ids):
            x1, y1, x2, y2 = (float(value) for value in coords)
            raw.append(
                {
                    "x1": x1 + tx,
                    "y1": y1 + ty,
                    "x2": x2 + tx,
                    "y2": y2 + ty,
                    "w": max(0.0, x2 - x1),
                    "h": max(0.0, y2 - y1),
                    "confidence": float(confidence),
                    "class_id": int(class_id),
                    "class_name": str(names.get(int(class_id), class_id)),
                }
            )
        return raw, roi

    def detect(self, frame: np.ndarray, frame_id: int) -> List[Detection]:
        raw, _ = self.predict_raw(frame)
        detections: List[Detection] = []
        for idx, item in enumerate(raw, start=1):
            w = float(item["w"])
            h = float(item["h"])
            if w <= 0 or h <= 0:
                continue
            detections.append(
                Detection(
                    frame_id=frame_id,
                    ball_id=-idx,
                    x=float(item["x1"]),
                    y=float(item["y1"]),
                    w=w,
                    h=h,
                    category=0,
                    confidence=float(item["confidence"]),
                )
            )
        return detections

    def draw_debug(self, frame: np.ndarray, raw_predictions: List[dict]) -> np.ndarray:
        out = frame.copy()
        tx, ty, tw, th = self._table_bbox if self._table_bbox else (0, 0, frame.shape[1], frame.shape[0])
        cv2.rectangle(out, (tx, ty), (tx + tw, ty + th), (180, 180, 180), 2)
        for item in raw_predictions:
            x1 = int(item["x1"])
            y1 = int(item["y1"])
            x2 = int(item["x2"])
            y2 = int(item["y2"])
            cv2.rectangle(out, (x1, y1), (x2, y2), (40, 210, 255), 2)
            cv2.putText(
                out,
                f"{item['class_name']} {item['confidence']:.2f}",
                (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (40, 210, 255),
                1,
                cv2.LINE_AA,
            )
        return out
