"""
OpenCV-based ball detector (no ML).

Strategy:
1. Estimate the table region via HSV colour thresholding (blue/teal felt).
2. Within the table mask, label every non-felt pixel as a candidate ball pixel.
3. Find connected-component contours and filter by:
     - area  → 50–800 px²  (radius ≈ 4–16 px for 1024×576 footage)
     - circularity > 0.40   (4πA / P²)
4. Exclude blobs too close to the table boundary (likely UI or cushion glare).
5. Assign category=0 (unknown); the tracker inherits known categories from
   the annotated first/last frames.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from .detection_loader import Detection
from .detector_base import BallDetector
from .felt_config import get_felt_mask

# Contour filter thresholds
_MIN_AREA = 55      # ≈ radius 4 px
_MAX_AREA = 750     # ≈ radius 15 px
_MIN_CIRC = 0.42    # circularity (1.0 = perfect circle)
_EDGE_MARGIN = 18   # ignore blobs within N px of the table edge


def estimate_table_bbox(frame: np.ndarray, felt: str = "blue") -> Optional[Tuple[int, int, int, int]]:
    """Return (x, y, w, h) bounding box of the table, or None if not found."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = get_felt_mask(hsv, felt)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 0.05 * frame.shape[0] * frame.shape[1]:
        return None
    return cv2.boundingRect(largest)


class OpenCVBallDetector(BallDetector):
    """
    Contour-based ball detector restricted to the table region.

    Parameters
    ----------
    table_bbox : optional (x, y, w, h)
        Pre-computed table bounding box. If None, estimated from the first frame.
    """

    def __init__(
        self,
        min_radius: int = 4,
        max_radius: int = 16,
        table_bbox: Optional[Tuple[int, int, int, int]] = None,
        felt: str = "blue",
    ):
        self._table_bbox = table_bbox
        self._table_estimated = table_bbox is not None
        self._felt = felt

    def _ensure_table(self, frame: np.ndarray):
        if not self._table_estimated:
            self._table_bbox = estimate_table_bbox(frame, self._felt)
            self._table_estimated = True

    def detect(self, frame: np.ndarray, frame_id: int) -> List[Detection]:
        self._ensure_table(frame)

        h_f, w_f = frame.shape[:2]
        tx, ty, tw, th = self._table_bbox if self._table_bbox else (0, 0, w_f, h_f)

        # ── 1. Crop to table ────────────────────────────────────────────
        roi = frame[ty: ty + th, tx: tx + tw]
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # ── 2. Ball mask = non-felt within the ROI ───────────────────────
        felt_mask = get_felt_mask(roi_hsv, self._felt)
        ball_mask = cv2.bitwise_not(felt_mask)

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        ball_mask = cv2.morphologyEx(ball_mask, cv2.MORPH_CLOSE, k)
        ball_mask = cv2.morphologyEx(ball_mask, cv2.MORPH_OPEN, k)

        # ── 3. Find contours ─────────────────────────────────────────────
        contours, _ = cv2.findContours(ball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections: List[Detection] = []
        m = _EDGE_MARGIN

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (_MIN_AREA <= area <= _MAX_AREA):
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4.0 * np.pi * area / (perimeter ** 2)
            if circularity < _MIN_CIRC:
                continue

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx_roi = M["m10"] / M["m00"]
            cy_roi = M["m01"] / M["m00"]

            # Skip blobs near table edge (cushion glare / UI artifacts)
            if cx_roi < m or cx_roi > tw - m or cy_roi < m or cy_roi > th - m:
                continue

            r = float(np.sqrt(area / np.pi))
            cx = cx_roi + tx
            cy = cy_roi + ty

            detections.append(
                Detection(
                    frame_id=frame_id,
                    ball_id=-1,          # assigned by tracker
                    x=float(cx - r),
                    y=float(cy - r),
                    w=float(r * 2),
                    h=float(r * 2),
                    category=0,          # unknown
                )
            )

        return detections
