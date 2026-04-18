"""
Lightweight per-video calibration helpers.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from .felt_config import get_felt_mask
from .opencv_detector import estimate_table_bbox


@dataclass
class VideoCalibration:
    felt: str
    table_bbox: Tuple[int, int, int, int]
    brightness_gain: float
    mean_brightness: float
    felt_coverage: float

    def to_json(self) -> dict:
        data = asdict(self)
        data["table_bbox"] = list(self.table_bbox)
        return data


def estimate_video_calibration(
    frame: np.ndarray,
    felt: str = "blue",
    table_bbox: Optional[Tuple[int, int, int, int]] = None,
) -> VideoCalibration:
    if table_bbox is None:
        table_bbox = estimate_table_bbox(frame, felt=felt)
    if table_bbox is None:
        h, w = frame.shape[:2]
        table_bbox = (0, 0, w, h)

    tx, ty, tw, th = table_bbox
    table = frame[ty: ty + th, tx: tx + tw]
    hsv = cv2.cvtColor(table, cv2.COLOR_BGR2HSV)
    felt_mask = get_felt_mask(hsv, felt)
    value = hsv[:, :, 2]
    felt_values = value[felt_mask > 0]
    mean_brightness = float(np.mean(felt_values)) if felt_values.size else float(np.mean(value))
    brightness_gain = float(np.clip(145.0 / max(mean_brightness, 1.0), 0.75, 1.35))
    felt_coverage = float(np.count_nonzero(felt_mask) / max(felt_mask.size, 1))
    return VideoCalibration(
        felt=felt,
        table_bbox=table_bbox,
        brightness_gain=round(brightness_gain, 4),
        mean_brightness=round(mean_brightness, 2),
        felt_coverage=round(felt_coverage, 4),
    )


def normalize_frame(frame: np.ndarray, calibration: VideoCalibration) -> np.ndarray:
    return cv2.convertScaleAbs(frame, alpha=calibration.brightness_gain, beta=0)


def save_calibration(calibration: VideoCalibration, path: str) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(calibration.to_json(), fh, indent=2)
