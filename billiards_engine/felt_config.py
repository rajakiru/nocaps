"""
Felt color configurations for different table types.

Red wraps around in HSV so it requires two masks combined with bitwise_or.
Blue/teal is a single contiguous HSV range.

Usage
-----
  from billiards_engine.felt_config import FELT_CONFIGS, get_felt_mask

  mask = get_felt_mask(hsv_frame, felt="red")
  mask = get_felt_mask(hsv_frame, felt="blue")
"""

from __future__ import annotations
from typing import Tuple

import cv2
import numpy as np

# Each entry: list of (lo, hi) HSV range pairs — all are OR'd together
FELT_CONFIGS = {
    "blue": [
        (np.array([85,  60,  80], dtype=np.uint8),
         np.array([110, 255, 255], dtype=np.uint8)),
    ],
    "red": [
        # lower red band (0-12)
        (np.array([0,   80,  60], dtype=np.uint8),
         np.array([12, 255, 255], dtype=np.uint8)),
        # upper red band (160-180)
        (np.array([160,  80,  60], dtype=np.uint8),
         np.array([180, 255, 255], dtype=np.uint8)),
    ],
    "green": [
        (np.array([40,  50,  40], dtype=np.uint8),
         np.array([80, 255, 255], dtype=np.uint8)),
    ],
}


def get_felt_mask(hsv: np.ndarray, felt: str = "blue") -> np.ndarray:
    """Return a binary mask of the felt pixels for the given color."""
    if felt not in FELT_CONFIGS:
        raise ValueError(f"Unknown felt color '{felt}'. Choose from: {list(FELT_CONFIGS)}")
    ranges = FELT_CONFIGS[felt]
    mask = cv2.inRange(hsv, ranges[0][0], ranges[0][1])
    for lo, hi in ranges[1:]:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))
    return mask
