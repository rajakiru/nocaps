"""
Abstract detector interface — plug in any ball detector here.

A detector takes a BGR frame and returns a list of Detection objects
with frame_id, x, y, w, h, and category filled in.  ball_id is left
as -1; the tracker will assign stable IDs across frames.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from .detection_loader import Detection


class BallDetector(ABC):
    """Base class for per-frame ball detectors."""

    @abstractmethod
    def detect(self, frame: np.ndarray, frame_id: int) -> List[Detection]:
        """Return detections for a single frame."""
        ...
