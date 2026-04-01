"""
Detection loader: reads pre-annotated bounding box files.

Annotation format (one ball per line, space-separated):
    x  y  w  h  category_id

Category IDs:
    1 = white cue ball
    2 = black 8-ball
    3 = solid color ball
    4 = striped ball
    5 = playing field (ignored here)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Detection:
    frame_id: int
    ball_id: int       # assigned from file row order (1-indexed)
    x: float           # top-left x
    y: float           # top-left y
    w: float
    h: float
    category: int      # 1=cue,2=8ball,3=solid,4=striped

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    @property
    def radius(self) -> float:
        return (self.w + self.h) / 4


class DetectionLoader:
    """
    Loads bounding box annotations from a clip's bounding_boxes/ directory.

    Only the first and last frame have annotations; all other frames return [].
    The first frame maps to frame_id=0, the last frame maps to frame_id=total_frames-1.
    """

    def __init__(self, bbox_dir: str, total_frames: int):
        self._detections: Dict[int, List[Detection]] = {}

        first_path = os.path.join(bbox_dir, "frame_first_bbox.txt")
        last_path = os.path.join(bbox_dir, "frame_last_bbox.txt")

        if os.path.isfile(first_path):
            self._detections[0] = self._parse(first_path, frame_id=0)

        last_frame_id = max(0, total_frames - 1)
        if os.path.isfile(last_path):
            self._detections[last_frame_id] = self._parse(last_path, frame_id=last_frame_id)

    @staticmethod
    def _parse(path: str, frame_id: int) -> List[Detection]:
        detections: List[Detection] = []
        with open(path, "r") as fh:
            for ball_id, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                x, y, w, h, cat = (float(p) for p in parts[:5])
                cat = int(cat)
                if cat == 5:           # skip "playing field" annotations
                    continue
                detections.append(
                    Detection(
                        frame_id=frame_id,
                        ball_id=ball_id,
                        x=x, y=y, w=w, h=h,
                        category=cat,
                    )
                )
        return detections

    def get(self, frame_id: int) -> List[Detection]:
        """Return detections for a specific frame (empty list if none)."""
        return self._detections.get(frame_id, [])

    def has_frame(self, frame_id: int) -> bool:
        return frame_id in self._detections

    @property
    def annotated_frames(self) -> List[int]:
        return sorted(self._detections.keys())
