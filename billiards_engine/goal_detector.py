"""
Goal detector — monitors pocket ROIs frame by frame.

Algorithm (no ball tracking, no annotations):
  1. Build a background model for each ROI from the first N quiet frames
     (median pixel value per channel).
  2. Each frame: compute mean absolute difference between the ROI and its
     background → "activity score".
  3. State machine per pocket:
       IDLE  → BALL_ENTERING  when activity > enter_threshold
       BALL_ENTERING → GOAL   when activity drops back below exit_threshold
                               (ball has fallen in — pocket returned to dark)
  4. The GOAL frame is reported as the exact moment the ball disappears.

Returns list of GoalEvent(pocket_idx, label, frame_id).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional

import cv2
import numpy as np


class _State(Enum):
    IDLE          = auto()
    BALL_ENTERING = auto()


@dataclass
class GoalEvent:
    pocket_idx: int
    label: str
    frame_id: int
    peak_activity: float    # max activity score seen during the entry


@dataclass
class _PocketTracker:
    idx: int
    label: str
    roi: dict                          # {x, y, w, h}
    background: Optional[np.ndarray] = None
    state: _State = _State.IDLE
    peak_activity: float = 0.0
    entry_frame: int = 0
    cooldown_until: int = 0            # suppress re-firing for N frames


class GoalDetector:
    """
    Parameters
    ----------
    rois : list of {label, x, y, w, h}
    background_frames : int
        Number of frames used to build the per-pocket background model.
    enter_threshold : float
        Mean-absolute-difference (0-255) that signals a ball entering the ROI.
    exit_threshold : float
        MAD below which we consider the ball gone (goal confirmed).
    min_entry_frames : int
        Ball must be present for at least this many frames to count as real.
    cooldown_frames : int
        Frames to wait before detecting another goal in the same pocket.
    """

    def __init__(
        self,
        rois: List[dict],
        background_frames: int = 15,
        enter_threshold: float = 18.0,
        exit_threshold: float = 10.0,
        min_entry_frames: int = 3,
        max_entry_frames: int = 18,
        cooldown_frames: int = 60,
    ):
        self._trackers = [
            _PocketTracker(idx=i, label=r["label"], roi=r)
            for i, r in enumerate(rois)
        ]
        self._bg_frames = background_frames
        self._enter_thr = enter_threshold
        self._exit_thr = exit_threshold
        self._min_entry = min_entry_frames
        self._max_entry = max_entry_frames
        self._cooldown = cooldown_frames

        # Buffer for background accumulation
        self._bg_buffer: List[List[np.ndarray]] = [[] for _ in rois]
        self._bg_built = False

        # Per-frame activity scores for debug/visualization
        self.activity_log: List[List[float]] = [[] for _ in rois]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray, frame_id: int) -> List[GoalEvent]:
        """Feed one frame. Returns any GoalEvents detected in this frame."""
        events: List[GoalEvent] = []

        for pt in self._trackers:
            roi_img = self._crop(frame, pt.roi)

            # ── Background accumulation phase ────────────────────────────
            if not self._bg_built:
                self._bg_buffer[pt.idx].append(roi_img.astype(np.float32))
                if len(self._bg_buffer[pt.idx]) >= self._bg_frames:
                    pt.background = np.median(
                        np.stack(self._bg_buffer[pt.idx]), axis=0
                    ).astype(np.float32)
                continue

            if pt.background is None:
                continue

            # ── Activity score ────────────────────────────────────────────
            diff = np.abs(roi_img.astype(np.float32) - pt.background)
            activity = float(diff.mean())
            self.activity_log[pt.idx].append(activity)

            # ── State machine ─────────────────────────────────────────────
            if frame_id < pt.cooldown_until:
                continue

            if pt.state == _State.IDLE:
                if activity >= self._enter_thr:
                    pt.state = _State.BALL_ENTERING
                    pt.entry_frame = frame_id
                    pt.peak_activity = activity

            elif pt.state == _State.BALL_ENTERING:
                pt.peak_activity = max(pt.peak_activity, activity)

                frames_in = frame_id - pt.entry_frame

                # Lingered too long → ball near pocket but not pocketed
                if frames_in > self._max_entry:
                    pt.state = _State.IDLE
                    continue

                if activity < self._exit_thr and frames_in >= self._min_entry:
                    # Ball appeared then vanished quickly → GOAL
                    ev = GoalEvent(
                        pocket_idx=pt.idx,
                        label=pt.label,
                        frame_id=frame_id,
                        peak_activity=round(pt.peak_activity, 1),
                    )
                    events.append(ev)
                    pt.state = _State.IDLE
                    pt.cooldown_until = frame_id + self._cooldown

                elif activity < self._exit_thr:
                    # Too brief — false positive (cue stick shadow etc.)
                    pt.state = _State.IDLE

        # Check if background is fully built after this frame
        if not self._bg_built and all(
            pt.background is not None for pt in self._trackers
        ):
            self._bg_built = True
            print(f"    Background model built ({self._bg_frames} frames)")

        return events

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _crop(frame: np.ndarray, roi: dict) -> np.ndarray:
        x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
        fh, fw = frame.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(fw, x + w), min(fh, y + h)
        return frame[y1:y2, x1:x2]
