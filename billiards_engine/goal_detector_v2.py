"""
goal_detector_v2.py — improved goal detector with false positive suppression.

Improvements over goal_detector.py:
  1. Temporal smoothing of activity score (rolling window)
  2. Optical flow confirmation before firing GOAL event
  3. Adaptive background model that tracks slow lighting changes
  4. Pixel-count guard to reject global illumination shifts
  5. GoalEvent now carries a confidence score (0.0–1.0)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import Deque, List, Optional

import cv2
import numpy as np


class _State(Enum):
    IDLE          = auto()
    BALL_ENTERING = auto()


@dataclass
class GoalEvent:
    pocket_idx:    int
    label:         str
    frame_id:      int
    peak_activity: float
    confidence:    float   # 0.0 – 1.0


@dataclass
class _PocketTracker:
    idx:           int
    label:         str
    roi:           dict
    background:    Optional[np.ndarray] = None
    state:         _State = _State.IDLE
    peak_activity: float  = 0.0
    peak_flow:     float  = 0.0   # max flow seen during BALL_ENTERING
    entry_frame:   int    = 0
    cooldown_until: int   = 0
    prev_gray_roi: Optional[np.ndarray] = None


class GoalDetectorV2:
    def __init__(
        self,
        rois: List[dict],
        background_frames: int  = 20,
        enter_threshold:   float = 12.0,
        exit_threshold:    float = 6.0,
        min_entry_frames:  int  = 3,
        max_entry_frames:  int  = 20,
        cooldown_frames:   int  = 60,
        smooth_window:     int  = 4,
        flow_threshold:    float = 0.8,
        min_bright_pixels: int  = 8,
        bg_adapt_rate:     float = 0.005,
    ):
        self._trackers = [
            _PocketTracker(idx=i, label=r["label"], roi=r)
            for i, r in enumerate(rois)
        ]
        self._bg_frames     = background_frames
        self._enter_thr     = enter_threshold
        self._exit_thr      = exit_threshold
        self._min_entry     = min_entry_frames
        self._max_entry     = max_entry_frames
        self._cooldown      = cooldown_frames
        self._smooth_window = smooth_window
        self._flow_thr      = flow_threshold
        self._min_bright    = min_bright_pixels
        self._bg_adapt_rate = bg_adapt_rate

        self._bg_buffer: List[List[np.ndarray]] = [[] for _ in rois]
        self._bg_built  = False
        self._activity_windows: List[Deque[float]] = [
            deque(maxlen=smooth_window) for _ in rois
        ]
        self.activity_log: List[List[float]] = [[] for _ in rois]

    # ------------------------------------------------------------------
    def process_frame(self, frame: np.ndarray, frame_id: int) -> List[GoalEvent]:
        events: List[GoalEvent] = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for pt in self._trackers:
            roi_img  = self._crop(frame, pt.roi)
            roi_gray = self._crop(gray, pt.roi)

            # Background accumulation phase
            if not self._bg_built:
                self._bg_buffer[pt.idx].append(roi_img.astype(np.float32))
                if len(self._bg_buffer[pt.idx]) >= self._bg_frames:
                    pt.background = np.median(
                        np.stack(self._bg_buffer[pt.idx]), axis=0
                    ).astype(np.float32)
                pt.prev_gray_roi = roi_gray.copy()
                continue

            if pt.background is None:
                continue

            # Activity score (mean absolute deviation vs background)
            diff         = np.abs(roi_img.astype(np.float32) - pt.background)
            raw_activity = float(diff.mean())

            # Temporal smoothing
            self._activity_windows[pt.idx].append(raw_activity)
            activity = float(np.mean(self._activity_windows[pt.idx]))
            self.activity_log[pt.idx].append(activity)

            # Adaptive background update (only when idle and not in cooldown)
            if pt.state == _State.IDLE and frame_id >= pt.cooldown_until:
                pt.background = (
                    (1.0 - self._bg_adapt_rate) * pt.background
                    + self._bg_adapt_rate * roi_img.astype(np.float32)
                )

            # Pixel-count guard: number of pixels meaningfully above threshold
            bright_pixels = int(np.sum(diff.mean(axis=2) > self._enter_thr))

            if frame_id < pt.cooldown_until:
                pt.prev_gray_roi = roi_gray.copy()
                continue

            # State machine
            if pt.state == _State.IDLE:
                if activity >= self._enter_thr and bright_pixels >= self._min_bright:
                    pt.state         = _State.BALL_ENTERING
                    pt.entry_frame   = frame_id
                    pt.peak_activity = activity
                    pt.peak_flow     = 0.0

            elif pt.state == _State.BALL_ENTERING:
                pt.peak_activity = max(pt.peak_activity, activity)
                # Accumulate peak flow throughout the entry — the ball moves
                # while entering, but may be gone by the time activity exits.
                # Checking flow only at the exit frame misses real goals.
                flow_now = self._optical_flow_magnitude(pt.prev_gray_roi, roi_gray)
                pt.peak_flow = max(pt.peak_flow, flow_now)
                frames_in = frame_id - pt.entry_frame

                if frames_in > self._max_entry:
                    # Took too long — probably a resting ball, not a pocketing
                    pt.state = _State.IDLE

                elif activity < self._exit_thr and frames_in >= self._min_entry:
                    # Activity returned to baseline — confirm with peak flow
                    if pt.peak_flow >= self._flow_thr:
                        confidence = self._compute_confidence(
                            pt.peak_activity, pt.peak_flow, frames_in
                        )
                        events.append(GoalEvent(
                            pocket_idx    = pt.idx,
                            label         = pt.label,
                            frame_id      = frame_id,
                            peak_activity = round(pt.peak_activity, 1),
                            confidence    = round(confidence, 2),
                        ))
                        pt.state           = _State.IDLE
                        pt.cooldown_until  = frame_id + self._cooldown
                    else:
                        # Low flow throughout — shadow, glare, or illumination shift
                        pt.state = _State.IDLE

                elif activity < self._exit_thr:
                    # Dropped too fast — noise spike, not a ball
                    pt.state = _State.IDLE

            pt.prev_gray_roi = roi_gray.copy()

        if not self._bg_built and all(pt.background is not None for pt in self._trackers):
            self._bg_built = True
            print(f"    [GoalDetectorV2] Background model built ({self._bg_frames} frames)")

        return events

    # ------------------------------------------------------------------
    @staticmethod
    def _optical_flow_magnitude(
        prev_gray: Optional[np.ndarray], curr_gray: np.ndarray
    ) -> float:
        if prev_gray is None or prev_gray.shape != curr_gray.shape:
            return 0.0
        h, w = curr_gray.shape
        if h < 8 or w < 8:
            return 0.0
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=2, winsize=7,
            iterations=2, poly_n=5, poly_sigma=1.1, flags=0,
        )
        return float(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).mean())

    @staticmethod
    def _compute_confidence(
        peak_activity: float, flow_mag: float, frames_in: int
    ) -> float:
        act_score  = min(peak_activity / 40.0, 1.0)
        flow_score = min(flow_mag / 3.0, 1.0)
        return (act_score + flow_score) / 2.0

    @staticmethod
    def _crop(frame: np.ndarray, roi: dict) -> np.ndarray:
        x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
        fh, fw     = frame.shape[:2]
        x1, y1     = max(0, x),     max(0, y)
        x2, y2     = min(fw, x + w), min(fh, y + h)
        return frame[y1:y2, x1:x2]
