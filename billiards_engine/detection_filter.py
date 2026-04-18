"""
Domain-specific filtering for billiards detections and tracks.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .detection_loader import Detection
from .tracker import Track
from .trajectory_builder import TrajectoryBuilder

MIN_CONFIDENCE = 0.12
MAX_ASPECT_RATIO = 1.35
RADIUS_LOW_FACTOR = 0.60
RADIUS_HIGH_FACTOR = 1.60
TABLE_EDGE_MARGIN = 16
POCKET_SUPPRESS_FACTOR = 0.45
MIN_TRACK_AGE_FOR_GOAL = 3
STATIONARY_SPEED_PX = 60.0  # px/s
STATIONARY_FRAMES_NEAR_POCKET = 15
NEAR_POCKET_FACTOR = 1.10
RECENT_MOTION_WINDOW = 12
RECENT_MOTION_TRIGGER_PX = 140.0


def estimate_expected_ball_radius(
    detections: Sequence[Detection],
    previous_radius: Optional[float] = None,
) -> Optional[float]:
    radii = [det.radius for det in detections if det.confidence >= MIN_CONFIDENCE]
    if len(radii) < 4:
        return previous_radius
    median = float(np.median(radii))
    if previous_radius is None:
        return median
    return 0.7 * previous_radius + 0.3 * median


def filter_detections(
    detections: Sequence[Detection],
    table_bbox: Tuple[int, int, int, int],
    rois: Sequence[dict],
    expected_radius: Optional[float],
) -> List[Detection]:
    tx, ty, tw, th = table_bbox
    filtered: List[Detection] = []

    for det in detections:
        if det.confidence < MIN_CONFIDENCE:
            continue

        aspect = max(det.w / max(det.h, 1e-6), det.h / max(det.w, 1e-6))
        if aspect > MAX_ASPECT_RATIO:
            continue

        if expected_radius is not None:
            if not (RADIUS_LOW_FACTOR * expected_radius <= det.radius <= RADIUS_HIGH_FACTOR * expected_radius):
                continue

        if tx + TABLE_EDGE_MARGIN < tx + tw - TABLE_EDGE_MARGIN:
            if det.cx < tx + TABLE_EDGE_MARGIN or det.cx > tx + tw - TABLE_EDGE_MARGIN:
                continue
        if ty + TABLE_EDGE_MARGIN < ty + th - TABLE_EDGE_MARGIN:
            if det.cy < ty + TABLE_EDGE_MARGIN or det.cy > ty + th - TABLE_EDGE_MARGIN:
                continue

        if _inside_pocket_suppression(det.cx, det.cy, rois):
            continue

        filtered.append(det)

    return filtered


def filter_tracks_for_events(
    tracks: Sequence[Track],
    rois: Sequence[dict],
    traj: TrajectoryBuilder,
) -> List[Track]:
    eligible: List[Track] = []
    for track in tracks:
        if len(track.positions) < MIN_TRACK_AGE_FOR_GOAL:
            continue
        if _stationary_near_pocket(track, rois, traj):
            continue
        eligible.append(track)
    return eligible


def _inside_pocket_suppression(cx: float, cy: float, rois: Sequence[dict]) -> bool:
    for roi in rois:
        limit = roi["radius"] * 0.28
        if np.hypot(cx - roi["cx"], cy - roi["cy"]) <= limit:
            return True
    return False


def _stationary_near_pocket(track: Track, rois: Sequence[dict], traj: TrajectoryBuilder) -> bool:
    if len(track.positions) < STATIONARY_FRAMES_NEAR_POCKET:
        return False

    recent_positions = track.positions[-STATIONARY_FRAMES_NEAR_POCKET:]
    mean_speed = traj.mean_speed(track, n=min(20, len(track.velocities)))
    if mean_speed > STATIONARY_SPEED_PX:
        return False
    if traj.max_recent_speed(track, n=RECENT_MOTION_WINDOW) >= RECENT_MOTION_TRIGGER_PX:
        return False

    for roi in rois:
        limit = roi["radius"] * NEAR_POCKET_FACTOR
        if all(np.hypot(x - roi["cx"], y - roi["cy"]) <= limit for _, x, y in recent_positions):
            return True
    return False
