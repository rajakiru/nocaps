"""
Trajectory builder: appends positions to tracks and computes smoothed velocities.

For each active track on each frame:
  1. Append (frame_id, cx, cy) to track.positions
  2. Compute instantaneous velocity from the last two positions
  3. Apply a moving-average smoothing window to velocities

Rolling window keeps only the last N frames in memory for streaming suitability.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Tuple

import numpy as np

from .tracker import Track


class TrajectoryBuilder:
    """
    Maintains per-track rolling windows of positions and velocities.

    Parameters
    ----------
    window : int
        Number of recent frames to keep per track (controls smoothing).
    smooth_k : int
        Moving-average kernel size for velocity smoothing (must be odd, >= 1).
    fps : float
        Video frame rate — used to express velocity in px/s.
    """

    def __init__(self, window: int = 30, smooth_k: int = 5, fps: float = 30.0):
        self.window = window
        self.smooth_k = smooth_k
        self.fps = fps
        # {track_id: deque of (frame_id, cx, cy)}
        self._pos_windows: Dict[int, deque] = {}

    def update(self, active_tracks: List[Track], frame_id: int) -> None:
        """
        Called every frame with the list of currently active tracks.
        Appends position data and recomputes velocities for each track.
        """
        for track in active_tracks:
            tid = track.id

            if tid not in self._pos_windows:
                self._pos_windows[tid] = deque(maxlen=self.window)

            window = self._pos_windows[tid]
            window.append((frame_id, track.cx, track.cy))

            # Mirror to track.positions (full history for event detection)
            track.positions.append((frame_id, track.cx, track.cy))

            # Recompute velocities from the rolling window
            track.velocities = self._compute_velocities(window)

    def _compute_velocities(
        self, window: deque
    ) -> List[Tuple[int, float, float]]:
        """
        Compute per-frame velocities (vx, vy) in px/s using finite differences,
        then apply a moving-average smoothing.
        """
        pts = list(window)
        if len(pts) < 2:
            return []

        raw: List[Tuple[int, float, float]] = []
        for i in range(1, len(pts)):
            f0, x0, y0 = pts[i - 1]
            f1, x1, y1 = pts[i]
            dt = (f1 - f0) / self.fps
            if dt <= 0:
                continue
            vx = (x1 - x0) / dt
            vy = (y1 - y0) / dt
            raw.append((f1, vx, vy))

        if not raw:
            return []

        return self._smooth(raw, self.smooth_k)

    @staticmethod
    def _smooth(
        velocities: List[Tuple[int, float, float]], k: int
    ) -> List[Tuple[int, float, float]]:
        """Apply a simple moving-average of width k to (vx, vy) sequences."""
        if k <= 1 or len(velocities) < k:
            return velocities

        half = k // 2
        result: List[Tuple[int, float, float]] = []
        frames = [v[0] for v in velocities]
        vxs = np.array([v[1] for v in velocities], dtype=float)
        vys = np.array([v[2] for v in velocities], dtype=float)

        kernel = np.ones(k) / k
        vxs_smooth = np.convolve(vxs, kernel, mode="same")
        vys_smooth = np.convolve(vys, kernel, mode="same")

        # Edges are biased by zero-padding in "same" mode; trim them
        for i in range(half, len(frames) - half):
            result.append((frames[i], float(vxs_smooth[i]), float(vys_smooth[i])))

        return result if result else velocities

    # ------------------------------------------------------------------
    # Helpers for event detection
    # ------------------------------------------------------------------

    def speed(self, track: Track) -> float:
        """Current speed (px/s) — magnitude of the most recent velocity."""
        if not track.velocities:
            return 0.0
        _, vx, vy = track.velocities[-1]
        return float(np.hypot(vx, vy))

    def recent_speeds(self, track: Track, n: int = 10) -> List[float]:
        """Last n speed values for the track."""
        return [float(np.hypot(vx, vy)) for _, vx, vy in track.velocities[-n:]]

    def mean_speed(self, track: Track, n: int = 20) -> float:
        """Mean speed over last n velocity samples."""
        speeds = self.recent_speeds(track, n)
        return float(np.mean(speeds)) if speeds else 0.0
