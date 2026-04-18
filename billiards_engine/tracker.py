"""
Centroid tracker with nearest-neighbor ID assignment.

Matches new detections to existing tracks based on Euclidean distance.
Handles:
  - New balls (first appearance)
  - Lost balls (disappeared for > max_missing frames)
  - Category inheritance from annotated frames
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .detection_loader import Detection


@dataclass
class Track:
    """One tracked ball across multiple frames."""
    id: int
    category: int                          # 1=cue,2=8ball,3=solid,4=striped,0=unknown
    last_seen: int                         # frame_id
    cx: float
    cy: float
    # Raw positions (frame_id, cx, cy) — appended by TrajectoryBuilder
    positions: List[Tuple[int, float, float]] = field(default_factory=list)
    # Instantaneous velocities (frame_id, vx, vy) — appended by TrajectoryBuilder
    velocities: List[Tuple[int, float, float]] = field(default_factory=list)
    missing_frames: int = 0
    radius: float = 0.0
    confidence: float = 1.0
    near_pocket_idx: Optional[int] = None
    pocket_grace_frames: int = 0

    def is_cue_ball(self) -> bool:
        return self.category == 1

    @property
    def visible(self) -> bool:
        return self.missing_frames == 0


class CentroidTracker:
    """
    Greedy nearest-neighbor tracker.

    Parameters
    ----------
    max_distance : float
        Max pixel distance allowed when matching a detection to a track.
    max_missing : int
        Frames a track can go undetected before being marked lost.
    """

    def __init__(
        self,
        max_distance: float = 50.0,
        max_missing: int = 10,
        rois: Optional[List[dict]] = None,
        near_pocket_margin: float = 24.0,
        pocket_bonus_missing: int = 6,
    ):
        self.max_distance = max_distance
        self.max_missing = max_missing
        self.rois = rois or []
        self.near_pocket_margin = near_pocket_margin
        self.pocket_bonus_missing = pocket_bonus_missing
        self._next_id = 1
        self._tracks: Dict[int, Track] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def seed(self, detections: List[Detection], frame_id: int):
        """
        Seed (or update) tracks from annotated detections (first/last frame).
        Detections carry stable ball_id values that the tracker preserves.
        """
        for det in detections:
            if det.ball_id in self._tracks:
                t = self._tracks[det.ball_id]
                t.cx, t.cy = det.cx, det.cy
                t.last_seen = frame_id
                t.missing_frames = 0
                t.radius = det.radius
                t.confidence = det.confidence
                self._refresh_pocket_context(t)
                if det.category != 0:
                    t.category = det.category
            else:
                track = Track(
                    id=det.ball_id,
                    category=det.category,
                    last_seen=frame_id,
                    cx=det.cx,
                    cy=det.cy,
                    radius=det.radius,
                    confidence=det.confidence,
                )
                self._refresh_pocket_context(track)
                self._tracks[det.ball_id] = track
                self._next_id = max(self._next_id, det.ball_id + 1)

    def update(self, detections: List[Detection], frame_id: int) -> List[Track]:
        """
        Match detections to existing tracks, create new tracks for unmatched
        detections, and age unmatched tracks.

        Returns list of all *active* tracks after the update.
        """
        active_ids = [tid for tid, t in self._tracks.items() if t.missing_frames <= self._effective_max_missing(t)]

        if not detections:
            for tid in active_ids:
                self._age_track(self._tracks[tid])
            return self._active_tracks()

        if not active_ids:
            for det in detections:
                self._create_track(det, frame_id)
            return self._active_tracks()

        # Build cost matrix (active tracks × detections)
        track_list = [self._tracks[tid] for tid in active_ids]
        det_centers = np.array([[d.cx, d.cy] for d in detections], dtype=float)
        track_centers = np.array([[t.cx, t.cy] for t in track_list], dtype=float)

        # Shape: (n_tracks, n_dets)
        diff = track_centers[:, np.newaxis, :] - det_centers[np.newaxis, :, :]
        cost = np.linalg.norm(diff, axis=2)

        matched_track_idxs = set()
        matched_det_idxs = set()

        # Greedy: always pick the smallest cost pair
        flat_order = np.argsort(cost.ravel())
        for flat_idx in flat_order:
            ti, di = divmod(int(flat_idx), len(detections))
            if ti in matched_track_idxs or di in matched_det_idxs:
                continue
            if cost[ti, di] > self.max_distance:
                break  # sorted — remaining are all worse
            track = track_list[ti]
            det = detections[di]
            track.cx, track.cy = det.cx, det.cy
            track.last_seen = frame_id
            track.missing_frames = 0
            track.radius = det.radius
            track.confidence = det.confidence
            self._refresh_pocket_context(track)
            if det.category != 0:
                track.category = det.category
            matched_track_idxs.add(ti)
            matched_det_idxs.add(di)

        # Age unmatched tracks
        for ti, track in enumerate(track_list):
            if ti not in matched_track_idxs:
                self._age_track(track)

        # Create new tracks for unmatched detections
        for di, det in enumerate(detections):
            if di not in matched_det_idxs:
                self._create_track(det, frame_id)

        return self._active_tracks()

    def get_track(self, track_id: int) -> Optional[Track]:
        return self._tracks.get(track_id)

    def all_tracks(self) -> List[Track]:
        return list(self._tracks.values())

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _create_track(self, det: Detection, frame_id: int) -> Track:
        tid = self._next_id
        self._next_id += 1
        track = Track(
            id=tid,
            category=det.category,
            last_seen=frame_id,
            cx=det.cx,
            cy=det.cy,
            radius=det.radius,
            confidence=det.confidence,
        )
        self._refresh_pocket_context(track)
        self._tracks[tid] = track
        return track

    def _active_tracks(self) -> List[Track]:
        return [t for t in self._tracks.values() if t.missing_frames <= self._effective_max_missing(t)]

    def _age_track(self, track: Track) -> None:
        track.missing_frames += 1
        if track.pocket_grace_frames > 0:
            track.pocket_grace_frames -= 1

    def _effective_max_missing(self, track: Track) -> int:
        return self.max_missing + track.pocket_grace_frames

    def _refresh_pocket_context(self, track: Track) -> None:
        pocket_idx = self._near_pocket_idx(track.cx, track.cy)
        track.near_pocket_idx = pocket_idx
        if pocket_idx is None:
            track.pocket_grace_frames = 0
            return
        track.pocket_grace_frames = self.pocket_bonus_missing

    def _near_pocket_idx(self, cx: float, cy: float) -> Optional[int]:
        best_idx: Optional[int] = None
        best_distance = float("inf")
        for idx, roi in enumerate(self.rois):
            limit = roi["radius"] + self.near_pocket_margin
            distance = float(np.hypot(cx - roi["cx"], cy - roi["cy"]))
            if distance <= limit and distance < best_distance:
                best_idx = idx
                best_distance = distance
        return best_idx
