"""
Fused pocket detector that combines ROI activity with track disappearance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Set

import numpy as np

from .tracker import Track

RECENT_MOTION_MIN_SAMPLES = 4
RECENT_MOTION_TRIGGER_PX = 140.0
ARM_MEMORY_FRAMES = 18
ARM_LINGER_FRAMES = 8
ACTIVITY_BASELINE_WINDOW = 24
MIN_ACTIVITY_RISE = 3.5
MIN_ACTIVITY_RATIO = 1.35
RECENT_APPROACH_SAMPLES = 4
MIN_APPROACH_DELTA_PX = 6.0
RECENT_APPROACH_SPEED_PX = 180.0
STABLE_POCKET_LINGER_FRAMES = 12
STABLE_POCKET_MAX_DISTANCE_FACTOR = 0.62
NEARBY_TRACK_RADIUS_FACTOR = 0.72
REJECTED_TRACK_COOLDOWN_FRAMES = 150
MOVING_START_DISTANCE_FACTOR = 0.82
STATIONARY_START_DISTANCE_FACTOR = 0.62
ARMED_START_DISTANCE_FACTOR = 0.70
STRONG_DISAPPEAR_DISTANCE_FACTOR = 0.70
STRONG_DISAPPEAR_PEAK_ACTIVITY = 10.0
REPLACEMENT_DISTANCE_GAIN_PX = 6.0
REPLACEMENT_MIN_AGE_FRAMES = 5
IMPACT_SUPPORT_SPEED_PX = 180.0
PREDICT_LOOKAHEAD_FRAMES = 2.0
RECENT_MISSING_SEED_FRAMES = 6
RECENT_MISSING_SPEED_PX = 90.0
RECENT_MISSING_START_DISTANCE_FACTOR = 1.10
FAST_NEAR_POCKET_SPEED_PX = 90.0
FAST_NEAR_POCKET_START_DISTANCE_FACTOR = 1.18
FAST_ARM_MIN_FRAMES = 2


@dataclass
class ArmedPocketTrack:
    pocket_idx: int
    track_id: int
    last_seen_frame: int
    frames_near_pocket: int = 1
    min_distance_px: float = 1e9


@dataclass
class PocketEventCandidate:
    pocket_idx: int
    label: str
    track_id: int
    start_frame: int
    end_frame: int
    last_visible_frame: int
    nearby_track_ids: List[int] = field(default_factory=list)
    peak_activity: float = 0.0
    last_activity: float = 0.0
    min_distance_px: float = 1e9
    impact_seen: bool = False
    status: str = "active"
    reject_reason: Optional[str] = None

    def to_json(self) -> dict:
        data = asdict(self)
        data["peak_activity"] = round(float(self.peak_activity), 3)
        data["last_activity"] = round(float(self.last_activity), 3)
        data["min_distance_px"] = round(float(self.min_distance_px), 3)
        return data


@dataclass
class PocketEvent:
    pocket_idx: int
    label: str
    frame_id: int
    time_s: float
    track_id: int
    confidence: float
    peak_activity: float
    roi_activity: float
    nearby_track_ids: List[int]

    def to_json(self) -> dict:
        return {
            "pocket": self.label,
            "pocket_idx": self.pocket_idx,
            "frame": self.frame_id,
            "time_s": round(float(self.time_s), 3),
            "track_id": self.track_id,
            "confidence": round(float(self.confidence), 3),
            "peak_activity": round(float(self.peak_activity), 3),
            "roi_activity": round(float(self.roi_activity), 3),
            "nearby_track_ids": self.nearby_track_ids,
        }


class FusedPocketDetector:
    def __init__(
        self,
        rois: List[dict],
        fps: float,
        background_frames: int = 15,
        enter_threshold: float = 4.0,
        exit_threshold: float = 3.0,
        near_pocket_margin: float = 30.0,
        min_candidate_frames: int = 3,
        max_candidate_frames: int = 15,
        confirm_missing_frames: int = 3,
        reappear_window_frames: int = 5,
        cooldown_frames: int = 90,
        min_track_age_frames: int = 3,
        min_peak_activity: float = 5.0,
        debug_pockets: Optional[List[str]] = None,
    ):
        self.rois = rois
        self.fps = fps
        self._bg_frames = background_frames
        self._enter_thr = enter_threshold
        self._exit_thr = exit_threshold
        self._near_margin = near_pocket_margin
        self._min_candidate = min_candidate_frames
        self._max_candidate = max_candidate_frames
        self._confirm_missing = confirm_missing_frames
        self._reappear_window = reappear_window_frames
        self._cooldown = cooldown_frames
        self._min_track_age = min_track_age_frames
        self._min_peak_activity = min_peak_activity
        self._bg_buffer: List[List[np.ndarray]] = [[] for _ in rois]
        self._backgrounds: List[Optional[np.ndarray]] = [None for _ in rois]
        self._bg_built = False
        self._pocket_cooldown_until: Dict[int, int] = {}
        self.activity_log: List[List[float]] = [[] for _ in rois]
        self.candidates: List[PocketEventCandidate] = []
        self.rejected_candidates: List[PocketEventCandidate] = []
        self._active_candidates: Dict[int, PocketEventCandidate] = {}
        self._armed_tracks: Dict[int, ArmedPocketTrack] = {}
        self._track_cooldown_until: Dict[tuple[int, int], int] = {}
        self._debug_pockets = {self._norm_label(label) for label in (debug_pockets or [])}
        self._debug_log: Dict[int, List[dict]] = {
            idx: [] for idx, roi in enumerate(rois) if self._should_debug(idx)
        }

    def process_frame(
        self,
        frame: np.ndarray,
        frame_id: int,
        visible_tracks: List[Track],
        all_tracks: List[Track],
    ) -> List[PocketEvent]:
        events: List[PocketEvent] = []
        tracks_by_id = {track.id: track for track in all_tracks}
        visible_all_tracks = [track for track in all_tracks if track.missing_frames <= self._confirm_missing]

        for pocket_idx, roi in enumerate(self.rois):
            roi_img = self._crop(frame, roi)
            if not self._bg_built:
                self._bg_buffer[pocket_idx].append(roi_img.astype(np.float32))
                if len(self._bg_buffer[pocket_idx]) >= self._bg_frames:
                    self._backgrounds[pocket_idx] = np.median(
                        np.stack(self._bg_buffer[pocket_idx]), axis=0
                    ).astype(np.float32)
                self._log_debug(
                    pocket_idx,
                    frame_id,
                    action="build_background",
                    extra={
                        "buffer_frames": len(self._bg_buffer[pocket_idx]),
                        "background_ready": self._backgrounds[pocket_idx] is not None,
                    },
                )
                continue

            background = self._backgrounds[pocket_idx]
            if background is None:
                continue

            diff = np.abs(roi_img.astype(np.float32) - background)
            activity = float(diff.mean())
            self.activity_log[pocket_idx].append(activity)
            activity_rise = self._activity_rise(pocket_idx)

            candidate = self._active_candidates.get(pocket_idx)
            nearby_visible = self._nearby_tracks(visible_tracks, roi)
            nearby_all_visible = self._nearby_tracks(visible_all_tracks, roi)
            self._update_armed_track(pocket_idx, frame_id, nearby_all_visible)

            if candidate is not None:
                replacement = self._replacement_seed(
                    pocket_idx=pocket_idx,
                    frame_id=frame_id,
                    candidate=candidate,
                    nearby_visible=nearby_visible,
                    tracks_by_id=tracks_by_id,
                    activity=activity,
                    activity_rise=activity_rise,
                )
                if replacement is not None:
                    self._reject_candidate(pocket_idx, "replaced_by_stronger_track")
                    lead_track, distance = replacement
                    candidate = self._start_candidate(
                        pocket_idx=pocket_idx,
                        frame_id=frame_id,
                        lead_track=lead_track,
                        distance=distance,
                        activity=activity,
                        nearby_visible=nearby_visible,
                    )
                    self._log_debug(
                        pocket_idx,
                        frame_id,
                        action="replace_candidate",
                        activity=activity,
                        activity_rise=activity_rise,
                        candidate=candidate,
                        nearby_visible=nearby_visible,
                        extra={
                            "selected_track_id": lead_track.id,
                            "selected_distance_px": round(float(distance), 3),
                        },
                    )
                    if candidate is not None:
                        continue

            if candidate is None:
                if frame_id < self._pocket_cooldown_until.get(pocket_idx, -1):
                    self._log_debug(
                        pocket_idx,
                        frame_id,
                        action="skip_start",
                        activity=activity,
                        activity_rise=activity_rise,
                        nearby_visible=nearby_visible,
                        extra={"reason": "pocket_cooldown"},
                    )
                    continue
                impact_support = self._has_impact_support(nearby_visible)
                can_start = activity >= self._enter_thr and (
                    self._has_activity_jump(activity, activity_rise) or impact_support
                )
                if not can_start:
                    self._log_debug(
                        pocket_idx,
                        frame_id,
                        action="skip_start",
                        activity=activity,
                        activity_rise=activity_rise,
                        nearby_visible=nearby_visible,
                        extra={
                            "reason": "start_conditions_not_met",
                            "impact_support": impact_support,
                        },
                    )
                    continue
                if can_start:
                    seeded = self._seed_candidate_track(
                        pocket_idx=pocket_idx,
                        frame_id=frame_id,
                        nearby_visible=nearby_visible,
                        tracks_by_id=tracks_by_id,
                        impact_support=impact_support,
                    )
                    if seeded is None:
                        self._log_debug(
                            pocket_idx,
                            frame_id,
                            action="skip_start",
                            activity=activity,
                            activity_rise=activity_rise,
                            nearby_visible=nearby_visible,
                            extra={
                                "reason": "no_seed_track",
                                "impact_support": impact_support,
                            },
                        )
                        continue
                    lead_track, distance = seeded
                    if (
                        frame_id < self._track_cooldown_until.get((pocket_idx, lead_track.id), -1)
                        and not impact_support
                    ):
                        self._log_debug(
                            pocket_idx,
                            frame_id,
                            action="skip_start",
                            activity=activity,
                            activity_rise=activity_rise,
                            nearby_visible=nearby_visible,
                            extra={
                                "reason": "track_cooldown",
                                "selected_track_id": lead_track.id,
                                "impact_support": impact_support,
                            },
                        )
                        continue
                    candidate = self._start_candidate(
                        pocket_idx=pocket_idx,
                        frame_id=frame_id,
                        lead_track=lead_track,
                        distance=distance,
                        activity=activity,
                        nearby_visible=nearby_visible,
                        impact_seen=impact_support,
                    )
                    self._log_debug(
                        pocket_idx,
                        frame_id,
                        action="start_candidate",
                        activity=activity,
                        activity_rise=activity_rise,
                        candidate=candidate,
                        nearby_visible=nearby_visible,
                        extra={
                            "selected_track_id": lead_track.id,
                            "selected_distance_px": round(float(distance), 3),
                            "impact_support": impact_support,
                        },
                    )
                continue

            candidate.end_frame = frame_id
            candidate.peak_activity = max(candidate.peak_activity, activity)
            candidate.last_activity = activity
            candidate.impact_seen = candidate.impact_seen or self._has_impact_support(
                nearby_visible,
                exclude_track_id=candidate.track_id,
            )
            candidate.nearby_track_ids = sorted(
                set(candidate.nearby_track_ids) | {track.id for track, _ in nearby_visible}
            )
            if nearby_visible:
                candidate.min_distance_px = min(
                    candidate.min_distance_px,
                    min(distance for _, distance in nearby_visible),
                )

            track = tracks_by_id.get(candidate.track_id)
            if track and track.visible:
                candidate.last_visible_frame = frame_id

            age = frame_id - candidate.start_frame
            if age > self._max_candidate:
                self._log_debug(
                    pocket_idx,
                    frame_id,
                    action="reject_candidate",
                    activity=activity,
                    activity_rise=activity_rise,
                    candidate=candidate,
                    nearby_visible=nearby_visible,
                    extra={"reason": "timed_out"},
                )
                self._reject_candidate(pocket_idx, "timed_out")
                continue

            if track and track.missing_frames >= self._confirm_missing and age >= self._min_candidate:
                if candidate.peak_activity < self._min_peak_activity:
                    self._log_debug(
                        pocket_idx,
                        frame_id,
                        action="reject_candidate",
                        activity=activity,
                        activity_rise=activity_rise,
                        candidate=candidate,
                        nearby_visible=nearby_visible,
                        extra={"reason": "weak_activity"},
                    )
                    self._reject_candidate(pocket_idx, "weak_activity")
                    continue
                if not self._has_confirming_motion(candidate, track, pocket_idx):
                    self._log_debug(
                        pocket_idx,
                        frame_id,
                        action="reject_candidate",
                        activity=activity,
                        activity_rise=activity_rise,
                        candidate=candidate,
                        nearby_visible=nearby_visible,
                        extra={"reason": "no_confirming_motion"},
                    )
                    self._reject_candidate(pocket_idx, "no_confirming_motion")
                    continue
                if self._track_reappeared_near_pocket(candidate, visible_tracks):
                    self._log_debug(
                        pocket_idx,
                        frame_id,
                        action="reject_candidate",
                        activity=activity,
                        activity_rise=activity_rise,
                        candidate=candidate,
                        nearby_visible=nearby_visible,
                        extra={"reason": "track_reappeared"},
                    )
                    self._reject_candidate(pocket_idx, "track_reappeared")
                    continue
                if frame_id - candidate.last_visible_frame <= self._reappear_window:
                    if activity <= (candidate.peak_activity + self._exit_thr) / 2.0:
                        events.append(self._confirm_candidate(candidate, frame_id))
                        self._log_debug(
                            pocket_idx,
                            frame_id,
                            action="confirm_candidate",
                            activity=activity,
                            activity_rise=activity_rise,
                            candidate=candidate,
                            nearby_visible=nearby_visible,
                            extra={"reason": "missing_and_settled"},
                        )
                        del self._active_candidates[pocket_idx]
                        self._pocket_cooldown_until[pocket_idx] = frame_id + self._cooldown
                        continue

            invisible_for = frame_id - candidate.last_visible_frame
            if (
                age >= self._min_candidate
                and invisible_for >= self._reappear_window
                and activity <= (candidate.peak_activity + self._exit_thr) / 2.0
            ):
                if self._track_reappeared_near_pocket(candidate, visible_tracks):
                    self._log_debug(
                        pocket_idx,
                        frame_id,
                        action="reject_candidate",
                        activity=activity,
                        activity_rise=activity_rise,
                        candidate=candidate,
                        nearby_visible=nearby_visible,
                        extra={"reason": "track_reappeared"},
                    )
                    self._reject_candidate(pocket_idx, "track_reappeared")
                    continue
                if self._has_strong_disappearance_signature(candidate):
                    events.append(self._confirm_candidate(candidate, frame_id))
                    self._log_debug(
                        pocket_idx,
                        frame_id,
                        action="confirm_candidate",
                        activity=activity,
                        activity_rise=activity_rise,
                        candidate=candidate,
                        nearby_visible=nearby_visible,
                        extra={"reason": "strong_disappearance"},
                    )
                    del self._active_candidates[pocket_idx]
                    self._pocket_cooldown_until[pocket_idx] = frame_id + self._cooldown
                    continue

            if age >= self._min_candidate and activity < self._exit_thr:
                if track and track.visible:
                    self._log_debug(
                        pocket_idx,
                        frame_id,
                        action="reject_candidate",
                        activity=activity,
                        activity_rise=activity_rise,
                        candidate=candidate,
                        nearby_visible=nearby_visible,
                        extra={"reason": "ball_still_visible"},
                    )
                    self._reject_candidate(pocket_idx, "ball_still_visible")
                elif not nearby_visible:
                    self._log_debug(
                        pocket_idx,
                        frame_id,
                        action="reject_candidate",
                        activity=activity,
                        activity_rise=activity_rise,
                        candidate=candidate,
                        nearby_visible=nearby_visible,
                        extra={"reason": "weak_activity"},
                    )
                    self._reject_candidate(pocket_idx, "weak_activity")
                continue

            self._log_debug(
                pocket_idx,
                frame_id,
                action="track_candidate",
                activity=activity,
                activity_rise=activity_rise,
                candidate=candidate,
                nearby_visible=nearby_visible,
            )

        if not self._bg_built and all(background is not None for background in self._backgrounds):
            self._bg_built = True
            print(f"    Fused detector background model built ({self._bg_frames} frames)")

        return events

    def activity_summary(self) -> List[dict]:
        summary = []
        for pocket_idx, roi in enumerate(self.rois):
            values = self.activity_log[pocket_idx]
            summary.append({
                "pocket": roi["label"],
                "num_frames": len(values),
                "mean_activity": round(float(np.mean(values)), 3) if values else 0.0,
                "max_activity": round(float(np.max(values)), 3) if values else 0.0,
                "activity_series": [round(float(value), 3) for value in values],
            })
        return summary

    def candidate_summary(self) -> List[dict]:
        return [candidate.to_json() for candidate in self.candidates]

    def rejected_summary(self) -> List[dict]:
        return [candidate.to_json() for candidate in self.rejected_candidates]

    def debug_summary(self) -> Dict[str, List[dict]]:
        return {
            self.rois[pocket_idx]["label"]: rows
            for pocket_idx, rows in self._debug_log.items()
        }

    @staticmethod
    def _crop(frame: np.ndarray, roi: dict) -> np.ndarray:
        x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
        fh, fw = frame.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(fw, x + w), min(fh, y + h)
        return frame[y1:y2, x1:x2]

    def _nearby_tracks(self, tracks: List[Track], roi: dict) -> List[tuple[Track, float]]:
        cx, cy = roi["cx"], roi["cy"]
        limit = roi["radius"] * NEARBY_TRACK_RADIUS_FACTOR + self._near_margin
        nearby = []
        for track in tracks:
            distance = self._effective_track_distance(track, roi)
            if distance <= limit:
                nearby.append((track, distance))
        nearby.sort(key=lambda item: item[1])
        return nearby

    def _track_reappeared_near_pocket(
        self,
        candidate: PocketEventCandidate,
        visible_tracks: List[Track],
    ) -> bool:
        roi = self.rois[candidate.pocket_idx]
        nearby_ids: Set[int] = {track.id for track, _ in self._nearby_tracks(visible_tracks, roi)}
        return candidate.track_id in nearby_ids

    def _activity_rise(self, pocket_idx: int) -> float:
        values = self.activity_log[pocket_idx]
        if len(values) <= 1:
            return 0.0
        history = values[:-1]
        if not history:
            return 0.0
        baseline = float(np.mean(history[-ACTIVITY_BASELINE_WINDOW:]))
        return values[-1] - baseline

    @staticmethod
    def _has_activity_jump(activity: float, rise: float) -> bool:
        if rise < MIN_ACTIVITY_RISE:
            return False
        baseline = max(activity - rise, 1e-6)
        return activity / baseline >= MIN_ACTIVITY_RATIO

    def _update_armed_track(
        self,
        pocket_idx: int,
        frame_id: int,
        nearby_tracks: List[tuple[Track, float]],
    ) -> None:
        armed = self._armed_tracks.get(pocket_idx)
        if not nearby_tracks:
            if armed and frame_id - armed.last_seen_frame > ARM_MEMORY_FRAMES:
                del self._armed_tracks[pocket_idx]
            return

        if armed:
            for track, distance in nearby_tracks:
                if track.id == armed.track_id:
                    armed.last_seen_frame = frame_id
                    armed.frames_near_pocket += 1
                    armed.min_distance_px = min(armed.min_distance_px, distance)
                    return
            if frame_id - armed.last_seen_frame <= ARM_MEMORY_FRAMES:
                return

        lead_track, distance = nearby_tracks[0]
        if armed and armed.track_id == lead_track.id:
            armed.last_seen_frame = frame_id
            armed.frames_near_pocket += 1
            armed.min_distance_px = min(armed.min_distance_px, distance)
            return

        self._armed_tracks[pocket_idx] = ArmedPocketTrack(
            pocket_idx=pocket_idx,
            track_id=lead_track.id,
            last_seen_frame=frame_id,
            frames_near_pocket=1,
            min_distance_px=distance,
        )

    def _seed_candidate_track(
        self,
        pocket_idx: int,
        frame_id: int,
        nearby_visible: List[tuple[Track, float]],
        tracks_by_id: Dict[int, Track],
        impact_support: bool = False,
    ) -> Optional[tuple[Track, float]]:
        armed = self._armed_tracks.get(pocket_idx)
        if armed is not None and impact_support:
            if frame_id - armed.last_seen_frame <= ARM_MEMORY_FRAMES:
                track = tracks_by_id.get(armed.track_id)
                if track is not None and len(track.positions) >= self._min_track_age:
                    required_frames = (
                        FAST_ARM_MIN_FRAMES
                        if self._is_fast_near_pocket_track(track, pocket_idx)
                        else ARM_LINGER_FRAMES
                    )
                    if armed.frames_near_pocket < required_frames:
                        pass
                    else:
                        moving = self._has_recent_motion(track)
                        distance = min(armed.min_distance_px, self._effective_track_distance(track, self.rois[pocket_idx]))
                        if self._within_dynamic_start_zone(
                            pocket_idx,
                            track,
                            distance,
                            moving=moving,
                            armed=True,
                        ):
                            return track, distance

        if nearby_visible:
            lead_track, distance = nearby_visible[0]
            if (
                len(lead_track.positions) >= self._min_track_age
                and (
                    self._has_recent_motion(lead_track)
                    or self._is_fast_near_pocket_track(lead_track, pocket_idx)
                )
                and self._within_dynamic_start_zone(
                    pocket_idx,
                    lead_track,
                    distance,
                    moving=True,
                )
            ):
                return lead_track, distance

        recent_missing = self._recent_missing_seed(
            pocket_idx=pocket_idx,
            tracks_by_id=tracks_by_id,
        )
        if recent_missing is not None:
            return recent_missing

        if armed is None:
            return None
        if frame_id - armed.last_seen_frame > ARM_MEMORY_FRAMES:
            return None

        track = tracks_by_id.get(armed.track_id)
        if track is None or len(track.positions) < self._min_track_age:
            return None
        required_frames = (
            FAST_ARM_MIN_FRAMES
            if self._is_fast_near_pocket_track(track, pocket_idx)
            else ARM_LINGER_FRAMES
        )
        if armed.frames_near_pocket < required_frames:
            return None
        moving = self._has_recent_motion(track)
        distance = min(armed.min_distance_px, self._effective_track_distance(track, self.rois[pocket_idx]))
        if not self._within_dynamic_start_zone(
            pocket_idx,
            track,
            distance,
            moving=moving,
            armed=True,
        ):
            return None
        return track, distance

    def _replacement_seed(
        self,
        pocket_idx: int,
        frame_id: int,
        candidate: PocketEventCandidate,
        nearby_visible: List[tuple[Track, float]],
        tracks_by_id: Dict[int, Track],
        activity: float,
        activity_rise: float,
    ) -> Optional[tuple[Track, float]]:
        if frame_id - candidate.start_frame < REPLACEMENT_MIN_AGE_FRAMES:
            return None
        if activity < self._enter_thr or not self._has_activity_jump(activity, activity_rise):
            return None
        if not nearby_visible:
            return None

        lead_track, distance = nearby_visible[0]
        if lead_track.id == candidate.track_id:
            return None
        if frame_id < self._track_cooldown_until.get((pocket_idx, lead_track.id), -1):
            return None
        if len(lead_track.positions) < self._min_track_age or not (
            self._has_recent_motion(lead_track)
            or self._is_fast_near_pocket_track(lead_track, pocket_idx)
        ):
            return None
        if not self._within_dynamic_start_zone(
            pocket_idx,
            lead_track,
            distance,
            moving=True,
        ):
            return None

        current_track = tracks_by_id.get(candidate.track_id)
        current_is_weak = (
            current_track is None
            or not current_track.visible
            or not self._has_recent_motion(current_track)
        )
        if not current_is_weak and distance > candidate.min_distance_px - REPLACEMENT_DISTANCE_GAIN_PX:
            return None
        return lead_track, distance

    def _start_candidate(
        self,
        pocket_idx: int,
        frame_id: int,
        lead_track: Track,
        distance: float,
        activity: float,
        nearby_visible: List[tuple[Track, float]],
        impact_seen: bool = False,
    ) -> PocketEventCandidate:
        candidate = PocketEventCandidate(
            pocket_idx=pocket_idx,
            label=self.rois[pocket_idx]["label"],
            track_id=lead_track.id,
            start_frame=frame_id,
            end_frame=frame_id,
            last_visible_frame=frame_id,
            nearby_track_ids=[track.id for track, _ in nearby_visible],
            peak_activity=activity,
            last_activity=activity,
            min_distance_px=distance,
            impact_seen=impact_seen,
        )
        self._active_candidates[pocket_idx] = candidate
        self.candidates.append(candidate)
        return candidate

    def _within_start_zone(
        self,
        pocket_idx: int,
        distance: float,
        moving: bool,
        armed: bool = False,
    ) -> bool:
        roi = self.rois[pocket_idx]
        if moving:
            factor = MOVING_START_DISTANCE_FACTOR
        elif armed:
            factor = ARMED_START_DISTANCE_FACTOR
        else:
            factor = STATIONARY_START_DISTANCE_FACTOR
        return distance <= roi["radius"] * factor

    def _within_dynamic_start_zone(
        self,
        pocket_idx: int,
        track: Track,
        distance: float,
        moving: bool,
        armed: bool = False,
    ) -> bool:
        if self._within_start_zone(pocket_idx, distance, moving=moving, armed=armed):
            return True
        if track.near_pocket_idx != pocket_idx:
            return False
        if self._recent_speed(track) < FAST_NEAR_POCKET_SPEED_PX:
            return False
        return distance <= self.rois[pocket_idx]["radius"] * FAST_NEAR_POCKET_START_DISTANCE_FACTOR

    def _is_fast_near_pocket_track(self, track: Track, pocket_idx: int) -> bool:
        return track.near_pocket_idx == pocket_idx and self._recent_speed(track) >= FAST_NEAR_POCKET_SPEED_PX

    def _effective_track_distance(self, track: Track, roi: dict) -> float:
        current = float(np.hypot(track.cx - roi["cx"], track.cy - roi["cy"]))
        if not track.velocities or self.fps <= 0:
            return current
        _, vx, vy = track.velocities[-1]
        lookahead_frames = PREDICT_LOOKAHEAD_FRAMES + max(track.missing_frames, 0)
        dt = lookahead_frames / self.fps
        pred_x = track.cx + vx * dt * 0.85
        pred_y = track.cy + vy * dt * 0.85
        predicted = float(np.hypot(pred_x - roi["cx"], pred_y - roi["cy"]))
        return min(current, predicted)

    def _recent_missing_seed(
        self,
        pocket_idx: int,
        tracks_by_id: Dict[int, Track],
    ) -> Optional[tuple[Track, float]]:
        roi = self.rois[pocket_idx]
        candidates: List[tuple[float, float, Track]] = []

        for track in tracks_by_id.values():
            if track.visible:
                continue
            if track.missing_frames <= 0 or track.missing_frames > RECENT_MISSING_SEED_FRAMES:
                continue
            if track.near_pocket_idx != pocket_idx:
                continue
            if len(track.positions) < self._min_track_age or not track.velocities:
                continue

            recent_speed = max(
                float(np.hypot(vx, vy))
                for _, vx, vy in track.velocities[-RECENT_MOTION_MIN_SAMPLES:]
            )
            if recent_speed < RECENT_MISSING_SPEED_PX:
                continue

            distance = self._effective_track_distance(track, roi)
            if distance > roi["radius"] * RECENT_MISSING_START_DISTANCE_FACTOR:
                continue

            candidates.append((distance, -recent_speed, track))

        if not candidates:
            return None

        candidates.sort(key=lambda item: (item[0], item[1], item[2].missing_frames))
        distance, _, track = candidates[0]
        return track, distance

    @staticmethod
    def _recent_speed(track: Track) -> float:
        if not track.velocities:
            return 0.0
        return max(
            float(np.hypot(vx, vy))
            for _, vx, vy in track.velocities[-RECENT_MOTION_MIN_SAMPLES:]
        )

    @staticmethod
    def _has_recent_approach(track: Track, roi: dict) -> bool:
        if len(track.positions) < RECENT_APPROACH_SAMPLES:
            return False
        recent = track.positions[-RECENT_APPROACH_SAMPLES:]
        distances = [
            float(np.hypot(x - roi["cx"], y - roi["cy"]))
            for _, x, y in recent
        ]
        if distances[0] - distances[-1] >= MIN_APPROACH_DELTA_PX:
            return True
        if not track.velocities:
            return False
        recent_velocities = track.velocities[-RECENT_APPROACH_SAMPLES:]
        return any(float(np.hypot(vx, vy)) >= RECENT_APPROACH_SPEED_PX for _, vx, vy in recent_velocities)

    def _has_confirming_motion(
        self,
        candidate: PocketEventCandidate,
        track: Track,
        pocket_idx: int,
    ) -> bool:
        roi = self.rois[pocket_idx]
        if self._has_recent_approach(track, roi):
            return True

        armed = self._armed_tracks.get(pocket_idx)
        if armed is None or armed.track_id != candidate.track_id:
            return False
        if armed.frames_near_pocket < STABLE_POCKET_LINGER_FRAMES:
            return False
        if candidate.min_distance_px > roi["radius"] * STABLE_POCKET_MAX_DISTANCE_FACTOR:
            return False
        if candidate.peak_activity < max(self._min_peak_activity + 1.0, self._enter_thr * 2.0):
            return False
        return candidate.impact_seen

    def _has_strong_disappearance_signature(self, candidate: PocketEventCandidate) -> bool:
        roi = self.rois[candidate.pocket_idx]
        if candidate.last_visible_frame > candidate.start_frame + 1:
            return False
        if candidate.min_distance_px > roi["radius"] * STRONG_DISAPPEAR_DISTANCE_FACTOR:
            return False
        return candidate.peak_activity >= max(self._min_peak_activity + 3.0, STRONG_DISAPPEAR_PEAK_ACTIVITY)

    @staticmethod
    def _has_recent_motion(track: Track) -> bool:
        if len(track.velocities) < RECENT_MOTION_MIN_SAMPLES:
            return False
        recent = track.velocities[-RECENT_MOTION_MIN_SAMPLES:]
        return any(np.hypot(vx, vy) >= RECENT_MOTION_TRIGGER_PX for _, vx, vy in recent)

    def _has_impact_support(
        self,
        nearby_visible: List[tuple[Track, float]],
        exclude_track_id: Optional[int] = None,
    ) -> bool:
        for track, _ in nearby_visible:
            if exclude_track_id is not None and track.id == exclude_track_id:
                continue
            if self._has_recent_motion(track):
                return True
            if not track.velocities:
                continue
            recent = track.velocities[-RECENT_APPROACH_SAMPLES:]
            if any(float(np.hypot(vx, vy)) >= IMPACT_SUPPORT_SPEED_PX for _, vx, vy in recent):
                return True
        return False

    def _confirm_candidate(self, candidate: PocketEventCandidate, frame_id: int) -> PocketEvent:
        radius = max(self.rois[candidate.pocket_idx]["radius"], 1)
        proximity_score = 1.0 - min(candidate.min_distance_px / radius, 1.0)
        activity_score = min(candidate.peak_activity / max(self._enter_thr * 2.0, 1.0), 1.0)
        confidence = 0.45 + 0.35 * proximity_score + 0.20 * activity_score
        candidate.status = "confirmed"
        return PocketEvent(
            pocket_idx=candidate.pocket_idx,
            label=candidate.label,
            frame_id=frame_id,
            time_s=frame_id / self.fps if self.fps > 0 else 0.0,
            track_id=candidate.track_id,
            confidence=float(np.clip(confidence, 0.0, 0.99)),
            peak_activity=candidate.peak_activity,
            roi_activity=candidate.last_activity,
            nearby_track_ids=candidate.nearby_track_ids,
        )

    def _reject_candidate(self, pocket_idx: int, reason: str) -> None:
        candidate = self._active_candidates[pocket_idx]
        candidate.status = "rejected"
        candidate.reject_reason = reason
        self.rejected_candidates.append(candidate)
        if reason in {"ball_still_visible", "no_confirming_motion", "timed_out", "replaced_by_stronger_track"}:
            self._track_cooldown_until[(candidate.pocket_idx, candidate.track_id)] = (
                candidate.end_frame + REJECTED_TRACK_COOLDOWN_FRAMES
            )
        del self._active_candidates[pocket_idx]

    @staticmethod
    def _norm_label(label: str) -> str:
        return label.strip().lower().replace("_", "-").replace(" ", "-")

    def _should_debug(self, pocket_idx: int) -> bool:
        if not self._debug_pockets:
            return False
        return self._norm_label(self.rois[pocket_idx]["label"]) in self._debug_pockets

    def _log_debug(
        self,
        pocket_idx: int,
        frame_id: int,
        action: str,
        activity: Optional[float] = None,
        activity_rise: Optional[float] = None,
        candidate: Optional[PocketEventCandidate] = None,
        nearby_visible: Optional[List[tuple[Track, float]]] = None,
        extra: Optional[dict] = None,
    ) -> None:
        if not self._should_debug(pocket_idx):
            return
        armed = self._armed_tracks.get(pocket_idx)
        rows = self._debug_log.setdefault(pocket_idx, [])
        entry = {
            "frame": frame_id,
            "time_s": round(frame_id / self.fps, 3) if self.fps > 0 else 0.0,
            "action": action,
            "activity": round(float(activity), 3) if activity is not None else None,
            "activity_rise": round(float(activity_rise), 3) if activity_rise is not None else None,
            "active_candidate_track_id": candidate.track_id if candidate else None,
            "active_candidate_age": frame_id - candidate.start_frame if candidate else None,
            "active_candidate_min_distance_px": (
                round(float(candidate.min_distance_px), 3) if candidate else None
            ),
            "active_candidate_impact_seen": candidate.impact_seen if candidate else None,
            "armed_track_id": armed.track_id if armed else None,
            "armed_last_seen_frame": armed.last_seen_frame if armed else None,
            "armed_frames_near_pocket": armed.frames_near_pocket if armed else None,
            "armed_min_distance_px": round(float(armed.min_distance_px), 3) if armed else None,
            "nearby_tracks": [
                {
                    "track_id": track.id,
                    "distance_px": round(float(distance), 3),
                    "missing_frames": track.missing_frames,
                    "visible": track.visible,
                }
                for track, distance in (nearby_visible or [])[:5]
            ],
        }
        if extra:
            entry.update(extra)
        rows.append(entry)
