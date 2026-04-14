"""
camera_manager.py — multi-camera GoalDetectorV2 coordinator.

Manages two camera configurations, each with its own pocket ROI set and
independent GoalDetectorV2 background model.  Both detectors process every
frame so their background models stay current even while inactive — this
ensures correct detection immediately after a camera switch.

Hard cut detection
------------------
A sudden jump in mean frame difference (> cut_threshold) is treated as a
camera cut.  On cut, _select_active() decides which configuration becomes
active.

Extension point
---------------
Override _select_active() to implement automatic "best view" selection —
e.g. score each camera config against the current frame using table visibility,
felt colour coverage, or a learned classifier.  The default simply toggles
between the two cameras.

ROI files
---------
ROIs are stored per-camera:
    <clip_dir>/pocket_rois_cam1.json   (camera 1)
    <clip_dir>/pocket_rois_cam2.json   (camera 2)

For backwards compatibility, if pocket_rois_cam1.json is absent but
pocket_rois.json exists, the legacy file is used for camera 1.

Annotation reference frame
--------------------------
pocket_roi_selector always uses frame 0 of the video for annotation, which
may be occluded by a player or otherwise unclear.  load_or_select_rois
automatically finds the frame with the highest felt coverage and creates a
temporary single-frame annotation clip from it, so the selector always shows
the clearest available table view.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from .goal_detector_v2 import GoalDetectorV2, GoalEvent
from .felt_config import get_felt_mask
from .pocket_roi_selector import select_pocket_rois

_ROI_RADIUS    = 28    # correct radius for this footage; selector default is 56
_SCAN_SAMPLES  = 30    # number of frames to sample when finding the best reference


# ── Reference frame selection ──────────────────────────────────────────────────

def find_best_reference_frame(video_path: str, felt: str = "blue") -> np.ndarray:
    """
    Sample ~_SCAN_SAMPLES evenly-spaced frames and return the one with the
    highest felt-pixel coverage — i.e., the clearest, least-occluded table view.
    """
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step  = max(1, total // _SCAN_SAMPLES)

    best_frame: Optional[np.ndarray] = None
    best_score = -1.0

    for i in range(_SCAN_SAMPLES):
        idx = i * step
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask  = get_felt_mask(hsv, felt)
        # mask pixels are 0 or 255 — count non-zero as fraction of total
        score = float((mask > 0).sum()) / mask.size
        if score > best_score:
            best_score = score
            best_frame = frame.copy()

    cap.release()

    if best_frame is None:
        cap = cv2.VideoCapture(video_path)
        _, best_frame = cap.read()
        cap.release()

    print(f"    [reference frame] best felt coverage = {best_score*100:.1f}%")
    return best_frame


def _write_reference_video(path: Path, frame: np.ndarray, fps: float) -> bool:
    """
    Write a short looping clip of a single frame.  Tries codecs in order
    until one produces a non-empty file.  Returns True on success.
    """
    h, w = frame.shape[:2]
    # avc1 (H.264) works best on macOS; mp4v and MJPG as fallbacks
    for fourcc_str in ("avc1", "mp4v", "MJPG"):
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*fourcc_str), fps, (w, h)
        )
        if not writer.isOpened():
            continue
        for _ in range(10):   # repeat the frame so the selector has something to display
            writer.write(frame)
        writer.release()
        if path.exists() and path.stat().st_size > 500:
            return True
        path.unlink(missing_ok=True)
    return False


def _make_annotation_dir(
    clip_dir: str,
    video_path: str,
    felt: str,
    cam_idx: int,
) -> str:
    """
    Create a subdirectory containing a short 'annotation clip' built from the
    clearest table frame found in the video.

    pocket_roi_selector expects the video at <dir>/<dirname>.mp4 so the
    directory and file are named to match.  The directory is reused on
    subsequent runs (idempotent).  If video writing fails for any reason,
    falls back to a hardlink/copy of the original video so the selector
    always has something to open.
    """
    ann_name  = f"_ann_cam{cam_idx}"
    ann_dir   = Path(clip_dir) / ann_name
    ann_dir.mkdir(exist_ok=True)
    ann_video = ann_dir / f"{ann_name}.mp4"

    if ann_video.exists() and ann_video.stat().st_size > 500:
        return str(ann_dir)

    print(f"  [Cam {cam_idx}] Scanning for clearest table frame...")
    best = find_best_reference_frame(video_path, felt)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    if not _write_reference_video(ann_video, best, fps):
        # VideoWriter failed on this platform — fall back to the original video
        print(f"    [reference frame] VideoWriter unavailable; using original video")
        try:
            os.link(video_path, str(ann_video))
        except OSError:
            shutil.copy2(video_path, str(ann_video))

    return str(ann_dir)


# ── ROI loading ────────────────────────────────────────────────────────────────

def load_or_select_rois(
    clip_dir: str,
    cam_idx: int,
    video_path: str,
    felt: str = "blue",
    force_reselect: bool = False,
    radius: int = _ROI_RADIUS,
) -> Optional[List[dict]]:
    """
    Load pocket ROIs for camera cam_idx (1-based).

    Resolution order:
      1. pocket_rois_cam{cam_idx}.json  (cam-specific file) — loaded directly
      2. pocket_rois.json               (legacy, cam1 only) — promoted then loaded
      3. Interactive selector           (if neither exists or force_reselect)
         — uses the clearest table frame as the annotation background

    After interactive selection the result is saved to pocket_rois_cam{cam_idx}.json
    so subsequent runs load instantly.
    """
    cam_file    = os.path.join(clip_dir, f"pocket_rois_cam{cam_idx}.json")
    legacy_file = os.path.join(clip_dir, "pocket_rois.json")

    if not force_reselect:
        if os.path.isfile(cam_file):
            with open(cam_file) as fh:
                rois = json.load(fh)
            print(f"  [Cam {cam_idx}] Loaded {cam_file}")
            for r in rois:
                print(f"    {r['label']:15s}  cx={r['cx']}  cy={r['cy']}  r={r['radius']}")
            return rois

        # Backwards compat: use pocket_rois.json for cam1
        if cam_idx == 1 and os.path.isfile(legacy_file):
            with open(legacy_file) as fh:
                rois = json.load(fh)
            with open(cam_file, "w") as fh:
                json.dump(rois, fh, indent=2)
            print(f"  [Cam {cam_idx}] Loaded (legacy) {legacy_file}")
            for r in rois:
                print(f"    {r['label']:15s}  cx={r['cx']}  cy={r['cy']}  r={r['radius']}")
            return rois

    # Need interactive annotation.
    # Build annotation dir with the clearest table frame so the selector shows
    # a clean view even if frame 0 has a player blocking the pockets.
    print(f"\n  Annotate pockets for Camera {cam_idx} "
          f"({'re-select' if force_reselect else 'first time'})...")
    ann_dir = _make_annotation_dir(clip_dir, video_path, felt, cam_idx)
    rois    = select_pocket_rois(ann_dir, force_reselect=True, radius=radius)
    if rois is None:
        return None

    # Persist to cam-specific file in the main clip dir
    with open(cam_file, "w") as fh:
        json.dump(rois, fh, indent=2)
    print(f"  [Cam {cam_idx}] Saved → {cam_file}")
    return rois


# ── Camera manager ─────────────────────────────────────────────────────────────

class CameraManager:
    """
    Coordinates two GoalDetectorV2 instances across camera switches.

    Parameters
    ----------
    detectors    : [GoalDetectorV2 for cam1, GoalDetectorV2 for cam2]
    rois_list    : [rois for cam1, rois for cam2]
    cut_threshold: mean-pixel-diff threshold that signals a hard camera cut
    """

    def __init__(
        self,
        detectors: List[GoalDetectorV2],
        rois_list: List[List[dict]],
        cut_threshold: float = 30.0,
    ):
        assert len(detectors) == len(rois_list), "detectors and rois_list must be the same length"
        self.detectors     = detectors
        self.rois_list     = rois_list
        self.active_idx    = 0
        self.cut_threshold = cut_threshold
        self.cut_count     = 0

        self._prev_gray: Optional[np.ndarray] = None

    # ── Public API ─────────────────────────────────────────────────────────

    @property
    def active_rois(self) -> List[dict]:
        return self.rois_list[self.active_idx]

    def process_frame(
        self, frame: np.ndarray, frame_id: int
    ) -> List[GoalEvent]:
        """
        Feed one frame to all detectors.
        Returns GoalEvents from the active camera only.
        """
        is_cut, diff = self._detect_cut(frame)
        if is_cut:
            new_idx = self._select_active(frame, frame_id, diff)
            if new_idx != self.active_idx:
                self.active_idx = new_idx
                self.cut_count += 1
                print(
                    f"  [CameraManager] Cut at frame {frame_id}"
                    f"  diff={diff:.1f}  → cam{self.active_idx + 1}"
                )

        events: List[GoalEvent] = []
        for i, det in enumerate(self.detectors):
            evs = det.process_frame(frame, frame_id)
            if i == self.active_idx:
                events.extend(evs)

        return events

    # ── Extension point ────────────────────────────────────────────────────

    def _select_active(
        self, frame: np.ndarray, frame_id: int, frame_diff: float
    ) -> int:
        """
        Decide which camera config to activate after a detected cut.

        Default behaviour: toggle between 0 and 1.

        Override this method for automatic "best view" selection, e.g.:
          - Measure felt-colour coverage in each config's table bbox
          - Score ball visibility against each config's expected table region
          - Use a learned view-quality classifier
        """
        return 1 - self.active_idx

    # ── Internals ──────────────────────────────────────────────────────────

    def _detect_cut(self, frame: np.ndarray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = 0.0
        if self._prev_gray is not None:
            diff = float(
                np.abs(
                    gray.astype(np.float32) - self._prev_gray.astype(np.float32)
                ).mean()
            )
        self._prev_gray = gray
        return diff > self.cut_threshold, diff
