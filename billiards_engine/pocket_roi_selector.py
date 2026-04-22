"""
Pocket ROI selector — click to place, circle shown around each point.

The user clicks once on each pocket center.  A circle of fixed radius is
drawn around the click; that circle becomes the detection ROI.

Saves to <clip_dir>/pocket_rois.json for reuse.
"""

from __future__ import annotations

import json
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .video_loader import VideoLoader

ROI_CONFIG_FILE = "pocket_rois.json"
ROI_RADIUS      = 56   # pixel radius of each pocket detection zone

POCKET_LABELS = [
    "Top-Left",
    "Top-Middle",
    "Top-Right",
    "Bottom-Left",
    "Bottom-Middle",
    "Bottom-Right",
]

_COLORS = [
    (0,   255, 80),
    (0,   220, 255),
    (0,   80,  255),
    (255, 100, 0),
    (255, 0,   200),
    (80,  255, 255),
]


class _State:
    def __init__(self):
        self.points: List[Optional[Tuple[int, int]]] = [None] * 6
        self.current: int = 0
        self.dirty: bool = True


_st = _State()


def _mouse_cb(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and _st.current < 6:
        _st.points[_st.current] = (x, y)
        _st.current = min(_st.current + 1, 6)
        _st.dirty = True


def _draw(base: np.ndarray, radius: int) -> np.ndarray:
    out = base.copy()
    h, w = out.shape[:2]

    for i, pt in enumerate(_st.points):
        if pt is None:
            continue
        cv2.circle(out, pt, radius, _COLORS[i], 2)
        cv2.circle(out, pt, 4, _COLORS[i], -1)
        cv2.putText(out, str(i + 1), (pt[0] + radius + 4, pt[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, _COLORS[i], 1, cv2.LINE_AA)

    idx = _st.current
    if idx < 6:
        msg  = f"Click pocket {idx+1}/6:  {POCKET_LABELS[idx]}"
        color = _COLORS[idx]
    else:
        msg  = "All 6 placed!  Enter/S = save    R = reset    Esc/Q = quit"
        color = (255, 255, 255)

    overlay = out.copy()
    cv2.rectangle(overlay, (0, h - 52), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, out, 0.35, 0, out)
    cv2.putText(out, msg, (14, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 1, cv2.LINE_AA)
    cv2.putText(out, "Backspace/Delete/U = undo    R = reset    Enter/S = save    Q/Esc = quit",
                (14, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1, cv2.LINE_AA)

    # legend
    for i, lbl in enumerate(POCKET_LABELS):
        c   = _COLORS[i] if _st.points[i] else (90, 90, 90)
        sym = "●" if _st.points[i] else f"{i+1}."
        cv2.putText(out, f" {sym} {lbl}", (w - 190, 22 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1, cv2.LINE_AA)

    return out


def select_pocket_rois(
    clip_dir: str,
    force_reselect: bool = False,
    radius: int = ROI_RADIUS,
) -> Optional[List[dict]]:
    """
    Returns list of ROI dicts [{label, x, y, w, h}, ...] (square around each
    clicked center), or None if cancelled.  Loads cached config if present.
    """
    config_path = os.path.join(clip_dir, ROI_CONFIG_FILE)

    if os.path.isfile(config_path) and not force_reselect:
        with open(config_path) as fh:
            rois = json.load(fh)
        print(f"  Loaded pocket ROIs: {config_path}")
        for r in rois:
            print(f"    {r['label']:15s}  cx={r['cx']}  cy={r['cy']}  r={r['radius']}")
        return rois

    clip_name = os.path.basename(clip_dir.rstrip("/\\"))
    video_path = os.path.join(clip_dir, f"{clip_name}.mp4")
    if not os.path.isfile(video_path):
        print(f"  Video not found: {video_path}")
        return None

    with VideoLoader(video_path) as loader:
        base = loader.get_frame(0)

    # reset state
    _st.points  = [None] * 6
    _st.current = 0
    _st.dirty   = True

    win = f"Pocket Selector — {clip_name}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, base.shape[1], base.shape[0])
    cv2.setMouseCallback(win, _mouse_cb)

    saved = False
    while True:
        if _st.dirty:
            cv2.imshow(win, _draw(base, radius))
            _st.dirty = False

        key = cv2.waitKey(20) & 0xFF

        if key in (13, ord("s")):                   # Enter / S
            if all(p is not None for p in _st.points):
                saved = True
                break
            print(f"  Still need {sum(1 for p in _st.points if p is None)} more.")

        elif key in (27, ord("q")):                 # Esc / Q
            break

        elif key in (8, 127, ord("u")):            # Backspace/Delete/U — undo
            idx = _st.current - 1
            if idx >= 0:
                _st.points[idx] = None
                _st.current = idx
                _st.dirty = True

        elif key == ord("r"):                       # R — reset
            _st.points  = [None] * 6
            _st.current = 0
            _st.dirty   = True

    cv2.destroyWindow(win)

    if not saved:
        print("  Cancelled — no config saved.")
        return None

    # Convert (cx, cy) → ROI dict with bounding square
    rois = []
    for i, pt in enumerate(_st.points):
        cx, cy = pt
        rois.append({
            "label":  POCKET_LABELS[i],
            "cx":     cx,
            "cy":     cy,
            "radius": radius,
            # bounding box for crop
            "x": cx - radius,
            "y": cy - radius,
            "w": radius * 2,
            "h": radius * 2,
        })

    with open(config_path, "w") as fh:
        json.dump(rois, fh, indent=2)
    print(f"  Saved pocket ROIs → {config_path}")
    for r in rois:
        print(f"    {r['label']:15s}  cx={r['cx']}  cy={r['cy']}")

    return rois
