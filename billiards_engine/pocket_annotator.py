"""
Interactive pocket location annotator.

Shows the first frame of a clip and lets the user click the 6 pocket
openings in order.  Saves coordinates to  <clip_dir>/pocket_config.json
so the pipeline can load them on subsequent runs without re-annotating.

Click order (shown as on-screen prompt):
  1. Top-left       2. Top-middle     3. Top-right
  4. Bottom-left    5. Bottom-middle  6. Bottom-right

Controls
--------
  Left-click   — place / replace the current pocket marker
  Backspace    — undo the last placed marker
  R            — reset all markers
  Enter / S    — save and exit
  Q / Esc      — quit without saving
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .video_loader import VideoLoader


# ── Pocket labels (in click order) ──────────────────────────────────────────
POCKET_LABELS = [
    "Top-Left",
    "Top-Middle",
    "Top-Right",
    "Bottom-Left",
    "Bottom-Middle",
    "Bottom-Right",
]

# BGR colours per pocket (for markers)
_POCKET_COLORS = [
    (0,   255, 80),    # top-left    — green
    (0,   220, 255),   # top-mid     — yellow
    (0,   80,  255),   # top-right   — red
    (255, 100, 0),     # bot-left    — blue
    (255, 0,   200),   # bot-mid     — magenta
    (80,  255, 255),   # bot-right   — cyan
]

CONFIG_FILENAME = "pocket_config.json"
POCKET_RADIUS   = 32   # detection radius stored in config


# ── State shared between OpenCV callbacks ────────────────────────────────────
class _AnnotatorState:
    def __init__(self):
        self.pockets: List[Optional[Tuple[int, int]]] = [None] * 6
        self.current_idx: int = 0          # next pocket to place
        self.base_frame: Optional[np.ndarray] = None
        self.dirty: bool = True            # needs redraw


_state = _AnnotatorState()


def _mouse_cb(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if _state.current_idx < 6:
            _state.pockets[_state.current_idx] = (x, y)
            _state.current_idx = min(_state.current_idx + 1, 6)
            _state.dirty = True


def _draw_overlay(base: np.ndarray) -> np.ndarray:
    out = base.copy()
    h, w = out.shape[:2]

    # ── Placed pocket markers ───────────────────────────────────────────
    for i, pt in enumerate(_state.pockets):
        if pt is None:
            continue
        color = _POCKET_COLORS[i]
        cv2.circle(out, pt, POCKET_RADIUS, color, 2)
        cv2.circle(out, pt, 4, color, -1)
        cv2.putText(
            out, str(i + 1),
            (pt[0] + POCKET_RADIUS + 4, pt[1] + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA,
        )

    # ── Current pocket prompt ───────────────────────────────────────────
    idx = _state.current_idx
    if idx < 6:
        label = POCKET_LABELS[idx]
        color = _POCKET_COLORS[idx]
        msg = f"Click pocket {idx + 1}/6:  {label}"
    else:
        msg = "All 6 pockets placed!  Press Enter/S to save, R to reset."
        color = (255, 255, 255)

    # Semi-transparent dark bar at bottom
    overlay = out.copy()
    cv2.rectangle(overlay, (0, h - 70), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, out, 0.35, 0, out)

    cv2.putText(
        out, msg,
        (14, h - 42),
        cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 1, cv2.LINE_AA,
    )
    cv2.putText(
        out, "Backspace=undo  R=reset  Enter/S=save  Q/Esc=quit",
        (14, h - 16),
        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160, 160, 160), 1, cv2.LINE_AA,
    )

    # ── Pocket legend (top-right) ────────────────────────────────────────
    for i, lbl in enumerate(POCKET_LABELS):
        status = "✓" if _state.pockets[i] else f"{i+1}."
        text = f" {status} {lbl}"
        clr = _POCKET_COLORS[i] if _state.pockets[i] else (100, 100, 100)
        cv2.putText(
            out, text,
            (w - 200, 22 + i * 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, clr, 1, cv2.LINE_AA,
        )

    return out


def annotate_pockets(
    clip_dir: str,
    pocket_radius: int = POCKET_RADIUS,
    force_reannotate: bool = False,
    show_ui: bool = True,
) -> Optional[List[Tuple[float, float, float]]]:
    """
    Open an interactive window to annotate the 6 pocket positions for a clip.

    Returns a list of (cx, cy, radius) tuples (same format as
    EventDetector.pockets), or None if the user quit without saving.

    If  <clip_dir>/pocket_config.json  already exists and
    force_reannotate=False, loads and returns those coordinates directly
    without opening a window.
    """
    config_path = os.path.join(clip_dir, CONFIG_FILENAME)

    # ── Load existing config ─────────────────────────────────────────────
    if os.path.isfile(config_path) and not force_reannotate:
        with open(config_path) as fh:
            data = json.load(fh)
        pockets = [(p["cx"], p["cy"], p["radius"]) for p in data["pockets"]]
        print(f"  Loaded pocket config: {config_path}")
        return pockets

    # ── No saved config and UI is disabled — skip silently ───────────────
    if not show_ui:
        print("  Pockets: no saved config and UI disabled — using estimated positions")
        return None

    # ── Find video ───────────────────────────────────────────────────────
    clip_name = os.path.basename(clip_dir.rstrip("/\\"))
    video_path = os.path.join(clip_dir, f"{clip_name}.mp4")
    if not os.path.isfile(video_path):
        print(f"  [pocket_annotator] Video not found: {video_path}")
        return None

    with VideoLoader(video_path) as loader:
        base_frame = loader.get_frame(0)

    # ── Reset state ──────────────────────────────────────────────────────
    _state.pockets = [None] * 6
    _state.current_idx = 0
    _state.base_frame = base_frame
    _state.dirty = True

    win = f"Pocket Annotator — {clip_name}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, base_frame.shape[1], base_frame.shape[0])
    cv2.setMouseCallback(win, _mouse_cb)

    print(f"\n  [pocket_annotator] Click the 6 pockets in order:")
    for i, lbl in enumerate(POCKET_LABELS):
        print(f"    {i+1}. {lbl}")
    print("  Controls: Backspace=undo  R=reset  Enter/S=save  Q/Esc=quit\n")

    saved = False
    while True:
        if _state.dirty:
            frame = _draw_overlay(base_frame)
            cv2.imshow(win, frame)
            _state.dirty = False

        key = cv2.waitKey(20) & 0xFF

        if key in (13, ord("s")):          # Enter or S — save
            if all(p is not None for p in _state.pockets):
                saved = True
                break
            else:
                remaining = sum(1 for p in _state.pockets if p is None)
                print(f"  Still need {remaining} more pocket(s).")

        elif key in (27, ord("q")):        # Esc or Q — quit
            break

        elif key == 8:                     # Backspace — undo
            idx = _state.current_idx - 1
            if idx >= 0:
                _state.pockets[idx] = None
                _state.current_idx = idx
                _state.dirty = True

        elif key == ord("r"):              # R — reset
            _state.pockets = [None] * 6
            _state.current_idx = 0
            _state.dirty = True

    cv2.destroyWindow(win)

    if not saved:
        print("  [pocket_annotator] Cancelled — no config saved.")
        return None

    # ── Save config ──────────────────────────────────────────────────────
    pocket_data = [
        {
            "label":  POCKET_LABELS[i],
            "cx":     float(_state.pockets[i][0]),
            "cy":     float(_state.pockets[i][1]),
            "radius": float(pocket_radius),
        }
        for i in range(6)
    ]
    config = {
        "clip":    clip_name,
        "pockets": pocket_data,
    }
    with open(config_path, "w") as fh:
        json.dump(config, fh, indent=2)

    print(f"  [pocket_annotator] Saved pocket config → {config_path}")
    for i, p in enumerate(pocket_data):
        print(f"    {i+1}. {p['label']:15s}  cx={p['cx']:.0f}  cy={p['cy']:.0f}")

    return [(p["cx"], p["cy"], p["radius"]) for p in pocket_data]
