"""
heatmap.py — BallHeatmap, PlayerHeatmap, and rendering utilities.

BallHeatmap   : accumulates Gaussian blobs at ball centroid positions each frame.
PlayerHeatmap : uses MOG2 background subtraction to locate large moving blobs
                (players/cue holders) and accumulates their positions.

Both render as a COLORMAP_JET overlay blended over a reference frame.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np


class BallHeatmap:
    """Accumulates ball positions as Gaussian blobs over the course of a video."""

    def __init__(self, width: int, height: int, sigma: float = 12.0):
        self._w      = width
        self._h      = height
        self._sigma  = sigma
        self.map     = np.zeros((height, width), dtype=np.float32)

    def update(self, active_tracks) -> None:
        """Add a Gaussian splat for every active track this frame."""
        for track in active_tracks:
            cx = int(round(track.cx))
            cy = int(round(track.cy))
            if 0 <= cx < self._w and 0 <= cy < self._h:
                self._splat(cx, cy)

    def reset(self) -> None:
        self.map[:] = 0.0

    def _splat(self, cx: int, cy: int) -> None:
        r  = int(self._sigma * 3)
        x0 = max(0, cx - r);  x1 = min(self._w, cx + r + 1)
        y0 = max(0, cy - r);  y1 = min(self._h, cy + r + 1)
        xs = np.arange(x0, x1) - cx
        ys = np.arange(y0, y1) - cy
        xx, yy = np.meshgrid(xs, ys)
        self.map[y0:y1, x0:x1] += np.exp(
            -(xx ** 2 + yy ** 2) / (2 * self._sigma ** 2)
        ).astype(np.float32)


class PlayerHeatmap:
    """
    Tracks large moving blobs (players) using MOG2 background subtraction
    on a downscaled copy of each frame.
    """

    def __init__(
        self,
        width: int,
        height: int,
        scale: float = 0.25,
        min_blob_area: int = 1500,
        bg_history: int = 120,
    ):
        self._w       = width
        self._h       = height
        self._scale   = scale
        # Scale the area threshold to the downscaled frame
        self._min_area = min_blob_area * (scale ** 2)
        self.map       = np.zeros((height, width), dtype=np.float32)
        self._bg_sub   = cv2.createBackgroundSubtractorMOG2(
            history=bg_history, varThreshold=40, detectShadows=False
        )
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def update(self, frame: np.ndarray) -> None:
        """Process one frame and accumulate player positions."""
        sw = int(self._w * self._scale)
        sh = int(self._h * self._scale)
        small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
        fg    = self._bg_sub.apply(small)
        fg    = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  self._kernel)
        fg    = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self._kernel)
        fg    = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)[1]

        contours, _ = cv2.findContours(
            fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for cnt in contours:
            if cv2.contourArea(cnt) < self._min_area:
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int((M["m10"] / M["m00"]) / self._scale)
            cy = int((M["m01"] / M["m00"]) / self._scale)
            if not (0 <= cx < self._w and 0 <= cy < self._h):
                continue
            r  = 30
            x0 = max(0, cx - r);  x1 = min(self._w, cx + r + 1)
            y0 = max(0, cy - r);  y1 = min(self._h, cy + r + 1)
            xs = np.arange(x0, x1) - cx
            ys = np.arange(y0, y1) - cy
            xx, yy = np.meshgrid(xs, ys)
            self.map[y0:y1, x0:x1] += np.exp(
                -(xx ** 2 + yy ** 2) / (2 * (r / 2) ** 2)
            ).astype(np.float32)

    def reset(self) -> None:
        self.map[:] = 0.0


# ── Rendering ──────────────────────────────────────────────────────────────────

def render_heatmap(
    heatmap: np.ndarray,
    reference_frame: Optional[np.ndarray] = None,
    alpha: float = 0.55,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Return a BGR image: heatmap colourised and blended over reference_frame."""
    h, w = heatmap.shape[:2]
    bg = (
        reference_frame.copy()
        if reference_frame is not None
        else np.zeros((h, w, 3), dtype=np.uint8)
    )

    if heatmap.max() < 1e-6:
        cv2.putText(
            bg, "No data", (w // 2 - 50, h // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2,
        )
        return bg

    norm    = (heatmap / heatmap.max() * 255).astype(np.uint8)
    colored = cv2.applyColorMap(norm, colormap)

    if bg.shape[:2] != (h, w):
        bg = cv2.resize(bg, (w, h))

    # Only blend where the heatmap has meaningful values
    mask = (heatmap / heatmap.max() > 0.02).astype(np.float32)
    blended = bg.copy()
    for c in range(3):
        blended[:, :, c] = (
            bg[:, :, c] * (1.0 - mask * alpha)
            + colored[:, :, c] * mask * alpha
        ).astype(np.uint8)
    return blended


def save_heatmaps(
    ball_heatmap: BallHeatmap,
    player_heatmap: PlayerHeatmap,
    out_dir: Path,
    reference_frame: Optional[np.ndarray] = None,
    prefix: str = "",
) -> dict:
    """Render and save both heatmaps; return a dict of {name: path}."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for name, hm in [("ball", ball_heatmap), ("player", player_heatmap)]:
        img  = render_heatmap(hm.map, reference_frame)
        path = out_dir / f"{prefix}{name}_heatmap.png"
        cv2.imwrite(str(path), img)
        results[f"{name}_heatmap"] = str(path)
        print(f"  {name.capitalize()} heatmap → {path}")

    return results
