"""
Runner for arbitrary MOV/MP4 videos with configurable felt color.

Steps:
  1. Optionally trim to a time window
  2. Prompt user to click 6 pocket ROIs
  3. Run goal detection
  4. Output full annotated video + goal highlight clip

Usage
-----
  # Trim first 60s, red felt
  python -m billiards_engine.run_mov --input /Users/kiruthikaraja/CMU/Spring 2026/Sports tech/billiard-video-analysis-main/IMG_4841.MOV --start 0 --end 60 --felt red

  # Full video, blue felt
  python -m billiards_engine.run_mov --input game.mp4 --felt blue

  # Re-annotate pockets
  python -m billiards_engine.run_mov --input IMG_4841.MOV --felt red --reselect
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .trim_video import trim_video
from .goal_pipeline import run_goal_pipeline, EVENTS_DIR
from .pocket_roi_selector import select_pocket_rois

# patch opencv_detector to use felt config — passed via env-like global
import billiards_engine.opencv_detector as _det_mod
from .felt_config import get_felt_mask as _get_felt_mask


def _patch_detector(felt: str):
    """Monkey-patch the detector's felt lookup to use the chosen color."""
    import numpy as np, cv2
    from .felt_config import get_felt_mask

    orig_estimate = _det_mod.estimate_table_bbox

    def patched_estimate(frame, felt_override=felt):
        return orig_estimate(frame, felt=felt_override)

    _det_mod.estimate_table_bbox = patched_estimate

    # Patch OpenCVBallDetector default felt
    _det_mod.OpenCVBallDetector.__init__.__defaults__ = (
        4, 16, None, felt
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run goal detection on a single video")
    parser.add_argument("--input",   required=True, help="Path to video file (.MOV, .mp4, etc.)")
    parser.add_argument("--felt",    default="blue", choices=["blue", "red", "green"],
                        help="Table felt color (default: blue)")
    parser.add_argument("--start",   type=float, default=None, help="Trim start time (seconds)")
    parser.add_argument("--end",     type=float, default=None, help="Trim end time (seconds)")
    parser.add_argument("--reselect", action="store_true", help="Force re-annotating pocket ROIs")
    args = parser.parse_args(argv)

    input_path = args.input
    if not os.path.isfile(input_path):
        sys.exit(f"File not found: {input_path}")

    # ── Step 1: trim if requested ────────────────────────────────────────
    if args.start is not None or args.end is not None:
        import cv2
        cap = cv2.VideoCapture(input_path)
        fps   = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        start_s = args.start or 0.0
        end_s   = args.end   or (total / fps)

        base    = os.path.splitext(input_path)[0]
        trimmed = f"{base}_trim_{int(start_s)}s_{int(end_s)}s.mp4"

        if not os.path.isfile(trimmed):
            print(f"\nTrimming {start_s:.0f}s → {end_s:.0f}s ...")
            trim_video(input_path, trimmed, start_s, end_s)
        else:
            print(f"\nUsing existing trimmed file: {trimmed}")

        input_path = trimmed

    # ── Step 2: set up output dir alongside the input file ───────────────
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    clip_dir  = Path(input_path).parent / base_name
    clip_dir.mkdir(exist_ok=True)

    # Symlink/copy video into clip_dir with matching name so pipeline finds it
    target_video = clip_dir / f"{base_name}.mp4"
    if not target_video.exists():
        # hard link so we don't double the disk space
        try:
            os.link(input_path, str(target_video))
        except OSError:
            import shutil
            shutil.copy2(input_path, str(target_video))

    # ── Step 3: patch felt color ─────────────────────────────────────────
    _patch_detector(args.felt)
    print(f"\n  Felt color: {args.felt}")

    # Override EVENTS_DIR output to sit next to the source file
    import billiards_engine.goal_pipeline as gp
    gp.EVENTS_DIR = clip_dir.parent / "events"

    # ── Step 4: run pipeline ─────────────────────────────────────────────
    events = run_goal_pipeline(str(clip_dir), force_reselect=args.reselect)

    print(f"\nDone. {len(events)} goal(s) detected.")
    out_base = gp.EVENTS_DIR / base_name
    print(f"Outputs → {out_base}")


if __name__ == "__main__":
    main()
