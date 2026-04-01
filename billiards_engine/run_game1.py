"""
Combined runner for all game1 clips.

For each clip:
  1. Open pocket ROI selector if no saved config exists
  2. Run goal detector
  3. If goal found  → save goal_clip.mp4 + still frames to events/
  4. If no goal     → run full annotated video pipeline as fallback

Usage
-----
  python -m billiards_engine.run_game1
  python -m billiards_engine.run_game1 --reselect   # redo pocket ROIs for all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .goal_pipeline import run_goal_pipeline, DATASET_DIR
from .pipeline import run_clip

GAME1_CLIPS = ["game1_clip1", "game1_clip2", "game1_clip3", "game1_clip4"]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run goal detection on all game1 clips")
    parser.add_argument("--dataset", default=str(DATASET_DIR))
    parser.add_argument("--reselect", action="store_true",
                        help="Force re-drawing pocket ROIs for every clip")
    args = parser.parse_args(argv)

    dataset = Path(args.dataset)
    results = {}

    for clip_name in GAME1_CLIPS:
        clip_dir = dataset / clip_name
        if not clip_dir.is_dir():
            print(f"\n  Skipping {clip_name} — folder not found")
            continue

        print(f"\n{'='*60}")
        print(f"  Processing: {clip_name}")

        # ── Step 1: goal detection ───────────────────────────────────────
        events = run_goal_pipeline(
            str(clip_dir),
            force_reselect=args.reselect,
        )

        if events:
            print(f"\n  ✓ {len(events)} goal(s) detected — event video saved")
            results[clip_name] = f"{len(events)} goal(s)"

        else:
            # ── Step 2: fallback — regular annotated video ───────────────
            print(f"\n  No goals detected — generating annotated video...")
            run_clip(
                clip_dir=str(clip_dir),
                save_video=True,
                annotate_pockets_ui=False,   # don't re-open UI
            )
            results[clip_name] = "no goal — annotated video saved"

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"  {'─'*40}")
    for clip, status in results.items():
        print(f"  {clip:15s}  {status}")
    print()


if __name__ == "__main__":
    main()
