"""
Run goal detection + full annotated video on every clip in the dataset.

For each clip:
  - Opens pocket ROI selector if no saved config exists
  - Generates:
      events/<clip>/<clip>_annotated.mp4     — full clip with pocket circles
      events/<clip>/goal_frameXXXX_*/        — goal highlight folder
        ├── goal_clip.mp4                    — ±30 frame highlight
        ├── EVENT_frameXXXX.png
        ├── pre_XXf_frameXXXX.png  (x5)
        └── post_XXf_frameXXXX.png (x5)
      events/<clip>/goals.json               — event summary

Usage
-----
  python -m billiards_engine.run_all
  python -m billiards_engine.run_all --reselect   # redo ROIs for all clips
  python -m billiards_engine.run_all --skip game2_clip1 game3_clip2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .goal_pipeline import run_goal_pipeline, DATASET_DIR

ALL_CLIPS = [
    "game1_clip1", "game1_clip2", "game1_clip3", "game1_clip4",
    "game2_clip1", "game2_clip2",
    "game3_clip1", "game3_clip2",
    "game4_clip1", "game4_clip2",
]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run all billiards clips")
    parser.add_argument("--dataset", default=str(DATASET_DIR))
    parser.add_argument("--reselect", action="store_true",
                        help="Force re-drawing pocket ROIs for every clip")
    parser.add_argument("--skip", nargs="*", default=[],
                        help="Clip names to skip")
    parser.add_argument("--only", nargs="*", default=[],
                        help="Only run these clips")
    args = parser.parse_args(argv)

    dataset = Path(args.dataset)
    clips = args.only if args.only else ALL_CLIPS
    clips = [c for c in clips if c not in args.skip]

    results = {}

    for clip_name in clips:
        clip_dir = dataset / clip_name
        if not clip_dir.is_dir():
            print(f"\n  Skipping {clip_name} — folder not found")
            continue

        events = run_goal_pipeline(str(clip_dir), force_reselect=args.reselect)
        results[clip_name] = len(events)

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  FINAL SUMMARY")
    print(f"  {'─'*40}")
    for clip, n in results.items():
        status = f"{n} goal(s) detected" if n else "no goals — annotated video only"
        print(f"  {clip:18s}  {status}")
    print()


if __name__ == "__main__":
    main()
