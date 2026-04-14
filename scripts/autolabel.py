"""
autolabel.py — propagate first-frame category labels through entire clips.

Strategy
--------
1. Seed CentroidTracker from the manually annotated first-frame bounding boxes.
   These detections carry the correct ball category (1=cue, 2=8ball, 3=solid,
   4=striped) and a stable ball_id from the annotation file row order.

2. For every subsequent frame, run OpenCVBallDetector to get ball positions.
   Because OpenCV returns category=0, the tracker preserves each seeded track's
   category on match — category is only overwritten when the incoming detection
   has category != 0.  (See CentroidTracker.update(), line 128-129.)

3. For each frame we keep only tracks that:
   - Were actually detected this frame (missing_frames == 0, not just predicted)
   - Have a known category (category != 0)

4. Write YOLO label files for all qualifying frames, images to yolo_dataset/.
   Frames are split 80/20 train/val per clip (deterministic, seed=42).

Output adds to yolo_dataset/ (created by prepare_yolo_data.py if you ran that
first, or created fresh here):
  yolo_dataset/images/{train,val}/game1_clipN_frameXXXX.png
  yolo_dataset/labels/{train,val}/game1_clipN_frameXXXX.txt
  yolo_dataset/data.yaml  (written/updated at end)

Usage:
  # Run from the nocaps/ root
  python scripts/autolabel.py
  python scripts/autolabel.py --stride 2 --felt blue --min-tracks 4
  python scripts/autolabel.py --clips billiards_dataset/game1_clip1 billiards_dataset/game1_clip3
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

# Add repo root to path so billiards_engine imports work when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from billiards_engine.detection_loader import Detection, DetectionLoader
from billiards_engine.opencv_detector import OpenCVBallDetector, estimate_table_bbox
from billiards_engine.tracker import CentroidTracker

_CLASS_NAMES  = ["cue", "8ball", "solid", "striped"]
_CAT_TO_CLASS = {1: 0, 2: 1, 3: 2, 4: 3}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_bbox_txt(path: Path, frame_id: int) -> list[Detection]:
    """Parse frame_*_bbox.txt → list of Detection with correct ball_ids/categories."""
    dets = []
    with open(path) as fh:
        for ball_id, line in enumerate(fh, start=1):
            parts = line.split()
            if len(parts) < 5:
                continue
            x, y, w, h, cat = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), int(parts[4])
            if cat == 5:   # "playing field" annotation — skip
                continue
            dets.append(Detection(
                frame_id=frame_id, ball_id=ball_id,
                x=x, y=y, w=w, h=h, category=cat,
            ))
    return dets


def _track_to_yolo(track, img_w: int, img_h: int) -> str | None:
    """Convert a Track to a YOLO label line, or None if category unknown."""
    if track.category not in _CAT_TO_CLASS:
        return None
    cls  = _CAT_TO_CLASS[track.category]
    # Estimate bbox from centroid; use a fixed radius matching typical ball size
    # (the tracker only stores cx/cy, not w/h — we use the median ball size ~18px)
    r    = 9.0   # half-width in pixels (ball diameter ≈ 18px at 1024×576)
    cx_n = track.cx / img_w
    cy_n = track.cy / img_h
    w_n  = (r * 2) / img_w
    h_n  = (r * 2) / img_h
    # Clamp to [0,1]
    cx_n = max(0.0, min(1.0, cx_n))
    cy_n = max(0.0, min(1.0, cy_n))
    return f"{cls} {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}"


def _write_data_yaml(out_dir: Path) -> None:
    yaml_data = {
        "path":  str(out_dir.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "nc":    len(_CLASS_NAMES),
        "names": _CLASS_NAMES,
    }
    yaml_path = out_dir / "data.yaml"
    yaml_path.write_text(yaml.dump(yaml_data, default_flow_style=False, sort_keys=False))
    print(f"  data.yaml → {yaml_path}")


# ── Per-clip autolabeler ───────────────────────────────────────────────────────

def autolabel_clip(
    clip_dir: Path,
    out_dir: Path,
    felt: str = "blue",
    stride: int = 2,
    min_known_tracks: int = 4,
    val_fraction: float = 0.2,
    rng: random.Random = None,
) -> dict:
    """
    Autolabel one clip.  Returns stats dict.
    """
    clip_name = clip_dir.name
    video_path = clip_dir / f"{clip_name}.mp4"
    bbox_first = clip_dir / "bounding_boxes" / "frame_first_bbox.txt"

    if not video_path.exists():
        print(f"  [skip] {clip_name}: video not found")
        return {}
    if not bbox_first.exists():
        print(f"  [skip] {clip_name}: no frame_first_bbox.txt")
        return {}

    cap   = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    print(f"\n  {clip_name}: {total} frames  {img_w}x{img_h}  {fps:.1f}fps")

    # Seed tracker from first frame manual annotations
    first_dets = _load_bbox_txt(bbox_first, frame_id=0)
    print(f"    Seeded {len(first_dets)} balls from frame_first_bbox.txt")

    tracker = CentroidTracker(max_distance=60.0, max_missing=30)
    tracker.seed(first_dets, frame_id=0)

    # Build detector (table bbox from first frame)
    cap = cv2.VideoCapture(str(video_path))
    ret, first_frame = cap.read()
    cap.release()
    table_bbox = estimate_table_bbox(first_frame, felt=felt) if ret else None
    if table_bbox is None:
        table_bbox = (0, 0, img_w, img_h)

    detector = OpenCVBallDetector(table_bbox=table_bbox, felt=felt)

    # Decide train/val split for this clip's frames upfront
    candidate_frames = list(range(0, total, stride))
    if rng:
        rng.shuffle(candidate_frames)
    n_val   = max(1, int(len(candidate_frames) * val_fraction))
    val_set = set(candidate_frames[:n_val])

    # Create output dirs
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Main pass
    cap = cv2.VideoCapture(str(video_path))
    stats = {"train": 0, "val": 0, "skipped_few_tracks": 0, "skipped_stride": 0}

    for frame_id in range(total):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id == 0:
            # Use seeded detections directly so frame 0 is consistent with annotation
            tracker.update(first_dets, frame_id)
        else:
            dets   = detector.detect(frame, frame_id)
            tracker.update(dets, frame_id)

        # Only write frames that are on stride boundary
        if frame_id % stride != 0:
            stats["skipped_stride"] += 1
            continue

        # Build YOLO labels — only tracks actively detected this frame with known category
        label_lines = []
        for t in tracker._active_tracks():
            if t.missing_frames > 0:
                continue   # not seen this frame — skip to avoid position drift
            line = _track_to_yolo(t, img_w, img_h)
            if line:
                label_lines.append(line)

        if len(label_lines) < min_known_tracks:
            stats["skipped_few_tracks"] += 1
            continue

        split = "val" if frame_id in val_set else "train"
        stem  = f"{clip_name}_frame{frame_id:04d}"

        img_dst = out_dir / "images" / split / f"{stem}.png"
        lbl_dst = out_dir / "labels" / split / f"{stem}.txt"

        cv2.imwrite(str(img_dst), frame)
        lbl_dst.write_text("\n".join(label_lines))
        stats[split] += 1

    cap.release()

    print(f"    train={stats['train']}  val={stats['val']}"
          f"  skipped(few_tracks)={stats['skipped_few_tracks']}"
          f"  skipped(stride)={stats['skipped_stride']}")
    return stats


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Auto-label clips via track propagation")
    parser.add_argument("--dataset-dir",  default="billiards_dataset", type=Path)
    parser.add_argument("--out",          default="yolo_dataset",       type=Path)
    parser.add_argument("--felt",         default="blue", choices=["blue", "red", "green"])
    parser.add_argument("--stride",       default=2, type=int,
                        help="Save every Nth frame (default 2 = every other frame)")
    parser.add_argument("--min-tracks",   default=4, type=int,
                        help="Skip frames with fewer than N known-category tracks")
    parser.add_argument("--val-fraction", default=0.2, type=float,
                        help="Fraction of frames to put in val split (default 0.2)")
    parser.add_argument("--seed",         default=42, type=int,
                        help="Random seed for train/val split")
    parser.add_argument("--clips",        nargs="*", type=Path,
                        help="Specific clip dirs to process (default: all game1_clip*)")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    if args.clips:
        clip_dirs = sorted(args.clips)
    else:
        clip_dirs = sorted(args.dataset_dir.glob("game1_clip*"))

    if not clip_dirs:
        sys.exit(f"No clips found in {args.dataset_dir}")

    print(f"Processing {len(clip_dirs)} clips → {args.out}")
    total_stats: dict = {"train": 0, "val": 0}

    for clip_dir in clip_dirs:
        s = autolabel_clip(
            clip_dir     = clip_dir,
            out_dir      = args.out,
            felt         = args.felt,
            stride       = args.stride,
            min_known_tracks = args.min_tracks,
            val_fraction = args.val_fraction,
            rng          = rng,
        )
        for k in ("train", "val"):
            total_stats[k] += s.get(k, 0)

    _write_data_yaml(args.out)
    print(f"\nTotal: train={total_stats['train']}  val={total_stats['val']}")
    print(f"Dataset ready at: {args.out.resolve()}")


if __name__ == "__main__":
    main()
