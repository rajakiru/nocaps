"""
prepare_yolo_data.py — convert manually annotated frames to YOLO format.

Reads the 8 hand-labeled frame/bbox pairs across the 4 game1 clips and
writes them into the canonical YOLO directory layout, plus data.yaml.

Input (per clip):
  billiards_dataset/game1_clipN/frames/frame_{first,last}.png
  billiards_dataset/game1_clipN/bounding_boxes/frame_{first,last}_bbox.txt

  bbox format (one ball per line):
    x  y  w  h  category
    (top-left pixel coords, absolute; category 1=cue 2=8ball 3=solid 4=striped)

Output:
  yolo_dataset/
    images/train/   ← 6 PNGs
    images/val/     ← 2 PNGs  (clip1_last, clip2_last)
    labels/train/   ← matching .txt files
    labels/val/
    data.yaml

Usage:
  python scripts/prepare_yolo_data.py
  python scripts/prepare_yolo_data.py --dataset-dir billiards_dataset --out yolo_dataset
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import yaml

# Val set: one frame per camera angle to keep val representative.
# Everything else goes to train.
_VAL_KEYS = {"game1_clip1_frame_last", "game1_clip2_frame_last"}

# YOLO class IDs are 0-indexed; our categories are 1-indexed.
_CAT_TO_CLASS = {1: 0, 2: 1, 3: 2, 4: 3}
_CLASS_NAMES   = ["cue", "8ball", "solid", "striped"]


def _parse_bbox_file(path: Path, img_w: int, img_h: int) -> list[str]:
    """Parse one bbox txt → list of YOLO label lines."""
    lines = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        x, y, w, h, cat = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
        if cat not in _CAT_TO_CLASS:
            continue
        cls  = _CAT_TO_CLASS[cat]
        cx_n = (x + w / 2) / img_w
        cy_n = (y + h / 2) / img_h
        w_n  = w / img_w
        h_n  = h / img_h
        lines.append(f"{cls} {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}")
    return lines


def prepare(dataset_dir: Path, out_dir: Path) -> None:
    clips = sorted(dataset_dir.glob("game1_clip*"))
    if not clips:
        raise FileNotFoundError(f"No game1_clip* directories found in {dataset_dir}")

    train_img = out_dir / "images" / "train"
    val_img   = out_dir / "images" / "val"
    train_lbl = out_dir / "labels" / "train"
    val_lbl   = out_dir / "labels" / "val"
    for d in (train_img, val_img, train_lbl, val_lbl):
        d.mkdir(parents=True, exist_ok=True)

    written = {"train": 0, "val": 0}

    for clip_dir in clips:
        clip_name = clip_dir.name

        for which in ("first", "last"):
            img_path  = clip_dir / "frames"       / f"frame_{which}.png"
            bbox_path = clip_dir / "bounding_boxes" / f"frame_{which}_bbox.txt"

            if not img_path.exists() or not bbox_path.exists():
                print(f"  [skip] {clip_name} frame_{which}: missing image or bbox")
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  [skip] {clip_name} frame_{which}: could not read image")
                continue
            img_h, img_w = img.shape[:2]

            label_lines = _parse_bbox_file(bbox_path, img_w, img_h)
            if not label_lines:
                print(f"  [skip] {clip_name} frame_{which}: no valid labels")
                continue

            key    = f"{clip_name}_frame_{which}"
            stem   = f"{clip_name}_{which}"
            split  = "val" if key in _VAL_KEYS else "train"

            dst_img = (val_img if split == "val" else train_img) / f"{stem}.png"
            dst_lbl = (val_lbl if split == "val" else train_lbl) / f"{stem}.txt"

            shutil.copy2(img_path, dst_img)
            dst_lbl.write_text("\n".join(label_lines))

            written[split] += 1
            print(f"  [{split:5s}] {stem}.png  ({len(label_lines)} balls)")

    # data.yaml — use absolute path so YOLOv8 can be called from anywhere
    yaml_data = {
        "path":  str(out_dir.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "nc":    len(_CLASS_NAMES),
        "names": _CLASS_NAMES,
    }
    yaml_path = out_dir / "data.yaml"
    yaml_path.write_text(yaml.dump(yaml_data, default_flow_style=False, sort_keys=False))

    print(f"\n  Written: train={written['train']}  val={written['val']}")
    print(f"  data.yaml → {yaml_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert manual annotations to YOLO format")
    parser.add_argument("--dataset-dir", default="billiards_dataset", type=Path)
    parser.add_argument("--out",         default="yolo_dataset",       type=Path)
    args = parser.parse_args()
    prepare(args.dataset_dir, args.out)


if __name__ == "__main__":
    main()
