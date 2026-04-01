"""
Trim a video to a specific time range and save as MP4.

Usage
-----
  python -m billiards_engine.trim_video --input IMG_4841.MOV --start 0 --end 30
  python -m billiards_engine.trim_video --input IMG_4841.MOV --start 10 --end 50 --out trimmed.mp4

Times are in seconds.  Default: first 60 seconds.
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2


def trim_video(
    input_path: str,
    output_path: str,
    start_s: float = 0.0,
    end_s: float = 60.0,
) -> str:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        sys.exit(f"Cannot open: {input_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dur    = total / fps

    start_f = int(start_s * fps)
    end_f   = min(int(end_s * fps), total - 1)

    print(f"  Input  : {input_path}")
    print(f"  Duration: {dur:.1f}s  ({total} frames @ {fps:.2f}fps  {w}x{h})")
    print(f"  Trimming: {start_s:.1f}s → {end_s:.1f}s  (frames {start_f}–{end_f})")
    print(f"  Output : {output_path}")

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (w, h),
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    for fid in range(start_f, end_f + 1):
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)

    cap.release()
    writer.release()
    print(f"  Done — {end_f - start_f + 1} frames written")
    return output_path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Trim a video to a time range")
    parser.add_argument("--input",  required=True, help="Input video path")
    parser.add_argument("--start",  type=float, default=0.0,  help="Start time in seconds")
    parser.add_argument("--end",    type=float, default=60.0, help="End time in seconds")
    parser.add_argument("--out",    default=None, help="Output path (default: <input>_trimmed.mp4)")
    args = parser.parse_args(argv)

    if args.out is None:
        base = os.path.splitext(args.input)[0]
        args.out = f"{base}_trimmed_{int(args.start)}s_{int(args.end)}s.mp4"

    trim_video(args.input, args.out, args.start, args.end)


if __name__ == "__main__":
    main()
