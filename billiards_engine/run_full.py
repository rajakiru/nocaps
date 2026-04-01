"""
Full pipeline: ball tracking + goal detection in a single streaming pass.

For each frame:
  - OpenCV ball detector finds balls
  - Centroid tracker assigns stable IDs + builds trajectories
  - Goal detector monitors pocket ROIs
  - Visualizer draws ball circles, trails, pocket circles, event flashes
  - Goal highlight clips extracted around any detected goals

Usage
-----
  python -m billiards_engine.run_full --input IMG_4835.MOV --felt red
  python -m billiards_engine.run_full --input IMG_4841.MOV --felt red --start 0 --end 60
  python -m billiards_engine.run_full --input game.mp4 --felt blue
  python -m billiards_engine.run_full --input IMG_4835.MOV --felt red --reselect
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import deque
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from .trim_video import trim_video
from .video_loader import VideoLoader
from .pocket_roi_selector import select_pocket_rois
from .goal_detector import GoalDetector, GoalEvent
from .opencv_detector import OpenCVBallDetector, estimate_table_bbox
from .tracker import CentroidTracker
from .trajectory_builder import TrajectoryBuilder
from .felt_config import get_felt_mask
from .goal_pipeline import _draw_rois, _label_frame, COLORS


def _draw_balls(frame: np.ndarray, active_tracks, trail_length: int = 20) -> np.ndarray:
    """Draw ball trajectories and markers onto the frame."""
    CAT_COLORS = {
        0: (0, 220, 220),
        1: (255, 255, 255),
        2: (100, 100, 100),
        3: (0, 140, 255),
        4: (255, 200, 0),
    }
    out = frame
    for track in active_tracks:
        color = CAT_COLORS.get(track.category, CAT_COLORS[0])
        positions = track.positions[-trail_length:]
        n = len(positions)
        for k in range(1, n):
            _, x0, y0 = positions[k - 1]
            _, x1, y1 = positions[k]
            alpha = (k / n) ** 0.7
            c = tuple(int(ch * alpha) for ch in color)
            cv2.line(out, (int(x0), int(y0)), (int(x1), int(y1)), c, 2 if track.category == 1 else 1)
        cx, cy = int(track.cx), int(track.cy)
        cv2.circle(out, (cx, cy), 9, color, 2)
        if track.category == 1:
            cv2.circle(out, (cx, cy), 4, color, -1)
        cv2.putText(out, str(track.id), (cx + 11, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
    return out


def run_full_pipeline(
    input_path: str,
    felt: str = "red",
    start_s: Optional[float] = None,
    end_s: Optional[float] = None,
    force_reselect: bool = False,
):
    # ── Trim if needed ────────────────────────────────────────────────────
    if start_s is not None or end_s is not None:
        cap = cv2.VideoCapture(input_path)
        fps_tmp = cap.get(cv2.CAP_PROP_FPS)
        total_tmp = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        s = start_s or 0.0
        e = end_s or (total_tmp / fps_tmp)
        base = os.path.splitext(input_path)[0]
        trimmed = f"{base}_trim_{int(s)}s_{int(e)}s.mp4"
        if not os.path.isfile(trimmed):
            print(f"Trimming {s:.0f}s → {e:.0f}s ...")
            trim_video(input_path, trimmed, s, e)
        else:
            print(f"Using existing trim: {trimmed}")
        input_path = trimmed

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    clip_dir  = Path(input_path).parent / base_name
    clip_dir.mkdir(exist_ok=True)

    target_video = clip_dir / f"{base_name}.mp4"
    if not target_video.exists():
        try:
            os.link(input_path, str(target_video))
        except OSError:
            import shutil; shutil.copy2(input_path, str(target_video))

    out_dir = clip_dir.parent / "events" / base_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Pocket ROIs ───────────────────────────────────────────────────────
    rois = select_pocket_rois(str(clip_dir), force_reselect=force_reselect)
    if not rois:
        sys.exit("No pocket ROIs — exiting.")

    # ── Open video ────────────────────────────────────────────────────────
    with VideoLoader(str(target_video)) as loader:
        info = loader.info
        total = info.frame_count
        pre_buf_size  = int(info.fps * 10)
        post_buf_size = int(info.fps * 10)

        print(f"\n  {total} frames @ {info.fps:.2f}fps  {info.width}x{info.height}")
        print(f"  Felt: {felt}  |  Pre/post buffer: {pre_buf_size/info.fps:.0f}s each")

        # ── Build components ──────────────────────────────────────────────
        table_bbox = estimate_table_bbox(loader.get_frame(0), felt=felt)
        if table_bbox is None:
            table_bbox = (0, 0, info.width, info.height)
        print(f"  Table bbox: {table_bbox}")

        ball_detector  = OpenCVBallDetector(table_bbox=table_bbox, felt=felt)
        tracker        = CentroidTracker(max_distance=80.0, max_missing=8)
        traj_builder   = TrajectoryBuilder(window=40, smooth_k=5, fps=info.fps)
        goal_detector  = GoalDetector(
            rois=rois,
            background_frames=15,
            enter_threshold=4.0,
            exit_threshold=3.0,
            min_entry_frames=2,
            max_entry_frames=25,
            cooldown_frames=60,
        )

        # ── Video writer ──────────────────────────────────────────────────
        annotated_path = str(out_dir / f"{base_name}_annotated.mp4")
        writer = cv2.VideoWriter(
            annotated_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            info.fps, (info.width, info.height),
        )

        all_goals: List[GoalEvent] = []
        active_flashes: List[tuple] = []
        post_collectors: dict = {}

        print(f"\n  Processing frames...")
        for frame_id, frame in loader.frames():
            # Ball tracking
            detections = ball_detector.detect(frame, frame_id)
            active     = tracker.update(detections, frame_id)
            traj_builder.update(active, frame_id)

            # Goal detection
            goal_events = goal_detector.process_frame(frame, frame_id)
            for ev in goal_events:
                all_goals.append(ev)
                ev_idx = len(all_goals) - 1
                active_flashes.append((frame_id + int(info.fps * 1.5), ev.pocket_idx))
                post_collectors[ev_idx] = {"ev": ev, "frames": [], "done": False}
                print(f"\n  *** GOAL ***  {ev.label}  frame={ev.frame_id}  t={frame_id/info.fps:.2f}s")

            # Annotate: pockets
            active_flashes = [(exp, idx) for exp, idx in active_flashes if frame_id <= exp]
            fired = {idx for _, idx in active_flashes}
            out_f = _draw_rois(frame, rois, fired_idxs=fired, flash=bool(fired))

            if fired:
                cv2.rectangle(out_f, (3,3), (info.width-3, info.height-3), (0,50,255), 3)

            # Annotate: balls + trajectories
            out_f = _draw_balls(out_f, active)

            # HUD
            ov = out_f.copy()
            cv2.rectangle(ov, (0,0), (info.width, 38), (10,10,10), -1)
            cv2.addWeighted(ov, 0.6, out_f, 0.4, 0, out_f)
            goal_labels = [ev.label for ev in all_goals if ev.frame_id == frame_id]
            tag = f"GOAL! {', '.join(goal_labels)}" if goal_labels else ""
            tc  = (0, 50, 255) if goal_labels else (200, 200, 200)
            cv2.putText(out_f, f"Frame {frame_id}  {frame_id/info.fps:.2f}s  tracks={len(active)}  {tag}",
                        (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, tc, 1, cv2.LINE_AA)

            writer.write(out_f)

            if frame_id % 300 == 0:
                print(f"    frame {frame_id}/{total}  ({frame_id/info.fps:.0f}s)  tracks={len(active)}", flush=True)

        writer.release()
        print(f"\n  Annotated video → {annotated_path}")

    # ── Goal highlight clips ──────────────────────────────────────────────
    summary = []
    for ev_idx, col in post_collectors.items():
        ev = col["ev"]
        folder_name = f"goal_frame{ev.frame_id:04d}_{ev.label.replace(' ', '_').lower()}"
        ev_dir = out_dir / folder_name
        ev_dir.mkdir(exist_ok=True)

        clip_start = max(0, ev.frame_id - pre_buf_size)
        clip_end   = min(total - 1, ev.frame_id + post_buf_size)

        goal_vid = str(ev_dir / "goal_clip.mp4")
        cap = cv2.VideoCapture(str(target_video))
        gw  = cv2.VideoWriter(goal_vid, cv2.VideoWriter_fourcc(*"mp4v"),
                              info.fps, (info.width, info.height))
        cap.set(cv2.CAP_PROP_POS_FRAMES, clip_start)
        saved_stills = []
        for fid in range(clip_start, clip_end + 1):
            ret, frame = cap.read()
            if not ret: break
            is_flash = abs(fid - ev.frame_id) <= int(info.fps * 0.5)
            out_f = _draw_rois(frame, rois,
                               fired_idxs={ev.pocket_idx} if is_flash else set(),
                               flash=is_flash)
            if fid == ev.frame_id:
                cv2.rectangle(out_f,(3,3),(info.width-3,info.height-3),(0,50,255),4)
                tag, tc = "GOAL!", (0,50,255)
            else:
                df = fid - ev.frame_id
                tag = f"t+{df}f" if df>0 else f"t{df}f"
                tc  = (0,120,255) if is_flash else (180,180,180)
            ov = out_f.copy()
            cv2.rectangle(ov,(0,0),(info.width,38),(10,10,10),-1)
            cv2.addWeighted(ov,0.6,out_f,0.4,0,out_f)
            cv2.putText(out_f, f"Frame {fid}  {fid/info.fps:.2f}s  [{tag}]",
                        (12,26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, tc, 2, cv2.LINE_AA)

            # Save key stills
            df = fid - ev.frame_id
            if df in (-(pre_buf_size//3), -(pre_buf_size//6), 0,
                       post_buf_size//6, post_buf_size//3):
                lbl = "EVENT" if df==0 else (f"pre_{abs(df):04d}f" if df<0 else f"post_{df:04d}f")
                fname = f"{lbl}_frame{fid:04d}.png"
                cv2.imwrite(str(ev_dir / fname), out_f)
                saved_stills.append(fname)

            gw.write(out_f)
        cap.release()
        gw.release()
        print(f"  Goal clip ({(clip_end-clip_start+1)/info.fps:.0f}s) → {ev_dir}")

        summary.append({
            "pocket": ev.label, "frame": ev.frame_id,
            "time_s": round(ev.frame_id / info.fps, 3),
            "peak_activity": ev.peak_activity,
            "goal_clip": goal_vid, "stills": saved_stills,
        })

    json_path = out_dir / "goals.json"
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n  Summary → {json_path}")
    print(f"  Total goals: {len(all_goals)}")
    print(f"\nOutputs:")
    print(f"  {annotated_path}")
    for s in summary:
        print(f"  {s['goal_clip']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Full billiards pipeline: balls + goals")
    parser.add_argument("--input",    required=True)
    parser.add_argument("--felt",     default="red", choices=["blue","red","green"])
    parser.add_argument("--start",    type=float, default=None)
    parser.add_argument("--end",      type=float, default=None)
    parser.add_argument("--reselect", action="store_true")
    args = parser.parse_args(argv)

    if not os.path.isfile(args.input):
        sys.exit(f"File not found: {args.input}")

    run_full_pipeline(args.input, args.felt, args.start, args.end, args.reselect)
    print("\nDone.")


if __name__ == "__main__":
    main()
