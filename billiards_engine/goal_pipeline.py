"""
Goal-only pipeline.

Input  : video + user-drawn pocket ROIs (no bounding box files, no ML)
Output : events/<clip_name>/goal_<frame>/  — frames around each goal event
         events/<clip_name>/goals.json     — event summary

Usage
-----
  python -m billiards_engine.goal_pipeline --clip game1_clip3
  python -m billiards_engine.goal_pipeline --clip game1_clip3 --reselect
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from collections import deque

from .video_loader import VideoLoader
from .pocket_roi_selector import select_pocket_rois
from .goal_detector import GoalDetector, GoalEvent

COLORS = [(0,255,80),(0,220,255),(0,80,255),(255,100,0),(255,0,200),(80,255,255)]

DATASET_DIR = Path(__file__).parent.parent / "billiards-orginal" / "dataset"
EVENTS_DIR  = Path(__file__).parent.parent / "events"

# How many frames to extract around each goal event
PRE_FRAMES  = [30, 20, 10, 5, 2]   # frames BEFORE the event
POST_FRAMES = [2, 5, 10, 20, 30]   # frames AFTER the event


def _draw_rois(
    frame: np.ndarray,
    rois: List[dict],
    fired_idxs: set = None,
    flash: bool = False,
) -> np.ndarray:
    out = frame.copy()
    fired_idxs = fired_idxs or set()
    for i, roi in enumerate(rois):
        cx, cy, r = roi["cx"], roi["cy"], roi["radius"]
        fired = i in fired_idxs
        color = (0, 50, 255) if (fired and flash) else COLORS[i]
        thick = 3 if (fired and flash) else 1
        cv2.circle(out, (cx, cy), r, color, thick)
        cv2.circle(out, (cx, cy), 3, color, -1)
        cv2.putText(out, f"P{i+1}", (cx + r + 3, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
    return out


def run_goal_pipeline(
    clip_dir: str,
    force_reselect: bool = False,
    pre_frames: List[int] = PRE_FRAMES,
    post_frames: List[int] = POST_FRAMES,
) -> List[GoalEvent]:
    clip_name = os.path.basename(clip_dir.rstrip("/\\"))
    video_path = os.path.join(clip_dir, f"{clip_name}.mp4")

    if not os.path.isfile(video_path):
        sys.exit(f"Video not found: {video_path}")

    print(f"\n{'='*60}")
    print(f"  Goal Detection Pipeline")
    print(f"  Clip : {clip_name}")

    # ── Step 1: pocket ROI selection ─────────────────────────────────────
    rois = select_pocket_rois(clip_dir, force_reselect=force_reselect)
    if not rois:
        sys.exit("No pocket ROIs — exiting.")

    # ── Step 2: open video (streaming — no full load into RAM) ──────────
    with VideoLoader(video_path) as loader:
        info = loader.info
        total = info.frame_count
        pre_buf_size = int(info.fps * 10)   # 10s rolling buffer
        post_buf_size = int(info.fps * 10)  # collect 10s after goal

        print(f"\n  {total} frames @ {info.fps:.2f}fps  ({info.width}x{info.height})")
        print(f"  Buffer: {pre_buf_size} frames ({pre_buf_size/info.fps:.0f}s) pre-goal")

        # ── Step 3: detector ─────────────────────────────────────────────
        print(f"\n  Running goal detector on {len(rois)} pocket ROIs...")
        detector = GoalDetector(
            rois=rois,
            background_frames=15,
            enter_threshold=4.0,
            exit_threshold=3.0,
            min_entry_frames=2,
            max_entry_frames=25,
            cooldown_frames=60,
        )

        # ── Step 4: streaming pass ────────────────────────────────────────
        out_base = EVENTS_DIR / clip_name
        out_base.mkdir(parents=True, exist_ok=True)

        full_video_path = str(out_base / f"{clip_name}_annotated.mp4")
        full_writer = cv2.VideoWriter(
            full_video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            info.fps, (info.width, info.height),
        )

        # Rolling buffer of (frame_id, annotated_frame) for pre-goal extraction
        pre_buffer: deque = deque(maxlen=pre_buf_size)

        all_events: List[GoalEvent] = []
        active_flashes: List[tuple] = []   # (expire_frame, pocket_idx)

        # Per-event post-goal collectors: {event_idx: [annotated_frames]}
        post_collectors: dict = {}

        for frame_id, frame in loader.frames():
            events = detector.process_frame(frame, frame_id)
            for ev in events:
                all_events.append(ev)
                ev_idx = len(all_events) - 1
                active_flashes.append((frame_id + int(info.fps * 1.5), ev.pocket_idx))
                post_collectors[ev_idx] = {"ev": ev, "frames": [], "done": False}
                print(f"\n  *** GOAL detected ***")
                print(f"      Pocket : {ev.label}")
                print(f"      Frame  : {ev.frame_id}  ({frame_id/info.fps:.2f}s)")
                print(f"      Peak activity: {ev.peak_activity}")

            # Annotate frame
            active_flashes = [(exp, idx) for exp, idx in active_flashes if frame_id <= exp]
            fired = {idx for _, idx in active_flashes}
            flash = bool(fired)

            out_f = _draw_rois(frame, rois, fired_idxs=fired, flash=flash)
            if flash:
                cv2.rectangle(out_f, (3,3), (info.width-3, info.height-3), (0,50,255), 3)

            ov = out_f.copy()
            cv2.rectangle(ov, (0,0), (info.width, 36), (10,10,10), -1)
            cv2.addWeighted(ov, 0.6, out_f, 0.4, 0, out_f)
            goal_labels = [ev.label for ev in all_events if ev.frame_id == frame_id]
            tag = f"GOAL! {', '.join(goal_labels)}" if goal_labels else ""
            tc  = (0, 50, 255) if goal_labels else (200, 200, 200)
            cv2.putText(out_f, f"Frame {frame_id}  {frame_id/info.fps:.2f}s  {tag}",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, tc, 1, cv2.LINE_AA)

            full_writer.write(out_f)

            # Store in rolling pre-buffer
            pre_buffer.append((frame_id, out_f))

            # Feed post-goal collectors
            for ev_idx, col in post_collectors.items():
                if not col["done"] and frame_id > col["ev"].frame_id:
                    col["frames"].append((frame_id, out_f))
                    if len(col["frames"]) >= post_buf_size:
                        col["done"] = True

            if frame_id % 300 == 0:
                print(f"    ...frame {frame_id}/{total}  ({frame_id/info.fps:.0f}s)", flush=True)

        full_writer.release()
        print(f"\n  Full annotated video → {full_video_path}")

    if not all_events:
        print("  No goal events detected.")
        print("  Tips: try --reselect and draw tighter ROIs around the pocket holes.")
        return []

    # ── Step 5: write goal highlight clips from collected buffers ─────────
    # Re-open video to grab raw pre-goal frames for the highlight clip
    summary = []

    for ev_idx, col in post_collectors.items():
        ev = col["ev"]
        folder_name = f"goal_frame{ev.frame_id:04d}_{ev.label.replace(' ', '_').lower()}"
        ev_dir = out_base / folder_name
        ev_dir.mkdir(exist_ok=True)

        t_event = ev.frame_id / info.fps
        saved_files = []

        # Re-read pre-goal frames directly from video (only ~300 frames)
        cap = cv2.VideoCapture(video_path)
        clip_start = max(0, ev.frame_id - pre_buf_size)
        clip_end   = min(total - 1, ev.frame_id + post_buf_size)

        goal_vid_path = str(ev_dir / "goal_clip.mp4")
        writer = cv2.VideoWriter(goal_vid_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                 info.fps, (info.width, info.height))

        cap.set(cv2.CAP_PROP_POS_FRAMES, clip_start)
        for fid in range(clip_start, clip_end + 1):
            ret, frame = cap.read()
            if not ret:
                break
            is_flash = abs(fid - ev.frame_id) <= int(info.fps * 0.5)
            out_f = _draw_rois(frame, rois,
                               fired_idxs={ev.pocket_idx} if is_flash else set(),
                               flash=is_flash)
            if fid == ev.frame_id:
                cv2.rectangle(out_f, (3,3), (info.width-3, info.height-3), (0,50,255), 4)
                tag, tc = "GOAL!", (0, 50, 255)
            else:
                df = fid - ev.frame_id
                tag = f"t+{df}f" if df > 0 else f"t{df}f"
                tc  = (0,120,255) if is_flash else (180,180,180)
            ov = out_f.copy()
            cv2.rectangle(ov, (0,0), (info.width, 38), (10,10,10), -1)
            cv2.addWeighted(ov, 0.6, out_f, 0.4, 0, out_f)
            cv2.putText(out_f, f"Frame {fid}  {fid/info.fps:.2f}s  [{tag}]",
                        (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, tc, 2, cv2.LINE_AA)

            # Save still frames at key offsets
            df = fid - ev.frame_id
            if df in (-(pre_buf_size//6), -(pre_buf_size//3), 0,
                      post_buf_size//6, post_buf_size//3):
                label = "EVENT" if df == 0 else (f"pre_{abs(df):04d}f" if df < 0 else f"post_{df:04d}f")
                fname = f"{label}_frame{fid:04d}.png"
                cv2.imwrite(str(ev_dir / fname), out_f)
                saved_files.append(fname)

            writer.write(out_f)
        cap.release()
        writer.release()
        saved_files.append("goal_clip.mp4")

        print(f"  Saved goal clip ({clip_end-clip_start+1} frames, {(clip_end-clip_start+1)/info.fps:.0f}s) → {ev_dir}")

        summary.append({
            "pocket": ev.label,
            "frame": ev.frame_id,
            "time_s": round(t_event, 3),
            "peak_activity": ev.peak_activity,
            "folder": str(ev_dir),
            "files_saved": saved_files,
        })

    json_path = out_base / "goals.json"
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n  Summary → {json_path}")
    print(f"  Total goals: {len(all_events)}")

    return all_events


def _label_frame(
    frame: np.ndarray,
    frame_id: int,
    fps: float,
    tag: str,
    color: tuple,
    big: bool = False,
):
    """Burn frame ID, timestamp, and event tag into the frame in-place."""
    h, w = frame.shape[:2]
    ts = frame_id / fps
    scale = 0.8 if big else 0.5
    thickness = 2 if big else 1

    # Dark bar at top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 36), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(
        frame, f"Frame {frame_id}  {ts:.2f}s  [{tag}]",
        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA,
    )


# ── CLI ──────────────────────────────────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser(description="Billiards goal detector")
    parser.add_argument("--dataset", default=str(DATASET_DIR))
    parser.add_argument("--clip", required=True, help="Clip folder name, e.g. game1_clip3")
    parser.add_argument("--reselect", action="store_true",
                        help="Force re-drawing pocket ROIs even if saved config exists")
    args = parser.parse_args(argv)

    clip_dir = Path(args.dataset) / args.clip
    if not clip_dir.is_dir():
        sys.exit(f"Clip not found: {clip_dir}")

    run_goal_pipeline(str(clip_dir), force_reselect=args.reselect)
    print("\nDone.")


if __name__ == "__main__":
    main()
