"""
Full pipeline v2: YOLO ball tracking + goal detection + heatmaps.

Changes from run_full.py:
  - YOLOBallDetector instead of OpenCVBallDetector (falls back to OpenCV if
    model is missing, so --model is always optional)
  - GoalDetectorV2 instead of GoalDetector (optical flow + smoothing)
  - CameraManager: supports two camera angles with automatic cut detection.
    Use --two-cameras to enable; each camera has its own pocket ROI file
    and independent background model.  Override CameraManager._select_active()
    for automatic "best view" selection in the future.
  - BallHeatmap + PlayerHeatmap built during the main pass
  - Output video  : {base_name}_annotated_v2.mp4
  - Output JSON   : goals_v2.json  (includes "confidence" and "camera" per event)
  - Output images : {base_name}_ball_heatmap.png / {base_name}_player_heatmap.png
  - New CLI args  : --model, --conf, --device, --two-cameras

Usage
-----
  # Single camera (default, backwards-compatible)
  python -m billiards_engine.run_full_v2 --input game.mp4 --felt blue

  # Two cameras (annotate both angles at first run, then cached)
  python -m billiards_engine.run_full_v2 --input game.mp4 --felt blue --two-cameras

  # With YOLO model
  python -m billiards_engine.run_full_v2 --input game.mp4 --felt blue \\
      --model models/best.pt --conf 0.35 --device cuda
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

from .trim_video import trim_video
from .video_loader import VideoLoader
from .goal_detector_v2 import GoalDetectorV2, GoalEvent
from .camera_manager import CameraManager, load_or_select_rois
from .yolo_detector import YOLOBallDetector
from .opencv_detector import estimate_table_bbox
from .heatmap import BallHeatmap, PlayerHeatmap, save_heatmaps
from .tracker import CentroidTracker
from .trajectory_builder import TrajectoryBuilder
from .goal_pipeline import _draw_rois, COLORS

_GOAL_DETECTOR_PARAMS = dict(
    background_frames = 15,
    enter_threshold   = 4.0,
    exit_threshold    = 2.5,
    min_entry_frames  = 2,
    max_entry_frames  = 60,    # allow slow rolls (clip2-style); 60f = 2s @ 30fps
    cooldown_frames   = 60,
    smooth_window     = 4,
    flow_threshold    = 0.3,   # peak flow during entry; real balls ~0.4-0.5, shadows ~0.002
    min_bright_pixels = 3,
    bg_adapt_rate     = 0.005,
)


def _draw_balls(frame: np.ndarray, active_tracks, trail_length: int = 20) -> np.ndarray:
    CAT_COLORS = {
        0: (0, 220, 220),
        1: (255, 255, 255),
        2: (100, 100, 100),
        3: (0, 140, 255),
        4: (255, 200, 0),
    }
    out = frame
    for track in active_tracks:
        color     = CAT_COLORS.get(track.category, CAT_COLORS[0])
        positions = track.positions[-trail_length:]
        n = len(positions)
        for k in range(1, n):
            _, x0, y0 = positions[k - 1]
            _, x1, y1 = positions[k]
            alpha = (k / n) ** 0.7
            c = tuple(int(ch * alpha) for ch in color)
            cv2.line(out, (int(x0), int(y0)), (int(x1), int(y1)),
                     c, 2 if track.category == 1 else 1)
        cx, cy = int(track.cx), int(track.cy)
        cv2.circle(out, (cx, cy), 9, color, 2)
        if track.category == 1:
            cv2.circle(out, (cx, cy), 4, color, -1)
        cv2.putText(out, str(track.id), (cx + 11, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
    return out


def _build_camera_manager(
    clip_dir: str,
    video_path: str,
    felt: str,
    two_cameras: bool,
    force_reselect: bool,
) -> CameraManager:
    """Load ROIs and create the CameraManager (single or dual camera)."""
    rois1 = load_or_select_rois(
        str(clip_dir), cam_idx=1,
        video_path=video_path, felt=felt,
        force_reselect=force_reselect,
    )
    if not rois1:
        sys.exit("No pocket ROIs for camera 1 — exiting.")

    det1 = GoalDetectorV2(rois=rois1, **_GOAL_DETECTOR_PARAMS)

    if not two_cameras:
        return CameraManager(detectors=[det1], rois_list=[rois1])

    rois2 = load_or_select_rois(
        str(clip_dir), cam_idx=2,
        video_path=video_path, felt=felt,
        force_reselect=force_reselect,
    )
    if not rois2:
        sys.exit("No pocket ROIs for camera 2 — exiting.")

    det2 = GoalDetectorV2(rois=rois2, **_GOAL_DETECTOR_PARAMS)
    print(f"  Two-camera mode enabled  (cut_threshold=30)")
    return CameraManager(detectors=[det1, det2], rois_list=[rois1, rois2])


def run_full_pipeline_v2(
    input_path: str,
    felt: str = "red",
    start_s: Optional[float] = None,
    end_s: Optional[float] = None,
    force_reselect: bool = False,
    model_path: Optional[str] = None,
    yolo_confidence: float = 0.35,
    device: str = "",
    two_cameras: bool = False,
):
    # ── Trim if needed ────────────────────────────────────────────────────
    if start_s is not None or end_s is not None:
        cap = cv2.VideoCapture(input_path)
        fps_tmp   = cap.get(cv2.CAP_PROP_FPS)
        total_tmp = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        s = start_s or 0.0
        e = end_s or (total_tmp / fps_tmp)
        base    = os.path.splitext(input_path)[0]
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

    # ── Camera manager + pocket ROIs ──────────────────────────────────────
    cam_mgr = _build_camera_manager(
        str(clip_dir), str(target_video), felt, two_cameras, force_reselect
    )

    # ── Open video ────────────────────────────────────────────────────────
    with VideoLoader(str(target_video)) as loader:
        info          = loader.info
        total         = info.frame_count
        pre_buf_size  = int(info.fps * 10)
        post_buf_size = int(info.fps * 10)

        print(f"\n  {total} frames @ {info.fps:.2f}fps  {info.width}x{info.height}")
        print(f"  Felt: {felt}  |  Pre/post buffer: {pre_buf_size/info.fps:.0f}s each")

        # ── Build components ──────────────────────────────────────────────
        table_bbox = estimate_table_bbox(loader.get_frame(0), felt=felt)
        if table_bbox is None:
            table_bbox = (0, 0, info.width, info.height)
        print(f"  Table bbox: {table_bbox}")

        ball_detector = YOLOBallDetector(
            model_path        = model_path or "__no_model__",
            confidence        = yolo_confidence,
            device            = device,
            table_bbox        = table_bbox,
            felt              = felt,
            fallback_on_error = True,
        )
        tracker      = CentroidTracker(max_distance=80.0, max_missing=8)
        traj_builder = TrajectoryBuilder(window=40, smooth_k=5, fps=info.fps)
        ball_heatmap   = BallHeatmap(info.width, info.height, sigma=12.0)
        player_heatmap = PlayerHeatmap(info.width, info.height)

        # ── Video writer ──────────────────────────────────────────────────
        annotated_path = str(out_dir / f"{base_name}_annotated_v2.mp4")
        writer = cv2.VideoWriter(
            annotated_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            info.fps, (info.width, info.height),
        )

        all_goals: List[GoalEvent] = []
        active_flashes: List[tuple] = []
        # post_collectors maps ev_idx → {ev, cam_rois, done}
        post_collectors: dict = {}
        reference_frame: Optional[np.ndarray] = None

        print(f"\n  Processing frames...")
        for frame_id, frame in loader.frames():
            if reference_frame is None:
                reference_frame = frame.copy()

            # Ball tracking
            detections   = ball_detector.detect(frame, frame_id)
            active       = tracker.update(detections, frame_id)
            traj_builder.update(active, frame_id)

            # Heatmaps
            ball_heatmap.update(active)
            player_heatmap.update(frame)

            # Goal detection (camera manager runs all detectors, reports active only)
            goal_events = cam_mgr.process_frame(frame, frame_id)
            for ev in goal_events:
                all_goals.append(ev)
                ev_idx = len(all_goals) - 1
                active_flashes.append((frame_id + int(info.fps * 1.5), ev.pocket_idx))
                # Snapshot the active ROIs at the moment of the goal for the highlight clip
                post_collectors[ev_idx] = {
                    "ev":       ev,
                    "cam_rois": list(cam_mgr.active_rois),
                    "cam_idx":  cam_mgr.active_idx + 1,
                    "done":     False,
                }
                print(
                    f"\n  *** GOAL ***  {ev.label}  frame={ev.frame_id}"
                    f"  t={frame_id/info.fps:.2f}s  conf={ev.confidence:.2f}"
                    f"  cam={cam_mgr.active_idx + 1}"
                )

            # Annotate: pockets (always use active camera's ROIs)
            active_flashes = [(exp, idx) for exp, idx in active_flashes if frame_id <= exp]
            fired  = {idx for _, idx in active_flashes}
            out_f  = _draw_rois(frame, cam_mgr.active_rois, fired_idxs=fired, flash=bool(fired))

            if fired:
                cv2.rectangle(out_f, (3, 3), (info.width - 3, info.height - 3), (0, 50, 255), 3)

            # Annotate: balls + trajectories
            out_f = _draw_balls(out_f, active)

            # HUD
            ov = out_f.copy()
            cv2.rectangle(ov, (0, 0), (info.width, 38), (10, 10, 10), -1)
            cv2.addWeighted(ov, 0.6, out_f, 0.4, 0, out_f)
            goal_labels = [ev.label for ev in all_goals if ev.frame_id == frame_id]
            tag = f"GOAL! {', '.join(goal_labels)}" if goal_labels else ""
            tc  = (0, 50, 255) if goal_labels else (200, 200, 200)
            cam_tag = f"  cam{cam_mgr.active_idx + 1}" if two_cameras else ""
            cv2.putText(
                out_f,
                f"Frame {frame_id}  {frame_id/info.fps:.2f}s  tracks={len(active)}{cam_tag}  {tag}",
                (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, tc, 1, cv2.LINE_AA,
            )

            writer.write(out_f)

            if frame_id % 300 == 0:
                print(
                    f"    frame {frame_id}/{total}  ({frame_id/info.fps:.0f}s)"
                    f"  tracks={len(active)}{cam_tag}",
                    flush=True,
                )

        writer.release()
        print(f"\n  Annotated video → {annotated_path}")

    # ── Heatmaps ──────────────────────────────────────────────────────────
    heatmap_paths = save_heatmaps(
        ball_heatmap, player_heatmap,
        out_dir         = out_dir,
        reference_frame = reference_frame,
        prefix          = f"{base_name}_",
    )

    # ── Goal highlight clips ──────────────────────────────────────────────
    summary = []
    for ev_idx, col in post_collectors.items():
        ev        = col["ev"]
        cam_rois  = col["cam_rois"]   # ROIs active when the goal fired
        cam_idx   = col["cam_idx"]
        folder_name = f"goal_frame{ev.frame_id:04d}_{ev.label.replace(' ', '_').lower()}"
        ev_dir      = out_dir / folder_name
        ev_dir.mkdir(exist_ok=True)

        clip_start = max(0, ev.frame_id - pre_buf_size)
        clip_end   = min(total - 1, ev.frame_id + post_buf_size)

        goal_vid = str(ev_dir / "goal_clip.mp4")
        cap = cv2.VideoCapture(str(target_video))
        gw  = cv2.VideoWriter(
            goal_vid, cv2.VideoWriter_fourcc(*"mp4v"),
            info.fps, (info.width, info.height),
        )
        cap.set(cv2.CAP_PROP_POS_FRAMES, clip_start)
        saved_stills = []
        for fid in range(clip_start, clip_end + 1):
            ret, frame = cap.read()
            if not ret:
                break
            is_flash = abs(fid - ev.frame_id) <= int(info.fps * 0.5)
            out_f = _draw_rois(
                frame, cam_rois,
                fired_idxs={ev.pocket_idx} if is_flash else set(),
                flash=is_flash,
            )
            if fid == ev.frame_id:
                cv2.rectangle(out_f, (3, 3), (info.width - 3, info.height - 3), (0, 50, 255), 4)
                tag = f"GOAL!  conf={ev.confidence:.2f}"
                tc  = (0, 50, 255)
            else:
                df  = fid - ev.frame_id
                tag = f"t+{df}f" if df > 0 else f"t{df}f"
                tc  = (0, 120, 255) if is_flash else (180, 180, 180)
            ov = out_f.copy()
            cv2.rectangle(ov, (0, 0), (info.width, 38), (10, 10, 10), -1)
            cv2.addWeighted(ov, 0.6, out_f, 0.4, 0, out_f)
            cv2.putText(
                out_f,
                f"Frame {fid}  {fid/info.fps:.2f}s  [{tag}]",
                (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, tc, 2, cv2.LINE_AA,
            )

            df = fid - ev.frame_id
            if df in (
                -(pre_buf_size // 3), -(pre_buf_size // 6),
                0,
                post_buf_size // 6, post_buf_size // 3,
            ):
                lbl   = "EVENT" if df == 0 else (
                    f"pre_{abs(df):04d}f" if df < 0 else f"post_{df:04d}f"
                )
                fname = f"{lbl}_frame{fid:04d}.png"
                cv2.imwrite(str(ev_dir / fname), out_f)
                saved_stills.append(fname)

            gw.write(out_f)
        cap.release()
        gw.release()
        print(f"  Goal clip ({(clip_end - clip_start + 1)/info.fps:.0f}s) → {ev_dir}")

        summary.append({
            "pocket":        ev.label,
            "frame":         ev.frame_id,
            "time_s":        round(ev.frame_id / info.fps, 3),
            "peak_activity": ev.peak_activity,
            "confidence":    ev.confidence,
            "camera":        cam_idx,
            "goal_clip":     goal_vid,
            "stills":        saved_stills,
        })

    json_path = out_dir / "goals_v2.json"
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n  Summary → {json_path}")
    print(f"  Total goals: {len(all_goals)}")
    if two_cameras:
        print(f"  Camera cuts detected: {cam_mgr.cut_count}")
    print(f"\nOutputs:")
    print(f"  {annotated_path}")
    for name, path in heatmap_paths.items():
        print(f"  {path}")
    for s in summary:
        print(f"  {s['goal_clip']}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Full billiards pipeline v2: YOLO balls + improved goals + heatmaps"
    )
    parser.add_argument("--input",       required=True)
    parser.add_argument("--felt",        default="red", choices=["blue", "red", "green"])
    parser.add_argument("--model",       default=None,  help="Path to YOLO .pt weights (optional)")
    parser.add_argument("--conf",        type=float, default=0.35, help="YOLO confidence threshold")
    parser.add_argument("--device",      default="",   help='Device: "" (auto) / "cpu" / "cuda"')
    parser.add_argument("--start",       type=float, default=None)
    parser.add_argument("--end",         type=float, default=None)
    parser.add_argument("--reselect",    action="store_true",
                        help="Re-annotate pocket ROIs even if cached files exist")
    parser.add_argument("--two-cameras", action="store_true",
                        help="Enable two-camera mode with automatic cut detection")
    args = parser.parse_args(argv)

    if not os.path.isfile(args.input):
        sys.exit(f"File not found: {args.input}")

    run_full_pipeline_v2(
        args.input, args.felt,
        args.start, args.end,
        args.reselect,
        model_path      = args.model,
        yolo_confidence = args.conf,
        device          = args.device,
        two_cameras     = args.two_cameras,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
