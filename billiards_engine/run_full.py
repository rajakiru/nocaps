"""
Full pipeline: calibrated ball tracking + fused pocket detection.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from .calibration import estimate_video_calibration, normalize_frame, save_calibration
from .detection_filter import (
    estimate_expected_ball_radius,
    filter_detections,
    filter_tracks_for_events,
)
from .fused_pocket_detector import FusedPocketDetector, PocketEvent
from .goal_pipeline import _draw_rois
from .opencv_detector import OpenCVBallDetector
from .pocket_roi_selector import select_pocket_rois
from .tracker import CentroidTracker
from .trajectory_builder import TrajectoryBuilder
from .trim_video import trim_video
from .video_loader import VideoLoader
from .yolo_detector import YOLOBallDetector


DEFAULT_YOLO_MODEL_PATH = str(
    Path(__file__).resolve().parent.parent / "models" / "generic_ball_model.pt"
)


def _draw_balls(frame: np.ndarray, tracks, trail_length: int = 20) -> np.ndarray:
    colors = {
        0: (0, 220, 220),
        1: (255, 255, 255),
        2: (100, 100, 100),
        3: (0, 140, 255),
        4: (255, 200, 0),
    }
    out = frame.copy()
    for track in tracks:
        color = colors.get(track.category, colors[0])
        positions = track.positions[-trail_length:]
        n = len(positions)
        for idx in range(1, n):
            _, x0, y0 = positions[idx - 1]
            _, x1, y1 = positions[idx]
            alpha = (idx / n) ** 0.7
            faded = tuple(int(channel * alpha) for channel in color)
            cv2.line(out, (int(x0), int(y0)), (int(x1), int(y1)), faded, 2 if track.category == 1 else 1)
        cx, cy = int(track.cx), int(track.cy)
        radius = max(6, int(round(track.radius))) if track.radius else 9
        cv2.circle(out, (cx, cy), radius, color, 2)
        if track.category == 1:
            cv2.circle(out, (cx, cy), max(3, radius // 2), color, -1)
        cv2.putText(
            out,
            f"{track.id}",
            (cx + radius + 2, cy + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            color,
            1,
            cv2.LINE_AA,
        )
    return out


def _draw_detector_debug(frame: np.ndarray, detections, visible_tracks, table_bbox) -> np.ndarray:
    out = frame.copy()
    tx, ty, tw, th = table_bbox
    cv2.rectangle(out, (tx, ty), (tx + tw, ty + th), (180, 180, 180), 2)
    for det in detections:
        x1 = int(det.x)
        y1 = int(det.y)
        x2 = int(det.x + det.w)
        y2 = int(det.y + det.h)
        cv2.rectangle(out, (x1, y1), (x2, y2), (40, 210, 255), 1)
        cv2.putText(
            out,
            f"{det.confidence:.2f}",
            (x1, max(12, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (40, 210, 255),
            1,
            cv2.LINE_AA,
        )
    for track in visible_tracks:
        cv2.putText(
            out,
            f"T{track.id}",
            (int(track.cx) + 8, int(track.cy) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 120),
            1,
            cv2.LINE_AA,
        )
    return out


def _save_event_clip(
    target_video: Path,
    out_dir: Path,
    rois: List[dict],
    event: PocketEvent,
    fps: float,
    frame_size: tuple[int, int],
    pre_buf_size: int,
    post_buf_size: int,
    label_prefix: str = "goal",
) -> dict:
    width, height = frame_size
    folder_name = f"{label_prefix}_frame{event.frame_id:04d}_{event.label.replace(' ', '_').lower()}"
    event_dir = out_dir / folder_name
    event_dir.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(str(target_video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    clip_start = max(0, event.frame_id - pre_buf_size)
    clip_end = min(total - 1, event.frame_id + post_buf_size)
    clip_path = str(event_dir / f"{label_prefix}_clip.mp4")
    writer = cv2.VideoWriter(
        clip_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    cap.set(cv2.CAP_PROP_POS_FRAMES, clip_start)
    saved_stills = []

    for fid in range(clip_start, clip_end + 1):
        ret, frame = cap.read()
        if not ret:
            break
        is_flash = abs(fid - event.frame_id) <= int(fps * 0.5)
        out_frame = _draw_rois(
            frame,
            rois,
            fired_idxs={event.pocket_idx} if is_flash else set(),
            flash=is_flash,
        )
        if fid == event.frame_id:
            cv2.rectangle(out_frame, (3, 3), (width - 3, height - 3), (0, 50, 255), 4)
            tag = f"{label_prefix.upper()} T{event.track_id} {event.confidence:.2f}"
            color = (0, 50, 255)
        else:
            delta = fid - event.frame_id
            tag = f"t{delta:+d}f"
            color = (0, 120, 255) if is_flash else (180, 180, 180)

        overlay = out_frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 38), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.6, out_frame, 0.4, 0, out_frame)
        cv2.putText(
            out_frame,
            f"Frame {fid}  {fid / fps:.2f}s  [{tag}]",
            (12, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )
        delta = fid - event.frame_id
        if delta in (-(pre_buf_size // 3), -(pre_buf_size // 6), 0, post_buf_size // 6, post_buf_size // 3):
            stem = "EVENT" if delta == 0 else (f"pre_{abs(delta):04d}f" if delta < 0 else f"post_{delta:04d}f")
            filename = f"{stem}_frame{fid:04d}.png"
            cv2.imwrite(str(event_dir / filename), out_frame)
            saved_stills.append(filename)
        writer.write(out_frame)

    cap.release()
    writer.release()
    return {
        "folder": str(event_dir),
        "clip": clip_path,
        "stills": saved_stills,
    }


def _print_goal_summary(summary: List[dict], out_dir: Path) -> None:
    print(f"\n  Summary → {out_dir / 'goals.json'}")
    print(f"  Total goals: {len(summary)}")
    if not summary:
        print("  No goals detected.")
        print(f"  Review debug outputs in: {out_dir}")
        return

    print("\n  Detected goals:")
    for idx, event in enumerate(summary, start=1):
        print(
            f"    {idx}. {event['pocket']}"
            f"  frame={event['frame']}"
            f"  time={event['time_s']:.3f}s"
            f"  track={event['track_id']}"
            f"  conf={event['confidence']:.2f}"
        )
    print(f"\n  Debug summaries → {out_dir}")


def _reset_output_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for child in out_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def run_full_pipeline(
    input_path: str,
    felt: str = "red",
    start_s: Optional[float] = None,
    end_s: Optional[float] = None,
    force_reselect: bool = False,
    detector_name: str = "opencv",
    model_path: Optional[str] = None,
    yolo_conf: float = 0.25,
    yolo_iou: float = 0.45,
    yolo_imgsz: int = 1024,
    yolo_device: Optional[str] = None,
):
    if start_s is not None or end_s is not None:
        cap = cv2.VideoCapture(input_path)
        fps_tmp = cap.get(cv2.CAP_PROP_FPS)
        total_tmp = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        start = start_s or 0.0
        end = end_s or (total_tmp / fps_tmp)
        trimmed = f"{os.path.splitext(input_path)[0]}_trim_{int(start)}s_{int(end)}s.mp4"
        if not os.path.isfile(trimmed):
            print(f"Trimming {start:.0f}s → {end:.0f}s ...")
            trim_video(input_path, trimmed, start, end)
        else:
            print(f"Using existing trim: {trimmed}")
        input_path = trimmed

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    clip_dir = Path(input_path).parent / base_name
    clip_dir.mkdir(exist_ok=True)

    target_video = clip_dir / f"{base_name}.mp4"
    if not target_video.exists():
        try:
            os.link(input_path, str(target_video))
        except OSError:
            import shutil
            shutil.copy2(input_path, str(target_video))

    out_dir = clip_dir.parent / "events" / base_name
    _reset_output_dir(out_dir)

    rois = select_pocket_rois(str(clip_dir), force_reselect=force_reselect)
    if not rois:
        sys.exit("No pocket ROIs — exiting.")

    with VideoLoader(str(target_video)) as loader:
        info = loader.info
        pre_buf_size = int(info.fps * 10)
        post_buf_size = int(info.fps * 10)
        first_frame = loader.get_frame(0)
        calibration = estimate_video_calibration(first_frame, felt=felt)
        table_bbox = calibration.table_bbox
        save_calibration(calibration, str(out_dir / "calibration.json"))

        print(f"\n  {info.frame_count} frames @ {info.fps:.2f}fps  {info.width}x{info.height}")
        print(f"  Felt: {felt}  |  Table bbox: {table_bbox}")
        print(
            f"  Calibration: brightness_gain={calibration.brightness_gain:.2f}"
            f" mean_brightness={calibration.mean_brightness:.1f}"
            f" felt_coverage={calibration.felt_coverage:.2f}"
        )
        if detector_name == "yolo":
            resolved_model_path = model_path or DEFAULT_YOLO_MODEL_PATH
            print(
                f"  Detector: yolo"
                f"  model={resolved_model_path}"
                f"  conf={yolo_conf:.2f}"
                f"  iou={yolo_iou:.2f}"
                f"  imgsz={yolo_imgsz}"
            )
            ball_detector = YOLOBallDetector(
                model_path=resolved_model_path,
                conf_threshold=yolo_conf,
                iou_threshold=yolo_iou,
                image_size=yolo_imgsz,
                table_bbox=table_bbox,
                device=yolo_device,
            )
        else:
            print("  Detector: opencv")
            ball_detector = OpenCVBallDetector(table_bbox=table_bbox, felt=felt)
        tracker = CentroidTracker(
            max_distance=80.0,
            max_missing=8,
            rois=rois,
            near_pocket_margin=28.0,
            pocket_bonus_missing=6,
        )
        trajectories = TrajectoryBuilder(window=40, smooth_k=5, fps=info.fps)
        pocket_detector = FusedPocketDetector(
            rois=rois,
            fps=info.fps,
            background_frames=15,
            enter_threshold=4.0,
            exit_threshold=3.0,
            min_candidate_frames=2,
            max_candidate_frames=25,
            confirm_missing_frames=2,
            reappear_window_frames=6,
            cooldown_frames=60,
        )

        annotated_path = str(out_dir / f"{base_name}_annotated.mp4")
        detector_debug_path = str(out_dir / f"{base_name}_detector_debug.mp4")
        writer = cv2.VideoWriter(
            annotated_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            info.fps,
            (info.width, info.height),
        )
        debug_writer = cv2.VideoWriter(
            detector_debug_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            info.fps,
            (info.width, info.height),
        )

        all_goals: List[PocketEvent] = []
        active_flashes: List[tuple[int, int]] = []
        expected_radius: Optional[float] = None

        print("\n  Processing frames...")
        for frame_id, frame in loader.frames():
            calibrated = normalize_frame(frame, calibration)
            raw_detections = ball_detector.detect(calibrated, frame_id)
            expected_radius = estimate_expected_ball_radius(raw_detections, expected_radius)
            detections = filter_detections(raw_detections, table_bbox, rois, expected_radius)
            tracks = tracker.update(detections, frame_id)
            visible_tracks = [track for track in tracks if track.visible]
            trajectories.update(visible_tracks, frame_id)
            event_tracks = filter_tracks_for_events(visible_tracks, rois, trajectories)

            goal_events = pocket_detector.process_frame(
                calibrated,
                frame_id,
                visible_tracks=event_tracks,
                all_tracks=tracker.all_tracks(),
            )
            for event in goal_events:
                all_goals.append(event)
                active_flashes.append((frame_id + int(info.fps * 1.5), event.pocket_idx))
                print(
                    f"\n  *** GOAL *** {event.label}"
                    f"  frame={event.frame_id}"
                    f"  track={event.track_id}"
                    f"  conf={event.confidence:.2f}"
                )

            active_flashes = [(expires, pocket_idx) for expires, pocket_idx in active_flashes if frame_id <= expires]
            fired = {pocket_idx for _, pocket_idx in active_flashes}
            annotated = _draw_rois(frame, rois, fired_idxs=fired, flash=bool(fired))
            if fired:
                cv2.rectangle(annotated, (3, 3), (info.width - 3, info.height - 3), (0, 50, 255), 3)
            annotated = _draw_balls(annotated, visible_tracks)
            overlay = annotated.copy()
            cv2.rectangle(overlay, (0, 0), (info.width, 38), (10, 10, 10), -1)
            cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)
            event_labels = [f"{event.label} (T{event.track_id})" for event in all_goals if event.frame_id == frame_id]
            hud_tag = f"GOAL! {', '.join(event_labels)}" if event_labels else ""
            hud_color = (0, 50, 255) if event_labels else (200, 200, 200)
            cv2.putText(
                annotated,
                f"Frame {frame_id}  {frame_id / info.fps:.2f}s  tracks={len(visible_tracks)}  {hud_tag}",
                (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                hud_color,
                1,
                cv2.LINE_AA,
            )
            writer.write(annotated)
            debug_writer.write(_draw_detector_debug(frame, detections, visible_tracks, table_bbox))

            if frame_id % 300 == 0:
                print(
                    f"    frame {frame_id}/{info.frame_count}"
                    f"  ({frame_id / info.fps:.0f}s)"
                    f"  visible_tracks={len(visible_tracks)}",
                    flush=True,
                )

        writer.release()
        debug_writer.release()
        print(f"\n  Annotated video → {annotated_path}")
        print(f"  Detector debug video → {detector_debug_path}")

    summary = []
    for event in all_goals:
        clip_meta = _save_event_clip(
            target_video=target_video,
            out_dir=out_dir,
            rois=rois,
            event=event,
            fps=info.fps,
            frame_size=(info.width, info.height),
            pre_buf_size=pre_buf_size,
            post_buf_size=post_buf_size,
            label_prefix="goal",
        )
        summary.append({
            **event.to_json(),
            "goal_clip": clip_meta["clip"],
            "stills": clip_meta["stills"],
            "folder": clip_meta["folder"],
        })

    rejected_review = []
    for candidate in pocket_detector.rejected_candidates:
        pseudo_event = PocketEvent(
            pocket_idx=candidate.pocket_idx,
            label=candidate.label,
            frame_id=candidate.end_frame,
            time_s=candidate.end_frame / info.fps if info.fps > 0 else 0.0,
            track_id=candidate.track_id,
            confidence=0.0,
            peak_activity=candidate.peak_activity,
            roi_activity=candidate.last_activity,
            nearby_track_ids=candidate.nearby_track_ids,
        )
        clip_meta = _save_event_clip(
            target_video=target_video,
            out_dir=out_dir,
            rois=rois,
            event=pseudo_event,
            fps=info.fps,
            frame_size=(info.width, info.height),
            pre_buf_size=min(pre_buf_size, int(info.fps * 5)),
            post_buf_size=min(post_buf_size, int(info.fps * 5)),
            label_prefix="candidate",
        )
        rejected_review.append({
            **candidate.to_json(),
            "review_clip": clip_meta["clip"],
            "stills": clip_meta["stills"],
            "folder": clip_meta["folder"],
        })

    with open(out_dir / "goals.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    with open(out_dir / "pocket_activity.json", "w") as fh:
        json.dump(pocket_detector.activity_summary(), fh, indent=2)
    with open(out_dir / "pocket_event_candidates.json", "w") as fh:
        json.dump(pocket_detector.candidate_summary(), fh, indent=2)
    with open(out_dir / "rejected_pocket_candidates.json", "w") as fh:
        json.dump(rejected_review, fh, indent=2)

    _print_goal_summary(summary, out_dir)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Full billiards pipeline: fused pocket events")
    parser.add_argument("--input", required=True)
    parser.add_argument("--felt", default="red", choices=["blue", "red", "green"])
    parser.add_argument("--start", type=float, default=None)
    parser.add_argument("--end", type=float, default=None)
    parser.add_argument("--reselect", action="store_true")
    parser.add_argument("--detector", default="opencv", choices=["opencv", "yolo"])
    parser.add_argument("--model-path", default=DEFAULT_YOLO_MODEL_PATH)
    parser.add_argument("--yolo-conf", type=float, default=0.25)
    parser.add_argument("--yolo-iou", type=float, default=0.45)
    parser.add_argument("--yolo-imgsz", type=int, default=1024)
    parser.add_argument("--yolo-device", default=None)
    args = parser.parse_args(argv)

    if not os.path.isfile(args.input):
        sys.exit(f"File not found: {args.input}")

    run_full_pipeline(
        args.input,
        args.felt,
        args.start,
        args.end,
        args.reselect,
        detector_name=args.detector,
        model_path=args.model_path,
        yolo_conf=args.yolo_conf,
        yolo_iou=args.yolo_iou,
        yolo_imgsz=args.yolo_imgsz,
        yolo_device=args.yolo_device,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
