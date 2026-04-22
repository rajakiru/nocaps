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
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

from .calibration import estimate_video_calibration, normalize_frame, save_calibration
from .detection_loader import Detection
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
GHOST_TRACK_SPEED_PX = 220.0
GHOST_TRACK_MAX_MISSING = 3
GHOST_TRACK_CONFIDENCE = 0.22
GHOST_NEAR_POCKET_SPEED_PX = 100.0


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
    parent_dir = out_dir / ("goals" if label_prefix == "goal" else "candidates")
    parent_dir.mkdir(exist_ok=True)
    safe_label = event.label.replace(" ", "_").lower()
    folder_name = (
        f"{label_prefix}_frame{event.frame_id:04d}_t{event.time_s:07.3f}s_{safe_label}"
    )
    event_dir = parent_dir / folder_name
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

    metadata = {
        "type": label_prefix,
        "pocket": event.label,
        "pocket_idx": event.pocket_idx,
        "frame": event.frame_id,
        "time_s": round(float(event.time_s), 3),
        "track_id": event.track_id,
        "confidence": round(float(event.confidence), 3),
        "peak_activity": round(float(event.peak_activity), 3),
        "roi_activity": round(float(event.roi_activity), 3),
        "nearby_track_ids": event.nearby_track_ids,
        "clip_start_frame": clip_start,
        "clip_end_frame": clip_end,
        "clip_path": clip_path,
        "stills": saved_stills,
    }
    with open(event_dir / "event_info.json", "w") as fh:
        json.dump(metadata, fh, indent=2)
    with open(event_dir / "event_info.txt", "w") as fh:
        fh.write(
            "\n".join(
                [
                    f"type: {label_prefix}",
                    f"pocket: {event.label}",
                    f"pocket_idx: {event.pocket_idx}",
                    f"frame: {event.frame_id}",
                    f"time_s: {event.time_s:.3f}",
                    f"track_id: {event.track_id}",
                    f"confidence: {event.confidence:.3f}",
                    f"peak_activity: {event.peak_activity:.3f}",
                    f"roi_activity: {event.roi_activity:.3f}",
                    f"clip_start_frame: {clip_start}",
                    f"clip_end_frame: {clip_end}",
                    f"clip_path: {clip_path}",
                ]
            )
        )

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


def _normalize_debug_token(token: str) -> str:
    return token.strip().lower().replace("_", "-").replace(" ", "-")


def _resolve_debug_pockets(requested: Optional[List[str]], rois: List[dict]) -> List[str]:
    if not requested:
        return []

    resolved: List[str] = []
    by_label = {_normalize_debug_token(roi["label"]): roi["label"] for roi in rois}

    for raw in requested:
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            if token.isdigit():
                idx = int(token) - 1
                if 0 <= idx < len(rois):
                    resolved.append(rois[idx]["label"])
                else:
                    print(f"  Warning: debug pocket index out of range: {token}")
                continue

            key = _normalize_debug_token(token)
            label = by_label.get(key)
            if label is None:
                print(f"  Warning: unknown debug pocket: {token}")
                continue
            resolved.append(label)

    return sorted(set(resolved))


def _augment_with_ghost_detections(
    detections: List[Detection],
    prior_tracks,
    frame_id: int,
    fps: float,
) -> List[Detection]:
    augmented = list(detections)
    for track in prior_tracks:
        if track.missing_frames > GHOST_TRACK_MAX_MISSING:
            continue
        if len(track.velocities) == 0:
            continue
        speed = max(float(np.hypot(vx, vy)) for _, vx, vy in track.velocities[-4:])
        speed_threshold = GHOST_NEAR_POCKET_SPEED_PX if track.near_pocket_idx is not None else GHOST_TRACK_SPEED_PX
        if speed < speed_threshold:
            continue

        radius = max(track.radius, 6.0)
        _, vx, vy = track.velocities[-1]
        dt = max(track.missing_frames + 1, 1) / max(fps, 1e-6)
        pred_cx = track.cx + vx * dt * 0.85
        pred_cy = track.cy + vy * dt * 0.85
        if any(np.hypot(det.cx - pred_cx, det.cy - pred_cy) <= radius * 1.8 for det in detections):
            continue

        augmented.append(
            Detection(
                frame_id=frame_id,
                ball_id=-100000 - track.id,
                x=pred_cx - radius,
                y=pred_cy - radius,
                w=radius * 2.0,
                h=radius * 2.0,
                category=track.category,
                confidence=GHOST_TRACK_CONFIDENCE,
            )
        )
    return augmented


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
    debug_pockets: Optional[List[str]] = None,
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
    resolved_debug_pockets = _resolve_debug_pockets(debug_pockets, rois)
    if resolved_debug_pockets:
        print(f"  Pocket debug enabled for: {', '.join(resolved_debug_pockets)}")

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
            cooldown_frames=18,
            debug_pockets=resolved_debug_pockets,
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
        progress = None
        try:
            for frame_id, frame in loader.frames():
                calibrated = normalize_frame(frame, calibration)
                prior_tracks = tracker.all_tracks()
                raw_detections = ball_detector.detect(calibrated, frame_id)
                raw_detections = _augment_with_ghost_detections(raw_detections, prior_tracks, frame_id, info.fps)
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

                if tqdm is not None and progress is None and pocket_detector._bg_built:
                    progress = tqdm(
                        total=info.frame_count,
                        desc="Frames",
                        unit="frame",
                        dynamic_ncols=True,
                        initial=frame_id + 1,
                    )
                    progress.set_postfix_str(
                        f"tracks={len(visible_tracks)} goals={len(all_goals)}",
                        refresh=False,
                    )
                elif progress is not None:
                    progress.update(1)
                    progress.set_postfix_str(f"tracks={len(visible_tracks)} goals={len(all_goals)}", refresh=False)
                elif frame_id % 300 == 0:
                    print(
                        f"    frame {frame_id}/{info.frame_count}"
                        f"  ({frame_id / info.fps:.0f}s)"
                        f"  visible_tracks={len(visible_tracks)}",
                        flush=True,
                    )
        finally:
            if progress is not None:
                progress.close()

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

    debug_summary = pocket_detector.debug_summary()
    if debug_summary:
        debug_dir = out_dir / "debug"
        debug_dir.mkdir(exist_ok=True)
        for pocket_label, rows in debug_summary.items():
            filename = f"pocket_{_normalize_debug_token(pocket_label)}.json"
            with open(debug_dir / filename, "w") as fh:
                json.dump(rows, fh, indent=2)

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
    parser.add_argument("--debug-pocket", action="append", default=[])
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
        debug_pockets=args.debug_pocket,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
