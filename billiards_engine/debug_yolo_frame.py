"""
Run YOLO on a single video frame and print/save raw detections.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2

from .calibration import estimate_video_calibration, normalize_frame
from .video_loader import VideoLoader
from .yolo_detector import YOLOBallDetector


DEFAULT_YOLO_MODEL_PATH = str(
    Path(__file__).resolve().parent.parent / "models" / "generic_ball_model.pt"
)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Debug YOLO detections on one billiards frame")
    parser.add_argument("--input", required=True)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--felt", default="red", choices=["blue", "red", "green"])
    parser.add_argument("--model-path", default=DEFAULT_YOLO_MODEL_PATH)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)

    if not os.path.isfile(args.input):
        raise SystemExit(f"File not found: {args.input}")

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.input).resolve().parent / "yolo_debug"
    output_dir.mkdir(parents=True, exist_ok=True)

    with VideoLoader(args.input) as loader:
        frame = loader.get_frame(args.frame)
        calibration = estimate_video_calibration(frame, felt=args.felt)
        normalized = normalize_frame(frame, calibration)
        detector = YOLOBallDetector(
            model_path=args.model_path,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            image_size=args.imgsz,
            table_bbox=calibration.table_bbox,
            device=args.device,
        )
        raw_predictions, _ = detector.predict_raw(normalized)
        debug_image = detector.draw_debug(frame, raw_predictions)

    image_path = output_dir / f"frame_{args.frame:04d}_yolo_debug.png"
    json_path = output_dir / f"frame_{args.frame:04d}_yolo_debug.json"
    cv2.imwrite(str(image_path), debug_image)
    with open(json_path, "w") as fh:
        json.dump(
            {
                "input": args.input,
                "frame": args.frame,
                "felt": args.felt,
                "model_path": args.model_path,
                "conf": args.conf,
                "iou": args.iou,
                "imgsz": args.imgsz,
                "table_bbox": list(calibration.table_bbox),
                "num_predictions": len(raw_predictions),
                "predictions": raw_predictions,
            },
            fh,
            indent=2,
        )

    print(f"Frame: {args.frame}")
    print(f"Model: {args.model_path}")
    print(f"Table bbox: {calibration.table_bbox}")
    print(f"Predictions: {len(raw_predictions)}")
    for idx, item in enumerate(raw_predictions, start=1):
        print(
            f"  {idx}. class={item['class_name']} ({item['class_id']})"
            f"  conf={item['confidence']:.3f}"
            f"  box=({item['x1']:.1f}, {item['y1']:.1f}, {item['x2']:.1f}, {item['y2']:.1f})"
        )
    print(f"Saved image → {image_path}")
    print(f"Saved JSON  → {json_path}")


if __name__ == "__main__":
    main()
