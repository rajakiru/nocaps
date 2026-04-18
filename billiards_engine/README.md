# Billiards Event Detection Engine

Real-time billiards event detection using **OpenCV + NumPy only** — no ML, no GPU required.

Detects:
- **Goal** — ball disappears into a pocket (primary focus)
- **Shot start** — cue ball velocity spike
- **Collision** — two balls make contact
- **Rail hit** — ball bounces off table boundary

---

## Setup

```bash
pip install opencv-python numpy
```

For YOLO inference with a trained Ultralytics model:

```bash
pip install ultralytics
```

Python 3.9+ required.

---

## Quick Start

### Run on a standard dataset clip (blue felt)
```bash
cd app/
python -m billiards_engine.main --clip game1_clip3
```

### Run goal detection only on a clip
```bash
python -m billiards_engine.goal_pipeline --clip game1_clip3
```

### Run on a custom video (red felt table)
```bash
python -m billiards_engine.run_full \
  --input /path/to/video.MOV \
  --felt red
```

### Run on a custom video with a YOLOv8 model
```bash
python -m billiards_engine.run_full \
  --input /path/to/video.MOV \
  --felt red \
  --detector yolo
```

By default, the YOLO path now points to `models/generic_ball_model.pt`. Use
`--model-path` to override it.

### Debug YOLO on one frame
```bash
python -m billiards_engine.debug_yolo_frame \
  --input /path/to/video.MOV \
  --frame 75 \
  --felt red \
  --conf 0.05
```

### Run on a custom video, trimmed to first 60 seconds
```bash
python -m billiards_engine.run_full \
  --input /path/to/video.MOV \
  --felt red \
  --start 0 --end 60
```

### Re-annotate pocket positions
```bash
python -m billiards_engine.run_full --input video.MOV --felt red --reselect
```

### Run all dataset clips
```bash
python -m billiards_engine.run_all
```

---

## Felt Color Options

| Table | `--felt` value |
|---|---|
| Blue/teal (competition tables) | `blue` (default) |
| Red/burgundy (casual tables) | `red` |
| Green (traditional) | `green` |

To add a new felt color, edit `felt_config.py` and add an HSV range entry.

---

## Pocket Annotation UI

On first run for any video, an interactive window opens showing the first frame.
**Click once on each pocket center in order:**

1. Top-Left → 2. Top-Middle → 3. Top-Right
4. Bottom-Left → 5. Bottom-Middle → 6. Bottom-Right

| Key | Action |
|---|---|
| Left-click | Place pocket marker |
| Backspace | Undo last marker |
| R | Reset all |
| Enter / S | Save and continue |
| Q / Esc | Quit |

Pocket locations are saved to `pocket_rois.json` in the clip folder and **reloaded automatically** on subsequent runs — no re-clicking needed.

---

## Output Structure

```
events/<clip_name>/
├── <clip_name>_annotated.mp4        # Full video with ball tracking + pocket circles
├── goals.json                        # Event summary
└── goal_frame<N>_<pocket>/
    ├── goal_clip.mp4                 # ±10s highlight clip around the goal
    ├── EVENT_frame<N>.png            # Frame at goal moment
    ├── pre_<N>f_frame<N>.png         # Frames before goal
    └── post_<N>f_frame<N>.png        # Frames after goal
```

---

## Module Overview

| File | Purpose |
|---|---|
| `run_full.py` | **Main entry point** — ball tracking + goal detection, any video |
| `run_all.py` | Batch runner for all dataset clips |
| `goal_pipeline.py` | Streaming goal detection pipeline |
| `goal_detector.py` | ROI-based background subtraction state machine |
| `pocket_roi_selector.py` | Click-to-annotate pocket UI |
| `opencv_detector.py` | Contour-based ball detector |
| `yolo_detector.py` | Ultralytics YOLO-based ball detector |
| `tracker.py` | Centroid nearest-neighbor tracker |
| `trajectory_builder.py` | Smoothed velocity computation |
| `event_detector.py` | Shot/collision/rail/pocket event logic |
| `felt_config.py` | HSV color ranges per felt type |
| `trim_video.py` | Trim long videos to a time window |
| `visualizer.py` | Drawing utilities |
| `video_loader.py` | OpenCV video wrapper |

---

## How Goal Detection Works

1. User clicks pocket centers on the first frame → saved as ROIs
2. Background model built from the first 15 frames (median pixel per ROI)
3. Each frame: `activity = mean(|current_roi - background|)`
4. State machine per pocket: `IDLE → BALL_ENTERING → GOAL`
   - Ball entering: `activity > 4.0`
   - Goal confirmed: activity drops back below `3.0` within 25 frames
   - Lingering ball (not pocketed): rejected if active for `> 25` frames

---

## Dataset

`billiards_dataset/` contains 4 game1 clips from the benchmark dataset:

| Clip | Frames | Duration |
|---|---|---|
| game1_clip1 | 187 | 6.2s |
| game1_clip2 | 157 | 5.2s |
| game1_clip3 | 158 | 5.3s |
| game1_clip4 | ~150 | ~5s |

Annotations (bounding boxes for first/last frame) are in `bounding_boxes/` — not used by the goal detector but available for the full tracking pipeline.

---

## Results (game1)

| Clip | Goal detected | Pocket | Time |
|---|---|---|---|
| game1_clip1 | ✓ | Bottom-Right | 2.50s |
| game1_clip2 | ✓ | Top-Right | 3.00s |
| game1_clip3 | ✓ | Bottom-Left | 1.07s |
| game1_clip4 | ✓ | Bottom-Left | 2.37s |

Full results (annotated videos + highlight clips) in `billiards_results/`.
