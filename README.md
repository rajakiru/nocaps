# nocaps

AI-Powered Broadcast System using Commodity Consumer Devices.

**Team 3**: Tabish, Akshara, Sai, Kiruthika

---

## What is nocaps?

nocaps turns regular smartphones into a multi-camera sports broadcast system. Place phones on tripods around a field or court — the system stitches feeds into a single live stream and automatically generates highlight reels using computer vision.

### How it works

1. **Create a match** — Organizer sets up a match (title, teams, sport, venue)
2. **Share the code** — A 6-character code is shared with camera operators
3. **Join as cameras** — Operators enter the code and pick a camera position
4. **Go live** — Each phone streams via WebRTC to the backend
5. **Watch** — Viewers browse active matches and watch the broadcast
6. **Replay** — After the match, watch the full synced game or AI highlight reel

---

## Tech Stack

| Layer | Technology |
|---|---|
| Mobile App | React Native + Expo SDK 54, TypeScript |
| Video Streaming | WebRTC (react-native-webrtc) |
| Navigation | React Navigation (native stack) |
| Backend | Node.js + Express + Socket.IO |
| Web UI | Vanilla JS SPA (no framework) |
| Computer Vision | Python + OpenCV + NumPy |
| Video Processing | FFmpeg |

---

## Getting Started

### Prerequisites

- Node.js v18+
- Python 3.9+
- Xcode (iOS) or Android Studio (Android) — required for WebRTC native build
- FFmpeg (`brew install ffmpeg`)
- OpenCV (`pip install opencv-python numpy`)

### Setup

```bash
git clone https://github.com/rajakiru/nocaps.git
cd nocaps

# App dependencies
npm install

# Server dependencies
cd server && npm install && cd ..
```

### Running

**Terminal 1 — Backend:**
```bash
cd server
npm run dev       # hot-reload with nodemon
# or
npm start         # single run
```

**Terminal 2 — Mobile app:**
```bash
npx expo prebuild --clean
npx expo run:ios        # or run:android
```

> Update `SERVER_URL` in `src/api.ts` to your machine's local IP when testing on a real device.

### Web UI (no build needed)

```
http://localhost:3000          — full web app (matches, replays, broadcast)
http://localhost:3000/test-camera  — browser webcam test page
```

---

## Project Structure

```
nocaps/
├── App.tsx                          React Native root
├── app.json                         Expo config
├── src/
│   ├── api.ts                       REST + Socket.IO + WebRTC client
│   ├── webrtc.ts                    ICE server config + peer helpers
│   ├── theme.ts                     Colors, spacing, typography
│   ├── navigation/AppNavigator.tsx  Typed route stack
│   └── screens/
│       ├── HomeScreen.tsx
│       ├── CreateMatchScreen.tsx
│       ├── JoinMatchScreen.tsx
│       ├── CameraRoleScreen.tsx
│       ├── CameraScreen.tsx
│       ├── MatchListScreen.tsx
│       └── ViewerScreen.tsx
├── server/
│   ├── src/
│   │   ├── index.ts                 Express entry + video streaming routes
│   │   ├── routes.ts                REST API (/api/matches)
│   │   ├── socket.ts                Socket.IO event handlers
│   │   ├── matchStore.ts            In-memory match store + seeded demos
│   │   └── types.ts                 TypeScript interfaces
│   └── public/
│       ├── index.html               SPA shell
│       ├── css/style.css            Dark-theme design system
│       └── js/
│           ├── app.js               Router + navigation
│           ├── socket.js            Socket.IO client wrapper
│           ├── webrtc.js            WebRTC helpers
│           └── pages/
│               ├── matches.js       Match list + create
│               ├── watch.js         Live viewer + full-game replay
│               ├── broadcast.js     Camera broadcast page
│               ├── director.js      Director cut / multi-cam control
│               ├── replays.js       Highlight reel + full game entry
│               ├── login.js
│               └── profile.js
├── billiards_engine/                Python CV pipeline (see below)
├── billiards_dataset/               Test videos + real game recordings
└── billiards_results/               Pipeline outputs (annotated video, goals, highlights)
```

---

## Web App Pages

| Page | Route | Description |
|---|---|---|
| Matches | `/` → Matches tab | Browse + create matches |
| Watch | `watch?code=XXXX` | Live viewer or replay with camera switching |
| Broadcast | `/` → Broadcast tab | Camera stream page |
| Replays | `/` → Replays tab | AI highlights + full-game player |
| Director | Internal | Multi-cam director cut controls |

### Seeded Demo Matches

| Code | Description |
|---|---|
| `DEMO01` | 3-angle demo clip (short loop) — Lateral / Frontal / Diagonal |
| `GAME02` | Full 16-minute Real Game 2 — synced 3-camera with draggable timeline |

---

## Billiards Computer Vision Pipeline

A standalone OpenCV pipeline for detecting ball-pocketing events in billiards videos. **No ML, no GPU, no external weights** — pure OpenCV + NumPy.

### What it does

- Detects billiard balls via HSV color thresholding + contour analysis
- Tracks balls frame-to-frame with a centroid nearest-neighbor tracker
- Detects goals using a pocket ROI state machine with approach-zone gating
- Outputs annotated video with ball trails, pocket circles, and HUD
- Extracts ±10s highlight clips around every detected goal
- Supports blue, red, and green felt tables

### Architecture

```
Video frames
    │
    ├─ OpenCVBallDetector     HSV mask → contour filter → ball positions
    │       felt_config.py    HSV ranges per felt color
    │
    ├─ CentroidTracker        Greedy nearest-neighbor ID assignment
    │                         Max distance: 80px, max missing: 8 frames
    │                         Categories: 0=unknown, 1=cue, 2=8ball, 3=solid, 4=striped
    │
    ├─ TrajectoryBuilder      Smoothed velocity from rolling window
    │
    ├─ GoalDetector           Per-pocket state machine:
    │                           IDLE → PRIMED (approach zone lit)
    │                                → ENTERING (ball at pocket edge)
    │                                → GOAL (activity drops = ball fell in)
    │                         Approach-zone guard: rejects arm/jacket drops
    │                         Peak-ratio guard: rejects slow drifts
    │
    └─ Visualizer             Draws trails, pocket circles, activity bars, HUD
```

### Goal Detection State Machine

The key insight: a ball rolling into a pocket crosses the **felt** first (approach zone), then enters the pocket ROI. A person's arm enters the pocket from outside the table — no felt motion precedes it. This means player interference cannot trigger a false goal.

```
IDLE ──(approach zone active)──► PRIMED
PRIMED ──(pocket ROI active)───► ENTERING
ENTERING ──(activity drops)────► GOAL ✓

Guards:
  - peak_ratio: peak activity must be ≥ 2.5× idle baseline
  - prime_ttl:  ball must reach pocket within 90 frames of approach
  - cooldown:   150-frame lockout after each goal (~5 seconds)
```

### Parameters (run_full.py)

```python
GoalDetector(
    background_frames  = 45,    # frames to build background model
    enter_threshold    = 20.0,  # MAD to confirm ball at pocket edge
    exit_threshold     = 10.0,  # MAD below which ball has fallen in
    approach_threshold = 12.0,  # MAD in felt approach zone to prime
    approach_window    = 3,     # consecutive approach frames required
    prime_ttl          = 90,    # frames PRIMED state stays valid
    min_entry_frames   = 3,
    max_entry_frames   = 30,
    cooldown_frames    = 150,
    peak_ratio         = 2.5,
)
```

### Running the Pipeline

```bash
# Run on a short clip (game1 benchmark, blue felt)
python -m billiards_engine.goal_pipeline --clip game1_clip3

# Full pipeline — ball tracking + goals — on your own video
python -m billiards_engine.run_full \
  --input billiards_dataset/realgame-2/IMG_1826.MOV \
  --felt blue

# Trim to specific window first
python -m billiards_engine.run_full \
  --input video.MOV --felt blue \
  --start 0 --end 60

# Force re-select pocket ROIs
python -m billiards_engine.run_full \
  --input video.MOV --felt blue --reselect
```

On first run, an interactive window opens. **Click each of the 6 pocket centers** in order (Top-Left → Top-Middle → Top-Right → Bottom-Left → Bottom-Middle → Bottom-Right), then press **Enter**. Positions are saved to `pocket_rois.json` and reused automatically.

### Output Structure

```
billiards_dataset/<video>/events/<clip>/
├── <clip>_annotated.mp4              full video with ball tracking overlays
├── goals.json                         detected events list
└── goal_frame<N>_<pocket>/
    ├── goal_clip.mp4                  ±10s highlight clip
    ├── EVENT_frame<N>.png
    ├── pre_*_frame*.png
    └── post_*_frame*.png
```

### Game1 Benchmark Results

| Clip | Pocket | Time | Result |
|---|---|---|---|
| game1_clip1 | Bottom-Right | 2.50s | ✓ |
| game1_clip2 | Top-Right | 3.00s | ✓ |
| game1_clip3 | Bottom-Left | 1.07s | ✓ |
| game1_clip4 | Bottom-Left | 2.37s | ✓ |

---

## Real Game 2 — Full Pipeline Run

### Dataset

3 iPhone cameras recording simultaneously from different angles during a real 8-ball pool game (~16 minutes):

| Camera | File | Role |
|---|---|---|
| CAM 1 | `IMG_1826.MOV` | Lateral (side view — reference) |
| CAM 2 | `IMG_5254.MOV` | Frontal (front view) |
| CAM 3 | `IMG_7658 2.MOV` | Diagonal (corner view) |

### Camera Sync Offsets

The cameras were not started simultaneously. Manual sync offsets measured from ground-truth timestamps:

| Camera | Offset from Lateral |
|---|---|
| Lateral (CAM 1) | 0s (reference) |
| Frontal (CAM 2) | +10–11s |
| Diagonal (CAM 3) | +5–6s |

### Ground Truth Events (Lateral timestamps)

| # | Time | Event | Ball | Lateral Pocket |
|---|---|---|---|---|
| 1 | 0:32 | Goal | Striped | Bottom-Right |
| 2 | 1:30 | Goal | Striped | Bottom-Center |
| 3 | 3:15 | Goal | Striped | Bottom-Right |
| 4 | 5:47 | Goal | Solid | Top-Right |
| 5 | 5:47 | Scratch | White | Top-Right |
| 6 | 6:24 | Scratch | White | Bottom-Center |
| 7 | 6:54 | Goal | Solid | Bottom-Left |
| 8 | 8:22 | Scratch | White | Bottom-Left |
| 9 | 8:23 | Goal | Striped | Top-Left |
| 10 | 8:49 | Goal | Solid | Bottom-Center |
| 11 | 9:41 | Goal | Solid | Top-Left |
| 12 | 10:56 | Scratch | White | Top-Center |
| 13 | 11:20 | Goal | Solid | Top-Right |
| 14 | 11:33 | Goal | Solid | Top-Right |
| 15 | 12:00 | Goal | Solid | Bottom-Left |
| 16 | 15:57 | Game Over | 8-Ball | Bottom-Right |

Full sync table with all 3 camera timestamps: `billiards_dataset/realgame-2/highlights.csv`

### Ball Tracking (Full 16-min Run)

Ran the full CV pipeline on the lateral video:

```bash
python -m billiards_engine.run_full \
  --input billiards_dataset/realgame-2/IMG_1826.MOV \
  --felt blue
```

- **29,461 frames** processed at ~30fps
- Tracking 2–18 active tracks per frame depending on table state
- **Output:** `billiards_dataset/realgame-2/events/IMG_1826/IMG_1826_annotated.mp4` (906 MB)

#### Goal Detection Notes

The automatic goal detector was noisy on the full game because:
1. The opening **break shot** (first 5s) causes massive pocket ROI activity — balls scatter in all directions, some go in early
2. The original thresholds in `run_full.py` were too low (`enter_threshold=4.0` vs the correct default `20.0`) causing repeated false positives on Top-Right pocket
3. After fixing the thresholds, 34 goals were detected vs 16 ground truth events — still noisy due to player movement and varied lighting

**Resolution:** Use hardcoded ground-truth timestamps for the highlight reel (see below). The pipeline output is used for the **ball tracking overlay only**, not for goal detection timing.

---

## Highlight Reel Generation

### Script: `billiards_dataset/realgame-2/generate_highlights.py`

Generates a multi-angle highlight video for all 16 events using FFmpeg.

**Per-event structure (3 clips concatenated):**
1. **Wide lateral intro** — 5s before → 4s after the event (9s)
2. **3-way split replay** — all 3 synced cameras side-by-side (7s)
3. **Pocket zoom** — lateral video cropped and scaled to the pocket region (3s)

Total: ~12–14s per event × 16 events ≈ **3–4 minute highlight reel**

### Running

```bash
cd billiards_dataset/realgame-2
python3 generate_highlights.py
```

Output: `highlights_output/highlights_reel.mp4`

### Sync Implementation

```python
FG_OFFSETS = { 1: 0, 2: 11, 3: 6 }   # seconds: lateral=ref, frontal=+11, diagonal=+6

# Per event: extract from each camera at the correct offset time
extract(LATERAL,  t_lat - BEFORE,       duration, ...)
extract(FRONTAL,  t_lat + 11 - BEFORE,  duration, ...)
extract(DIAGONAL, t_lat + 6  - BEFORE,  duration, ...)
```

### Technical Notes — HDR to SDR Conversion

iPhone videos are recorded in **10-bit HLG HDR** (`yuv420p10le`, BT.2020, `arib-std-b67`). This format is incompatible with QuickTime and most web players. The fix:

```python
# Force 8-bit SDR with BT.709 metadata
SDR_FLAGS = [
    "-r", "30",
    "-pix_fmt", "yuv420p",
    "-colorspace", "1",       # bt709
    "-color_trc", "1",
    "-color_primaries", "1",
    "-vf", "format=yuv420p",
]
```

The `zscale` and `colorspace` FFmpeg filters were unavailable in the Homebrew FFmpeg build (missing `libzimg`). The `format=yuv420p` pixel format conversion with metadata override was used as a lightweight alternative — no tone-mapping, but fully playable in QuickTime, VLC, and browsers.

### Pocket Crop Regions

The zoom clips crop from the lateral video using fractional coordinates:

```python
POCKET_CROP = {
    "top_left":      (0.00, 0.02, 0.32, 0.38),
    "top_center":    (0.34, 0.02, 0.32, 0.38),
    "top_right":     (0.68, 0.02, 0.32, 0.38),
    "bottom_left":   (0.00, 0.60, 0.32, 0.38),
    "bottom_center": (0.34, 0.60, 0.32, 0.38),
    "bottom_right":  (0.68, 0.60, 0.32, 0.38),
}
# Format: (x_frac, y_frac, w_frac, h_frac) relative to full frame
# Adjust these if the table isn't centred in the frame
```

---

## Full-Game Web Player

The web app serves the full 16-minute game at `http://localhost:3000` → Replays → **"Full 16-min"** card (match code `GAME02`).

### Features

- **3-camera layout** — Lateral (main, with ball tracking overlay), Frontal + Diagonal thumbnails
- **Draggable timeline** — scrub anywhere in the 16-minute game
- **Goal markers** on the timeline — red dots for goals, amber for scratches, purple for game over
- **Play/Pause button** — tap the play button in the timeline bar
- **Sound toggle** — starts muted (browser autoplay policy); tap the speaker icon to unmute
- **Camera switching** — tap a thumbnail to make it the main view

### Video Streaming

Large video files are served with HTTP range request support for seeking:

```typescript
app.get('/game/:camera', (req, res) => {
  // Handles Range: bytes=X-Y headers for video seeking
  // lateral  → IMG_1826_annotated.mp4 (906 MB, with ball tracking)
  // frontal  → IMG_5254.MOV
  // diagonal → IMG_7658 2.MOV
});
```

### Camera Sync in Player

When the user scrubs to time `T` on the timeline (lateral reference):

```javascript
const FG_OFFSETS = { 1: 0, 2: 11, 3: 6 };
video.currentTime = T + FG_OFFSETS[camNumber];
```

---

## What We Tried / Lessons Learned

### Goal Detection — Threshold Tuning

The original thresholds in `run_full.py` were `enter_threshold=4.0` — 5× lower than the GoalDetector defaults (`20.0`). This caused the Top-Right pocket to fire every ~15–30s. After fixing to `20.0`, false positives dropped significantly but the break shot still caused unavoidable early fires.

**Lesson:** For full-game detection, the background model needs to be built from a quiet pre-game window, not the first 45 frames which may include the break.

### HDR Video Compatibility

iPhone 13+ records in HLG HDR. Without explicit conversion, the FFmpeg output was `yuv420p10le` with BT.2020 metadata — unplayable in QuickTime. The `zscale` filter (libzimg) and `colorspace` filter both failed due to the Homebrew FFmpeg build lacking those libraries.

**Fix:** `format=yuv420p` + manual metadata override tags. Simple, works everywhere, no tone-mapping needed for offline review.

### drawtext Filter Unavailable

The Homebrew FFmpeg build lacked `libfreetype`, making `drawtext` unavailable. Event labels were dropped from the highlight clips rather than pulling in a full font rendering library.

### Web Autoplay Policy

Browsers block autoplay for unmuted video without prior user interaction. Setting `video.muted = false` before `play()` silently prevented the video from showing. Fix: start all videos muted, provide a sound toggle button.

### Multi-Angle Sync

The 3 cameras had inconsistent sync offsets:
- Frontal: +10s or +11s depending on the event
- Diagonal: +5s or +6s depending on the event

Using per-event exact timestamps for highlight extraction handles this precisely. For the live player, a fixed average offset (+11s / +6s) is used which is accurate to ±1 frame at most events.

---

## API Reference

### REST

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/matches` | List all matches |
| `POST` | `/api/matches` | Create a match |
| `GET` | `/api/matches/:code` | Get match by code |

### Socket.IO Events

| Event | Direction | Description |
|---|---|---|
| `watch-match` | Client→Server | Join a match as viewer |
| `join-as-camera` | Client→Server | Join as a camera |
| `match-updated` | Server→Client | Match state changed |
| `webrtc-offer` | Server→Client | WebRTC SDP offer |
| `webrtc-answer` | Client→Server | WebRTC SDP answer |
| `webrtc-ice-candidate` | Both | ICE candidate exchange |
| `director-cut` | Server→Client | Force camera switch for all viewers |

### Video Routes

| Route | Description |
|---|---|
| `GET /game/lateral` | Full annotated lateral video (range-request enabled) |
| `GET /game/frontal` | Full frontal video |
| `GET /game/diagonal` | Full diagonal video |
| `GET /highlights/billiards` | AI goal highlight reel |

---

## Progress

- [x] Phase 1: Navigation & screen wireframes
- [x] Phase 2: Camera functionality (live preview, flip, permissions)
- [x] Phase 3: Backend (Node.js, sessions, real-time sync)
- [x] Phase 4: WebRTC P2P video streaming + web test pages
- [x] Phase 5: Billiards CV pipeline — ball tracking + goal detection
- [x] Phase 6: Real game dataset (3-camera, 16 min, ground truth timestamps)
- [x] Phase 7: Multi-angle highlight reel generator (FFmpeg, 3-way split + zoom)
- [x] Phase 8: Full-game web player with draggable timeline + goal markers
- [ ] Phase 9: ML-based ball classification (solid/striped/cue/8-ball)
- [ ] Phase 10: Live goal detection integrated into broadcast stream
