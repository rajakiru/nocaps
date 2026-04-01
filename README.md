# nocaps

AI-Powered Broadcast System using Commodity Consumer Devices.

**Team 3**: Tabish, Akshara, Sai, Kiruthika

## What is nocaps?

nocaps turns regular smartphones into a multi-camera sports broadcast system. Place phones on tripods around a field or court, and the system stitches their video feeds into a single professional-quality live stream using AI.

### How it works

1. **Create a match** — An organizer sets up a match (title, teams, sport, venue)
2. **Share the code** — A 6-character match code is generated to share with camera operators
3. **Join as cameras** — Camera operators enter the code and pick a camera position (Main, Side, Close-up, Wide)
4. **Go live** — Each phone streams its camera feed to the backend
5. **Watch** — Viewers browse active matches and watch the AI-stitched broadcast

## Tech Stack

- **Mobile App**: React Native + Expo SDK 54, TypeScript
- **Video**: react-native-webrtc (P2P streaming via WebRTC)
- **Navigation**: React Navigation (native stack)
- **Backend**: Node.js + Express + Socket.IO (signaling relay)
- **Web Test Pages**: Browser-based camera + viewer for testing

## Getting Started

### Prerequisites

- [Node.js](https://nodejs.org/) v18+ (we use v22)
- Xcode (for iOS) or Android Studio (for Android) — required for native WebRTC build

### Setup

```bash
# Clone the repo
git clone https://github.com/rajakiru/nocaps.git
cd nocaps

# Install app dependencies
npm install

# Install server dependencies
cd server
npm install
cd ..
```

### Running

You need two terminals:

**Terminal 1 — Start the backend:**
```bash
cd server
npm run dev
```

**Terminal 2 — Start the app:**
```bash
npx expo prebuild --clean
npx expo run:ios        # or npx expo run:android
```

> **Note**: This app uses react-native-webrtc which requires a custom dev client (Expo Go won't work). You need Xcode or Android Studio installed.

> **Tip**: Update the `SERVER_URL` in `src/api.ts` to your computer's local IP (e.g. `http://192.168.x.x:3000`) when testing on a real device.

### Web Test Pages (no build needed)

You can test WebRTC streaming entirely in the browser:

```
http://localhost:3000/camera   — use laptop/phone webcam as a camera
http://localhost:3000/viewer   — watch a live stream
```

Open the camera page, create a match, start streaming. Then open the viewer page in another tab, enter the match code, and watch.

## Project Structure

```
├── App.tsx                         # Root — NavigationContainer + SafeAreaProvider
├── app.json                        # Expo config (dark theme, camera plugin)
├── assets/
│   └── logo.png                    # nocaps logo
├── src/
│   ├── api.ts                      # REST + Socket.IO + WebRTC signaling client
│   ├── webrtc.ts                   # ICE servers, peer connection helpers
│   ├── theme.ts                    # Colors, spacing, font sizes
│   ├── navigation/
│   │   └── AppNavigator.tsx        # Stack navigator with typed routes
│   └── screens/
│       ├── HomeScreen.tsx          # Landing — Create / Join / Watch
│       ├── CreateMatchScreen.tsx   # Match setup form → generates code
│       ├── JoinMatchScreen.tsx     # Enter match code to join as camera
│       ├── CameraRoleScreen.tsx    # Pick camera position (Cam 1-4)
│       ├── CameraScreen.tsx        # Live camera preview + controls
│       ├── MatchListScreen.tsx     # Browse live/upcoming matches
│       └── ViewerScreen.tsx        # Watch the broadcast
├── server/
│   └── src/
│       ├── index.ts                # Express + Socket.IO entry point
│       ├── routes.ts               # REST API endpoints
│       ├── socket.ts               # Socket.IO event handlers
│       ├── matchStore.ts           # In-memory match/session store
│       └── types.ts                # TypeScript interfaces
```

## Server / API

See [server/SERVER.md](server/SERVER.md) for full backend documentation — REST endpoints, Socket.IO events, testing instructions.

## Progress

- [x] Phase 1: Navigation & screen wireframes
- [x] Phase 2: Camera functionality (live preview, flip, permissions)
- [x] Phase 3: Backend (Node.js, sessions, real-time sync)
- [x] Phase 4: WebRTC P2P video streaming + web test pages
- [ ] Phase 5: AI stitching & field calibration

---

## Billiards Event Detection Engine

A standalone computer vision pipeline for detecting ball-pocketing events in billiards videos. Built with **OpenCV + NumPy only** — no ML, no GPU required.

### What it does

- Detects when a ball goes into a pocket (goal) with exact timestamp
- Tracks all balls across frames with trajectory trails
- Outputs a full annotated video + a ±10s highlight clip around each goal
- Works with any felt color (blue, red, green)

### Folder structure

```
billiards_engine/     — Python pipeline source code
billiards_dataset/    — game1 benchmark clips + annotations (4 clips)
billiards_results/    — detected goals, highlight clips, still frames (game1)
```

### Setup

```bash
pip install opencv-python numpy
```

### Run on the included dataset (blue felt)

```bash
# Run goal detection on one clip
python -m billiards_engine.goal_pipeline --clip game1_clip3

# Run full pipeline (ball tracking + goals) on one clip
python -m billiards_engine.main --clip game1_clip3

# Run all 4 game1 clips
python -m billiards_engine.run_all
```

### Run on your own video (red felt table)

```bash
# Full pipeline — ball tracking + goal detection
python -m billiards_engine.run_full \
  --input /path/to/video.MOV \
  --felt red

# Trim to first 60 seconds before processing
python -m billiards_engine.run_full \
  --input /path/to/video.MOV \
  --felt red \
  --start 0 --end 60
```

### Pocket annotation

On first run an interactive window opens. **Click once on each of the 6 pocket centers** in order (top-left → top-mid → top-right → bottom-left → bottom-mid → bottom-right), then press **Enter**. Positions are saved and reloaded automatically on future runs.

### Output

```
events/<clip>/
├── <clip>_annotated.mp4          full video with ball tracking + pocket circles
├── goals.json                     event list with pocket, frame, timestamp
└── goal_frame<N>_<pocket>/
    ├── goal_clip.mp4              ±10s highlight clip
    ├── EVENT_frame<N>.png
    ├── pre_*f_frame*.png          frames before goal
    └── post_*f_frame*.png         frames after goal
```

### Game1 results

| Clip | Pocket | Time |
|---|---|---|
| game1_clip1 | Bottom-Right | 2.50s |
| game1_clip2 | Top-Right | 3.00s |
| game1_clip3 | Bottom-Left | 1.07s |
| game1_clip4 | Bottom-Left | 2.37s |

See [`billiards_engine/README.md`](billiards_engine/README.md) for full documentation.
