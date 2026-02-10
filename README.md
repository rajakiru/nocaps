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
- **Camera**: expo-camera v17
- **Navigation**: React Navigation (native stack)
- **Backend**: Node.js + Express + Socket.IO
- **Streaming**: WebRTC (coming soon)

## Getting Started

### Prerequisites

- [Node.js](https://nodejs.org/) v18+ (we use v22)
- [Expo Go](https://expo.dev/go) app on your phone (iOS or Android)

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
npx expo start
```

Then scan the QR code with Expo Go on your phone.

> **Note**: When testing on a real device, update the `SERVER_URL` in `src/api.ts` to your computer's local IP (e.g. `http://192.168.x.x:3000`) instead of `localhost`.

## Project Structure

```
├── App.tsx                         # Root — NavigationContainer + SafeAreaProvider
├── app.json                        # Expo config (dark theme, camera plugin)
├── assets/
│   └── logo.png                    # nocaps logo
├── src/
│   ├── api.ts                      # REST + Socket.IO client
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
- [ ] Phase 4: Video streaming (WebRTC, HLS playback)
- [ ] Phase 5: AI stitching & field calibration
