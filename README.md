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
- **Backend**: Node.js + Socket.IO (coming soon)
- **Streaming**: WebRTC (coming soon)

## Getting Started

### Prerequisites

- [Node.js](https://nodejs.org/) v18+ (we use v22)
- [Expo Go](https://expo.dev/go) app on your phone (iOS or Android)

### Setup

```bash
# Clone the repo
git clone https://github.com/rajakiru/nocap.git
cd nocap

# Install dependencies
npm install

# Start the dev server
npx expo start
```

### Running the app

- **On your phone**: Scan the QR code with Expo Go
- **iOS Simulator**: Press `i` (requires Xcode)
- **Android Emulator**: Press `a` (requires Android Studio)
- **Web**: Press `w` (camera features won't work in web)

## Project Structure

```
app/
├── App.tsx                         # Root — NavigationContainer + SafeAreaProvider
├── app.json                        # Expo config (dark theme, camera plugin)
├── assets/
│   └── logo.png                    # nocaps logo
├── src/
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
```

## Progress

- [x] Phase 1: Navigation & screen wireframes
- [x] Phase 2: Camera functionality (live preview, flip, permissions)
- [ ] Phase 3: Backend (Node.js, sessions, real-time sync)
- [ ] Phase 4: Video streaming (WebRTC, HLS playback)
- [ ] Phase 5: AI stitching & field calibration
