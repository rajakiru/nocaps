# nocaps server

Backend for the nocaps broadcast system. Handles match sessions and real-time camera coordination.

## Tech Stack

- Node.js + Express 5
- Socket.IO for real-time events
- TypeScript
- In-memory storage (no database yet)

## Setup

```bash
cd server
npm install
```

## Running

```bash
# Development (auto-restart on changes)
npm run dev

# Production
npm start
```

Server runs on `http://localhost:3000` by default. Set the `PORT` env variable to change it.

## Testing the API

### Health check
```bash
curl http://localhost:3000/
```

### Create a match
```bash
curl -X POST http://localhost:3000/api/matches \
  -H "Content-Type: application/json" \
  -d '{"title":"CMU vs Pitt","teamA":"CMU Tartans","teamB":"Pitt Panthers","sport":"Basketball","venue":"Gesling Stadium"}'
```

### List all matches
```bash
curl http://localhost:3000/api/matches
```

### Get match by code
```bash
curl http://localhost:3000/api/matches/ABC123
```

## REST Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Health check |
| POST | `/api/matches` | Create a new match |
| GET | `/api/matches` | List all matches |
| GET | `/api/matches/:code` | Get match by code |

### POST `/api/matches` body

```json
{
  "title": "CMU vs Pitt",
  "teamA": "CMU Tartans",
  "teamB": "Pitt Panthers",
  "sport": "Basketball",
  "venue": "Gesling Stadium"
}
```

`title`, `teamA`, `teamB` are required. `sport` and `venue` are optional.

### Response format

```json
{
  "code": "CPC2JP",
  "title": "CMU vs Pitt",
  "teamA": "CMU Tartans",
  "teamB": "Pitt Panthers",
  "sport": "Basketball",
  "venue": "Gesling Stadium",
  "createdAt": "2026-02-10T05:17:28.274Z",
  "isLive": false,
  "cameras": []
}
```

## Socket.IO Events

### Client → Server

| Event | Payload | Callback | Description |
|-------|---------|----------|-------------|
| `join-match` | `{ code, cameraNumber, cameraRole }` | `{ ok: true }` or `{ error }` | Camera operator joins a match |
| `stream-toggle` | `{ code, cameraNumber, isStreaming }` | — | Camera starts/stops streaming |
| `watch-match` | `{ code }` | `{ match }` or `{ error }` | Viewer subscribes to match updates |

### Server → Client

| Event | Payload | Description |
|-------|---------|-------------|
| `match-updated` | Full match object | Sent to all clients in a match room whenever cameras join/leave/toggle streaming |

## File Structure

```
server/src/
├── index.ts        # Entry point — Express + Socket.IO setup
├── routes.ts       # REST API route handlers
├── socket.ts       # Socket.IO event handlers
├── matchStore.ts   # In-memory match CRUD + code generation
└── types.ts        # TypeScript interfaces (Match, Camera, DTOs)
```

## Connecting from the app

The React Native app connects via `src/api.ts`. When testing on a real device, update `SERVER_URL` to your computer's local IP:

```bash
# Find your IP
ipconfig getifaddr en0
```

Then in `src/api.ts`:
```ts
const SERVER_URL = 'http://192.168.x.x:3000';
```
