import { Server } from 'socket.io';
import { getRawMatch, getMatch, removeSocketFromAllMatches } from './matchStore';

export function setupSocket(io: Server) {
  io.on('connection', (socket) => {
    console.log(`[socket] connected: ${socket.id}`);

    // Camera joins a match room
    socket.on('join-match', (data: { code: string; cameraNumber: number; cameraRole: string }, callback) => {
      const match = getRawMatch(data.code);
      if (!match) {
        callback?.({ error: 'match not found' });
        return;
      }

      // Check if camera slot is taken
      const existing = match.cameras.get(data.cameraNumber);
      if (existing && existing.socketId !== socket.id) {
        callback?.({ error: `camera ${data.cameraNumber} is already taken` });
        return;
      }

      // Claim the camera slot
      match.cameras.set(data.cameraNumber, {
        socketId: socket.id,
        number: data.cameraNumber,
        role: data.cameraRole,
        isStreaming: false,
      });

      // Join the socket.io room for this match
      socket.join(data.code);

      // Notify everyone in the match
      io.to(data.code).emit('match-updated', getMatch(data.code));
      callback?.({ ok: true });

      console.log(`[socket] ${socket.id} joined match ${data.code} as CAM ${data.cameraNumber}`);
    });

    // Camera starts/stops streaming
    socket.on('stream-toggle', (data: { code: string; cameraNumber: number; isStreaming: boolean }) => {
      const match = getRawMatch(data.code);
      if (!match) return;

      const cam = match.cameras.get(data.cameraNumber);
      if (!cam || cam.socketId !== socket.id) return;

      cam.isStreaming = data.isStreaming;
      match.isLive = Array.from(match.cameras.values()).some((c) => c.isStreaming);

      io.to(data.code).emit('match-updated', getMatch(data.code));

      console.log(`[socket] CAM ${data.cameraNumber} in ${data.code} streaming: ${data.isStreaming}`);
    });

    // Viewer joins to watch a match
    socket.on('watch-match', (data: { code: string }, callback) => {
      const match = getMatch(data.code);
      if (!match) {
        callback?.({ error: 'match not found' });
        return;
      }
      socket.join(data.code);
      callback?.({ match });

      console.log(`[socket] ${socket.id} watching match ${data.code}`);
    });

    // Handle disconnect
    socket.on('disconnect', () => {
      const removed = removeSocketFromAllMatches(socket.id);
      if (removed) {
        io.to(removed.code).emit('match-updated', getMatch(removed.code));
        console.log(`[socket] ${socket.id} disconnected, removed CAM ${removed.cameraNumber} from ${removed.code}`);
      } else {
        console.log(`[socket] disconnected: ${socket.id}`);
      }
    });
  });
}
