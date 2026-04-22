import express from 'express';
import { createServer } from 'http';
import { Server } from 'socket.io';
import cors from 'cors';
import path from 'path';
import fs from 'fs';
import routes from './routes';
import { setupSocket } from './socket';

const app = express();
const httpServer = createServer(app);

const io = new Server(httpServer, {
  cors: { origin: '*' },
});

app.use(cors());
app.use(express.json());
app.use('/api', routes);

// Serve web app
app.use(express.static(path.join(__dirname, '..', 'public')));

// Full game video streaming with range-request support (for seeking)
const GAME_VIDEOS: Record<string, string> = {
  lateral:  path.resolve(__dirname, '..', '..', 'billiards_dataset', 'realgame-2', 'events', 'IMG_1826', 'IMG_1826_annotated.mp4'),
  frontal:  path.resolve(__dirname, '..', '..', 'billiards_dataset', 'realgame-2', 'IMG_5254.MOV'),
  diagonal: path.resolve(__dirname, '..', '..', 'billiards_dataset', 'realgame-2', 'IMG_7658 2.MOV'),
};

app.get('/game/:camera', (req, res) => {
  const filePath = GAME_VIDEOS[req.params.camera];
  if (!filePath || !fs.existsSync(filePath)) { res.status(404).send('Video not found'); return; }
  const stat      = fs.statSync(filePath);
  const fileSize  = stat.size;
  const mimeType  = filePath.endsWith('.MOV') ? 'video/mp4' : 'video/mp4';
  const range     = req.headers.range;
  if (range) {
    const [startStr, endStr] = range.replace(/bytes=/, '').split('-');
    const start = parseInt(startStr, 10);
    const end   = endStr ? parseInt(endStr, 10) : Math.min(start + 10 * 1024 * 1024, fileSize - 1);
    res.writeHead(206, {
      'Content-Range':  `bytes ${start}-${end}/${fileSize}`,
      'Accept-Ranges':  'bytes',
      'Content-Length': end - start + 1,
      'Content-Type':   mimeType,
    });
    fs.createReadStream(filePath, { start, end }).pipe(res);
  } else {
    res.writeHead(200, { 'Content-Length': fileSize, 'Content-Type': mimeType, 'Accept-Ranges': 'bytes' });
    fs.createReadStream(filePath).pipe(res);
  }
});

// Billiards highlight video
app.get('/highlights/billiards', (_req, res) => {
  res.sendFile(
    path.resolve(__dirname, '..', '..', 'billiards_dataset',
      'realgame-1', 'events', 'IMG_5253', 'highlights.mp4'),
    { headers: { 'Content-Type': 'video/mp4' } }
  );
});

// Legacy test pages
app.get('/test-viewer', (_req, res) => {
  res.sendFile(path.join(__dirname, '..', 'test-viewer.html'));
});
app.get('/test-camera', (_req, res) => {
  res.sendFile(path.join(__dirname, '..', 'test-camera.html'));
});

// SPA catch-all — serve index.html for any non-API route
app.get(/^(?!\/api).*/, (_req, res) => {
  res.sendFile(path.join(__dirname, '..', 'public', 'index.html'));
});

setupSocket(io);

const PORT = process.env.PORT || 3000;
httpServer.listen(PORT, () => {
  console.log(`[nocaps] server running on http://localhost:${PORT}`);
});
