import express from 'express';
import { createServer } from 'http';
import { Server } from 'socket.io';
import cors from 'cors';
import path from 'path';
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

// Test pages
app.get('/viewer', (_req, res) => {
  res.sendFile(path.join(__dirname, '..', 'test-viewer.html'));
});
app.get('/camera', (_req, res) => {
  res.sendFile(path.join(__dirname, '..', 'test-camera.html'));
});

// Health check
app.get('/', (_req, res) => {
  res.json({ status: 'nocaps server running' });
});

setupSocket(io);

const PORT = process.env.PORT || 3000;
httpServer.listen(PORT, () => {
  console.log(`[nocaps] server running on http://localhost:${PORT}`);
});
