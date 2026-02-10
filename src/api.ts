import { io, Socket } from 'socket.io-client';

// Change this to your server's IP when testing on a real device.
// Use your computer's local IP (e.g. 192.168.x.x), not localhost,
// since the phone is a separate device on the network.
const SERVER_URL = 'http://localhost:3000';

// --- REST API ---

export interface MatchDTO {
  code: string;
  title: string;
  teamA: string;
  teamB: string;
  sport: string;
  venue: string;
  createdAt: string;
  isLive: boolean;
  cameras: { number: number; role: string; isStreaming: boolean }[];
}

export async function createMatch(data: {
  title: string;
  teamA: string;
  teamB: string;
  sport?: string;
  venue?: string;
}): Promise<MatchDTO> {
  const res = await fetch(`${SERVER_URL}/api/matches`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error('Failed to create match');
  return res.json();
}

export async function getMatch(code: string): Promise<MatchDTO | null> {
  const res = await fetch(`${SERVER_URL}/api/matches/${code}`);
  if (res.status === 404) return null;
  if (!res.ok) throw new Error('Failed to get match');
  return res.json();
}

export async function listMatches(): Promise<MatchDTO[]> {
  const res = await fetch(`${SERVER_URL}/api/matches`);
  if (!res.ok) throw new Error('Failed to list matches');
  return res.json();
}

// --- Socket.IO ---

let socket: Socket | null = null;

export function getSocket(): Socket {
  if (!socket) {
    socket = io(SERVER_URL, { transports: ['websocket'] });
  }
  return socket;
}

export function joinMatchAsCamera(
  code: string,
  cameraNumber: number,
  cameraRole: string
): Promise<{ ok?: boolean; error?: string }> {
  return new Promise((resolve) => {
    getSocket().emit('join-match', { code, cameraNumber, cameraRole }, resolve);
  });
}

export function toggleStream(code: string, cameraNumber: number, isStreaming: boolean) {
  getSocket().emit('stream-toggle', { code, cameraNumber, isStreaming });
}

export function watchMatch(code: string): Promise<{ match?: MatchDTO; error?: string }> {
  return new Promise((resolve) => {
    getSocket().emit('watch-match', { code }, resolve);
  });
}

export function onMatchUpdated(callback: (match: MatchDTO) => void) {
  getSocket().on('match-updated', callback);
  return () => {
    getSocket().off('match-updated', callback);
  };
}
