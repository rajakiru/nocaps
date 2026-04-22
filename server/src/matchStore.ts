import { Match, MatchDTO, CameraDTO } from './types';

// In-memory store — replace with a database in production
const matches = new Map<string, Match>();

// Pre-seeded billiards demo match (3 real camera angles, synced 5-min clip)
export const DEMO_CODE = 'DEMO01';

function seedDemo() {
  const demo: Match = {
    code: DEMO_CODE,
    title: 'Billiards – Real Game 2',
    teamA: 'Stripes',
    teamB: 'Solids',
    sport: 'Billiards',
    venue: 'CMU Game Room',
    createdAt: new Date('2026-04-19T00:00:00Z'),
    isLive: true,
    isDemo: true,
    cameras: new Map([
      [1, { socketId: 'demo-1', number: 1, role: 'Lateral',   isStreaming: true, videoSrc: '/demo/lateral.mp4'  }],
      [2, { socketId: 'demo-2', number: 2, role: 'Frontal',   isStreaming: true, videoSrc: '/demo/frontal.mp4'  }],
      [3, { socketId: 'demo-3', number: 3, role: 'Diagonal',  isStreaming: true, videoSrc: '/demo/diagonal.mp4' }],
    ]),
  };
  matches.set(DEMO_CODE, demo);
}

function seedFullGame() {
  const fullGame: Match = {
    code: 'GAME02',
    title: 'Billiards – Real Game 2',
    teamA: 'Stripes',
    teamB: 'Solids',
    sport: 'Billiards',
    venue: 'CMU Game Room',
    createdAt: new Date('2026-04-20T00:00:00Z'),
    isLive: false,
    isDemo: true,
    isFullGame: true,
    cameras: new Map([
      [1, { socketId: 'full-1', number: 1, role: 'Lateral',  isStreaming: true, videoSrc: '/game/lateral'  }],
      [2, { socketId: 'full-2', number: 2, role: 'Frontal',  isStreaming: true, videoSrc: '/game/frontal'  }],
      [3, { socketId: 'full-3', number: 3, role: 'Diagonal', isStreaming: true, videoSrc: '/game/diagonal' }],
    ]),
  };
  matches.set('GAME02', fullGame);
}

seedDemo();
seedFullGame();

function generateCode(): string {
  const chars = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789';
  let code = '';
  for (let i = 0; i < 6; i++) {
    code += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  // Ensure uniqueness
  if (matches.has(code)) return generateCode();
  return code;
}

function toDTO(match: Match): MatchDTO {
  const cameras: CameraDTO[] = [];
  match.cameras.forEach((cam) => {
    cameras.push({
      number: cam.number,
      role: cam.role,
      isStreaming: cam.isStreaming,
      ...(cam.videoSrc ? { videoSrc: cam.videoSrc } : {}),
    });
  });
  return {
    code: match.code,
    title: match.title,
    teamA: match.teamA,
    teamB: match.teamB,
    sport: match.sport,
    venue: match.venue,
    createdAt: match.createdAt.toISOString(),
    isLive: match.isLive,
    cameras,
    ...(match.isDemo     ? { isDemo: true }     : {}),
    ...(match.isFullGame ? { isFullGame: true }  : {}),
  };
}

export function createMatch(data: {
  title: string;
  teamA: string;
  teamB: string;
  sport?: string;
  venue?: string;
}): MatchDTO {
  const code = generateCode();
  const match: Match = {
    code,
    title: data.title,
    teamA: data.teamA,
    teamB: data.teamB,
    sport: data.sport || '',
    venue: data.venue || '',
    createdAt: new Date(),
    isLive: false,
    cameras: new Map(),
  };
  matches.set(code, match);
  return toDTO(match);
}

export function getMatch(code: string): MatchDTO | null {
  const match = matches.get(code.toUpperCase());
  return match ? toDTO(match) : null;
}

export function getRawMatch(code: string): Match | null {
  return matches.get(code.toUpperCase()) || null;
}

export function listMatches(): MatchDTO[] {
  return Array.from(matches.values())
    .sort((a, b) => b.createdAt.getTime() - a.createdAt.getTime())
    .map(toDTO);
}

export function removeSocketFromAllMatches(socketId: string): { code: string; cameraNumber: number } | null {
  for (const [code, match] of matches) {
    for (const [num, cam] of match.cameras) {
      if (cam.socketId === socketId) {
        match.cameras.delete(num);
        if (match.cameras.size === 0) {
          match.isLive = false;
        }
        return { code, cameraNumber: num };
      }
    }
  }
  return null;
}
