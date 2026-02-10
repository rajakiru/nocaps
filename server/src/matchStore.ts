import { Match, MatchDTO, CameraDTO } from './types';

// In-memory store — replace with a database in production
const matches = new Map<string, Match>();

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
