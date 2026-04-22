export interface Camera {
  socketId: string;
  number: number;
  role: string;
  isStreaming: boolean;
  videoSrc?: string;   // set for pre-recorded demo cameras
}

export interface Match {
  code: string;
  title: string;
  teamA: string;
  teamB: string;
  sport: string;
  venue: string;
  createdAt: Date;
  isLive: boolean;
  cameras: Map<number, Camera>;
  isDemo?: boolean;
  isFullGame?: boolean;
}

export interface MatchDTO {
  code: string;
  title: string;
  teamA: string;
  teamB: string;
  sport: string;
  venue: string;
  createdAt: string;
  isLive: boolean;
  cameras: CameraDTO[];
  isDemo?: boolean;
  isFullGame?: boolean;
}

export interface CameraDTO {
  number: number;
  role: string;
  isStreaming: boolean;
  videoSrc?: string;
}
