import {
  RTCPeerConnection,
  RTCSessionDescription,
  RTCIceCandidate,
  mediaDevices,
  MediaStream,
} from 'react-native-webrtc';

export const ICE_SERVERS = [
  { urls: 'stun:stun.l.google.com:19302' },
  { urls: 'stun:stun1.l.google.com:19302' },
];

export function createPeerConnection(): RTCPeerConnection {
  return new RTCPeerConnection({ iceServers: ICE_SERVERS });
}

export async function getLocalStream(
  facingMode: 'environment' | 'user' = 'environment'
): Promise<MediaStream> {
  const stream = await mediaDevices.getUserMedia({
    video: {
      facingMode,
      width: 1280,
      height: 720,
      frameRate: 30,
    },
    audio: true,
  });
  return stream as MediaStream;
}

export {
  RTCPeerConnection,
  RTCSessionDescription,
  RTCIceCandidate,
  MediaStream,
  mediaDevices,
};
export { RTCView } from 'react-native-webrtc';
