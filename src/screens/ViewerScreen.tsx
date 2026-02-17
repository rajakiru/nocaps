import { useEffect, useRef, useState } from 'react';
import { StyleSheet, Text, TouchableOpacity, View, ScrollView } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { RootStackParamList } from '../navigation/AppNavigator';
import { colors, fontSize, spacing } from '../theme';
import {
  watchMatch,
  onMatchUpdated,
  requestStream,
  sendAnswer,
  sendIceCandidate,
  onWebRTCOffer,
  onWebRTCIceCandidate,
  type MatchDTO,
  type CameraDTO,
} from '../api';
import {
  RTCView,
  RTCPeerConnection,
  RTCSessionDescription,
  RTCIceCandidate,
  MediaStream,
  createPeerConnection,
} from '../webrtc';

type Props = NativeStackScreenProps<RootStackParamList, 'Viewer'>;

export default function ViewerScreen({ navigation, route }: Props) {
  const { matchTitle, matchCode, teamA, teamB } = route.params;
  const [cameras, setCameras] = useState<CameraDTO[]>([]);
  const [isLive, setIsLive] = useState(false);
  const [selectedCamera, setSelectedCamera] = useState<number | null>(null);
  const [remoteStream, setRemoteStream] = useState<MediaStream | null>(null);
  const [connectionState, setConnectionState] = useState<'idle' | 'connecting' | 'connected'>('idle');
  const pcRef = useRef<RTCPeerConnection | null>(null);
  const selectedCameraRef = useRef<number | null>(null);
  const cameraSocketIdRef = useRef<string | null>(null);

  // Keep ref in sync for use inside callbacks
  useEffect(() => {
    selectedCameraRef.current = selectedCamera;
  }, [selectedCamera]);

  // Join match as viewer
  useEffect(() => {
    watchMatch(matchCode).then((response) => {
      if (response.match) {
        setCameras(response.match.cameras);
        setIsLive(response.match.isLive);
      }
    });

    const unsubMatch = onMatchUpdated((match: MatchDTO) => {
      setCameras(match.cameras);
      setIsLive(match.isLive);
    });

    return unsubMatch;
  }, [matchCode]);

  // Handle incoming WebRTC offers from cameras
  useEffect(() => {
    const unsubOffer = onWebRTCOffer(async (data) => {
      if (data.cameraNumber !== selectedCameraRef.current) return;

      const pc = pcRef.current;
      if (!pc) return;

      cameraSocketIdRef.current = data.cameraSocketId;
      await pc.setRemoteDescription(new RTCSessionDescription(data.sdp as { sdp: string; type: string | null }));
      const answer = await pc.createAnswer();
      await pc.setLocalDescription(answer);
      sendAnswer(data.cameraSocketId, pc.localDescription);
    });

    const unsubIce = onWebRTCIceCandidate((data) => {
      const pc = pcRef.current;
      if (pc) {
        pc.addIceCandidate(new RTCIceCandidate(data.candidate as RTCIceCandidateInit));
      }
    });

    return () => {
      unsubOffer();
      unsubIce();
    };
  }, []);

  // Clean up peer connection on unmount
  useEffect(() => {
    return () => {
      if (pcRef.current) {
        pcRef.current.close();
        pcRef.current = null;
      }
    };
  }, []);

  const connectToCamera = (cameraNumber: number) => {
    // Close existing connection
    if (pcRef.current) {
      pcRef.current.close();
      pcRef.current = null;
      setRemoteStream(null);
    }

    setSelectedCamera(cameraNumber);
    setConnectionState('connecting');

    const pc = createPeerConnection();
    pcRef.current = pc;

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const pcAny = pc as any;

    pcAny.addEventListener('track', (event: any) => {
      if (event.streams && event.streams[0]) {
        setRemoteStream(event.streams[0]);
        setConnectionState('connected');
      }
    });

    pcAny.addEventListener('icecandidate', (event: any) => {
      if (event.candidate && cameraSocketIdRef.current) {
        sendIceCandidate(cameraSocketIdRef.current, event.candidate.toJSON());
      }
    });

    pcAny.addEventListener('iceconnectionstatechange', () => {
      if (pc.iceConnectionState === 'disconnected' || pc.iceConnectionState === 'failed') {
        setConnectionState('idle');
        setRemoteStream(null);
      }
    });

    // Ask server to tell the camera we want its stream
    requestStream(matchCode, cameraNumber);
  };

  const streamingCameras = cameras.filter((c) => c.isStreaming);

  return (
    <View style={styles.container}>
      {/* Video area */}
      {remoteStream ? (
        <RTCView
          streamURL={remoteStream.toURL()}
          style={styles.videoArea}
          objectFit="cover"
        />
      ) : (
        <View style={styles.videoArea}>
          {connectionState === 'connecting' ? (
            <Text style={styles.videoPlaceholder}>Connecting...</Text>
          ) : streamingCameras.length > 0 ? (
            <>
              <Text style={styles.videoPlaceholder}>Select a camera</Text>
              <Text style={styles.videoHint}>
                {streamingCameras.length} camera{streamingCameras.length !== 1 ? 's' : ''} streaming
              </Text>
            </>
          ) : (
            <>
              <Text style={styles.videoPlaceholder}>
                {isLive ? 'Waiting for streams...' : 'Match not live yet'}
              </Text>
              <Text style={styles.videoHint}>
                {cameras.length} camera{cameras.length !== 1 ? 's' : ''} connected
              </Text>
            </>
          )}
        </View>
      )}

      {/* Top overlay */}
      <SafeAreaView style={styles.overlay} edges={['top']}>
        <View style={styles.topBar}>
          <TouchableOpacity onPress={() => navigation.goBack()}>
            <Text style={styles.backArrow}>{'<'}</Text>
          </TouchableOpacity>
          <View style={styles.topInfo}>
            <Text style={styles.matchLabel}>{matchTitle}</Text>
            <Text style={styles.teamsLabel}>{teamA} vs {teamB}</Text>
          </View>
          {isLive && (
            <View style={styles.liveBadge}>
              <View style={styles.liveDot} />
              <Text style={styles.liveText}>LIVE</Text>
            </View>
          )}
        </View>
      </SafeAreaView>

      {/* Bottom panel — camera selector + stats */}
      <SafeAreaView style={styles.bottomPanel} edges={['bottom']}>
        {streamingCameras.length > 0 && (
          <View style={styles.cameraSelector}>
            <Text style={styles.selectorTitle}>Camera Angles</Text>
            <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.cameraList}>
              {streamingCameras.map((cam) => (
                <TouchableOpacity
                  key={cam.number}
                  style={[
                    styles.cameraChip,
                    selectedCamera === cam.number && styles.cameraChipActive,
                  ]}
                  onPress={() => connectToCamera(cam.number)}
                >
                  <Text
                    style={[
                      styles.cameraChipText,
                      selectedCamera === cam.number && styles.cameraChipTextActive,
                    ]}
                  >
                    CAM {cam.number}
                  </Text>
                  <Text
                    style={[
                      styles.cameraChipRole,
                      selectedCamera === cam.number && styles.cameraChipRoleActive,
                    ]}
                  >
                    {cam.role}
                  </Text>
                </TouchableOpacity>
              ))}
            </ScrollView>
          </View>
        )}

        <View style={styles.statsRow}>
          <View style={styles.statItem}>
            <Text style={styles.statValue}>{cameras.length}</Text>
            <Text style={styles.statLabel}>Cameras</Text>
          </View>
          <View style={styles.statDivider} />
          <View style={styles.statItem}>
            <Text style={styles.statValue}>{streamingCameras.length}</Text>
            <Text style={styles.statLabel}>Streaming</Text>
          </View>
          <View style={styles.statDivider} />
          <View style={styles.statItem}>
            <Text style={styles.statValue}>{connectionState === 'connected' ? 'HD' : '--'}</Text>
            <Text style={styles.statLabel}>Quality</Text>
          </View>
        </View>
      </SafeAreaView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#000' },
  videoArea: {
    flex: 1,
    backgroundColor: colors.surface,
    justifyContent: 'center',
    alignItems: 'center',
  },
  videoPlaceholder: { fontSize: fontSize.xl, color: colors.textMuted, fontWeight: '600' },
  videoHint: { fontSize: fontSize.xs, color: colors.textMuted, marginTop: spacing.sm },
  overlay: { position: 'absolute', top: 0, left: 0, right: 0 },
  topBar: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: spacing.md,
    paddingTop: spacing.sm,
    gap: spacing.sm,
  },
  backArrow: { fontSize: fontSize.xxl, color: '#fff', fontWeight: '300', paddingRight: spacing.sm },
  topInfo: { flex: 1 },
  matchLabel: { fontSize: fontSize.md, fontWeight: '600', color: '#fff' },
  teamsLabel: { fontSize: fontSize.xs, color: 'rgba(255,255,255,0.6)', marginTop: 2 },
  liveBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(255, 77, 77, 0.9)',
    borderRadius: 6,
    paddingHorizontal: spacing.sm,
    paddingVertical: spacing.xs,
  },
  liveDot: { width: 6, height: 6, borderRadius: 3, backgroundColor: '#fff', marginRight: spacing.xs },
  liveText: { fontSize: fontSize.xs, fontWeight: '800', color: '#fff', letterSpacing: 1 },
  bottomPanel: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    backgroundColor: 'rgba(13, 13, 13, 0.95)',
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    paddingHorizontal: spacing.lg,
    paddingTop: spacing.lg,
  },
  cameraSelector: { marginBottom: spacing.md },
  selectorTitle: { fontSize: fontSize.sm, fontWeight: '600', color: colors.textSecondary, marginBottom: spacing.sm },
  cameraList: { flexDirection: 'row' },
  cameraChip: {
    backgroundColor: colors.surface,
    borderRadius: 12,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    marginRight: spacing.sm,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: colors.border,
  },
  cameraChipActive: { backgroundColor: colors.primary, borderColor: colors.primary },
  cameraChipText: { fontSize: fontSize.sm, fontWeight: '700', color: colors.textPrimary },
  cameraChipTextActive: { color: colors.background },
  cameraChipRole: { fontSize: fontSize.xs, color: colors.textMuted, marginTop: 2 },
  cameraChipRoleActive: { color: 'rgba(0,0,0,0.6)' },
  statsRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingBottom: spacing.md,
  },
  statItem: { flex: 1, alignItems: 'center' },
  statValue: { fontSize: fontSize.lg, fontWeight: '700', color: colors.primary },
  statLabel: { fontSize: fontSize.xs, color: colors.textMuted, marginTop: 2 },
  statDivider: { width: 1, height: 30, backgroundColor: colors.border },
});
