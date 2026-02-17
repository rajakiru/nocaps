import { useEffect, useRef, useState } from 'react';
import { StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { RootStackParamList } from '../navigation/AppNavigator';
import { colors, fontSize, spacing } from '../theme';
import {
  joinMatchAsCamera,
  toggleStream,
  onMatchUpdated,
  sendOffer,
  sendIceCandidate,
  onWebRTCIncomingRequest,
  onWebRTCAnswer,
  onWebRTCIceCandidate,
  type MatchDTO,
} from '../api';
import {
  RTCView,
  RTCPeerConnection,
  RTCSessionDescription,
  RTCIceCandidate,
  MediaStream,
  createPeerConnection,
  getLocalStream,
} from '../webrtc';

type Props = NativeStackScreenProps<RootStackParamList, 'Camera'>;
type CameraFacing = 'front' | 'back';

export default function CameraScreen({ navigation, route }: Props) {
  const { matchTitle, matchCode, cameraRole, cameraNumber } = route.params;
  const [facing, setFacing] = useState<CameraFacing>('back');
  const [isStreaming, setIsStreaming] = useState(false);
  const [connectedCameras, setConnectedCameras] = useState(1);
  const [localStream, setLocalStream] = useState<MediaStream | null>(null);
  const [permissionDenied, setPermissionDenied] = useState(false);
  const peerConnections = useRef<Map<string, RTCPeerConnection>>(new Map());

  // Start camera capture on mount
  useEffect(() => {
    let mounted = true;
    getLocalStream('environment')
      .then((stream) => { if (mounted) setLocalStream(stream); })
      .catch(() => { if (mounted) setPermissionDenied(true); });
    return () => { mounted = false; };
  }, []);

  // Clean up stream on unmount
  useEffect(() => {
    return () => {
      if (localStream) {
        localStream.getTracks().forEach((t) => t.stop());
      }
      peerConnections.current.forEach((pc) => pc.close());
      peerConnections.current.clear();
    };
  }, [localStream]);

  // Join match via Socket.IO
  useEffect(() => {
    joinMatchAsCamera(matchCode, cameraNumber, cameraRole);
    const unsubMatch = onMatchUpdated((match: MatchDTO) => {
      setConnectedCameras(match.cameras.length);
    });
    return unsubMatch;
  }, [matchCode, cameraNumber, cameraRole]);

  // Handle WebRTC signaling
  useEffect(() => {
    if (!localStream || !isStreaming) return;

    const unsubRequest = onWebRTCIncomingRequest(async (data) => {
      if (data.cameraNumber !== cameraNumber) return;

      const pc = createPeerConnection();
      peerConnections.current.set(data.viewerSocketId, pc);

      localStream.getTracks().forEach((track) => {
        pc.addTrack(track, localStream);
      });

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (pc as any).addEventListener('icecandidate', (event: any) => {
        if (event.candidate) {
          sendIceCandidate(data.viewerSocketId, event.candidate.toJSON());
        }
      });

      const offer = await pc.createOffer({});
      await pc.setLocalDescription(offer);
      sendOffer(data.viewerSocketId, cameraNumber, pc.localDescription);
    });

    const unsubAnswer = onWebRTCAnswer(async (data: { viewerSocketId: string; sdp: unknown }) => {
      const pc = peerConnections.current.get(data.viewerSocketId);
      if (pc) {
        await pc.setRemoteDescription(new RTCSessionDescription(data.sdp as { sdp: string; type: string | null }));
      }
    });

    const unsubIce = onWebRTCIceCandidate((data: { candidate: unknown }) => {
      peerConnections.current.forEach((pc) => {
        pc.addIceCandidate(new RTCIceCandidate(data.candidate as RTCIceCandidateInit));
      });
    });

    return () => {
      unsubRequest();
      unsubAnswer();
      unsubIce();
    };
  }, [localStream, isStreaming, cameraNumber, matchCode]);

  if (permissionDenied) {
    return (
      <View style={styles.container}>
        <View style={styles.permissionScreen}>
          <Text style={styles.permissionTitle}>Camera Access Required</Text>
          <Text style={styles.permissionText}>
            nocaps needs camera and microphone access to broadcast
          </Text>
          <TouchableOpacity style={styles.backButton} onPress={() => navigation.goBack()}>
            <Text style={styles.backButtonText}>Go Back</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  }

  const toggleFacing = async () => {
    if (!localStream) return;
    const newFacing: CameraFacing = facing === 'back' ? 'front' : 'back';
    try {
      const newStream = await getLocalStream(newFacing === 'back' ? 'environment' : 'user');
      const newVideoTrack = newStream.getVideoTracks()[0];

      peerConnections.current.forEach((pc) => {
        const senders = pc.getSenders();
        const videoSender = senders.find((s: { track: { kind: string } | null }) => s.track?.kind === 'video');
        if (videoSender) videoSender.replaceTrack(newVideoTrack);
      });

      localStream.getVideoTracks().forEach((t) => t.stop());
      setLocalStream(newStream);
      setFacing(newFacing);
    } catch (err) {
      console.error('Failed to flip camera:', err);
    }
  };

  const toggleStreaming = () => {
    const next = !isStreaming;
    setIsStreaming(next);
    toggleStream(matchCode, cameraNumber, next);
    if (!next) {
      peerConnections.current.forEach((pc) => pc.close());
      peerConnections.current.clear();
    }
  };

  return (
    <View style={styles.container}>
      {localStream ? (
        <RTCView
          streamURL={localStream.toURL()}
          style={styles.camera}
          objectFit="cover"
          mirror={facing === 'front'}
        />
      ) : (
        <View style={[styles.camera, styles.loadingCamera]}>
          <Text style={styles.loadingText}>Starting camera...</Text>
        </View>
      )}

      <SafeAreaView style={styles.topOverlay} edges={['top']}>
        <View style={styles.topBar}>
          <TouchableOpacity onPress={() => navigation.goBack()}>
            <Text style={styles.backArrow}>{'<'}</Text>
          </TouchableOpacity>
          <View style={styles.topInfo}>
            <Text style={styles.matchLabel}>{matchTitle}</Text>
            <Text style={styles.matchCodeLabel}>{matchCode}</Text>
          </View>
          <View style={styles.cameraBadge}>
            <Text style={styles.cameraBadgeText}>CAM {cameraNumber}</Text>
          </View>
        </View>
        {isStreaming && (
          <View style={styles.liveBadge}>
            <View style={styles.liveDot} />
            <Text style={styles.liveText}>LIVE</Text>
          </View>
        )}
      </SafeAreaView>

      <SafeAreaView style={styles.bottomOverlay} edges={['bottom']}>
        <View style={styles.roleBanner}>
          <Text style={styles.roleText}>{cameraRole}</Text>
        </View>
        <View style={styles.controls}>
          <TouchableOpacity style={styles.controlButton} onPress={toggleFacing}>
            <View style={styles.controlCircle}>
              <Text style={styles.controlIcon}>{'↻'}</Text>
            </View>
            <Text style={styles.controlLabel}>Flip</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.streamButton, isStreaming && styles.streamButtonActive]}
            onPress={toggleStreaming}
          >
            <View style={[styles.streamButtonInner, isStreaming && styles.streamButtonInnerActive]} />
          </TouchableOpacity>

          <TouchableOpacity style={styles.controlButton}>
            <View style={styles.controlCircle}>
              <Text style={styles.controlIcon}>{'⚙'}</Text>
            </View>
            <Text style={styles.controlLabel}>Settings</Text>
          </TouchableOpacity>
        </View>
        <View style={styles.statusBar}>
          <Text style={styles.statusText}>
            {isStreaming ? 'Streaming to viewers...' : 'Ready to stream'}
          </Text>
          <Text style={styles.statusText}>
            {connectedCameras} camera{connectedCameras !== 1 ? 's' : ''} connected
          </Text>
        </View>
      </SafeAreaView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#000' },
  camera: { ...StyleSheet.absoluteFillObject },
  loadingCamera: { backgroundColor: colors.surface, justifyContent: 'center', alignItems: 'center' },
  loadingText: { color: colors.textMuted, fontSize: fontSize.md },
  permissionScreen: { flex: 1, backgroundColor: colors.background, justifyContent: 'center', alignItems: 'center', padding: spacing.lg },
  permissionTitle: { fontSize: fontSize.xl, fontWeight: '700', color: colors.textPrimary, marginBottom: spacing.sm },
  permissionText: { fontSize: fontSize.md, color: colors.textSecondary, textAlign: 'center' },
  backButton: { marginTop: spacing.xl, padding: spacing.md },
  backButtonText: { fontSize: fontSize.md, color: colors.textMuted },
  topOverlay: { position: 'absolute', top: 0, left: 0, right: 0 },
  topBar: { flexDirection: 'row', alignItems: 'center', paddingHorizontal: spacing.md, paddingTop: spacing.sm, gap: spacing.sm },
  backArrow: { fontSize: fontSize.xxl, color: '#fff', fontWeight: '300', paddingRight: spacing.sm },
  topInfo: { flex: 1 },
  matchLabel: { fontSize: fontSize.md, fontWeight: '600', color: '#fff' },
  matchCodeLabel: { fontSize: fontSize.xs, color: 'rgba(255,255,255,0.6)', marginTop: 2 },
  cameraBadge: { backgroundColor: colors.primary, borderRadius: 8, paddingHorizontal: spacing.md, paddingVertical: spacing.xs },
  cameraBadgeText: { fontSize: fontSize.xs, fontWeight: '700', color: colors.background },
  liveBadge: { flexDirection: 'row', alignItems: 'center', alignSelf: 'center', backgroundColor: 'rgba(255, 77, 77, 0.9)', borderRadius: 6, paddingHorizontal: spacing.sm, paddingVertical: spacing.xs, marginTop: spacing.sm },
  liveDot: { width: 8, height: 8, borderRadius: 4, backgroundColor: '#fff', marginRight: spacing.xs },
  liveText: { fontSize: fontSize.xs, fontWeight: '800', color: '#fff', letterSpacing: 1 },
  bottomOverlay: { position: 'absolute', bottom: 0, left: 0, right: 0 },
  roleBanner: { alignSelf: 'center', backgroundColor: 'rgba(0,0,0,0.6)', borderRadius: 8, paddingHorizontal: spacing.md, paddingVertical: spacing.xs, marginBottom: spacing.sm },
  roleText: { fontSize: fontSize.sm, fontWeight: '600', color: colors.primary, textTransform: 'uppercase', letterSpacing: 1 },
  controls: { flexDirection: 'row', justifyContent: 'center', alignItems: 'center', gap: spacing.xl, paddingVertical: spacing.md },
  controlButton: { alignItems: 'center' },
  controlCircle: { width: 44, height: 44, borderRadius: 22, backgroundColor: 'rgba(255,255,255,0.15)', justifyContent: 'center', alignItems: 'center' },
  controlIcon: { fontSize: fontSize.xl, color: '#fff' },
  controlLabel: { fontSize: fontSize.xs, color: 'rgba(255,255,255,0.6)', marginTop: 4 },
  streamButton: { width: 72, height: 72, borderRadius: 36, borderWidth: 4, borderColor: '#fff', justifyContent: 'center', alignItems: 'center' },
  streamButtonActive: { borderColor: colors.error },
  streamButtonInner: { width: 56, height: 56, borderRadius: 28, backgroundColor: colors.error },
  streamButtonInnerActive: { width: 28, height: 28, borderRadius: 6 },
  statusBar: { flexDirection: 'row', justifyContent: 'space-between', paddingHorizontal: spacing.lg, paddingBottom: spacing.sm },
  statusText: { fontSize: fontSize.xs, color: 'rgba(255,255,255,0.5)' },
});
