import { useRef, useState } from 'react';
import { StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { CameraView, useCameraPermissions } from 'expo-camera';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { RootStackParamList } from '../navigation/AppNavigator';
import { colors, fontSize, spacing } from '../theme';

type Props = NativeStackScreenProps<RootStackParamList, 'Camera'>;
type CameraFacing = 'front' | 'back';

export default function CameraScreen({ navigation, route }: Props) {
  const { matchTitle, matchCode, cameraRole, cameraNumber } = route.params;
  const [facing, setFacing] = useState<CameraFacing>('back');
  const [isStreaming, setIsStreaming] = useState(false);
  const [permission, requestPermission] = useCameraPermissions();
  const cameraRef = useRef<CameraView>(null);

  // Permission not yet determined
  if (!permission) {
    return (
      <View style={styles.container}>
        <View style={styles.permissionScreen}>
          <Text style={styles.permissionText}>Loading camera...</Text>
        </View>
      </View>
    );
  }

  // Permission denied — show request UI
  if (!permission.granted) {
    return (
      <View style={styles.container}>
        <View style={styles.permissionScreen}>
          <Text style={styles.permissionTitle}>Camera Access Required</Text>
          <Text style={styles.permissionText}>
            nocaps needs camera access to broadcast this match
          </Text>
          <TouchableOpacity style={styles.permissionButton} onPress={requestPermission}>
            <Text style={styles.permissionButtonText}>Grant Access</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={styles.backButton}
            onPress={() => navigation.goBack()}
          >
            <Text style={styles.backButtonText}>Go Back</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  }

  const toggleFacing = () => {
    setFacing((prev) => (prev === 'back' ? 'front' : 'back'));
  };

  const toggleStreaming = () => {
    setIsStreaming((prev) => !prev);
    // In Phase 4, this will start/stop WebRTC streaming
  };

  return (
    <View style={styles.container}>
      {/* Live camera preview */}
      <CameraView
        ref={cameraRef}
        style={styles.camera}
        facing={facing}
      />

      {/* Top overlay */}
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
            <Text style={styles.cameraBadgeText}>
              CAM {cameraNumber}
            </Text>
          </View>
        </View>

        {isStreaming && (
          <View style={styles.liveBadge}>
            <View style={styles.liveDot} />
            <Text style={styles.liveText}>LIVE</Text>
          </View>
        )}
      </SafeAreaView>

      {/* Bottom controls */}
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
            <View
              style={[
                styles.streamButtonInner,
                isStreaming && styles.streamButtonInnerActive,
              ]}
            />
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
            {isStreaming ? 'Streaming to server...' : 'Ready to stream'}
          </Text>
          <Text style={styles.statusText}>
            {facing === 'back' ? 'Rear cam' : 'Front cam'}
          </Text>
        </View>
      </SafeAreaView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  camera: {
    ...StyleSheet.absoluteFillObject,
  },
  // Permission screens
  permissionScreen: {
    flex: 1,
    backgroundColor: colors.background,
    justifyContent: 'center',
    alignItems: 'center',
    padding: spacing.lg,
  },
  permissionTitle: {
    fontSize: fontSize.xl,
    fontWeight: '700',
    color: colors.textPrimary,
    marginBottom: spacing.sm,
  },
  permissionText: {
    fontSize: fontSize.md,
    color: colors.textSecondary,
    textAlign: 'center',
  },
  permissionButton: {
    backgroundColor: colors.primary,
    borderRadius: 12,
    paddingHorizontal: spacing.xl,
    paddingVertical: spacing.md,
    marginTop: spacing.xl,
  },
  permissionButtonText: {
    fontSize: fontSize.md,
    fontWeight: '700',
    color: colors.background,
  },
  backButton: {
    marginTop: spacing.md,
    padding: spacing.md,
  },
  backButtonText: {
    fontSize: fontSize.md,
    color: colors.textMuted,
  },
  // Top overlay
  topOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
  },
  topBar: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: spacing.md,
    paddingTop: spacing.sm,
    gap: spacing.sm,
  },
  backArrow: {
    fontSize: fontSize.xxl,
    color: '#fff',
    fontWeight: '300',
    paddingRight: spacing.sm,
  },
  topInfo: {
    flex: 1,
  },
  matchLabel: {
    fontSize: fontSize.md,
    fontWeight: '600',
    color: '#fff',
  },
  matchCodeLabel: {
    fontSize: fontSize.xs,
    color: 'rgba(255,255,255,0.6)',
    marginTop: 2,
  },
  cameraBadge: {
    backgroundColor: colors.primary,
    borderRadius: 8,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.xs,
  },
  cameraBadgeText: {
    fontSize: fontSize.xs,
    fontWeight: '700',
    color: colors.background,
  },
  liveBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    alignSelf: 'center',
    backgroundColor: 'rgba(255, 77, 77, 0.9)',
    borderRadius: 6,
    paddingHorizontal: spacing.sm,
    paddingVertical: spacing.xs,
    marginTop: spacing.sm,
  },
  liveDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: '#fff',
    marginRight: spacing.xs,
  },
  liveText: {
    fontSize: fontSize.xs,
    fontWeight: '800',
    color: '#fff',
    letterSpacing: 1,
  },
  // Bottom overlay
  bottomOverlay: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
  },
  roleBanner: {
    alignSelf: 'center',
    backgroundColor: 'rgba(0,0,0,0.6)',
    borderRadius: 8,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.xs,
    marginBottom: spacing.sm,
  },
  roleText: {
    fontSize: fontSize.sm,
    fontWeight: '600',
    color: colors.primary,
    textTransform: 'uppercase',
    letterSpacing: 1,
  },
  controls: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    gap: spacing.xl,
    paddingVertical: spacing.md,
  },
  controlButton: {
    alignItems: 'center',
  },
  controlCircle: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: 'rgba(255,255,255,0.15)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  controlIcon: {
    fontSize: fontSize.xl,
    color: '#fff',
  },
  controlLabel: {
    fontSize: fontSize.xs,
    color: 'rgba(255,255,255,0.6)',
    marginTop: 4,
  },
  streamButton: {
    width: 72,
    height: 72,
    borderRadius: 36,
    borderWidth: 4,
    borderColor: '#fff',
    justifyContent: 'center',
    alignItems: 'center',
  },
  streamButtonActive: {
    borderColor: colors.error,
  },
  streamButtonInner: {
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: colors.error,
  },
  streamButtonInnerActive: {
    width: 28,
    height: 28,
    borderRadius: 6,
  },
  statusBar: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingHorizontal: spacing.lg,
    paddingBottom: spacing.sm,
  },
  statusText: {
    fontSize: fontSize.xs,
    color: 'rgba(255,255,255,0.5)',
  },
});
