import { StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { RootStackParamList } from '../navigation/AppNavigator';
import { colors, fontSize, spacing } from '../theme';

type Props = NativeStackScreenProps<RootStackParamList, 'CameraRole'>;

const CAMERA_ROLES = [
  { number: 1, label: 'Main', description: 'Center court, wide angle' },
  { number: 2, label: 'Side', description: 'Sideline perspective' },
  { number: 3, label: 'Close-up', description: 'Player tracking, zoom' },
  { number: 4, label: 'Wide', description: 'Full field overview' },
];

export default function CameraRoleScreen({ navigation, route }: Props) {
  const { matchTitle, matchCode, teamA, teamB } = route.params;

  const handleSelectRole = (role: typeof CAMERA_ROLES[number]) => {
    navigation.navigate('Camera', {
      matchTitle,
      matchCode,
      cameraRole: role.label,
      cameraNumber: role.number,
    });
  };

  return (
    <View style={styles.container}>
      <View style={styles.matchBar}>
        <Text style={styles.matchTitle}>{matchTitle}</Text>
        <Text style={styles.matchTeams}>{teamA} vs {teamB}</Text>
        <View style={styles.codeBadge}>
          <Text style={styles.codeText}>{matchCode}</Text>
        </View>
      </View>

      <Text style={styles.sectionTitle}>Choose Your Camera Position</Text>

      <View style={styles.grid}>
        {CAMERA_ROLES.map((role) => (
          <TouchableOpacity
            key={role.number}
            style={styles.roleCard}
            activeOpacity={0.7}
            onPress={() => handleSelectRole(role)}
          >
            <View style={styles.roleNumber}>
              <Text style={styles.roleNumberText}>{role.number}</Text>
            </View>
            <Text style={styles.roleLabel}>CAM {role.number}</Text>
            <Text style={styles.roleName}>{role.label}</Text>
            <Text style={styles.roleDescription}>{role.description}</Text>
          </TouchableOpacity>
        ))}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background,
    padding: spacing.lg,
  },
  matchBar: {
    backgroundColor: colors.surface,
    borderRadius: 16,
    padding: spacing.lg,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: colors.border,
    marginBottom: spacing.lg,
  },
  matchTitle: {
    fontSize: fontSize.lg,
    fontWeight: '700',
    color: colors.textPrimary,
  },
  matchTeams: {
    fontSize: fontSize.sm,
    color: colors.textSecondary,
    marginTop: 4,
  },
  codeBadge: {
    backgroundColor: colors.primaryDark,
    borderRadius: 8,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.xs,
    marginTop: spacing.sm,
  },
  codeText: {
    fontSize: fontSize.sm,
    fontWeight: '700',
    color: colors.primary,
    letterSpacing: 2,
  },
  sectionTitle: {
    fontSize: fontSize.md,
    fontWeight: '600',
    color: colors.textSecondary,
    marginBottom: spacing.md,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  grid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: spacing.md,
  },
  roleCard: {
    width: '47%',
    backgroundColor: colors.surface,
    borderRadius: 16,
    padding: spacing.lg,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: colors.border,
  },
  roleNumber: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: colors.primaryDark,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: spacing.sm,
  },
  roleNumberText: {
    fontSize: fontSize.xl,
    fontWeight: '800',
    color: colors.primary,
  },
  roleLabel: {
    fontSize: fontSize.xs,
    fontWeight: '700',
    color: colors.textMuted,
    letterSpacing: 1,
  },
  roleName: {
    fontSize: fontSize.lg,
    fontWeight: '600',
    color: colors.textPrimary,
    marginTop: 2,
  },
  roleDescription: {
    fontSize: fontSize.xs,
    color: colors.textSecondary,
    textAlign: 'center',
    marginTop: 4,
  },
});
