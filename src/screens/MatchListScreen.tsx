import { FlatList, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { RootStackParamList } from '../navigation/AppNavigator';
import { colors, fontSize, spacing } from '../theme';

type Props = NativeStackScreenProps<RootStackParamList, 'MatchList'>;

// Mock data — replaced by backend in Phase 3
const MOCK_MATCHES = [
  {
    id: '1',
    title: 'CMU vs Pitt',
    teamA: 'CMU Tartans',
    teamB: 'Pitt Panthers',
    sport: 'Basketball',
    venue: 'Gesling Stadium',
    isLive: true,
    cameras: 3,
  },
  {
    id: '2',
    title: 'Intramural Finals',
    teamA: 'Team Alpha',
    teamB: 'Team Bravo',
    sport: 'Soccer',
    venue: 'Cut Field',
    isLive: true,
    cameras: 2,
  },
  {
    id: '3',
    title: 'Spring Invitational',
    teamA: 'CMU',
    teamB: 'Case Western',
    sport: 'Volleyball',
    venue: 'Wiegand Gym',
    isLive: false,
    cameras: 0,
  },
];

export default function MatchListScreen({ navigation }: Props) {
  return (
    <View style={styles.container}>
      <FlatList
        data={MOCK_MATCHES}
        keyExtractor={(item) => item.id}
        contentContainerStyle={styles.list}
        renderItem={({ item }) => (
          <TouchableOpacity
            style={styles.card}
            activeOpacity={0.7}
            onPress={() =>
              navigation.navigate('Viewer', {
                matchTitle: item.title,
                teamA: item.teamA,
                teamB: item.teamB,
              })
            }
          >
            <View style={styles.cardHeader}>
              <View style={styles.sportBadge}>
                <Text style={styles.sportText}>{item.sport}</Text>
              </View>
              {item.isLive && (
                <View style={styles.liveBadge}>
                  <View style={styles.liveDot} />
                  <Text style={styles.liveText}>LIVE</Text>
                </View>
              )}
            </View>

            <Text style={styles.matchTitle}>{item.title}</Text>
            <Text style={styles.teams}>{item.teamA} vs {item.teamB}</Text>
            <Text style={styles.venue}>{item.venue}</Text>

            {item.isLive && (
              <Text style={styles.cameras}>{item.cameras} cameras broadcasting</Text>
            )}
          </TouchableOpacity>
        )}
        ListEmptyComponent={
          <View style={styles.empty}>
            <Text style={styles.emptyText}>No active matches</Text>
            <Text style={styles.emptyHint}>Matches will appear here when broadcasters go live</Text>
          </View>
        }
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background,
  },
  list: {
    padding: spacing.lg,
    gap: spacing.md,
  },
  card: {
    backgroundColor: colors.surface,
    borderRadius: 16,
    padding: spacing.lg,
    borderWidth: 1,
    borderColor: colors.border,
  },
  cardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: spacing.sm,
  },
  sportBadge: {
    backgroundColor: colors.surfaceLight,
    borderRadius: 8,
    paddingHorizontal: spacing.sm,
    paddingVertical: spacing.xs,
  },
  sportText: {
    fontSize: fontSize.xs,
    color: colors.textSecondary,
    fontWeight: '600',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  liveBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(255, 77, 77, 0.15)',
    borderRadius: 6,
    paddingHorizontal: spacing.sm,
    paddingVertical: spacing.xs,
  },
  liveDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: colors.error,
    marginRight: spacing.xs,
  },
  liveText: {
    fontSize: fontSize.xs,
    fontWeight: '800',
    color: colors.error,
    letterSpacing: 1,
  },
  matchTitle: {
    fontSize: fontSize.lg,
    fontWeight: '700',
    color: colors.textPrimary,
  },
  teams: {
    fontSize: fontSize.md,
    color: colors.textSecondary,
    marginTop: 4,
  },
  venue: {
    fontSize: fontSize.sm,
    color: colors.textMuted,
    marginTop: 4,
  },
  cameras: {
    fontSize: fontSize.xs,
    color: colors.primary,
    marginTop: spacing.sm,
    fontWeight: '600',
  },
  empty: {
    alignItems: 'center',
    paddingTop: spacing.xxl,
  },
  emptyText: {
    fontSize: fontSize.lg,
    color: colors.textSecondary,
    fontWeight: '600',
  },
  emptyHint: {
    fontSize: fontSize.sm,
    color: colors.textMuted,
    marginTop: spacing.sm,
    textAlign: 'center',
  },
});
