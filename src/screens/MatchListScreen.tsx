import { useCallback, useState } from 'react';
import { ActivityIndicator, FlatList, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { useFocusEffect } from '@react-navigation/native';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { RootStackParamList } from '../navigation/AppNavigator';
import { colors, fontSize, spacing } from '../theme';
import { listMatches, type MatchDTO } from '../api';

type Props = NativeStackScreenProps<RootStackParamList, 'MatchList'>;

export default function MatchListScreen({ navigation }: Props) {
  const [matches, setMatches] = useState<MatchDTO[]>([]);
  const [loading, setLoading] = useState(true);

  useFocusEffect(
    useCallback(() => {
      let active = true;
      setLoading(true);
      listMatches()
        .then((data) => { if (active) setMatches(data); })
        .catch(() => { if (active) setMatches([]); })
        .finally(() => { if (active) setLoading(false); });
      return () => { active = false; };
    }, [])
  );

  if (loading) {
    return (
      <View style={[styles.container, styles.center]}>
        <ActivityIndicator size="large" color={colors.primary} />
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <FlatList
        data={matches}
        keyExtractor={(item) => item.code}
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
              {item.sport ? (
                <View style={styles.sportBadge}>
                  <Text style={styles.sportText}>{item.sport}</Text>
                </View>
              ) : null}
              {item.isLive && (
                <View style={styles.liveBadge}>
                  <View style={styles.liveDot} />
                  <Text style={styles.liveText}>LIVE</Text>
                </View>
              )}
            </View>

            <Text style={styles.matchTitle}>{item.title}</Text>
            <Text style={styles.teams}>{item.teamA} vs {item.teamB}</Text>
            {item.venue ? <Text style={styles.venue}>{item.venue}</Text> : null}

            {item.cameras.length > 0 && (
              <Text style={styles.cameras}>
                {item.cameras.length} camera{item.cameras.length !== 1 ? 's' : ''} broadcasting
              </Text>
            )}
          </TouchableOpacity>
        )}
        ListEmptyComponent={
          <View style={styles.empty}>
            <Text style={styles.emptyText}>No active matches</Text>
            <Text style={styles.emptyHint}>Matches will appear here when broadcasters create them</Text>
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
  center: {
    justifyContent: 'center',
    alignItems: 'center',
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
