import { useState } from 'react';
import {
  Alert,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from 'react-native';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { RootStackParamList } from '../navigation/AppNavigator';
import { colors, fontSize, spacing } from '../theme';
import { createMatch as apiCreateMatch } from '../api';

type Props = NativeStackScreenProps<RootStackParamList, 'CreateMatch'>;

const SPORTS = ['Basketball', 'Soccer', 'Football', 'Tennis', 'Volleyball', 'Baseball', 'Other'];

export default function CreateMatchScreen({ navigation }: Props) {
  const [title, setTitle] = useState('');
  const [teamA, setTeamA] = useState('');
  const [teamB, setTeamB] = useState('');
  const [sport, setSport] = useState('');
  const [venue, setVenue] = useState('');
  const [matchCode, setMatchCode] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleCreate = async () => {
    if (!title.trim() || !teamA.trim() || !teamB.trim()) {
      Alert.alert('Missing Info', 'Please fill in match title and both team names.');
      return;
    }
    setLoading(true);
    try {
      const match = await apiCreateMatch({ title, teamA, teamB, sport, venue });
      setMatchCode(match.code);
    } catch {
      Alert.alert('Error', 'Could not create match. Is the server running?');
    } finally {
      setLoading(false);
    }
  };

  const handleContinue = () => {
    if (!matchCode) return;
    navigation.navigate('CameraRole', {
      matchTitle: title || `${teamA} vs ${teamB}`,
      matchCode,
      teamA,
      teamB,
    });
  };

  if (matchCode) {
    return (
      <View style={styles.container}>
        <View style={styles.codeContainer}>
          <Text style={styles.codeLabel}>Your Match Code</Text>
          <Text style={styles.codeText}>{matchCode}</Text>
          <Text style={styles.codeHint}>
            Share this code with camera operators to join your broadcast
          </Text>

          <View style={styles.matchSummary}>
            <Text style={styles.summaryTitle}>{title || `${teamA} vs ${teamB}`}</Text>
            <Text style={styles.summaryDetail}>{teamA} vs {teamB}</Text>
            {sport ? <Text style={styles.summaryDetail}>{sport}</Text> : null}
            {venue ? <Text style={styles.summaryDetail}>{venue}</Text> : null}
          </View>

          <TouchableOpacity style={styles.primaryButton} onPress={handleContinue}>
            <Text style={styles.primaryButtonText}>Continue as Camera</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  }

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.scrollContent}>
      <Text style={styles.sectionTitle}>Match Details</Text>

      <Text style={styles.label}>Match Title</Text>
      <TextInput
        style={styles.input}
        placeholder="e.g. CMU vs Pitt"
        placeholderTextColor={colors.textMuted}
        value={title}
        onChangeText={setTitle}
      />

      <Text style={styles.label}>Team A</Text>
      <TextInput
        style={styles.input}
        placeholder="Home team name"
        placeholderTextColor={colors.textMuted}
        value={teamA}
        onChangeText={setTeamA}
      />

      <Text style={styles.label}>Team B</Text>
      <TextInput
        style={styles.input}
        placeholder="Away team name"
        placeholderTextColor={colors.textMuted}
        value={teamB}
        onChangeText={setTeamB}
      />

      <Text style={styles.label}>Sport</Text>
      <View style={styles.sportGrid}>
        {SPORTS.map((s) => (
          <TouchableOpacity
            key={s}
            style={[styles.sportChip, sport === s && styles.sportChipActive]}
            onPress={() => setSport(s)}
          >
            <Text style={[styles.sportChipText, sport === s && styles.sportChipTextActive]}>
              {s}
            </Text>
          </TouchableOpacity>
        ))}
      </View>

      <Text style={styles.label}>Venue / Court</Text>
      <TextInput
        style={styles.input}
        placeholder="e.g. Gesling Stadium"
        placeholderTextColor={colors.textMuted}
        value={venue}
        onChangeText={setVenue}
      />

      <TouchableOpacity
        style={[styles.primaryButton, loading && { opacity: 0.5 }]}
        onPress={handleCreate}
        disabled={loading}
      >
        <Text style={styles.primaryButtonText}>{loading ? 'Creating...' : 'Create Match'}</Text>
      </TouchableOpacity>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background,
  },
  scrollContent: {
    padding: spacing.lg,
    paddingBottom: spacing.xxl,
  },
  sectionTitle: {
    fontSize: fontSize.xl,
    fontWeight: '700',
    color: colors.textPrimary,
    marginBottom: spacing.lg,
  },
  label: {
    fontSize: fontSize.sm,
    fontWeight: '600',
    color: colors.textSecondary,
    marginBottom: spacing.xs,
    marginTop: spacing.md,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  input: {
    backgroundColor: colors.surface,
    borderRadius: 12,
    padding: spacing.md,
    fontSize: fontSize.md,
    color: colors.textPrimary,
    borderWidth: 1,
    borderColor: colors.border,
  },
  sportGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: spacing.sm,
  },
  sportChip: {
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    borderRadius: 20,
    backgroundColor: colors.surface,
    borderWidth: 1,
    borderColor: colors.border,
  },
  sportChipActive: {
    backgroundColor: colors.primaryDark,
    borderColor: colors.primary,
  },
  sportChipText: {
    fontSize: fontSize.sm,
    color: colors.textSecondary,
  },
  sportChipTextActive: {
    color: colors.primary,
    fontWeight: '600',
  },
  primaryButton: {
    backgroundColor: colors.primary,
    borderRadius: 12,
    padding: spacing.md,
    alignItems: 'center',
    marginTop: spacing.xl,
  },
  primaryButtonText: {
    fontSize: fontSize.md,
    fontWeight: '700',
    color: colors.background,
  },
  codeContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: spacing.lg,
  },
  codeLabel: {
    fontSize: fontSize.md,
    color: colors.textSecondary,
    textTransform: 'uppercase',
    letterSpacing: 1,
  },
  codeText: {
    fontSize: 56,
    fontWeight: '800',
    color: colors.primary,
    letterSpacing: 8,
    marginTop: spacing.sm,
  },
  codeHint: {
    fontSize: fontSize.sm,
    color: colors.textMuted,
    textAlign: 'center',
    marginTop: spacing.md,
    maxWidth: 280,
  },
  matchSummary: {
    backgroundColor: colors.surface,
    borderRadius: 16,
    padding: spacing.lg,
    marginTop: spacing.xl,
    width: '100%',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: colors.border,
  },
  summaryTitle: {
    fontSize: fontSize.lg,
    fontWeight: '700',
    color: colors.textPrimary,
  },
  summaryDetail: {
    fontSize: fontSize.sm,
    color: colors.textSecondary,
    marginTop: 4,
  },
});
