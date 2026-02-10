import { useState } from 'react';
import { StyleSheet, Text, TextInput, TouchableOpacity, View } from 'react-native';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { RootStackParamList } from '../navigation/AppNavigator';
import { colors, fontSize, spacing } from '../theme';

type Props = NativeStackScreenProps<RootStackParamList, 'JoinMatch'>;

export default function JoinMatchScreen({ navigation }: Props) {
  const [code, setCode] = useState('');

  const handleJoin = () => {
    if (code.trim().length < 6) return;
    // In Phase 3, this will validate against the backend.
    // For now, navigate with placeholder match info.
    navigation.navigate('CameraRole', {
      matchTitle: 'Live Match',
      matchCode: code.toUpperCase(),
      teamA: 'Team A',
      teamB: 'Team B',
    });
  };

  return (
    <View style={styles.container}>
      <View style={styles.content}>
        <Text style={styles.title}>Enter Match Code</Text>
        <Text style={styles.hint}>
          Get the 6-character code from the match organizer
        </Text>

        <TextInput
          style={styles.codeInput}
          placeholder="ABC123"
          placeholderTextColor={colors.textMuted}
          value={code}
          onChangeText={(text) => setCode(text.toUpperCase())}
          maxLength={6}
          autoCapitalize="characters"
          autoCorrect={false}
          textAlign="center"
        />

        <TouchableOpacity
          style={[styles.joinButton, code.length < 6 && styles.joinButtonDisabled]}
          onPress={handleJoin}
          disabled={code.length < 6}
        >
          <Text style={styles.joinButtonText}>Join Match</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background,
  },
  content: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: spacing.lg,
  },
  title: {
    fontSize: fontSize.xxl,
    fontWeight: '700',
    color: colors.textPrimary,
  },
  hint: {
    fontSize: fontSize.sm,
    color: colors.textMuted,
    marginTop: spacing.sm,
    textAlign: 'center',
  },
  codeInput: {
    backgroundColor: colors.surface,
    borderRadius: 16,
    paddingVertical: spacing.lg,
    paddingHorizontal: spacing.xl,
    fontSize: 36,
    fontWeight: '800',
    color: colors.primary,
    letterSpacing: 8,
    marginTop: spacing.xl,
    width: '100%',
    borderWidth: 2,
    borderColor: colors.border,
  },
  joinButton: {
    backgroundColor: colors.primary,
    borderRadius: 12,
    padding: spacing.md,
    alignItems: 'center',
    marginTop: spacing.xl,
    width: '100%',
  },
  joinButtonDisabled: {
    opacity: 0.4,
  },
  joinButtonText: {
    fontSize: fontSize.md,
    fontWeight: '700',
    color: colors.background,
  },
});
