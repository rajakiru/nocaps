import { Image, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { RootStackParamList } from '../navigation/AppNavigator';
import { colors, fontSize, spacing } from '../theme';

type Props = NativeStackScreenProps<RootStackParamList, 'Home'>;

const actions = [
  {
    key: 'create',
    title: 'Create Match',
    description: 'Set up a new broadcast',
    icon: '+',
    route: 'CreateMatch' as const,
  },
  {
    key: 'join',
    title: 'Join as Camera',
    description: 'Enter a match code',
    icon: '~',
    route: 'JoinMatch' as const,
  },
  {
    key: 'watch',
    title: 'Watch a Match',
    description: 'Browse live streams',
    icon: '>',
    route: 'MatchList' as const,
  },
];

export default function HomeScreen({ navigation }: Props) {
  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <Image
          source={require('../../assets/logo.png')}
          style={styles.logo}
          resizeMode="contain"
        />
        <Text style={styles.tagline}>AI-Powered Sports Broadcasting</Text>
      </View>

      <View style={styles.actions}>
        {actions.map((action) => (
          <TouchableOpacity
            key={action.key}
            style={styles.card}
            activeOpacity={0.7}
            onPress={() => navigation.navigate(action.route)}
          >
            <View style={styles.cardIcon}>
              <Text style={styles.cardIconText}>{action.icon}</Text>
            </View>
            <View style={styles.cardContent}>
              <Text style={styles.cardTitle}>{action.title}</Text>
              <Text style={styles.cardDescription}>{action.description}</Text>
            </View>
            <Text style={styles.chevron}>{'>'}</Text>
          </TouchableOpacity>
        ))}
      </View>

      <Text style={styles.footer}>Team 3: Tabish, Akshara, Sai, Kiruthika</Text>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background,
    paddingHorizontal: spacing.lg,
  },
  header: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  logo: {
    width: 260,
    height: 75,
  },
  tagline: {
    fontSize: fontSize.sm,
    color: colors.textMuted,
    marginTop: spacing.sm,
    letterSpacing: 1,
    textTransform: 'uppercase',
  },
  actions: {
    gap: spacing.md,
    paddingBottom: spacing.xl,
  },
  card: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.surface,
    borderRadius: 16,
    padding: spacing.lg,
    borderWidth: 1,
    borderColor: colors.border,
  },
  cardIcon: {
    width: 48,
    height: 48,
    borderRadius: 12,
    backgroundColor: colors.primaryDark,
    justifyContent: 'center',
    alignItems: 'center',
  },
  cardIconText: {
    fontSize: fontSize.xl,
    color: colors.primary,
    fontWeight: '700',
  },
  cardContent: {
    flex: 1,
    marginLeft: spacing.md,
  },
  cardTitle: {
    fontSize: fontSize.lg,
    fontWeight: '600',
    color: colors.textPrimary,
  },
  cardDescription: {
    fontSize: fontSize.sm,
    color: colors.textSecondary,
    marginTop: 2,
  },
  chevron: {
    fontSize: fontSize.lg,
    color: colors.textMuted,
  },
  footer: {
    fontSize: fontSize.xs,
    color: colors.textMuted,
    textAlign: 'center',
    paddingBottom: spacing.md,
  },
});
