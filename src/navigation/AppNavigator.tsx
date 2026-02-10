import { createNativeStackNavigator } from '@react-navigation/native-stack';
import HomeScreen from '../screens/HomeScreen';
import CreateMatchScreen from '../screens/CreateMatchScreen';
import JoinMatchScreen from '../screens/JoinMatchScreen';
import CameraRoleScreen from '../screens/CameraRoleScreen';
import CameraScreen from '../screens/CameraScreen';
import MatchListScreen from '../screens/MatchListScreen';
import ViewerScreen from '../screens/ViewerScreen';
import { colors } from '../theme';

export type RootStackParamList = {
  Home: undefined;
  CreateMatch: undefined;
  JoinMatch: undefined;
  CameraRole: { matchTitle: string; matchCode: string; teamA: string; teamB: string };
  Camera: { matchTitle: string; matchCode: string; cameraRole: string; cameraNumber: number };
  MatchList: undefined;
  Viewer: { matchTitle: string; teamA: string; teamB: string };
};

const Stack = createNativeStackNavigator<RootStackParamList>();

export default function AppNavigator() {
  return (
    <Stack.Navigator
      screenOptions={{
        headerStyle: { backgroundColor: colors.surface },
        headerTintColor: colors.textPrimary,
        headerTitleStyle: { fontWeight: '600' },
        contentStyle: { backgroundColor: colors.background },
        animation: 'slide_from_right',
      }}
    >
      <Stack.Screen
        name="Home"
        component={HomeScreen}
        options={{ headerShown: false }}
      />
      <Stack.Screen
        name="CreateMatch"
        component={CreateMatchScreen}
        options={{ title: 'Create Match' }}
      />
      <Stack.Screen
        name="JoinMatch"
        component={JoinMatchScreen}
        options={{ title: 'Join as Camera' }}
      />
      <Stack.Screen
        name="CameraRole"
        component={CameraRoleScreen}
        options={{ title: 'Select Camera Position' }}
      />
      <Stack.Screen
        name="Camera"
        component={CameraScreen}
        options={{ headerShown: false }}
      />
      <Stack.Screen
        name="MatchList"
        component={MatchListScreen}
        options={{ title: 'Live Matches' }}
      />
      <Stack.Screen
        name="Viewer"
        component={ViewerScreen}
        options={{ headerShown: false }}
      />
    </Stack.Navigator>
  );
}
