import copy
import numpy as np
import gymnasium as gym
from nhl94_ai.game.constants import GameConsts
from nhl94_ai.game.geometry import aim_pass
from nhl94_ai.env.intents import HOCKEY_INTENT_CARRY_PUCK, HOCKEY_INTENT_CHANGE_PLAYER, HOCKEY_INTENT_DPAD_ACTIONS, HOCKEY_INTENT_DPAD_ACTION_SPACE, HOCKEY_INTENT_NORMAL_SHOOT, HOCKEY_INTENT_ONE_TIMER, HOCKEY_INTENT_PASS_START, HOCKEY_INTENT_POKE_CHECK, HOCKEY_INTENT_SLAPSHOT
HOCKEY_PASS_RELEASE_FRAMES = 5
HOCKEY_PASS_PRESS_FRAMES = 4
HOCKEY_ONE_TIMER_SHOT_DELAY_FRAMES = 1
HOCKEY_ONE_TIMER_SHOT_FRAMES = 12
HOCKEY_SLAPSHOT_HOLD_FRAMES = 18
HOCKEY_POKE_DISTANCE = 18.0
HOCKEY_CATCH_MAX_LOOKAHEAD_FRAMES = 36
HOCKEY_CATCH_PLAYER_SPEED_ESTIMATE = 7.2
HOCKEY_CATCH_BURST_DISTANCE = 22.0


class HockeyActionController:
    """Extracted behavior with an explicit environment-state context."""

    def __init__(self, context):
        self.context = context

    @property
    def action_type(self):
        return self.context.action_type

    @property
    def env(self):
        return self.context.env

    @property
    def game_state(self):
        return self.context.game_state

    def _build_action_space(self):
        if self.action_type == 'TARGET_POSITION':
            return gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
        if self.action_type == 'FILTERED':
            return gym.spaces.MultiBinary(GameConsts.INPUT_MAX)
        if self.action_type == 'MULTI_DISCRETE':
            return gym.spaces.MultiDiscrete([3, 3, 3])
        if self.action_type == 'HOCKEY_INTENT_DPAD':
            return gym.spaces.MultiDiscrete(HOCKEY_INTENT_DPAD_ACTION_SPACE)
        return self.env.action_space

    def _default_env_action(self):
        if self.action_type == 'TARGET_POSITION':
            return np.zeros(2, dtype=np.float32)
        if self.action_type == 'MULTI_DISCRETE':
            return np.zeros(3, dtype=np.int8)
        if self.action_type == 'HOCKEY_INTENT_DPAD':
            return np.zeros(len(HOCKEY_INTENT_DPAD_ACTION_SPACE), dtype=np.int8)
        return np.zeros(GameConsts.INPUT_MAX, dtype=np.int8)

    def _default_controller_action(self):
        return np.zeros(GameConsts.INPUT_MAX, dtype=np.int8)

    def _default_reset_action(self):
        if self.action_type in ('HOCKEY_INTENT_DPAD', 'TARGET_POSITION'):
            return self._default_controller_action()
        return self._default_env_action()

    def _new_action_state(self):
        return {
            'slapshot_frames': 0,
            'last_env_action': self._default_env_action(),
            'last_gamestate_action': [0] * 6,
            'hockey_macro': 'none',
            'pass_release_frames': 0,
            'pass_press_frames': 0,
            'pass_aim': (0, 0, 0, 0),
            'one_timer_shot_delay': 0,
            'one_timer_shot_frames': 0,
            'slapshot_charge_frames': 0,
        }

    def _reset_action_state(self, action_state):
        action_state['slapshot_frames'] = 0
        action_state['last_env_action'] = self._default_env_action()
        action_state['last_gamestate_action'] = [0] * 6
        action_state['hockey_macro'] = 'none'
        action_state['pass_release_frames'] = 0
        action_state['pass_press_frames'] = 0
        action_state['pass_aim'] = (0, 0, 0, 0)
        action_state['one_timer_shot_delay'] = 0
        action_state['one_timer_shot_frames'] = 0
        action_state['slapshot_charge_frames'] = 0

    def _team_has_puck(self, team):
        engine = getattr(self.game_state, 'engine', None)
        if getattr(engine, 'puck_owner_known', False):
            return team.owns_scnum(engine.puck_owner)
        return bool(getattr(team, 'player_haspuck', False) or getattr(team, 'goalie_haspuck', False))

    def _player_has_puck(self, team):
        engine = getattr(self.game_state, 'engine', None)
        if getattr(engine, 'puck_owner_known', False):
            return team.owns_scnum(engine.puck_owner) and engine.puck_owner != team.goalie_scnum()
        return bool(getattr(team, 'player_haspuck', False))

    def _opponent_has_puck(self):
        return self._team_has_puck(self.game_state.team2)

    def _hockey_controlled_player(self, team):
        return team.get_controlled_player()

    def _hockey_distance(self, first, second):
        return GameConsts.Distance(
            (getattr(first, 'x', 0) or 0, getattr(first, 'y', 0) or 0),
            (getattr(second, 'x', 0) or 0, getattr(second, 'y', 0) or 0),
        )

    def _hockey_distance_xy(self, first_x, first_y, second_x, second_y):
        return GameConsts.Distance((first_x, first_y), (second_x, second_y))

    def _set_controller_gamestate(self, processed_ac, gamestate_ac):
        gamestate_ac[0] = bool(processed_ac[GameConsts.INPUT_UP])
        gamestate_ac[1] = bool(processed_ac[GameConsts.INPUT_DOWN])
        gamestate_ac[2] = bool(processed_ac[GameConsts.INPUT_LEFT])
        gamestate_ac[3] = bool(processed_ac[GameConsts.INPUT_RIGHT])
        gamestate_ac[4] = bool(processed_ac[GameConsts.INPUT_B])
        gamestate_ac[5] = bool(processed_ac[GameConsts.INPUT_C])

    def _steer_hockey_toward(self, processed_ac, player, target, deadzone=4):
        target_x, target_y = target
        delta_x = target_x - (getattr(player, 'x', 0) or 0)
        delta_y = target_y - (getattr(player, 'y', 0) or 0)

        if delta_x <= -deadzone:
            processed_ac[GameConsts.INPUT_LEFT] = 1
            processed_ac[GameConsts.INPUT_RIGHT] = 0
        elif delta_x >= deadzone:
            processed_ac[GameConsts.INPUT_RIGHT] = 1
            processed_ac[GameConsts.INPUT_LEFT] = 0

        if delta_y <= -deadzone:
            processed_ac[GameConsts.INPUT_DOWN] = 1
            processed_ac[GameConsts.INPUT_UP] = 0
        elif delta_y >= deadzone:
            processed_ac[GameConsts.INPUT_UP] = 1
            processed_ac[GameConsts.INPUT_DOWN] = 0

    def _hockey_predict_puck_position(self, puck_x, puck_y, puck_vx, puck_vy, frames):
        predicted_x = puck_x + puck_vx * frames * 0.45
        predicted_y = puck_y + puck_vy * frames * 0.45

        if predicted_x < -GameConsts.MAX_PLAYER_X:
            predicted_x = -GameConsts.MAX_PLAYER_X + (-GameConsts.MAX_PLAYER_X - predicted_x) * 0.65
        elif predicted_x > GameConsts.MAX_PLAYER_X:
            predicted_x = GameConsts.MAX_PLAYER_X - (predicted_x - GameConsts.MAX_PLAYER_X) * 0.65

        if predicted_y < -GameConsts.MAX_PLAYER_Y:
            predicted_y = -GameConsts.MAX_PLAYER_Y + (-GameConsts.MAX_PLAYER_Y - predicted_y) * 0.45
        elif predicted_y > GameConsts.MAX_PLAYER_Y:
            predicted_y = GameConsts.MAX_PLAYER_Y - (predicted_y - GameConsts.MAX_PLAYER_Y) * 0.45

        return predicted_x, predicted_y

    def _hockey_opponent_puck_carrier(self):
        opponents = self.game_state.team2
        if self._team_has_puck(opponents):
            return opponents.get_controlled_player()
        return None

    def _hockey_carrier_stick_target(self, carrier, puck):
        carrier_x = getattr(carrier, 'x', 0) or 0
        carrier_y = getattr(carrier, 'y', 0) or 0
        puck_x = getattr(puck, 'x', 0) or 0
        puck_y = getattr(puck, 'y', 0) or 0
        toward_puck_x = puck_x - carrier_x
        toward_puck_y = puck_y - carrier_y
        length = max(1.0, self._hockey_distance_xy(carrier_x, carrier_y, puck_x, puck_y))
        stick_x = carrier_x + toward_puck_x / length * 10.0
        stick_y = carrier_y + toward_puck_y / length * 10.0
        return (
            stick_x + (getattr(carrier, 'vx', 0) or 0) * 0.65,
            stick_y + (getattr(carrier, 'vy', 0) or 0) * 0.65,
        )

    def _hockey_choose_intercept_target(self, controlled):
        puck = self.game_state.puck
        player_x = getattr(controlled, 'x', 0) or 0
        player_y = getattr(controlled, 'y', 0) or 0
        player_vx = getattr(controlled, 'vx', 0) or 0
        player_vy = getattr(controlled, 'vy', 0) or 0
        puck_x = getattr(puck, 'x', 0) or 0
        puck_y = getattr(puck, 'y', 0) or 0
        puck_vx = getattr(puck, 'vx', 0) or 0
        puck_vy = getattr(puck, 'vy', 0) or 0

        best_target = (puck_x, puck_y)
        best_error = float('inf')
        for frames in range(1, HOCKEY_CATCH_MAX_LOOKAHEAD_FRAMES + 1):
            predicted_puck_x, predicted_puck_y = self._hockey_predict_puck_position(
                puck_x, puck_y, puck_vx, puck_vy, frames,
            )
            desired_player_x = player_x + player_vx * min(frames, 8) * 0.20
            desired_player_y = player_y + player_vy * min(frames, 8) * 0.20
            distance = self._hockey_distance_xy(
                desired_player_x, desired_player_y, predicted_puck_x, predicted_puck_y,
            )
            reachable_distance = HOCKEY_CATCH_PLAYER_SPEED_ESTIMATE * frames
            error = abs(distance - reachable_distance) + frames * 0.15
            if error < best_error:
                best_error = error
                best_target = (predicted_puck_x, predicted_puck_y)

        carrier = self._hockey_opponent_puck_carrier()
        if carrier is not None:
            carrier_target = self._hockey_carrier_stick_target(carrier, puck)
            carrier_weight = 0.65 if self._hockey_distance(controlled, carrier) < 42 else 0.35
            best_target = (
                best_target[0] * (1.0 - carrier_weight) + carrier_target[0] * carrier_weight,
                best_target[1] * (1.0 - carrier_weight) + carrier_target[1] * carrier_weight,
            )

        return (
            int(np.clip(best_target[0], -GameConsts.MAX_PLAYER_X, GameConsts.MAX_PLAYER_X)),
            int(np.clip(best_target[1], -GameConsts.MAX_PLAYER_Y, GameConsts.MAX_PLAYER_Y)),
        )

    def _hockey_closing_speed(self, controlled):
        puck = self.game_state.puck
        delta_x = (getattr(puck, 'x', 0) or 0) - (getattr(controlled, 'x', 0) or 0)
        delta_y = (getattr(puck, 'y', 0) or 0) - (getattr(controlled, 'y', 0) or 0)
        distance = max(1.0, self._hockey_distance_xy(0, 0, delta_x, delta_y))
        relative_vx = (getattr(controlled, 'vx', 0) or 0) - (getattr(puck, 'vx', 0) or 0)
        relative_vy = (getattr(controlled, 'vy', 0) or 0) - (getattr(puck, 'vy', 0) or 0)
        return (relative_vx * delta_x + relative_vy * delta_y) / distance

    def _hockey_pass_target(self, intent):
        target_index = intent - HOCKEY_INTENT_PASS_START
        players = list(getattr(self.game_state.team1, 'players', []) or [])
        controlled_index = getattr(self.game_state.team1, 'control', 0) - 1
        teammates = [player for index, player in enumerate(players) if index != controlled_index]
        if target_index < 0 or target_index >= len(teammates):
            return None
        return teammates[target_index]

    def _best_one_timer_target(self):
        team = self.game_state.team1
        opponents = self.game_state.team2
        controlled = self._hockey_controlled_player(team)
        controlled_index = getattr(team, 'control', 0) - 1
        opponent_group = list(getattr(opponents, 'players', []) or []) + [getattr(opponents, 'goalie', None)]
        opponent_group = [player for player in opponent_group if player is not None]

        best_player = None
        best_score = -float('inf')
        for index, player in enumerate(team.players):
            if index == controlled_index:
                continue
            if getattr(player, 'is_falling', 0.0) or getattr(player, 'is_dive', 0.0):
                continue

            receiver_space = min(
                [self._hockey_distance(player, opponent) for opponent in opponent_group] or [999.0]
            )
            lateral_separation = abs((getattr(player, 'x', 0) or 0) - (getattr(controlled, 'x', 0) or 0))
            vertical_separation = abs((getattr(player, 'y', 0) or 0) - (getattr(controlled, 'y', 0) or 0))
            score = receiver_space + lateral_separation * 0.45 + vertical_separation * 0.20
            if getattr(player, 'passing_lane_clear', False):
                score += 24.0
            if abs(getattr(player, 'x', 0) or 0) > 96:
                score -= 18.0

            if score > best_score:
                best_score = score
                best_player = player

        return best_player

    def _apply_pending_hockey_macro(self, processed_ac, action_state):
        if self._opponent_has_puck():
            action_state['one_timer_shot_delay'] = 0
            action_state['one_timer_shot_frames'] = 0
            action_state['pass_press_frames'] = 0

        if action_state['pass_release_frames'] > 0:
            action_state['pass_release_frames'] -= 1

        if action_state['pass_press_frames'] > 0:
            # An immediate B release skips the ROM's directional pass update;
            # keep B and the chosen aim stable across its input sampling ticks.
            processed_ac[GameConsts.INPUT_B] = 1
            processed_ac[4:8] = action_state['pass_aim']
            action_state['pass_press_frames'] -= 1
            action_state['hockey_macro'] = 'pass_press'
            return True

        if action_state['one_timer_shot_frames'] <= 0:
            return False

        action_state['hockey_macro'] = 'one_timer_wait'
        if action_state['one_timer_shot_delay'] > 0:
            action_state['one_timer_shot_delay'] -= 1
            return True

        processed_ac[GameConsts.INPUT_C] = 1
        action_state['one_timer_shot_frames'] -= 1
        action_state['hockey_macro'] = 'one_timer_shoot'
        return True

    def _apply_hockey_intent(self, intent_ac, processed_ac, gamestate_ac, action_state):
        action_state['hockey_macro'] = 'none'

        if self._apply_pending_hockey_macro(processed_ac, action_state):
            self._set_controller_gamestate(processed_ac, gamestate_ac)
            return

        intent = int(np.clip(intent_ac[0], 0, len(HOCKEY_INTENT_DPAD_ACTIONS) - 1))
        boost = bool(intent_ac[5])
        team = self.game_state.team1
        controlled = self._hockey_controlled_player(team)
        own_player_has_puck = self._player_has_puck(team)
        own_team_has_puck = self._team_has_puck(team)

        if intent == HOCKEY_INTENT_CARRY_PUCK:
            if own_player_has_puck:
                action_state['hockey_macro'] = 'carry_puck'

        elif intent == HOCKEY_INTENT_NORMAL_SHOOT:
            if own_player_has_puck:
                processed_ac[GameConsts.INPUT_C] = 1
                action_state['hockey_macro'] = 'normal_shoot'

        elif intent == HOCKEY_INTENT_SLAPSHOT:
            if own_player_has_puck and action_state['slapshot_charge_frames'] < HOCKEY_SLAPSHOT_HOLD_FRAMES:
                processed_ac[GameConsts.INPUT_C] = 1
                action_state['slapshot_charge_frames'] += 1
                action_state['hockey_macro'] = 'slapshot_charge'
            else:
                if action_state['slapshot_charge_frames'] > 0:
                    action_state['hockey_macro'] = 'slapshot_release'
                action_state['slapshot_charge_frames'] = 0
        else:
            action_state['slapshot_charge_frames'] = 0

        if intent == HOCKEY_INTENT_POKE_CHECK:
            distance_to_puck = self._hockey_distance(controlled, self.game_state.puck)
            if not own_team_has_puck and (self._opponent_has_puck() or distance_to_puck <= HOCKEY_POKE_DISTANCE):
                processed_ac[GameConsts.INPUT_B] = 1
                action_state['hockey_macro'] = 'poke_check'

        elif intent == HOCKEY_INTENT_CHANGE_PLAYER:
            if not own_team_has_puck:
                processed_ac[GameConsts.INPUT_B] = 1
                action_state['hockey_macro'] = 'change_player'

        elif intent >= HOCKEY_INTENT_PASS_START:
            target = self._hockey_pass_target(intent)
            if target is not None:
                aim_pass(processed_ac, controlled, target)
                if own_team_has_puck and action_state['pass_release_frames'] <= 0:
                    processed_ac[GameConsts.INPUT_B] = 1
                    action_state['pass_release_frames'] = HOCKEY_PASS_RELEASE_FRAMES
                    action_state['pass_press_frames'] = HOCKEY_PASS_PRESS_FRAMES - 1
                    action_state['pass_aim'] = tuple(processed_ac[4:8])
                    action_state['hockey_macro'] = HOCKEY_INTENT_DPAD_ACTIONS[intent].lower()

        elif intent == HOCKEY_INTENT_ONE_TIMER:
            target = self._best_one_timer_target()
            if target is not None:
                aim_pass(processed_ac, controlled, target)
                if own_team_has_puck and action_state['pass_release_frames'] <= 0:
                    processed_ac[GameConsts.INPUT_B] = 1
                    action_state['pass_release_frames'] = HOCKEY_PASS_RELEASE_FRAMES
                    action_state['pass_press_frames'] = HOCKEY_PASS_PRESS_FRAMES - 1
                    action_state['pass_aim'] = tuple(processed_ac[4:8])
                    action_state['one_timer_shot_delay'] = HOCKEY_ONE_TIMER_SHOT_DELAY_FRAMES
                    action_state['one_timer_shot_frames'] = HOCKEY_ONE_TIMER_SHOT_FRAMES
                    action_state['hockey_macro'] = 'one_timer_pass'

        if boost and not own_team_has_puck and not processed_ac[GameConsts.INPUT_C]:
            processed_ac[GameConsts.INPUT_C] = 1
            if action_state['hockey_macro'] == 'none':
                action_state['hockey_macro'] = 'boost'

        self._set_controller_gamestate(processed_ac, gamestate_ac)

    def _process_action(self, ac, action_state):
        gamestate_ac = [0] * 6

        if self.action_type == 'TARGET_POSITION':
            processed_ac = self.context.target_controller.step(ac, self.game_state, self.context.target_info)
            recorded_ac = np.asarray(ac, dtype=np.float32)
            self._set_controller_gamestate(processed_ac, gamestate_ac)
        elif isinstance(ac, (list, np.ndarray)) and len(ac) == GameConsts.INPUT_MAX:
            processed_ac = np.asarray(ac, dtype=np.int8).copy()
            recorded_ac = processed_ac

            gamestate_ac[0] = bool(processed_ac[GameConsts.INPUT_UP])
            gamestate_ac[1] = bool(processed_ac[GameConsts.INPUT_DOWN])
            gamestate_ac[2] = bool(processed_ac[GameConsts.INPUT_LEFT])
            gamestate_ac[3] = bool(processed_ac[GameConsts.INPUT_RIGHT])
            gamestate_ac[4] = bool(processed_ac[GameConsts.INPUT_B])
            gamestate_ac[5] = bool(processed_ac[GameConsts.INPUT_C])

        elif isinstance(ac, (list, np.ndarray)) and len(ac) == 3:
            processed_ac = np.asarray(ac, dtype=np.int8).copy()
            recorded_ac = processed_ac

            gamestate_ac[0] = processed_ac[0] == 1
            gamestate_ac[1] = processed_ac[0] == 2
            gamestate_ac[2] = processed_ac[1] == 1
            gamestate_ac[3] = processed_ac[1] == 2
            gamestate_ac[4] = processed_ac[2] == 1
            gamestate_ac[5] = processed_ac[2] == 2

        elif self.action_type == 'HOCKEY_INTENT_DPAD' and isinstance(ac, (list, np.ndarray)) and len(ac) == 6:
            intent_ac = np.asarray(ac, dtype=np.int8).copy()
            recorded_ac = intent_ac
            processed_ac = np.zeros(GameConsts.INPUT_MAX, dtype=np.int8)

            up = bool(intent_ac[1])
            down = bool(intent_ac[2])
            left = bool(intent_ac[3])
            right = bool(intent_ac[4])

            if up != down:
                processed_ac[GameConsts.INPUT_UP if up else GameConsts.INPUT_DOWN] = 1
            if left != right:
                processed_ac[GameConsts.INPUT_LEFT if left else GameConsts.INPUT_RIGHT] = 1

            self._apply_hockey_intent(intent_ac, processed_ac, gamestate_ac, action_state)
        else:
            raise ValueError(f"Unsupported action format: {ac}")

        if gamestate_ac[5]:
            action_state['slapshot_frames'] += 1
        else:
            action_state['slapshot_frames'] = 0
        action_state['last_env_action'] = recorded_ac.copy()
        action_state['last_gamestate_action'] = list(gamestate_ac)
        return processed_ac, gamestate_ac

    def _gamestate_action_from_env_action(self, env_action):
        gamestate_ac = [0] * 6

        if isinstance(env_action, (list, np.ndarray)) and len(env_action) == GameConsts.INPUT_MAX:
            env_action = np.asarray(env_action, dtype=np.int8)
            gamestate_ac[0] = bool(env_action[GameConsts.INPUT_UP])
            gamestate_ac[1] = bool(env_action[GameConsts.INPUT_DOWN])
            gamestate_ac[2] = bool(env_action[GameConsts.INPUT_LEFT])
            gamestate_ac[3] = bool(env_action[GameConsts.INPUT_RIGHT])
            gamestate_ac[4] = bool(env_action[GameConsts.INPUT_B])
            gamestate_ac[5] = bool(env_action[GameConsts.INPUT_C])
            return gamestate_ac

        if isinstance(env_action, (list, np.ndarray)) and len(env_action) == 3:
            env_action = np.asarray(env_action, dtype=np.int8)
            gamestate_ac[0] = env_action[0] == 1
            gamestate_ac[1] = env_action[0] == 2
            gamestate_ac[2] = env_action[1] == 1
            gamestate_ac[3] = env_action[1] == 2
            gamestate_ac[4] = env_action[2] == 1
            gamestate_ac[5] = env_action[2] == 2
            return gamestate_ac

        raise ValueError(f"Unsupported env action format: {env_action}")

    def _finalize_masked_action(self, env_action, action_state):
        masked_gamestate_ac = self._gamestate_action_from_env_action(env_action)
        if self.action_type == 'TARGET_POSITION':
            self.context.target_controller.buttons = np.asarray(env_action, dtype=np.int8).copy()

        if not masked_gamestate_ac[5]:
            action_state['slapshot_frames'] = 0
            action_state['slapshot_charge_frames'] = 0
            action_state['one_timer_shot_delay'] = 0
            action_state['one_timer_shot_frames'] = 0
            if action_state['hockey_macro'] in {
                'normal_shoot',
                'slapshot_charge',
                'slapshot_release',
                'one_timer_wait',
                'one_timer_shoot',
                'boost',
            }:
                action_state['hockey_macro'] = 'none'

        action_state['last_env_action'] = np.asarray(env_action, dtype=np.int8).copy()
        action_state['last_gamestate_action'] = list(masked_gamestate_ac)
        return masked_gamestate_ac
