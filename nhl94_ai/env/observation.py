"""
NHL94 Observation wrapper
"""

import copy
from collections import deque
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from nhl94_ai.game.constants import GameConsts
from nhl94_ai.env.intents import HOCKEY_INTENT_DPAD_ACTIONS, HOCKEY_INTENT_DPAD_ACTION_SPACE, HOCKEY_INTENT_CARRY_PUCK, HOCKEY_INTENT_NORMAL_SHOOT, HOCKEY_INTENT_SLAPSHOT, HOCKEY_INTENT_ONE_TIMER, HOCKEY_INTENT_POKE_CHECK, HOCKEY_INTENT_CHANGE_PLAYER, HOCKEY_INTENT_PASS_START
from nhl94_ai.tasks.registry import get_task
from nhl94_ai.env.actions import HockeyActionController
from nhl94_ai.env.perspective import OpponentPerspective
from nhl94_ai.game.specs import get_game
from nhl94_ai.agents.multi_model import NHL94AISystem
from nhl94_ai.game.state import NHL94GameState
from nhl94_ai.game.ram import controller_team_info, pass_geometry_info, restore_away_control, select_cpu_side
from nhl94_ai.models.factory import load_model_for_inference


HOCKEY_PASS_RELEASE_FRAMES = 5
HOCKEY_ONE_TIMER_SHOT_DELAY_FRAMES = 1
HOCKEY_ONE_TIMER_SHOT_FRAMES = 12
HOCKEY_SLAPSHOT_HOLD_FRAMES = 18
HOCKEY_POKE_DISTANCE = 18.0
HOCKEY_CATCH_MAX_LOOKAHEAD_FRAMES = 36
HOCKEY_CATCH_PLAYER_SPEED_ESTIMATE = 7.2
HOCKEY_CATCH_BURST_DISTANCE = 22.0


class NHL94Observation2PEnv(gym.Wrapper):
    def __init__(self, env, args, num_players, rf_name):
        gym.Wrapper.__init__(self, env)

        self.nn = args.nn
        self.env_name = args.env
        self.play_side = getattr(args, 'side', 'home')
        self.action_type = getattr(args, 'action_type', 'FILTERED').upper()
        self.selfplay_enabled = bool(getattr(args, 'selfplay', False))
        self.selfplay_role = self._resolve_selfplay_role(args, rf_name)
        self.deterministic = bool(getattr(args, 'deterministic', True))
        self.opponent_model_alg = getattr(args, 'alg', 'ppo2')
        self.model_input_config = (getattr(args, 'hyperparams_dict', {}) or {}).get('model_input')

        self.rf_name = rf_name
        self.task = get_task(self.rf_name)
        self.init_function = self.task.initialize
        self.reward_function = self.task.reward
        self.done_function = self.task.done
        self.init_model = self.task.observation_size
        self.set_model_input = self.task.encode
        self.input_overide = self.task.restrict_action
        self.opponent_model = None
        self.opponent_agent = None
        self.opponent_model_path = ''
        self.opponent_set_model_input = None
        self.opponent_input_overide = None

        self.num_players_per_team = get_game(args.env).skaters_per_team

        from nhl94_ai.env.encoding import ObservationEncoder
        self.encoder = ObservationEncoder(self.task, self.num_players_per_team, self.model_input_config)
        self.NUM_PARAMS = self.encoder.size
        self.target_controller = None
        self.target_info = {}
        self.defense_camera = {}
        if self.action_type == 'TARGET_POSITION':
            from nhl94_ai.config import EnvironmentConfig
            from nhl94_ai.env.target_control import TargetPositionController, CONTROLLER_FIELDS
            EnvironmentConfig.from_args(args)
            if num_players != 1:
                raise ValueError('TARGET_POSITION requires one emulator controller')
            self.target_controller = TargetPositionController(self.task.target_control, self.task.target_bounds)
            self.NUM_PARAMS += len(CONTROLLER_FIELDS)

        self.game_state = NHL94GameState(self.num_players_per_team)
        self.uses_sequence_obs = self.nn in ('HybridMambaPolicy', 'GRUMlpPolicy')
        self.frame_stack_size = max(1, int(getattr(args, 'seq_len', 16)))
        self.frame_buffer = deque(maxlen=self.frame_stack_size) if self.uses_sequence_obs else None

        low = np.array([-1] * self.NUM_PARAMS, dtype=np.float32)
        high = np.array([1] * self.NUM_PARAMS, dtype=np.float32)

        if self.nn == 'CombinedPolicy':
            self.observation_space = spaces.Dict({
                'image': spaces.Box(low=0, high=255, shape=(224, 256, 3), dtype=np.uint8),
                'scalar': spaces.Box(low, high, dtype=np.float32)
            })
        elif self.uses_sequence_obs:
            self.observation_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.frame_stack_size, self.NUM_PARAMS),
                dtype=np.float32,
            )
        else:
            self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.target_xy = [-1, -1]

        self.num_players = num_players
        self.external_opponent = num_players == 2 and getattr(args, 'mode', None) == 'player_vs_model'
        if self.external_opponent and (self.action_type != 'FILTERED' or self.selfplay_enabled):
            raise ValueError('player_vs_model requires FILTERED buttons without self-play')
        if self.selfplay_enabled and self.num_players != 2:
            raise ValueError('Self-play requires a two-player emulator environment.')
        if self.selfplay_enabled and self.action_type == 'DISCRETE':
            raise NotImplementedError('Self-play does not yet support DISCRETE action_type.')
        if self.action_type == 'HOCKEY_INTENT_DPAD':
            if self.env_name != 'NHL94-Genesis-v0' or self.num_players_per_team != 5:
                raise ValueError('HOCKEY_INTENT_DPAD is only supported for NHL94-Genesis-v0 full 5v5.')
            if self.num_players != 1:
                raise ValueError('HOCKEY_INTENT_DPAD currently supports one learner controller only.')

        self.actions = HockeyActionController(self)
        self.perspective = OpponentPerspective(self)
        self.action_space = self._build_action_space()
        if self.external_opponent:
            self.action_space = spaces.MultiBinary(GameConsts.INPUT_MAX * 2)

        self.ai_sys = NHL94AISystem(args, env, None)

        self.ram_inited = False
        self.learner_action_state = self._new_action_state()
        self.opponent_action_state = self._new_action_state()

        if self.selfplay_enabled:
            self.set_selfplay_role(self.selfplay_role)
            self.set_opponent_model(getattr(args, 'load_opponent_model', ''))

    def _resolve_selfplay_role(self, args, rf_name):
        if rf_name == 'SelfPlayDefenseFinetune':
            return 'defense'
        if rf_name == 'SelfPlayOffenseFinetune':
            return 'offense'
        return getattr(args, 'selfplay_role', 'offense')

    def _build_action_space(self, *args, **kwargs):
        return self.actions._build_action_space(*args, **kwargs)

    def _default_env_action(self, *args, **kwargs):
        return self.actions._default_env_action(*args, **kwargs)

    def _default_controller_action(self, *args, **kwargs):
        return self.actions._default_controller_action(*args, **kwargs)

    def _default_reset_action(self, *args, **kwargs):
        return self.actions._default_reset_action(*args, **kwargs)

    def get_action_names(self):
        if self.target_controller is not None:
            return ['TARGET_X', 'TARGET_Y']
        if self.action_type == 'HOCKEY_INTENT_DPAD':
            return ['INTENT', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'BOOST']
        return getattr(self.env, 'buttons', [])

    def _new_action_state(self, *args, **kwargs):
        return self.actions._new_action_state(*args, **kwargs)

    def _reset_action_state(self, *args, **kwargs):
        return self.actions._reset_action_state(*args, **kwargs)

    def set_selfplay_role(self, role):
        if role not in ('offense', 'defense'):
            raise ValueError(f"Unsupported self-play role: {role}")

        self.selfplay_role = role
        opponent_rf = 'DefenseZone' if role == 'offense' else 'ScoreGoal'
        opponent_task = get_task(opponent_rf)
        self.opponent_set_model_input = opponent_task.encode
        self.opponent_input_overide = opponent_task.restrict_action
        return self.selfplay_role

    def set_opponent_model(self, path):
        if not path:
            self.opponent_model = None
            self.opponent_agent = None
            self.opponent_model_path = ''
            return None

        if path == self.opponent_model_path and self.opponent_model is not None:
            return self.opponent_model_path

        self.opponent_model = load_model_for_inference(path, self.opponent_model_alg)
        if getattr(self.opponent_model, '_nhl94_metadata', {}).get('schema', {}).get('action_type') == 'TARGET_POSITION':
            self.opponent_model = None
            raise ValueError('TARGET_POSITION opponents are not supported in self-play')
        from nhl94_ai.agents.base import LearnedAgent
        self.opponent_agent = LearnedAgent(self.opponent_model, self.action_type)
        self.opponent_model_path = path
        return self.opponent_model_path

    def _get_scalar_state_array(self):
        return np.asarray(self.state, dtype=np.float32)

    def _set_model_input(self, game_state):
        encoded = self.encoder.encode(game_state)
        if self.target_controller is not None:
            return np.concatenate((encoded, self.target_controller.observation())).astype(np.float32)
        return encoded

    def _reset_frame_buffer(self):
        if not self.uses_sequence_obs:
            return

        self.frame_buffer.clear()
        current_state = self._get_scalar_state_array()
        for _ in range(self.frame_stack_size):
            self.frame_buffer.append(current_state.copy())

    def _get_obs(self, image_obs=None):
        if self.nn == 'CombinedPolicy':
            return {
                'image': image_obs,
                'scalar': self.state
            }
        if self.uses_sequence_obs:
            return np.array(self.frame_buffer, dtype=np.float32, copy=True)
        return self.state

    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)

        self.init_function(self.env, self.env_name)
        if self.play_side == 'away':
            select_cpu_side(self.env.data, self.env.data.lookup_all(), 2)
        if self.target_controller is not None:
            self.target_info = self.env.data.lookup_all()

        reset_action = self._default_reset_action()
        if self.num_players == 2:
            reset_action = np.concatenate([reset_action, reset_action])

        state, _, _, _, info = self.env.step(reset_action)
        info = pass_geometry_info(self.env, info)
        if self.play_side == 'away' and (info.get('defense_team1'), info.get('defense_team2')) != (2, 0):
            raise ValueError('Away playback did not leave the home team under CPU control.')
        if self.play_side == 'away':
            info = controller_team_info(info)
        if self.external_opponent and (info.get('defense_team1'), info.get('defense_team2')) != (1, 2):
            raise ValueError('player_vs_model requires a two-controller save with P1 home and P2 away; use a compatible .2P state')

        self.game_state = NHL94GameState(self.num_players_per_team)
        self.ram_inited = True
        self._reset_action_state(self.learner_action_state)
        self._reset_action_state(self.opponent_action_state)
        if self.opponent_agent is not None:
            self.opponent_agent.reset()

        self.game_state.BeginFrame(info, [0] * 6)
        self.defense_camera = info.copy()
        if self.target_controller is not None:
            self.target_controller.reset()
            self._update_target_feedback(info)
        self.game_state.EndFrame()
        self.state = self._set_model_input(self.game_state)
        self._reset_frame_buffer()
        if self.uses_sequence_obs:
            self.frame_buffer.append(self._get_scalar_state_array().copy())

        return self._get_obs(state), info

    def _team_has_puck(self, *args, **kwargs):
        return self.actions._team_has_puck(*args, **kwargs)

    def _player_has_puck(self, *args, **kwargs):
        return self.actions._player_has_puck(*args, **kwargs)

    def _opponent_has_puck(self, *args, **kwargs):
        return self.actions._opponent_has_puck(*args, **kwargs)

    def _hockey_controlled_player(self, *args, **kwargs):
        return self.actions._hockey_controlled_player(*args, **kwargs)

    def _hockey_distance(self, *args, **kwargs):
        return self.actions._hockey_distance(*args, **kwargs)

    def _hockey_distance_xy(self, *args, **kwargs):
        return self.actions._hockey_distance_xy(*args, **kwargs)

    def _set_controller_gamestate(self, *args, **kwargs):
        return self.actions._set_controller_gamestate(*args, **kwargs)

    def _steer_hockey_toward(self, *args, **kwargs):
        return self.actions._steer_hockey_toward(*args, **kwargs)

    def _hockey_predict_puck_position(self, *args, **kwargs):
        return self.actions._hockey_predict_puck_position(*args, **kwargs)

    def _hockey_opponent_puck_carrier(self, *args, **kwargs):
        return self.actions._hockey_opponent_puck_carrier(*args, **kwargs)

    def _hockey_carrier_stick_target(self, *args, **kwargs):
        return self.actions._hockey_carrier_stick_target(*args, **kwargs)

    def _hockey_choose_intercept_target(self, *args, **kwargs):
        return self.actions._hockey_choose_intercept_target(*args, **kwargs)

    def _hockey_closing_speed(self, *args, **kwargs):
        return self.actions._hockey_closing_speed(*args, **kwargs)

    def _hockey_pass_target(self, *args, **kwargs):
        return self.actions._hockey_pass_target(*args, **kwargs)

    def _best_one_timer_target(self, *args, **kwargs):
        return self.actions._best_one_timer_target(*args, **kwargs)

    def _apply_pending_hockey_macro(self, *args, **kwargs):
        return self.actions._apply_pending_hockey_macro(*args, **kwargs)

    def _apply_hockey_intent(self, *args, **kwargs):
        return self.actions._apply_hockey_intent(*args, **kwargs)

    def _process_action(self, *args, **kwargs):
        return self.actions._process_action(*args, **kwargs)

    def _gamestate_action_from_env_action(self, *args, **kwargs):
        return self.actions._gamestate_action_from_env_action(*args, **kwargs)

    def _finalize_masked_action(self, *args, **kwargs):
        return self.actions._finalize_masked_action(*args, **kwargs)

    def _mirror_env_action(self, *args, **kwargs):
        return self.perspective._mirror_env_action(*args, **kwargs)

    def _mirror_gamestate_action(self, *args, **kwargs):
        return self.perspective._mirror_gamestate_action(*args, **kwargs)

    def _should_flip_zones_for_opponent(self, *args, **kwargs):
        return self.perspective._should_flip_zones_for_opponent(*args, **kwargs)

    def _rotate_player_180(self, *args, **kwargs):
        return self.perspective._rotate_player_180(*args, **kwargs)

    def _rotate_team_180(self, *args, **kwargs):
        return self.perspective._rotate_team_180(*args, **kwargs)

    def _recompute_team_relationships(self, *args, **kwargs):
        return self.perspective._recompute_team_relationships(*args, **kwargs)

    def _refresh_derived_state(self, *args, **kwargs):
        return self.perspective._refresh_derived_state(*args, **kwargs)

    def _build_opponent_view_state(self, *args, **kwargs):
        return self.perspective._build_opponent_view_state(*args, **kwargs)

    def _compute_opponent_action(self):
        if self.opponent_model is None:
            if self.opponent_model_path:
                self.set_opponent_model(self.opponent_model_path)
            if self.opponent_model is None:
                raise ValueError('Self-play requires load_opponent_model to be set.')

        opponent_state = self._build_opponent_view_state()
        opponent_obs = self.opponent_set_model_input(opponent_state)
        from nhl94_ai.agents.base import AgentInput
        opponent_action = self.opponent_agent.act(AgentInput(opponent_state, opponent_obs), self.deterministic).action
        if self._should_flip_zones_for_opponent():
            opponent_action = self._mirror_env_action(opponent_action)
        self.opponent_input_overide(opponent_action)
        return opponent_action

    def _update_target_feedback(self, info):
        # The queued scroll values before this step are what the VDP displays,
        # not Hpos/Vpos, which may already describe a future game frame.
        info['target_camera_x'] = -64 - self.target_info['target_scroll_x']
        info['target_camera_y'] = -self.target_info['target_scroll_y']
        self.target_info = info.copy()
        info['target_control'] = self.target_controller.observe(self.game_state, info)

    def step(self, ac):
        human_action = None
        if self.external_opponent:
            ac = np.asarray(ac)
            if not self.action_space.contains(ac):
                raise ValueError('player_vs_model requires 24 binary buttons: 12 AI buttons followed by 12 human buttons')
            ac, human_action = ac[:GameConsts.INPUT_MAX], ac[GameConsts.INPUT_MAX:]
        learner_action, gamestate_ac = self._process_action(ac, self.learner_action_state)
        self.input_overide(learner_action)
        gamestate_ac = self._finalize_masked_action(learner_action, self.learner_action_state)

        ac2 = learner_action
        if self.num_players == 2:
            if self.external_opponent:
                opponent_action, _ = self._process_action(human_action, self.opponent_action_state)
                self._finalize_masked_action(opponent_action, self.opponent_action_state)
            elif self.selfplay_enabled:
                opponent_action = self._compute_opponent_action()
                opponent_action, _ = self._process_action(opponent_action, self.opponent_action_state)
                self.opponent_input_overide(opponent_action)
                self._finalize_masked_action(opponent_action, self.opponent_action_state)
            else:
                opponent_action = np.zeros_like(learner_action)

            ac2 = np.concatenate([learner_action, opponent_action])

        ob, rew, terminated, truncated, info = self.env.step(ac2)
        info = pass_geometry_info(self.env, info)
        if self.play_side == 'away':
            info = controller_team_info(restore_away_control(self.env.data, info))
        if 'defense_scroll_x' in self.defense_camera:
            info['target_camera_x'] = -64 - self.defense_camera['defense_scroll_x']
            info['target_camera_y'] = -self.defense_camera['defense_scroll_y']
        self.defense_camera = info.copy()

        self.game_state.BeginFrame(info, gamestate_ac)
        if self.target_controller is not None:
            self._update_target_feedback(info)

        rew = self.reward_function(self.game_state)
        terminated = self.done_function(self.game_state)

        self.game_state.EndFrame()

        self.state = self._set_model_input(self.game_state)
        if self.uses_sequence_obs:
            self.frame_buffer.append(self._get_scalar_state_array().copy())

        return self._get_obs(ob), rew, terminated, truncated, info

    def seed(self, s):
        if hasattr(self.env, 'seed'):
            return self.env.seed(s)
        return None
