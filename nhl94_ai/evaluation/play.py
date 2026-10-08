#!/usr/bin/env python3
"""
Play modes:
- Player vs Model
- Model vs Model
- Model vs Game
- Player vs Game (new mode)
"""
from nhl94_ai.agents.registry import CONTROLLERS, add_classic_arguments

from nhl94_ai.config import default_config_path, EnvironmentConfig
from nhl94_ai.model_inputs import add_model_input_arguments
from nhl94_ai.agents.base import AgentInput, LearnedAgent, FrameRepeatAgent

import sys
import argparse
import os
import time
import numpy as np
from nhl94_ai.training.logging import com_print, init_logger
from nhl94_ai.env.factory import init_env, init_play_env
from nhl94_ai.models.factory import init_model, get_model_probabilities, get_num_parameters
import nhl94_ai.env.components as games
from nhl94_ai.config import load_hyperparams, resolve_hyperparams_for_model
from nhl94_ai.game.ram import validate_rom_seed
from nhl94_ai.game.specs import get_game

def build_parser():
    parser = argparse.ArgumentParser(description='Play with your model in different modes')

    # Mode selection
    parser.add_argument('--mode', type=str, default='player_vs_model',
                       choices=['player_vs_model', 'model_vs_model', 'model_vs_game', 'player_vs_game'],
                       help='Game mode: player_vs_model, model_vs_model, model_vs_game, or player_vs_game')

    # Common arguments
    parser.add_argument('--env', type=str, default='NHL941on1-Genesis-v0')
    parser.add_argument('--state', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None,
                       help='Opt-in NHL94 ROM seed; each episode uses the next uint32 seed. Omit to retain the saved RNG.')
    parser.add_argument('--side', choices=['home', 'away'], default='home',
                       help='Classic team in model_vs_game; away uses a home-only full-team save')
    parser.add_argument('--goalie-policy', choices=['off', 'selective', 'always'], default='off',
                        help='Opt-in Classic manual goalie AI; requires full-team FILTERED controls and a manual-goalie save')
    parser.add_argument('--num_players', type=int, default=2)
    parser.add_argument('--num_env', type=int, default=1)
    parser.add_argument('--output_basedir', type=str, default='~/OUTPUT')
    parser.add_argument('--display_width', type=int, default=1920)
    parser.add_argument('--display_height', type=int, default=1080)
    parser.add_argument('--fullscreen', default=False, action='store_true')
    parser.add_argument('--deterministic', default=True, action='store_true')
    parser.add_argument('--rf', type=str, default='')
    parser.add_argument('--hyperparams', type=str, default=default_config_path("default"))
    parser.add_argument('--video', default=False, action='store_true')
    parser.add_argument('--video_path', type=str, default='../retro_game.avi')
    parser.add_argument('--record-mp4', metavar='PATH',
                       help='Record the entire 1920x1080 debug display to MP4 at 60 fps, including pauses (no audio).')
    parser.add_argument('--single_session', default=False, action='store_true',
                       help='Exit after the first completed game instead of continuing to the next session')
    parser.add_argument('--seq_len', type=int, default=16,
                       help='Frame history length for temporal policies such as HybridMambaPolicy or GRUMlpPolicy')
    parser.add_argument('--max_playback_speed', type=float, default=2.0,
                       help='Maximum watched playback speed relative to normal game speed. Capped at 2.0.')

    # Model-related arguments
    parser.add_argument('--alg', type=str, default='ppo2', help='Algorithm for single model')
    parser.add_argument('--p1_alg', type=str, default='ppo2', help='Algorithm for player 1 model')
    parser.add_argument('--p2_alg', type=str, default='ppo2', help='Algorithm for player 2 model')
    parser.add_argument('--nn', type=str, default='MlpPolicy')
    parser.add_argument('--model1_desc', type=str, default='CNN')
    parser.add_argument('--model2_desc', type=str, default='MLP')
    parser.add_argument('--nnsize', type=int, default=256)
    parser.add_argument('--model_1', type=str, default='', help='Model path for player 1')
    parser.add_argument('--model_2', type=str, default='', help='Model path for player 2')
    parser.add_argument('--load_p1_model', type=str, default='', help='Model path for player 1 in model_vs_model mode')
    parser.add_argument('--load_p2_model', type=str, default='', help='Model path for player 2 in model_vs_model mode')

    parser.add_argument('--action_type', type=str, default='FILTERED',
                       choices=['FILTERED', 'DISCRETE', 'MULTI_DISCRETE', 'HOCKEY_INTENT_DPAD', 'TARGET_POSITION'],
                       help='Buttons, hockey intents, or target-only positioning')

    return add_classic_arguments(add_model_input_arguments(parser))


def parse_cmdline(argv):
    args = build_parser().parse_args(argv)

    # Set default num_players based on mode
    if args.mode == 'model_vs_game' or args.mode == 'player_vs_game':
        args.num_players = 1
    elif args.mode == 'player_vs_model':
        args.num_players = 2

    return args

class NHL94Player:
    def __init__(self, args, logger, need_display=True):
        self.args = args
        if args.mode == 'player_vs_model':
            args.num_players = 2
        self.logger = logger
        self.need_display = need_display
        record_mp4 = getattr(args, 'record_mp4', None)
        if record_mp4 and (not need_display or args.mode == 'model_vs_model'):
            raise ValueError('--record-mp4 requires the debug display in model_vs_game, '
                             'player_vs_game, or player_vs_model mode')
        if getattr(args, 'seed', None) is not None:
            validate_rom_seed(args.seed)
            get_game(args.env)
        if getattr(args, 'cross_crease', False) or getattr(args, 'deke', False):
            EnvironmentConfig.from_args(args)
        if getattr(args, 'goalie_policy', 'off') != 'off':
            EnvironmentConfig.from_args(args)
            if args.mode not in ('model_vs_game', 'player_vs_model') or args.model_2:
                raise ValueError('Manual goalie AI supports one Classic agent versus CPU or human')
        if getattr(args, 'side', 'home') != 'home':
            requirements = (
                args.side == 'away', args.mode == 'model_vs_game', args.nn in CONTROLLERS,
                args.env == 'NHL94-Genesis-v0', args.action_type.upper() == 'FILTERED',
                args.rf == 'PostPlay', not args.model_2,
            )
            if not all(requirements):
                raise ValueError('--side away requires full-team Classic model_vs_game playback '
                                 'with FILTERED buttons, PostPlay, and one agent')
        self.max_playback_speed = max(0.1, min(float(getattr(args, 'max_playback_speed', 2.0)), 2.0))
        self.display_frame_interval = 1.0 / (60.0 * self.max_playback_speed) if need_display else 0.0
        self.next_display_frame_time = None

        if getattr(args, 'action_type', 'FILTERED').upper() == 'TARGET_POSITION':
            self.init_target_mode()
        elif args.mode == 'model_vs_model':
            self.init_model_vs_model()
        else:
            self.init_player_or_game_mode()

        if record_mp4:
            try:
                self.display_env.start_recording(record_mp4)
            except BaseException:
                self.close()
                raise

    def init_target_mode(self):
        EnvironmentConfig.from_args(self.args)
        if not self.args.model_1:
            raise ValueError('TARGET_POSITION playback requires --model_1 with a target-policy checkpoint')
        self.p1_env = init_env(
            None, 1, self.args.state, 1, self.args, self.args.hyperparams_dict,
            use_sticky_action=False, use_frame_skip=False,
            episode_rom_seed=getattr(self.args, 'seed', None),
        )
        try:
            model = init_model(None, self.args.model_1, self.args.alg, self.args, self.p1_env,
                               self.logger, self.args.hyperparams_dict)
            self.target_agent = FrameRepeatAgent(
                LearnedAgent(model, 'TARGET_POSITION'), self.args.hyperparams_dict.get('frame_skip', 4),
            )
            self.display_env = self.p1_env
            if self.need_display:
                self.display_env = games.get_bindings(self.args).sp_display_env(
                    self.p1_env, self.args, get_num_parameters(model), self.args.nn, ['TARGET_X', 'TARGET_Y'],
                )
        except BaseException:
            self.p1_env.close()
            raise

    def play_target_mode(self, continuous):
        observation = self.display_env.reset()
        self.target_agent.reset()
        self._reset_playback_timer()
        total_reward = 0.0
        while True:
            self._wait_for_display()
            action = self.target_agent.act(AgentInput(None, observation), self.args.deterministic).action
            observation, reward, done, info = self.display_env.step(action)
            total_reward += float(reward[0])
            self._throttle_display_frame()
            if np.any(done):
                if not continuous:
                    return info, total_reward
                # VecEnv already returned the next episode's reset observation.
                self.target_agent.reset()
                self._reset_playback_timer()

    def _reset_playback_timer(self):
        if not self.need_display:
            return
        self.next_display_frame_time = time.monotonic()

    def _wait_for_display(self, env=None):
        if not self.need_display:
            return
        display = self.display_env if env is None else env
        if hasattr(display, 'wait_until_running') and display.wait_until_running():
            self._reset_playback_timer()

    def close(self):
        for name in ('display_env', 'play_env', 'p1_env', 'p2_env'):
            env = getattr(self, name, None)
            if env is not None:
                env.close()

    def _throttle_display_frame(self):
        if not self.need_display:
            return

        now = time.monotonic()
        if self.next_display_frame_time is None:
            self.next_display_frame_time = now

        self.next_display_frame_time += self.display_frame_interval
        sleep_time = self.next_display_frame_time - now
        if sleep_time > 0:
            time.sleep(sleep_time)
        elif sleep_time < -0.5:
            # If rendering falls far behind, reset instead of trying to catch up forever.
            self.next_display_frame_time = now

    def init_model_vs_model(self):
        """Initialize for model vs model mode"""
        self.play_env = init_play_env(self.args, 2, self.args.hyperparams_dict, True)
        self.p1_env = init_env(None, 1, None, 1, self.args, self.args.hyperparams_dict, use_sticky_action=False)
        self.p2_env = init_env(None, 1, None, 1, self.args, self.args.hyperparams_dict, use_sticky_action=False)

        self.p1_model = init_model(None, self.args.load_p1_model, self.args.p1_alg, self.args, self.p1_env, self.logger, self.args.hyperparams_dict)
        self.p2_model = init_model(None, self.args.load_p2_model, self.args.p2_alg, self.args, self.p2_env, self.logger, self.args.hyperparams_dict)

        self.play_env.model1_params = get_num_parameters(self.p1_model)
        self.play_env.model2_params = get_num_parameters(self.p2_model)

    def init_player_or_game_mode(self):
        """Initialize for player vs model or model vs game modes"""
        num_players = 1 if self.args.mode in ['model_vs_game', 'player_vs_game'] else 2
        self.p1_env = init_env(None, 1, self.args.state, 1, self.args, self.args.hyperparams_dict, True)
        self.display_env = init_play_env(self.args, num_players, self.args.hyperparams_dict, False, self.need_display, False)

        if self.args.mode != 'player_vs_game':
            self.ai_sys = games.get_bindings(self.args).ai_sys(self.args, self.p1_env, self.logger)
            if self.args.nn in CONTROLLERS or self.args.model_1 != '' or self.args.model_2 != '':
                models = [self.args.model_1, self.args.model_2]
                self.ai_sys.SetModels(models)
                if self.args.nn in CONTROLLERS:
                    for model in self.ai_sys.models:
                        if model is not None:
                            model.frame_skip = 4

    def play(self, continuous=True, need_reset=True):
        """Main game loop"""
        if getattr(self.args, 'action_type', 'FILTERED').upper() == 'TARGET_POSITION':
            return self.play_target_mode(continuous)
        if self.args.mode == 'model_vs_model':
            self.play_model_vs_model(continuous, need_reset)
        else:
            self.play_player_or_game_mode(continuous, need_reset)

    def play_model_vs_model(self, continuous, need_reset):
        """Game loop for model vs model mode"""
        state = self.play_env.reset()
        self._reset_playback_timer()

        while True:
            self._wait_for_display(self.play_env)
            p1_actions = self.p1_model.predict(state)[0]  # Get the action array directly
            p2_actions = self.p2_model.predict(state)[0]  # Get the action array directly

            self.play_env.p1_action_probabilities = get_model_probabilities(self.p1_model, state)[0]
            self.play_env.p2_action_probabilities = get_model_probabilities(self.p2_model, state)[0]

            # Combine actions properly for 2-player game
            actions = [p1_actions, p2_actions]

            state, _, done, _ = self.play_env.step(actions)
            self._throttle_display_frame()

            if done:
                if continuous:
                    if need_reset:
                        state = self.play_env.reset()
                        self._reset_playback_timer()
                else:
                    return

    def play_player_or_game_mode(self, continuous, need_reset):
        """Game loop for player vs model or model vs game or player vs game modes"""
        state = self.display_env.reset()
        self._reset_playback_timer()
        total_rewards = 0
        scripted = self.args.nn in CONTROLLERS and self.args.mode != 'player_vs_game'
        info = self._scripted_reset_info() if scripted else None

        while True:
            self._wait_for_display()
            initialized = info is not None
            if self.args.mode == 'player_vs_game':
                # Player vs Game mode - just use player inputs
                # Ensure we pass a valid action format
                actions = self.display_env.player_actions
                if not isinstance(actions, list) or len(actions) not in [12, 24]:
                    actions = [0] * 12  # Default no-op action
            else:
                # Get actions from AI system for other modes
                p1_actions = self.ai_sys.predict(state, info=info, deterministic=self.args.deterministic)

                # The display samples human controller 2 independently every frame.
                actions = [p1_actions[0]]

            self.display_env.action_probabilities = []

            for i in range(4):
                if i:
                    self._wait_for_display()
                if i and scripted and initialized:
                    actions[0] = self.ai_sys.predict(state, info=info, deterministic=self.args.deterministic)[0]
                if self.need_display and self.args.mode != 'player_vs_game':
                    self.display_env.set_ai_sys_info(self.ai_sys)
                try:
                    state, reward, done, info = self.display_env.step(actions)
                    total_rewards += reward
                    self._throttle_display_frame()
                except ValueError as e:
                    if scripted or self.args.mode == 'player_vs_model':
                        raise
                    print(f"Invalid action format: {actions}")
                    actions = [0] * 6  # Fallback to no-op
                    state, reward, done, info = self.display_env.step(actions)
                    total_rewards += reward
                    self._throttle_display_frame()
                if scripted and np.any(done):
                    for model in self.ai_sys.models:
                        if model is not None:
                            model.reset()
                    self.ai_sys.last_diagnostics = {}
                    break

            if done:
                if continuous:
                    if need_reset:
                        state = self.display_env.reset()
                        self._reset_playback_timer()
                    if scripted:
                        info = self._scripted_reset_info()
                else:
                    return info, total_rewards

    def _scripted_reset_info(self):
        vector = self.display_env.env if self.need_display else self.display_env
        if not vector.reset_infos or not vector.reset_infos[0]:
            raise ValueError('Scripted playback requires fresh RAM feedback from reset.')
        return vector.reset_infos

def main(argv):
    return run(parse_cmdline(argv[1:]))


def run(args):
    if args.nn in CONTROLLERS and args.env in ('NHL941on1-Genesis-v0', 'NHL942on2-Genesis-v0', 'NHL94-Genesis-v0') and not args.rf:
        args.rf = 'PostPlay'
    args.hyperparams_dict = resolve_hyperparams_for_model(load_hyperparams(
        args.hyperparams,
        required=True,
        base_dir=os.path.dirname(__file__),
    ), args.nn)
    logger = init_logger(args)

    if args.mode != 'model_vs_model' and args.mode != 'player_vs_game':
        games.get_bindings(args)

    player = NHL94Player(args, logger)

    com_print('========= Start of Game Loop ==========')
    com_print('Debug display: Space/P pauses playback; 1-7 overlays, 8 AI planner, 9 teammate scores.')
    if args.mode in ['player_vs_model', 'player_vs_game']:
        com_print('Arrows: skate | X: pass/switch | C: shoot/check | Z: clear/hold | Enter: pause | ESC: quit')
    if args.mode == 'player_vs_model':
        com_print('AI: controller 1 / home team. Keyboard: controller 2 / away team.')
    elif args.mode == 'model_vs_game':
        com_print(f"AI: controller 1 / {getattr(args, 'side', 'home')} team. Opponent: built-in CPU.")

    try:
        player.play(continuous=not args.single_session, need_reset=False)
    finally:
        player.close()

if __name__ == '__main__':
    main(sys.argv)
