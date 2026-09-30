"""Separate keyboard and AI controllers without changing single-policy schemas."""
from collections import defaultdict
import importlib.util
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np

from nhl94_ai.cli import main
from nhl94_ai.config import EnvironmentConfig
from nhl94_ai.env.observation import NHL94Observation2PEnv
from nhl94_ai.evaluation.play import parse_cmdline
from nhl94_ai.game.constants import GameConsts as Buttons
from tests.unit.test_environment_contracts import FixtureEnv


def human_args(**values):
    return SimpleNamespace(**dict({
        'mode': 'player_vs_model', 'env': 'NHL94-Genesis-v0', 'nn': 'ClassicAIV1',
        'num_players': 2, 'action_type': 'FILTERED', 'seq_len': 4, 'alg': 'ppo2',
        'hyperparams_dict': {},
    }, **values))


def human_env(*, controllers=2, mode='player_vs_model', truncate=False):
    base = FixtureEnv(truncate=truncate)
    base.render_mode = 'rgb_array'
    base.action_space = gym.spaces.MultiBinary(12 * controllers)
    base.info.update(defense_team1=1, defense_team2=2 if controllers == 2 else 0,
                     defense_scroll_x=-64, defense_scroll_y=179)
    return NHL94Observation2PEnv(base, human_args(mode=mode), controllers, 'PostPlay')


class HumanPlayContracts(unittest.TestCase):
    def test_installed_cli_and_legacy_parser_choose_two_controllers(self):
        with patch('nhl94_ai.evaluation.play.run') as run:
            main(['play', '--agent', 'classic-v1', '--mode', 'player_vs_model'])
        self.assertEqual(run.call_args.args[0].num_players, 2)
        self.assertEqual(run.call_args.args[0].nn, 'ClassicAIV1')
        self.assertEqual(parse_cmdline(['--mode=player_vs_model', '--num_players=1']).num_players, 2)
        for mode in ('model_vs_game', 'player_vs_game'):
            with patch('nhl94_ai.evaluation.play.run') as run:
                main(['play', '--mode', mode])
            self.assertEqual(run.call_args.args[0].num_players, 1)

    def test_unsupported_human_schemas_fail_before_emulator_creation(self):
        for action in ('DISCRETE', 'MULTI_DISCRETE', 'HOCKEY_INTENT_DPAD', 'TARGET_POSITION'):
            with self.subTest(action=action), self.assertRaises(ValueError):
                EnvironmentConfig.from_args(human_args(action_type=action))
        with self.assertRaisesRegex(ValueError, 'without self-play'):
            EnvironmentConfig.from_args(human_args(selfplay=True))

    def test_human_buttons_are_forwarded_independently_and_inputs_are_not_mutated(self):
        env = human_env()
        self.addCleanup(env.close)
        env.reset()
        action = np.zeros(24, dtype=np.int8)
        action[[Buttons.INPUT_LEFT, 12 + Buttons.INPUT_RIGHT, 12 + Buttons.INPUT_C]] = 1
        expected = action.copy()
        with patch.object(env.env, 'step', wraps=env.env.step) as step:
            env.step(action)
        np.testing.assert_array_equal(step.call_args.args[0], expected)
        np.testing.assert_array_equal(action, expected)
        np.testing.assert_array_equal(env.learner_action_state['last_env_action'], expected[:12])
        np.testing.assert_array_equal(env.opponent_action_state['last_env_action'], expected[12:])
        env.reset()
        self.assertFalse(env.learner_action_state['last_env_action'].any())
        self.assertFalse(env.opponent_action_state['last_env_action'].any())

    def test_only_external_match_env_uses_24_buttons(self):
        for controllers, mode, shape in (
            (2, 'player_vs_model', (24,)), (1, 'player_vs_model', (12,)),
            (2, 'model_vs_game', (12,)), (1, 'model_vs_game', (12,)),
        ):
            with self.subTest(controllers=controllers, mode=mode):
                env = human_env(controllers=controllers, mode=mode)
                self.addCleanup(env.close)
                self.assertEqual(env.action_space.shape, shape)
                observation, _ = env.reset()
                self.assertEqual(np.asarray(observation).shape, (env.NUM_PARAMS,))
                if not env.external_opponent:
                    with patch.object(env.env, 'step', wraps=env.env.step) as step:
                        env.step(np.ones(12, dtype=np.int8))
                    sent = step.call_args.args[0]
                    if controllers == 2:
                        self.assertFalse(sent[12:].any())
                    else:
                        self.assertEqual(sent.shape, (12,))

    def test_bad_combined_actions_fail_instead_of_making_an_idle_opponent(self):
        env = human_env()
        self.addCleanup(env.close)
        env.reset()
        for action in (np.zeros(12), np.zeros((2, 12)), np.full(24, 2), np.full(24, np.nan)):
            with self.subTest(shape=action.shape), patch.object(env.env, 'step') as step:
                with self.assertRaisesRegex(ValueError, '24 binary buttons'):
                    env.step(action)
                step.assert_not_called()

    def test_single_controller_or_swapped_team_save_is_rejected(self):
        env = human_env()
        self.addCleanup(env.close)
        for teams in ((1, 0), (1, 1), (2, 1)):
            env.env.info.update(defense_team1=teams[0], defense_team2=teams[1])
            with self.subTest(teams=teams), self.assertRaisesRegex(ValueError, 'two-controller save'):
                env.reset()

    def test_selfplay_still_uses_its_opponent_not_external_buttons(self):
        env = human_env(mode='model_vs_game')
        self.addCleanup(env.close)
        env.reset()
        env.selfplay_enabled = True
        env.opponent_input_overide = lambda _: None
        opponent = np.zeros(12, dtype=np.int8)
        opponent[Buttons.INPUT_DOWN] = 1
        with patch.object(env, '_compute_opponent_action', return_value=opponent), \
                patch.object(env.env, 'step', wraps=env.env.step) as step:
            env.step(np.zeros(12, dtype=np.int8))
        np.testing.assert_array_equal(step.call_args.args[0][12:], opponent)


@unittest.skipUnless(importlib.util.find_spec('pygame'), 'optional display extra is not installed')
class HumanDisplayContracts(unittest.TestCase):
    def setUp(self):
        variables = patch.dict(os.environ, {'SDL_VIDEODRIVER': 'dummy', 'SDL_AUDIODRIVER': 'dummy'})
        variables.start()
        self.addCleanup(variables.stop)
        import pygame
        pygame.init()
        self.addCleanup(pygame.quit)

    def display(self, *, truncate=False):
        from stable_baselines3.common.vec_env import DummyVecEnv
        from nhl94_ai.ui.debug import NHL94DebugDisplay
        inner = human_env(truncate=truncate)
        vector = DummyVecEnv([lambda: inner])
        self.addCleanup(vector.close)
        vector.reset()
        return NHL94DebugDisplay(vector, human_args(), 0, 'ClassicAIV1', []), inner

    def test_keyboard_goes_to_p2_each_frame_and_never_overwrites_ai(self):
        import pygame
        display, inner = self.display()
        ai = np.zeros(12, dtype=np.int8)
        ai[Buttons.INPUT_LEFT] = 1
        for pressed in ({pygame.K_RIGHT: True, pygame.K_c: True}, {}, {pygame.K_x: True}):
            with self.subTest(pressed=pressed), \
                    patch('pygame.key.get_pressed', return_value=defaultdict(bool, pressed)), \
                    patch.object(inner.env, 'step', wraps=inner.env.step) as step:
                display.step([ai])
                sent = step.call_args.args[0]
                np.testing.assert_array_equal(sent[:12], ai)
                self.assertEqual(sent[12 + Buttons.INPUT_RIGHT], bool(pressed.get(pygame.K_RIGHT)))
                self.assertEqual(sent[12 + Buttons.INPUT_C], bool(pressed.get(pygame.K_c)))
                self.assertEqual(sent[12 + Buttons.INPUT_B], bool(pressed.get(pygame.K_x)))
        display.human_control = False
        with patch.object(display, 'get_human_input') as human, \
                patch.object(inner.env, 'step', wraps=inner.env.step) as step:
            display.step([ai])
            human.assert_not_called()
            np.testing.assert_array_equal(step.call_args.args[0][:12], ai)
            self.assertFalse(step.call_args.args[0][12:].any())

    def test_invalid_ai_shape_is_not_silently_replaced(self):
        display, _ = self.display()
        for action in (None, np.zeros(24), np.ones((2, 12)), np.full(12, 2)):
            with self.subTest(action=action), patch.object(display.env, 'step') as step:
                with self.assertRaisesRegex(ValueError, 'one 12-button AI action'):
                    display.step(action)
                step.assert_not_called()

    def test_green_target_remains_visible_while_human_controls_opponent(self):
        import pygame
        from nhl94_ai.agents.defense import DefenseController
        from nhl94_ai.ui.targets import TARGET_GREEN
        from tests.unit.test_classic_defense import defense_state
        for terminal in (False, True):
            with self.subTest(terminal=terminal):
                display, _ = self.display(truncate=terminal)
                planner = DefenseController()
                planner.step(defense_state())
                display.set_ai_sys_info(SimpleNamespace(last_diagnostics={'classic_defense': planner.diagnostics}))
                with patch.object(display, 'get_human_input', return_value=[0] * 12):
                    display.step([np.zeros(12, dtype=np.int8)])
                pixels = pygame.surfarray.array3d(display.game_surf)
                self.assertEqual(bool(np.any(np.all(pixels == TARGET_GREEN, axis=2))), not terminal)

    def test_close_releases_workers_in_button_play(self):
        display, _ = self.display()
        with patch.object(display.env, 'close', wraps=display.env.close) as close:
            display.close()
            close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
