"""Target action schemas, PPO checkpoints and display adapters."""
import copy
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from stable_baselines3 import PPO

from nhl94_ai.agents.base import AgentInput, FrameRepeatAgent, LearnedAgent
from nhl94_ai.artifacts import load_policy, run_metadata, save_checkpoint, validate_schema
from nhl94_ai.config import EnvironmentConfig
from nhl94_ai.env.factory import build_single_nhl94_env, resolve_backend_action_type
from nhl94_ai.env.target_control import CONTROLLER_FIELDS, target_schema
from nhl94_ai.tasks.registry import TASKS, get_task, register_task
from tests.unit.test_target_control import target_env


def target_args(**overrides):
    return SimpleNamespace(**{
        'env': 'NHL94-Genesis-v0', 'rf': 'DefenseZone', 'nn': 'MlpPolicy',
        'action_type': 'TARGET_POSITION', 'num_players': 1, 'alg': 'ppo2',
        **overrides,
    })


class TargetInterfaceTests(unittest.TestCase):
    def test_entry_points_accept_target_mode_without_changing_defaults(self):
        from nhl94_ai.training import live, rl
        from nhl94_ai.evaluation import play
        for module in (live, rl, play):
            parser = module.build_parser()
            self.assertEqual(parser.parse_args([]).action_type, 'FILTERED')
            self.assertEqual(parser.parse_args(['--action_type', 'TARGET_POSITION']).action_type, 'TARGET_POSITION')
        self.assertEqual(resolve_backend_action_type(target_args(), 1), 'FILTERED')

    def test_unsupported_combinations_fail_before_creating_emulator(self):
        combinations = (
            {'env': 'NHL942on2-Genesis-v0'}, {'num_players': 2}, {'selfplay': True},
            {'nn': 'GRUMlpPolicy'}, {'nn': 'ClassicAIV1'}, {'rf': 'PostPlay'},
            {'rf': 'SelfPlayDefenseFinetune'}, {'alg': 'es'},
            {'mode': 'player_vs_game'}, {'mode': 'model_vs_model'}, {'model_2': 'second.zip'},
        )
        for values in combinations:
            with self.subTest(values=values), patch('nhl94_ai.env.factory.make_retro') as make:
                with self.assertRaises(ValueError):
                    build_single_nhl94_env(target_args(**values), {})
                make.assert_not_called()
        for interval in (0, -1, 0.5, True):
            with self.subTest(interval=interval), patch('nhl94_ai.env.factory.make_retro') as make:
                with self.assertRaisesRegex(ValueError, 'positive integer'):
                    build_single_nhl94_env(target_args(), {'frame_skip': interval})
                make.assert_not_called()

    def test_another_task_can_opt_in_without_copying_controller(self):
        register_task('TargetTestTask', replace(get_task('DefenseZone'), target_control='defense'))
        self.addCleanup(TASKS.pop, 'TargetTestTask')
        config = EnvironmentConfig.from_args(target_args(rf='TargetTestTask'))
        self.assertEqual(config.action_type, 'TARGET_POSITION')
        schema = run_metadata(target_args(rf='TargetTestTask'), {})['schema']
        self.assertEqual(schema['target_controller']['profile'], 'defense')
        self.assertEqual(schema['target_controller']['observation_fields'], list(CONTROLLER_FIELDS))

    def test_controller_schema_mismatches_fail_but_legacy_schema_is_unchanged(self):
        legacy = run_metadata(target_args(action_type='FILTERED'), {})
        self.assertNotIn('target_controller', legacy['schema'])
        validate_schema(legacy, copy.deepcopy(legacy))
        actual = run_metadata(target_args(), {})
        for key in ('version', 'boost_cooldown', 'scale_x'):
            expected = copy.deepcopy(actual)
            expected['schema']['target_controller']['settings'][key] += 1
            with self.assertRaisesRegex(ValueError, 'target_controller'):
                validate_schema(actual, expected)
        with self.assertRaisesRegex(ValueError, 'action_type'):
            validate_schema(actual, legacy)

    def test_frame_cadence_preserves_floats_and_resets_immediately(self):
        calls = []

        class Model:
            def predict(self, obs, **kwargs):
                calls.append(obs)
                return np.array([0.125 * len(calls), -0.25], dtype=np.float32), None

        agent = FrameRepeatAgent(LearnedAgent(Model(), 'TARGET_POSITION'), 4)
        outputs = [agent.act(AgentInput(None, index)).action for index in range(9)]
        self.assertEqual(calls, [0, 4, 8])
        np.testing.assert_array_equal(outputs[3], [0.125, -0.25])
        np.testing.assert_array_equal(outputs[4], [0.25, -0.25])
        agent.reset()
        agent.act(AgentInput(None, 20))
        self.assertEqual(calls[-1], 20)
        with self.assertRaises(ValueError):
            FrameRepeatAgent(agent, 0)

    def test_small_ppo_update_save_reload_and_legacy_rejection(self):
        env = target_env()
        self.addCleanup(env.close)
        params = {'frame_skip': 4, 'clip_reward': False}
        model = PPO('MlpPolicy', env, n_steps=8, batch_size=4, n_epochs=1, seed=0,
                    device='cpu', policy_kwargs={'net_arch': {'pi': [16], 'vf': [16]}})
        model.learn(total_timesteps=16)
        observation, _ = env.reset(seed=7)
        action, _ = model.predict(observation, deterministic=True)
        self.assertEqual(action.shape, (2,))
        self.assertTrue(env.action_space.contains(action))
        with tempfile.TemporaryDirectory() as root:
            path = save_checkpoint(model, str(Path(root) / 'target'), target_args(), params)
            metadata = run_metadata(target_args(), params)
            restored = load_policy(path, env=env, expected=metadata, device='cpu')
            np.testing.assert_array_equal(action, restored.predict(observation, deterministic=True)[0])
            env.env.info['p1_control_slot'] = -1
            unselected_observation, _, _, _, _ = env.step(action)
            self.assertTrue(np.all(np.isfinite(restored.predict(unselected_observation, deterministic=False)[0])))
            manifest = Path(path + '.json')
            original = json.loads(manifest.read_text(encoding='utf-8'))
            old_mapping = copy.deepcopy(original)
            old_mapping['schema']['target_controller'] = target_schema('defense')
            manifest.write_text(json.dumps(old_mapping), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'target_controller'):
                load_policy(path, env=env, expected=metadata)
            broken = copy.deepcopy(original)
            broken['schema']['target_controller']['settings']['version'] += 1
            manifest.write_text(json.dumps(broken), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'target controller'):
                load_policy(path, env=env, expected=metadata)
            manifest.unlink()
            with self.assertRaisesRegex(ValueError, 'requires checkpoint metadata'):
                load_policy(path, env=env, expected=metadata)

    def test_scripted_and_runtime_export_paths_reject_targets(self):
        from nhl94_ai.agents.registry import create_scripted
        from nhl94_ai.export import export_onnx
        with self.assertRaisesRegex(ValueError, 'not TARGET_POSITION'):
            create_scripted('classic-v1', target_args())
        model = SimpleNamespace(_nhl94_metadata=run_metadata(target_args(), {}))
        with patch('nhl94_ai.export.load_policy', return_value=model):
            with self.assertRaisesRegex(ValueError, 'matching runtime controller'):
                export_onnx(SimpleNamespace(src='target.zip'))


@unittest.skipUnless(importlib.util.find_spec('pygame'), 'optional display extra is not installed')
class TargetDisplayTests(unittest.TestCase):
    def test_debug_adapter_preserves_targets_and_rejects_buttons(self):
        from nhl94_ai.ui.debug import NHL94DebugDisplay
        display = NHL94DebugDisplay.__new__(NHL94DebugDisplay)
        display.args = target_args()
        for value in ([0.125, -0.25], [[0.125, -0.25]], np.array([[0.125, -0.25]])):
            self.assertEqual(display._normalize_env_action(value), [[0.125, -0.25]])
        for value in (None, np.zeros(12), [np.nan, 0]):
            with self.assertRaises(ValueError):
                display._normalize_env_action(value)
        display.args.action_type = 'FILTERED'
        self.assertEqual(display._normalize_env_action(np.ones(12)), [[1] * 12])

    def test_shared_overlay_renders_target_and_player_markers(self):
        with patch.dict(os.environ, {'SDL_VIDEODRIVER': 'dummy', 'SDL_AUDIODRIVER': 'dummy'}):
            import pygame
            from nhl94_ai.ui.targets import draw_target_rink, TARGET_GREEN
            pygame.init()
            self.addCleanup(pygame.quit)
            env = target_env()
            self.addCleanup(env.close)
            env.reset(seed=0)
            _, _, _, _, info = env.step([0.5, 0])
            surface = pygame.Surface((300, 600))
            draw_target_rink(surface, surface.get_rect(), env.game_state, info['target_control'],
                             pygame.font.SysFont('Arial', 14))
            pixels = pygame.surfarray.array3d(surface)
            self.assertTrue(np.any(np.all(pixels == TARGET_GREEN, axis=2)))
            env.env.info['p1_control_slot'] = -1
            _, _, _, _, info = env.step([0.5, 0])
            draw_target_rink(surface, surface.get_rect(), env.game_state, info['target_control'],
                             pygame.font.SysFont('Arial', 14))
            from nhl94_ai.ui.debug import NHL94DebugDisplay
            display = NHL94DebugDisplay.__new__(NHL94DebugDisplay)
            display.show_distances = True
            display._draw_controlled_distances(env.game_state.team1, (255, 0, 0))


if __name__ == '__main__':
    unittest.main()
