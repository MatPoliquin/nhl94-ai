"""Latest replay snapshots advance independently of best-model selection."""
from collections import deque
import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from nhl94_ai.artifacts import load_policy, run_metadata, save_checkpoint
from nhl94_ai.agents.base import AgentInput
from nhl94_ai.evaluation.metrics import LiveTeamTotals
from nhl94_ai.models.factory import build_MlpPolicy
from nhl94_ai.training.events import LiveTrainingState
from nhl94_ai.training.live import LiveTrainingCallback
from tests.unit.test_target_control import target_env
from tests.unit.test_target_interfaces import target_args


class LiveSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.params = {'frame_skip': 4}
        self.args = target_args(hyperparams_dict=self.params)
        self.env = target_env()
        self.addCleanup(self.env.close)
        self.model = PPO('MlpPolicy', self.env, n_steps=8, batch_size=4, device='cpu',
                         policy_kwargs={'net_arch': {'pi': [16], 'vf': [16]}})
        self.model._nhl94_metadata = run_metadata(self.args, self.params)
        self.state = LiveTrainingState(
            str(self.root / 'policy_best_live'), deque(),
            latest_model_path=str(self.root / 'policy_latest_live'),
        )
        self.callback = LiveTrainingCallback(self.state, None, self.args.env, 1, 8, 5)
        self.callback.init_callback(self.model)

    def evaluate(self, reward, steps, bias):
        self.callback.num_timesteps = self.model.num_timesteps = steps
        with torch.no_grad():
            self.model.policy.action_net.bias.fill_(bias)

        def save_locked(model, path):
            self.assertTrue(self.state.lock.locked())
            return save_checkpoint(model, path)

        result = reward, 0.0, LiveTeamTotals(), LiveTeamTotals()
        with patch('nhl94_ai.training.live.evaluate_policy_with_totals', return_value=result), \
                patch('nhl94_ai.training.live.save_checkpoint', side_effect=save_locked):
            self.callback._run_evaluation()

    def test_latest_advances_on_worse_and_equal_rewards_without_replacing_best(self):
        for reward, steps, bias in ((1.0, 8, 0.1), (0.559, 16, 0.2), (1.0, 24, 0.3)):
            self.evaluate(reward, steps, bias)
            self.assertEqual(self.state.model_version, steps // 8)
            self.assertEqual(self.state.latest_model_timesteps, steps)
            self.assertEqual(self.state.best_mean_reward, 1.0)
            latest = load_policy(self.state.latest_model_path, expected=self.model._nhl94_metadata, device='cpu')
            best = load_policy(self.state.best_model_path, expected=self.model._nhl94_metadata, device='cpu')
            self.assertEqual(latest.num_timesteps, steps)
            self.assertEqual(best.num_timesteps, 8)
            torch.testing.assert_close(latest.policy.action_net.bias, torch.full((2,), bias))
            torch.testing.assert_close(best.policy.action_net.bias, torch.full((2,), 0.1))
        self.assertEqual(list(self.state.reward_history), [(8, 1.0), (16, 0.559), (24, 1.0)])
        self.assertEqual(len(list(self.root.glob('*.zip'))), 2)
        self.assertEqual(len(list(self.root.glob('*.zip.json'))), 2)

    def test_better_reward_still_replaces_the_best_checkpoint(self):
        self.evaluate(0.4, 8, 0.1)
        self.evaluate(0.8, 16, 0.2)
        self.assertEqual(self.state.best_mean_reward, 0.8)
        self.assertEqual(load_policy(self.state.best_model_path, device='cpu').num_timesteps, 16)
        self.assertEqual(load_policy(self.state.latest_model_path, device='cpu').num_timesteps, 16)

    def test_headless_callbacks_do_not_write_a_viewer_snapshot(self):
        self.state.latest_model_path = None
        self.evaluate(0.8, 8, 0.1)
        self.evaluate(0.4, 16, 0.2)
        self.assertEqual([path.name for path in self.root.glob('*.zip')], ['policy_best_live.zip'])
        self.assertEqual(load_policy(self.state.best_model_path, device='cpu').num_timesteps, 8)

    def test_failed_snapshot_is_not_published_as_a_new_version(self):
        self.evaluate(1.0, 8, 0.1)
        self.callback.num_timesteps = 16
        with patch('nhl94_ai.training.live.evaluate_policy_with_totals',
                   return_value=(0.5, 0.0, LiveTeamTotals(), LiveTeamTotals())), \
                patch('nhl94_ai.training.live.save_checkpoint', side_effect=OSError('disk full')):
            with self.assertRaisesRegex(OSError, 'disk full'):
                self.callback._run_evaluation()
        self.assertEqual((self.state.model_version, self.state.latest_model_timesteps), (1, 8))

    @unittest.skipUnless(importlib.util.find_spec('pygame'), 'optional display extra is not installed')
    def test_viewer_reloads_latest_and_keeps_old_version_if_loading_fails(self):
        variables = patch.dict(os.environ, {'SDL_VIDEODRIVER': 'dummy', 'SDL_AUDIODRIVER': 'dummy'})
        variables.start()
        self.addCleanup(variables.stop)
        import pygame
        from nhl94_ai.ui.live import LiveTrainingDisplay
        pygame.init()
        self.addCleanup(pygame.quit)
        replay = target_env()
        replay.unwrapped.render_mode = 'rgb_array'
        vector = DummyVecEnv([lambda: replay])
        self.addCleanup(vector.close)
        display = LiveTrainingDisplay(self.args, self.state, fps=60, graph_width=260)
        display.env = vector

        def load_locked(path, **kwargs):
            self.assertTrue(self.state.lock.locked())
            self.assertEqual(path, self.state.latest_model_path + '.zip')
            return load_policy(path, device='cpu', **kwargs)

        with patch('nhl94_ai.ui.live.load_policy', side_effect=load_locked) as load, \
                patch('nhl94_ai.ui.live.com_print') as log:
            display._ensure_model_loaded()
            load.assert_not_called()
            self.evaluate(1.0, 8, 0.1)
            display._ensure_model_loaded()
            self.assertEqual(display.loaded_model_timesteps, 8)
            self.assertEqual(display.model_version_loaded, 1)
            display._ensure_model_loaded()
            self.assertEqual(load.call_count, 1)

            self.evaluate(0.5, 16, 0.2)
            sidecar = Path(self.state.latest_model_path + '.zip.json')
            metadata = json.loads(sidecar.read_text(encoding='utf-8'))
            incompatible = json.loads(sidecar.read_text(encoding='utf-8'))
            incompatible['schema']['target_controller']['settings']['version'] += 1
            sidecar.write_text(json.dumps(incompatible), encoding='utf-8')
            display._ensure_model_loaded()
            self.assertEqual(display.model_version_loaded, 1)
            self.assertEqual(display.loaded_model_timesteps, 8)
            torch.testing.assert_close(display.display_model.policy.action_net.bias, torch.full((2,), 0.1))
            self.assertIn('Failed to load latest model', log.call_args.args[0])

            sidecar.write_text(json.dumps(metadata), encoding='utf-8')
            display._ensure_model_loaded()
            self.assertEqual(display.model_version_loaded, 2)
            self.assertEqual(display.loaded_model_timesteps, 16)
            self.assertEqual(display.loaded_model_name, 'policy_latest_live.zip')
            torch.testing.assert_close(display.display_model.policy.action_net.bias, torch.full((2,), 0.2))
            self.assertEqual(display.target_agent.frames, 4)
            self.assertIsNone(display.target_diagnostics)
            self.assertEqual(display.target_frame_info, {})

    @unittest.skipUnless(importlib.util.find_spec('pygame'), 'optional display extra is not installed')
    def test_viewer_preserves_squashed_policy_predictions(self):
        from nhl94_ai.ui.live import LiveTrainingDisplay
        self.params.update(squash_output=True, log_std_init=math.log(0.5),
                           net_arch={'pi': [16], 'vf': [16]})
        policy, kwargs = build_MlpPolicy(256, self.params)
        self.model = PPO(policy, self.env, n_steps=8, batch_size=4, device='cpu', policy_kwargs=kwargs)
        self.model._nhl94_metadata = run_metadata(self.args, self.params)
        with torch.no_grad():
            self.model.policy.action_net.weight.zero_()
        self.callback.init_callback(self.model)
        self.evaluate(0.5, 8, 1.5)
        replay = target_env()
        vector = DummyVecEnv([lambda: replay])
        self.addCleanup(vector.close)
        display = LiveTrainingDisplay(self.args, self.state, fps=60, graph_width=260)
        display.env = vector

        def load_cpu(path, **load_kwargs):
            return load_policy(path, device='cpu', **load_kwargs)

        with patch('nhl94_ai.ui.live.load_policy', side_effect=load_cpu), \
                patch('nhl94_ai.ui.live.com_print'):
            display._ensure_model_loaded()
        self.assertTrue(display.display_model.policy.squash_output)
        self.assertEqual(display.model_version_loaded, 1)
        action = display.target_agent.act(AgentInput(replay.game_state, display.obs)).action
        torch.testing.assert_close(torch.as_tensor(action).reshape(2),
                                   torch.full((2,), math.tanh(1.5)))
        self.assertEqual(display.target_agent.frames, 4)

    @unittest.skipUnless(importlib.util.find_spec('pygame'), 'optional display extra is not installed')
    def test_session_configures_latest_snapshots_only_for_visible_runs(self):
        from nhl94_ai.training.live import parse_cmdline, run_training_session
        for headless in (False, True):
            with self.subTest(headless=headless):
                args = parse_cmdline(['--env=NHL94-Genesis-v0', '--nn=MlpPolicy', '--rf=DefenseZone',
                                      '--action_type=TARGET_POSITION', '--state=PenguinsVsSenators.DefenseZone'])
                args.hyperparams_dict = self.params
                args.headless = headless
                trainer = SimpleNamespace(
                    best_model_savepath=self.state.best_model_path,
                    latest_model_savepath=self.state.latest_model_path,
                    model_savepath=str(self.root / 'policy'), output_fullpath=str(self.root),
                    build_callback=Mock(), train=Mock(), close=Mock(),
                )
                with patch('nhl94_ai.training.live.LiveTrainer', return_value=trainer), \
                        patch('nhl94_ai.ui.live.LiveTrainingDisplay') as display, \
                        patch('nhl94_ai.training.live.com_print'):
                    display.return_value.ident = None
                    result = run_training_session(args, logger=Mock())
                shared = trainer.build_callback.call_args.args[0]
                self.assertEqual(shared.latest_model_path, None if headless else self.state.latest_model_path)
                self.assertEqual(shared.best_model_path, self.state.best_model_path)
                self.assertEqual(display.call_count, 0 if headless else 1)
                self.assertEqual(result.best_model_path, self.state.best_model_path + '.zip')
                self.assertEqual(result.final_model_path, str(self.root / 'policy.zip'))
                trainer.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
