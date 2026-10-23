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
from tests.support.environments import target_args, target_env


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

    def publish_snapshot(self, steps, bias):
        self.model.num_timesteps = steps
        with torch.no_grad():
            self.model.policy.action_net.bias.fill_(bias)
        with self.state.lock:
            save_checkpoint(self.model, self.state.latest_model_path)
            self.state.latest_model_timesteps = steps
            self.state.model_version += 1


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
            self.publish_snapshot(8, 0.1)
            display._ensure_model_loaded()
            self.assertEqual(display.loaded_model_timesteps, 8)
            self.assertEqual(display.model_version_loaded, 1)
            display._ensure_model_loaded()
            self.assertEqual(load.call_count, 1)

            self.publish_snapshot(16, 0.2)
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
        self.publish_snapshot(8, 1.5)
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


if __name__ == '__main__':
    unittest.main()
