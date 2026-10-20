"""Temporal integration checks for finite-window offense policies and PPO."""
import unittest

import numpy as np
import torch

from nhl94_ai.agents.learned_offense import N_ACTIONS
from stable_baselines3 import PPO
from nhl94_ai.models.offense import OffensePolicy
from tests.unit.test_offense_decisions import TickEnv


class OffenseTemporalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.model = PPO(OffensePolicy, TickEnv(), n_steps=4, batch_size=2, seed=81, device='cpu',
                         policy_kwargs=dict(features_extractor_kwargs=dict(
                             features_dim=128, hidden_dim=64, spatial_hidden_dim=64, gru_layers=1),
                             net_arch=dict(pi=[64], vf=[64])))
        self.policy = self.model.policy
        self.extractor = self.policy.features_extractor
        self.observations = torch.randn(3, 16, 331) * 0.2
        self.observations[..., -N_ACTIONS:] = 0
        self.observations[..., -N_ACTIONS:-N_ACTIONS + 9] = 1

    def test_compact_policy_has_the_requested_recurrence(self):
        gru = self.extractor.temporal_model
        self.assertIsInstance(gru, torch.nn.GRU)
        self.assertEqual((gru.input_size, gru.hidden_size, gru.num_layers), (64, 64, 1))
        self.assertTrue(gru.batch_first)
        self.assertFalse(gru.bidirectional)
        self.assertEqual(sum(p.numel() for p in self.policy.parameters()), 110868)

    def test_window_matches_chronological_one_step_unrolling(self):
        # A different execution route catches batch/time transposition and an
        # incorrect temporal summary without reimplementing PyTorch's gates.
        projected = self.extractor.temporal_input_proj(self.observations)
        hidden = None
        for frame in range(16):
            _, hidden = self.extractor.temporal_model(projected[:, frame:frame + 1], hidden)
        expected = self.extractor.combiner(torch.cat((
            self.extractor.spatial_encoder(self.observations[:, -1]),
            self.extractor.temporal_proj(hidden[-1])), dim=-1))
        torch.testing.assert_close(self.extractor(self.observations), expected, atol=1e-6, rtol=1e-5)

    def test_same_current_frame_retains_ordered_history_information(self):
        reordered = self.observations.clone()
        reordered[:, :-1] = reordered[:, :-1].flip(1)
        torch.testing.assert_close(reordered[:, -1], self.observations[:, -1])
        original = self.policy.get_distribution(self.observations).distribution.logits
        changed = self.policy.get_distribution(reordered).distribution.logits
        self.assertGreater(float((original[:, :9] - changed[:, :9]).abs().max().detach()), 1e-7)

    def test_actor_and_critic_backpropagate_into_old_frames_and_gru(self):
        for branch in ('actor', 'critic'):
            with self.subTest(branch=branch):
                obs = self.observations.clone().requires_grad_()
                self.policy.zero_grad(set_to_none=True)
                values, log_probability, _ = self.policy.evaluate_actions(obs, torch.tensor([0, 2, 4]))
                loss = -log_probability.mean() if branch == 'actor' else (values - 1).square().mean()
                loss.backward()
                for frame in (0, 7, 14):
                    gradient = obs.grad[:, frame, :-N_ACTIONS]
                    self.assertTrue(torch.isfinite(gradient).all())
                    self.assertGreater(float(gradient.abs().sum()), 0)
                for name, parameter in self.extractor.temporal_model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                    self.assertGreater(float(parameter.grad.abs().sum()), 0, name)

    def test_batch_shuffle_and_other_calls_do_not_change_a_window(self):
        self.policy.set_training_mode(False)
        actions, values, old_log_probability = self.policy(self.observations, deterministic=True)
        self.policy(torch.flip(self.observations, (1,)))
        permutation = torch.tensor([2, 0, 1])
        self.policy.set_training_mode(True)
        new_values, new_log_probability, _ = self.policy.evaluate_actions(
            self.observations[permutation], actions[permutation])
        torch.testing.assert_close(values[permutation], new_values)
        torch.testing.assert_close(old_log_probability[permutation], new_log_probability)
        for row in range(3):
            value, log_probability, _ = self.policy.evaluate_actions(
                self.observations[row:row + 1], actions[row:row + 1])
            torch.testing.assert_close(value, values[row:row + 1], atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(log_probability, old_log_probability[row:row + 1])

    def test_rollout_buffer_shuffle_preserves_complete_ordered_windows(self):
        buffer = self.model.rollout_buffer
        originals = []
        for step in range(buffer.buffer_size):
            obs = self.observations[:1].numpy().copy()
            obs[0, :, 0] = step * 16 + np.arange(16)
            originals.append(obs[0])
            buffer.add(obs, np.zeros((1, 1)), np.zeros(1), np.zeros(1),
                       torch.zeros(1), torch.zeros(1))
        expected = {tuple(obs[:, 0]): obs for obs in originals}
        seen = set()
        for batch in buffer.get(batch_size=2):
            for obs in batch.observations.numpy():
                key = tuple(obs[:, 0])
                np.testing.assert_array_equal(obs, expected[key])
                seen.add(key)
        self.assertEqual(seen, set(expected))
