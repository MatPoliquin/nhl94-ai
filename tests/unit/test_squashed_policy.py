"""Bounded PPO actions, transformed entropy, and checkpoint compatibility."""
import math
from pathlib import Path
import tempfile
import unittest

from gymnasium import spaces
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure

from nhl94_ai.artifacts import load_policy, read_metadata, run_metadata, save_checkpoint
from nhl94_ai.models.factory import build_MlpPolicy, init_model
from nhl94_ai.models.mlp import SquashedGaussianDistribution, SquashedMlpPolicy
from tests.unit.test_target_control import target_env
from tests.unit.test_target_interfaces import target_args


class SquashedDistributionTests(unittest.TestCase):
    def setUp(self):
        rng = torch.random.fork_rng(devices=[])
        rng.__enter__()
        self.addCleanup(rng.__exit__, None, None, None)
        torch.manual_seed(7)

    def test_samples_modes_and_log_probabilities_follow_tanh_transform(self):
        mean = torch.tensor([[0.2, -0.4], [-0.3, 0.5]], dtype=torch.float64)
        log_std = torch.full((2,), math.log(0.5), dtype=torch.float64)
        distribution = SquashedGaussianDistribution(2).proba_distribution(mean, log_std)
        raw = torch.tensor([[-0.7, 0.3], [0.8, -1.2]], dtype=torch.float64)
        actions = raw.tanh()
        reference = torch.distributions.TransformedDistribution(
            torch.distributions.Normal(mean, log_std.exp()), [torch.distributions.TanhTransform()],
        )
        torch.testing.assert_close(distribution.mode(), mean.tanh())
        torch.testing.assert_close(distribution.log_prob(actions), reference.log_prob(actions).sum(dim=1),
                                   rtol=1e-5, atol=1e-5)
        sampled = distribution.sample()
        self.assertTrue(torch.all(sampled.abs() < 1))
        torch.testing.assert_close(sampled, distribution.gaussian_actions.tanh())

    def test_entropy_matches_bounded_density_not_raw_gaussian(self):
        mean = torch.zeros((16384, 2))
        log_std = torch.full((2,), math.log(0.5))
        distribution = SquashedGaussianDistribution(2).proba_distribution(mean, log_std)
        entropy = distribution.entropy()
        self.assertEqual(entropy.shape, (16384,))
        reference = torch.distributions.TransformedDistribution(
            torch.distributions.Normal(mean, log_std.exp()), [torch.distributions.TanhTransform(cache_size=1)],
        )
        reference_entropy = -reference.log_prob(reference.rsample()).sum(dim=1).mean()
        self.assertAlmostEqual(entropy.mean().item(), reference_entropy.item(), delta=0.03)
        self.assertLess(entropy.mean().item(), 2 * math.log(2))
        self.assertLess(entropy.mean().item(), distribution.distribution.entropy().sum(dim=1).mean().item())

    def test_entropy_gradient_opposes_runaway_noise_and_saturated_means(self):
        for mean_value, std in ((0.0, 0.1), (0.0, 22.7), (20.0, 0.5)):
            with self.subTest(mean=mean_value, std=std):
                mean = torch.full((16384, 2), mean_value, requires_grad=True)
                log_std = torch.full((2,), math.log(std), requires_grad=True)
                distribution = SquashedGaussianDistribution(2).proba_distribution(mean, log_std)
                entropy = distribution.entropy().mean()
                self.assertTrue(torch.isfinite(entropy))
                entropy.backward()
                self.assertTrue(torch.all(torch.isfinite(log_std.grad)))
                self.assertTrue(torch.all(torch.isfinite(mean.grad)))
                if std == 22.7:
                    self.assertTrue(torch.all(log_std.grad < 0))
                elif std == 0.1:
                    self.assertTrue(torch.all(log_std.grad > 0))
                else:
                    self.assertLess(mean.grad.sum().item(), 0)


class SquashedPolicyTests(unittest.TestCase):
    def setUp(self):
        self.env = target_env()
        self.addCleanup(self.env.close)
        self.params = {
            'net_arch': {'pi': [16], 'vf': [16]}, 'squash_output': True,
            'log_std_init': math.log(0.5), 'frame_skip': 4, 'clip_reward': False,
            'ent_coef': 0.01, 'n_steps': 8, 'batch_size': 4, 'n_epochs': 2,
        }

    def make_policy(self):
        policy_class, kwargs = build_MlpPolicy(256, self.params)
        return policy_class(self.env.observation_space, self.env.action_space, lambda _: 0.00025, **kwargs)

    def test_policy_and_replay_use_the_same_bounded_coordinates(self):
        policy = self.make_policy()
        self.assertTrue(policy.squash_output)
        torch.testing.assert_close(policy.log_std.exp(), torch.full((2,), 0.5))
        with torch.no_grad():
            policy.action_net.weight.zero_()
            policy.action_net.bias.copy_(torch.tensor([2.0, -2.0]))
        observation, _ = self.env.reset(seed=0)
        tensor = policy.obs_to_tensor(observation)[0]
        actions, _, log_prob = policy(tensor, deterministic=True)
        predicted, _ = policy.predict(observation, deterministic=True)
        np.testing.assert_allclose(predicted, np.tanh([2.0, -2.0]), rtol=1e-6)
        np.testing.assert_allclose(actions.detach().numpy()[0], predicted, rtol=1e-6)
        _, evaluated, entropy = policy.evaluate_actions(tensor, actions.detach())
        torch.testing.assert_close(log_prob, evaluated)
        self.assertTrue(torch.all(torch.isfinite(entropy)))
        for _ in range(16):
            action, _ = policy.predict(observation, deterministic=False)
            self.assertTrue(self.env.action_space.contains(action))
            observation, _, _, _, _ = self.env.step(action)

    def test_log_probabilities_remain_finite_and_consistent_near_saturation(self):
        policy = self.make_policy()
        with torch.no_grad():
            policy.log_std.fill_(math.log(22.7))
            policy.action_net.weight.zero_()
            policy.action_net.bias.copy_(torch.tensor([20.0, -20.0]))
        observations = torch.zeros((256, 322))
        with torch.no_grad():
            actions, _, rollout_log_prob = policy(observations)
        _, update_log_prob, entropy = policy.evaluate_actions(observations, actions)
        self.assertTrue(torch.all(actions.abs() <= 1))
        self.assertTrue(torch.all(torch.isfinite(update_log_prob)))
        self.assertTrue(torch.all(torch.isfinite(entropy)))
        torch.testing.assert_close(rollout_log_prob, update_log_prob)
        torch.testing.assert_close((update_log_prob - rollout_log_prob).exp(), torch.ones(256))
        (-entropy.mean()).backward()
        self.assertTrue(torch.all(torch.isfinite(policy.log_std.grad)))

    def test_real_ppo_update_checkpoint_and_policy_round_trip(self):
        args = target_args(nnsize=256, policy_seed=0, alg_verbose=False)
        model = init_model(None, '', 'ppo2', args, self.env, configure(format_strings=[]), self.params)
        self.assertIsInstance(model.policy, SquashedMlpPolicy)
        self.assertEqual(model.ent_coef, 0.01)
        self.assertTrue(model.policy.squash_output)
        torch.testing.assert_close(model.policy.log_std.exp(), torch.full((2,), 0.5, device=model.device))
        before = model.policy.log_std.detach().clone()
        model.learn(total_timesteps=32)
        self.assertTrue(all(torch.all(torch.isfinite(parameter)) for parameter in model.policy.parameters()))
        self.assertFalse(torch.equal(before, model.policy.log_std))
        self.assertTrue(np.all(np.abs(model.rollout_buffer.actions) <= 1))
        self.assertTrue(np.all(np.isfinite(model.rollout_buffer.log_probs)))
        observation, _ = self.env.reset(seed=3)
        expected = model.predict(observation, deterministic=True)[0]
        with tempfile.TemporaryDirectory() as directory:
            path = save_checkpoint(model, str(Path(directory) / 'squashed'))
            metadata = read_metadata(path)
            self.assertTrue(metadata['hyperparams']['squash_output'])
            self.assertAlmostEqual(metadata['hyperparams']['log_std_init'], math.log(0.5))
            restored = load_policy(path, env=self.env, expected=run_metadata(args, self.params), device='cpu')
            self.assertIsInstance(restored.policy, SquashedMlpPolicy)
            np.testing.assert_allclose(restored.predict(observation, deterministic=True)[0], expected, atol=1e-6)
            torch.testing.assert_close(model.policy.log_std.cpu(), restored.policy.log_std)
            policy_path = str(Path(directory) / 'policy.pt')
            restored.policy.save(policy_path)
            policy = SquashedMlpPolicy.load(policy_path, device='cpu')
            np.testing.assert_array_equal(policy.predict(observation, deterministic=True)[0],
                                          restored.predict(observation, deterministic=True)[0])
            restored.learn(total_timesteps=8)

    def test_old_target_checkpoints_keep_their_original_clipped_distribution(self):
        old_params = {'frame_skip': 4}
        old = PPO('MlpPolicy', self.env, n_steps=8, batch_size=4, device='cpu',
                  policy_kwargs={'net_arch': {'pi': [16], 'vf': [16]}})
        with torch.no_grad():
            old.policy.action_net.weight.zero_()
            old.policy.action_net.bias.copy_(torch.tensor([2.0, -2.0]))
        with tempfile.TemporaryDirectory() as directory:
            path = save_checkpoint(old, str(Path(directory) / 'old'), target_args(), old_params)
            restored = load_policy(path, env=self.env, expected=run_metadata(target_args(), self.params), device='cpu')
            self.assertFalse(restored.policy.squash_output)
            observation, _ = self.env.reset(seed=0)
            np.testing.assert_array_equal(restored.predict(observation, deterministic=True)[0], [1.0, -1.0])
        policy_name, kwargs = build_MlpPolicy(128, {})
        self.assertEqual(policy_name, 'MlpPolicy')
        self.assertNotIn('log_std_init', kwargs)

    def test_invalid_settings_and_non_target_spaces_fail_explicitly(self):
        with self.assertRaisesRegex(TypeError, 'JSON boolean'):
            build_MlpPolicy(256, {'squash_output': 'true'})
        for value in (float('nan'), float('inf'), -float('inf')):
            with self.subTest(log_std_init=value):
                with self.assertRaisesRegex(ValueError, 'finite'):
                    build_MlpPolicy(256, {'log_std_init': value})
        for action_space in (spaces.MultiBinary(12), spaces.Box(-2, 2, (2,)),
                             spaces.Box(-1, 1, (3,))):
            with self.subTest(action_space=action_space):
                with self.assertRaisesRegex(ValueError, 'target action space'):
                    SquashedMlpPolicy(self.env.observation_space, action_space, lambda _: 0.00025)
        with self.assertRaisesRegex(ValueError, 'not gSDE'):
            SquashedMlpPolicy(self.env.observation_space, self.env.action_space, lambda _: 0.00025, use_sde=True)


if __name__ == '__main__':
    unittest.main()
