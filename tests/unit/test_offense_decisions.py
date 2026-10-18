"""Variable-duration credit assignment and offensive-attempt boundaries."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

from nhl94_ai.agents.learned_offense import N_ACTIONS, OBSERVATION_FIELDS
from nhl94_ai.env.offense_decisions import DecisionEnv
from nhl94_ai.models.offense import OffensePolicy, load_offense_model, save_offense_model
from nhl94_ai.training.decision_ppo import DecisionPPO, DecisionRolloutBuffer
from tests.unit.test_classic_offense import live_one_timer_state
from tests.unit import test_learned_offense as offense_tests


class TickEnv(gym.Env):
    """Decisions at ticks zero and three; a delayed goal at tick five."""
    def __init__(self, cutoff=5, truncated=False):
        super().__init__()
        self.observation_space = gym.spaces.Box(-1, 1, (16, len(OBSERVATION_FIELDS)), np.float32)
        self.action_space = gym.spaces.Discrete(N_ACTIONS)
        self.cutoff, self.truncated, self.ticks = cutoff, truncated, 0
        self.actions = []

    def observation(self):
        obs = np.zeros(self.observation_space.shape, np.float32)
        obs[-1, -N_ACTIONS:] = int(self.ticks in (0, 3))
        obs[-1, -N_ACTIONS] = 1
        return obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.ticks = 0
        return self.observation(), {}

    def step(self, action):
        active = self.ticks in (0, 3)
        self.actions.append(int(action))
        self.ticks += 1
        done = self.ticks == self.cutoff
        return (self.observation(), float(done and not self.truncated),
                done and not self.truncated, done and self.truncated,
                dict(native_frames=4, actor_active=active, command_used=active))

    def render(self):
        return None


class DecisionLearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def model(self, env, value=0):
        model = DecisionPPO(OffensePolicy, env, n_steps=2, batch_size=2, n_epochs=1,
                            gamma=0.9, gae_lambda=1, learning_rate=0, device='cpu')
        with torch.no_grad():
            model.policy.value_net.weight.zero_()
            model.policy.value_net.bias.fill_(value)
        return model

    def test_wrapper_keeps_pass_outcome_and_updates_through_inactive_ticks(self):
        base = TickEnv()
        env = DecisionEnv(base, gamma=0.9)
        env.reset()
        _, reward, end, cut, info = env.step(9)
        self.assertEqual(base.actions, [9, 0, 0])
        self.assertEqual(reward, 0)
        self.assertFalse(end or cut)
        self.assertEqual(info['elapsed_steps'], 3)
        self.assertEqual(info['native_frames'], 12)
        _, reward, end, _, info = env.step(17)
        self.assertAlmostEqual(reward, 0.9)
        self.assertTrue(end)
        self.assertEqual(info['elapsed_steps'], 2)
        self.assertEqual(info['undiscounted_reward'], 1)
        with self.assertRaises(RuntimeError):
            env.step(0)

    def test_decision_returns_match_uncompressed_discounted_goal(self):
        model = self.model(DecisionEnv(TickEnv(), gamma=0.9))
        model.learn(total_timesteps=2)
        np.testing.assert_allclose(model.rollout_buffer.returns.flatten(), [0.9**4, 0.9], atol=1e-6)
        np.testing.assert_array_equal(model.rollout_buffer.durations.flatten(), [3, 2])
        self.assertEqual(model.num_timesteps, 2)

    def test_time_limit_bootstraps_with_actual_duration(self):
        model = self.model(DecisionEnv(TickEnv(cutoff=2, truncated=True), gamma=0.9), value=2)
        model.learn(total_timesteps=2)
        np.testing.assert_allclose(model.rollout_buffer.rewards, 0.9**2 * 2, atol=1e-6)

    def test_terminal_goal_does_not_bootstrap(self):
        model = self.model(DecisionEnv(TickEnv(cutoff=2), gamma=0.9), value=2)
        model.learn(total_timesteps=2)
        np.testing.assert_allclose(model.rollout_buffer.rewards, 0.9, atol=1e-6)

    def test_each_vector_worker_uses_its_own_duration(self):
        env = TickEnv()
        buffer = DecisionRolloutBuffer(2, env.observation_space, env.action_space,
                                       n_envs=2, gamma=0.5, gae_lambda=1, device='cpu')
        obs = np.stack([env.observation()] * 2)
        for rewards, durations, starts in (([0, 0], [1, 3], [1, 1]), ([1, 1], [1, 1], [0, 0])):
            buffer.add(obs, np.zeros((2, 1)), np.asarray(rewards), np.asarray(starts),
                       torch.zeros(2), torch.zeros(2), durations=durations)
        buffer.compute_returns_and_advantage(torch.zeros(2), np.ones(2))
        np.testing.assert_allclose(buffer.returns, [[0.5, 0.125], [1, 1]])

    def test_checkpoint_preserves_duration_aware_algorithm(self):
        model = self.model(DecisionEnv(TickEnv(), gamma=0.9))
        with tempfile.TemporaryDirectory() as directory:
            path = save_offense_model(model, Path(directory) / 'decision', training_contract=dict(task='attack'))
            restored = load_offense_model(path)
            self.assertIsInstance(restored, DecisionPPO)
            self.assertIsInstance(restored.rollout_buffer, DecisionRolloutBuffer)
            with self.assertRaisesRegex(ValueError, 'Checkpoint task'):
                load_offense_model(path, expected_task='period')
            load_offense_model(path, expected_task='period', allow_task_transfer=True)


class AttackBoundaryTests(unittest.TestCase):
    def env(self, scores):
        env = offense_tests.OffenseTransitionTests().make_env(iter(scores))
        env.task, env.attack_start, env.attack_frames, env.snapshot = 'attack', 1, 600, 'test'
        state = live_one_timer_state()
        env.views = [state, state]
        return env

    def test_native_goal_ends_attempt_before_extra_actions(self):
        env = self.env([(1, 0)])
        env.views[0].engine.puck_owner = env.views[0].team2.skater_scnum_base()
        _, reward, end, cut, info = env.step(0)
        self.assertEqual((reward, end, cut), (1, True, False))
        self.assertEqual(info['native_frames'], 1)
        self.assertEqual(info['outcome'], 'goal')

    def test_turnover_and_retreat_end_before_classic_can_rescue_attempt(self):
        for outcome in ('turnover', 'retreat'):
            env = self.env([(0, 0)])
            state = env.views[0]
            if outcome == 'turnover':
                state.engine.puck_owner = state.team2.skater_scnum_base()
            else:
                state.team1.get_player_by_scnum(state.engine.puck_owner).y = 99
            _, reward, end, _, info = env.step(0)
            self.assertEqual((reward, end), (0, True))
            self.assertEqual(info['outcome'], outcome)
            self.assertEqual(info['native_frames'], 1)

    def test_loose_puck_keeps_delayed_goal_credit(self):
        env = self.env([(0, 0)] * 4 + [(1, 0)])
        env.views[0].engine.puck_owner = -1
        _, reward, end, _, _ = env.step(0)
        self.assertEqual((reward, end), (0, False))
        _, reward, end, _, info = env.step(0)
        self.assertEqual((reward, end), (1, True))
        self.assertEqual(info['outcome'], 'goal')

    def test_task_horizon_is_terminal_failure_not_value_bootstrap(self):
        env = self.env([(0, 0)] * 3)
        env.attack_frames = 3
        _, reward, end, cut, info = env.step(0)
        self.assertEqual((reward, end, cut), (0, True, False))
        self.assertEqual(info['outcome'], 'timeout')


if __name__ == '__main__':
    unittest.main()
