"""Variable-duration credit assignment and offensive-attempt boundaries."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from stable_baselines3.common.callbacks import BaseCallback

from nhl94_ai.env.offense_decisions import DecisionEnv
from nhl94_ai.models.offense import OffensePolicy, load_offense_model, save_offense_model
from nhl94_ai.models.decision_ppo import DecisionPPO, DecisionRolloutBuffer
from tests.support.learned_offense import TickEnv, make_offense_env
from tests.support.offense import live_one_timer_state


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


    def test_checkpoint_preserves_legacy_algorithm_and_predictions(self):
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
        env = make_offense_env(iter(scores))
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
