"""Decision semantics, perspective, reward, and ownership learning contracts."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO

from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.agents.learned_offense import (
    N_ACTIONS, OBSERVATION_FIELDS, ENTITY_FIELDS, PossessionController, encode_offense,
)
from nhl94_ai.env.offense import OffenseEnv, score_reward
from nhl94_ai.game.constants import GameConsts as Buttons
from nhl94_ai.models.offense import OffensePolicy, ReceiverOrderedExtractor
from nhl94_ai.models.offense import load_offense_model, save_offense_model
from tests.support.learned_offense import DelayedOutcomeEnv, make_offense_env
from tests.support.offense import live_one_timer_state


class OffenseControllerTests(unittest.TestCase):
    def test_disabled_adapter_preserves_classic_actions_and_lifecycle(self):
        state = live_one_timer_state()
        classic = ClassicAIV1Model(SimpleNamespace(action_type='FILTERED', goalie_policy='off'))
        adapter = PossessionController(mode='classic', goalie_policy='off')
        for _ in range(80):
            np.testing.assert_array_equal(classic.predict_frame(state, 4)[0], adapter.frame(state))
            self.assertEqual(classic._last_decision, adapter._last_decision)
            self.assertEqual(classic.scheduler.frames, adapter.scheduler.frames)

    def test_model_can_carry_instead_of_classic_one_timer(self):
        state = live_one_timer_state()
        controller = PossessionController(goalie_policy='off')
        self.assertTrue(controller.begin_step(state, 5))
        buttons = controller.frame(state)
        self.assertEqual(buttons[Buttons.INPUT_DOWN], 1)
        self.assertEqual(buttons[Buttons.INPUT_B], 0)
        self.assertEqual(buttons[Buttons.INPUT_C], 0)
        self.assertTrue(controller.command_used)

    def test_every_receiver_can_be_requested_without_tactical_gating(self):
        for action in range(9, 17):
            state = live_one_timer_state()
            controller = PossessionController(goalie_policy='off')
            self.assertTrue(controller.action_mask(state)[action])
            controller.begin_step(state, action)
            buttons = controller.frame(state)
            self.assertEqual(buttons[Buttons.INPUT_B], 1)
            self.assertEqual(controller._last_pass_request['receiver'], 1 + (action - 9) % 4)
            self.assertEqual(controller.one_timer.pending is not None, action >= 13)

    def test_target_order_survives_teammate_movement(self):
        state = live_one_timer_state()
        for player in state.team1.players[1:]:
            player.x, player.y = -player.x, -player.y
        controller = PossessionController(goalie_policy='off')
        controller.begin_step(state, 14)
        controller.frame(state)
        self.assertEqual(controller.one_timer.pending.receiver, 2)

    def test_away_movement_rotates_both_axes(self):
        state = live_one_timer_state(away=True)
        controller = PossessionController(goalie_policy='off')
        controller.begin_step(state, 2)
        buttons = controller.frame(state)
        self.assertEqual(buttons[Buttons.INPUT_DOWN], 1)
        self.assertEqual(buttons[Buttons.INPUT_LEFT], 1)

    def test_rotation_preserves_normalized_observation(self):
        home, away = live_one_timer_state(), live_one_timer_state(away=True)
        home.team2.defense_control = home.team2.skater_scnum_base()
        a, b = (PossessionController(goalie_policy='off') for _ in range(2))
        x = encode_offense(home, a, clock=300, score_difference=1)
        y = encode_offense(away, b, clock=300, score_difference=1)
        np.testing.assert_allclose(x, y, atol=1e-6)
        self.assertEqual(x.shape, (len(OBSERVATION_FIELDS),))

    def test_defense_and_commitments_allow_only_forced_coast(self):
        state = live_one_timer_state()
        controller = PossessionController(goalie_policy='off')
        controller.begin_step(state, 13)
        controller.frame(state)
        self.assertEqual(controller.action_mask(state).sum(), 1)
        self.assertFalse(controller.begin_step(state, 18))
        self.assertIsNone(controller.command)
        state.engine.puck_owner = 6
        self.assertEqual(controller.action_mask(state).sum(), 1)

    def test_slapshot_commitment_holds_and_releases(self):
        state = live_one_timer_state()
        controller = PossessionController(goalie_policy='off')
        controller.begin_step(state, 18)
        output = [controller.frame(state)[Buttons.INPUT_C] for _ in range(32)]
        self.assertEqual(output[:18], [1] * 18)
        self.assertTrue(all(value == 0 for value in output[18:]))

    def test_teacher_waits_for_the_same_decision_boundary_as_learner(self):
        state = live_one_timer_state()
        teacher = PossessionController(mode='teacher', goalie_policy='off')
        learner = PossessionController(mode='learned', goalie_policy='off')
        np.testing.assert_array_equal(teacher.frame(state), learner.frame(state))
        self.assertIsNone(teacher.teacher_command)
        teacher.begin_step(state)
        teacher.frame(state)
        self.assertIsNotNone(teacher.teacher_command)

    def test_pass_executor_uses_moving_contact_and_native_deadline(self):
        state = live_one_timer_state()
        with patch('nhl94_ai.agents.learned_offense.pass_contact', return_value=((0, 0), (35, 205), 8, 4)):
            for command in (9, 13):
                controller = PossessionController(goalie_policy='off')
                controller.begin_step(state, command)
                controller.frame(state)
                if command == 9:
                    self.assertEqual(controller.offense.pass_action.pending.point, (35, 205))
                    pending = controller.offense.pass_action.pending
                    self.assertEqual(pending.deadline - pending.frame, 28)
                else:
                    from nhl94_ai.agents.passing import pass_release_frames
                    from nhl94_ai.agents.lifecycle import ONE_TIMER_CONTACT_GRACE
                    self.assertEqual(controller.one_timer.pending.deadline - controller.scheduler.frames,
                                     8 + pass_release_frames(state.team1.players[1]) + ONE_TIMER_CONTACT_GRACE)


class OffenseLearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def make_policy(self):
        return OffensePolicy(gym.spaces.Box(-1, 1, (16, len(OBSERVATION_FIELDS)), np.float32),
                             gym.spaces.Discrete(N_ACTIONS), lambda _: 0.0003)


    def test_receiver_features_match_command_order_for_every_carrier(self):
        width = len(ENTITY_FIELDS)
        for carrier in range(5):
            obs = torch.zeros(1, 16, len(OBSERVATION_FIELDS))
            for slot in range(5):
                obs[..., slot * width] = slot
            obs[..., carrier * width + ENTITY_FIELDS.index('has_puck')] = 1
            obs[..., -N_ACTIONS:] = 1
            ordered = ReceiverOrderedExtractor.ordered(obs)
            expected = [carrier, *[slot for slot in range(5) if slot != carrier]]
            self.assertEqual(ordered[0, -1, :5 * width:width].tolist(), expected)
            torch.testing.assert_close(ordered[..., 5 * width:], obs[..., 5 * width:])


    def test_forced_actions_have_zero_actor_gradient_but_train_critic(self):
        policy = self.make_policy()
        obs = torch.zeros(2, 16, len(OBSERVATION_FIELDS))
        obs[:, -1, -N_ACTIONS] = 1
        actions, values, log_prob = policy(obs)
        entropy = policy.get_distribution(obs).entropy()
        self.assertEqual(actions.tolist(), [0, 0])
        self.assertEqual(log_prob.tolist(), [0, 0])
        (log_prob.sum() + entropy.sum()).backward()
        self.assertEqual(sum(float(p.grad.abs().sum()) for p in policy.parameters() if p.grad is not None), 0)
        policy.zero_grad()
        values = policy.predict_values(obs)
        ((values - 1) ** 2).mean().backward()
        self.assertGreater(float(policy.value_net.weight.grad.abs().sum()), 0)

    def test_rollout_and_training_use_the_same_masked_distribution(self):
        policy = self.make_policy()
        obs = torch.randn(4, 16, len(OBSERVATION_FIELDS))
        obs[:, -1, -N_ACTIONS:] = 0
        obs[:, -1, -N_ACTIONS + 4] = 1
        obs[:, -1, -N_ACTIONS + 8] = 1
        actions, values, log_prob = policy(obs)
        new_values, new_log_prob, _ = policy.evaluate_actions(obs, actions)
        self.assertTrue(set(actions.tolist()).issubset({4, 8}))
        torch.testing.assert_close(log_prob, new_log_prob)
        torch.testing.assert_close(values, new_values)


    def test_score_reward_counts_only_changes_and_is_zero_sum(self):
        self.assertEqual(score_reward((1, 2), (1, 2), 1), 0)
        self.assertEqual(score_reward((1, 2), (2, 2), 1), 1)
        self.assertEqual(score_reward((1, 2), (2, 2), 2), -1)
        self.assertEqual(score_reward((1, 2), (1, 3), 1), -1)

    def test_checkpoint_rejects_wrong_schema_and_keeps_predictions(self):
        env = OffenseEnv()
        model = PPO(OffensePolicy, env, n_steps=8, batch_size=4, device='cpu')
        obs = np.zeros(env.observation_space.shape, np.float32)
        obs[-1, -N_ACTIONS:] = 1
        with tempfile.TemporaryDirectory() as directory:
            path = save_offense_model(model, Path(directory) / 'model')
            restored = load_offense_model(path)
            np.testing.assert_array_equal(model.predict(obs, deterministic=True)[0], restored.predict(obs, deterministic=True)[0])
            meta_path = Path(path + '.offense.json')
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta['schema']['interval'] = 20
            meta_path.write_text(json.dumps(meta), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, 'schema'):
                load_offense_model(path)
        env.close()

    def test_ppo_credits_goal_after_two_forced_execution_ticks(self):
        env = DelayedOutcomeEnv()
        model = PPO(OffensePolicy, env, n_steps=3, batch_size=3, n_epochs=1,
                    gamma=0.997, gae_lambda=1, learning_rate=0, device='cpu')
        with torch.no_grad():
            model.policy.value_net.weight.zero_()
            model.policy.value_net.bias.zero_()
        model.learn(total_timesteps=3)
        np.testing.assert_allclose(model.rollout_buffer.returns.flatten(), [0.997**2, 0.997, 1], atol=1e-6)
        np.testing.assert_array_equal(model.rollout_buffer.actions.flatten()[1:], [0, 0])

    def test_ppo_bootstraps_time_limit_but_not_period_end(self):
        for truncation, expected in ((True, 0.997 * 2), (False, 0)):
            env = DelayedOutcomeEnv(cutoff=1, truncation=truncation)
            model = PPO(OffensePolicy, env, n_steps=2, batch_size=2, n_epochs=1,
                        gamma=0.997, learning_rate=0, device='cpu')
            with torch.no_grad():
                model.policy.value_net.weight.zero_()
                model.policy.value_net.bias.fill_(2)
            model.learn(total_timesteps=2)
            np.testing.assert_allclose(model.rollout_buffer.rewards, expected, atol=1e-6)


class OffenseTransitionTests(unittest.TestCase):
    def make_env(self, scores, *, budget=None):
        return make_offense_env(scores, budget=budget)

    def test_reward_survives_classic_handoff_without_ending_episode(self):
        env = self.make_env(iter([(0, 0), (1, 0), (1, 0), (1, 0),
                                  (1, 0), (1, 1), (1, 1), (1, 1)]))
        _, reward, end, cut, info = env.step(0)
        self.assertEqual(reward, 1)
        self.assertFalse(end or cut or info['actor_active'])
        _, reward, end, cut, _ = env.step(0)
        self.assertEqual(reward, -1)
        self.assertFalse(end or cut)

    def test_time_budget_is_truncation_not_terminal_failure(self):
        env = self.make_env(iter([(0, 0)] * 4), budget=5)
        _, reward, terminated, truncated, info = env.step(0)
        self.assertEqual(reward, 0)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertFalse(info['completed'])

    def test_pool_change_does_not_change_current_match(self):
        env = self.make_env(iter([]))
        env.set_pool(['classic', 'historical.zip'])
        self.assertEqual(env.opponent, 'cpu')
        self.assertEqual(env.pool, ('classic', 'historical.zip'))


if __name__ == '__main__':
    unittest.main()
