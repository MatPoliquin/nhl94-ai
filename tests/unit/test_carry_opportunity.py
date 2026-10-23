"""Carry-only authority, Classic handoff and non-repeatable opportunity credit."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from stable_baselines3 import PPO

from nhl94_ai.agents.learned_offense import PossessionController
from nhl94_ai.env.offense import OffenseEnv
from nhl94_ai.env.opportunity import OpportunityReward, REWARD_SPEC, classic_opportunity
from nhl94_ai.models.offense import OffensePolicy, load_offense_model, save_offense_model
from tests.support.learned_offense import make_offense_env
from tests.support.offense import live_one_timer_state


class CarryControlTests(unittest.TestCase):
    def test_finishing_is_identical_for_every_movement_action_at_both_ends(self):
        for away in (False, True):
            state = live_one_timer_state(away=away)
            teacher = PossessionController(mode='teacher', control_scope='carry', goalie_policy='off')
            teacher.begin_step(state)
            expected = teacher.frame(state)
            self.assertIsNotNone(teacher.one_timer.pending)
            for command in range(9):
                learner = PossessionController(control_scope='carry', goalie_policy='off')
                self.assertEqual(learner.action_mask(state).sum(), 9)
                learner.begin_step(state, command)
                np.testing.assert_array_equal(learner.frame(state), expected)
                self.assertEqual(learner.one_timer.pending, teacher.one_timer.pending)
                self.assertFalse(learner.command_used)

    def test_external_pass_and_shot_commands_are_rejected(self):
        state = live_one_timer_state()
        learner = PossessionController(control_scope='carry')
        for command in range(9, 19):
            with self.assertRaisesRegex(ValueError, 'Unavailable'):
                learner.begin_step(state, command)

    def test_carry_is_replaced_and_respects_decision_clock(self):
        state = live_one_timer_state()
        learner = PossessionController(control_scope='carry', goalie_policy='off')

        def carry_only(controller, _state, action, *_args):
            controller.input_owner = 'offense'
            action[4] = 1
            return action, 0

        with patch('nhl94_ai.agents.classic_v1.ClassicAIV1Model._attack', carry_only):
            learner.begin_step(state, 7)
            for _ in range(4):
                buttons = learner.frame(state)
                self.assertEqual(buttons[6], 1)
                self.assertEqual(buttons.sum(), 1)
            self.assertTrue(learner.command_used)

    def test_reward_evaluation_uses_unmodified_classic_admission_without_mutation(self):
        state = live_one_timer_state()
        controller = PossessionController(control_scope='carry')
        choices, _ = controller.offense.passes(state, 'one-timer')
        before = copy.deepcopy(controller.one_timer)
        self.assertAlmostEqual(classic_opportunity(state, controller), choices[0].shot_value / 100)
        self.assertEqual(controller.one_timer, before)
        controller.one_timer.retry_at_frame = 10
        self.assertEqual(classic_opportunity(state, controller), 0)

    def test_checkpoint_requires_explicit_scope_transfer(self):
        torch.set_num_threads(1)
        env = OffenseEnv()
        model = PPO(OffensePolicy, env, n_steps=2, batch_size=2, device='cpu')
        with tempfile.TemporaryDirectory() as directory:
            path = save_offense_model(model, Path(directory) / 'carry',
                                      training_contract=dict(task='period', control_scope='carry'))
            with self.assertRaisesRegex(ValueError, 'control scope'):
                load_offense_model(path, expected_control_scope='offense')
            self.assertEqual(load_offense_model(path, expected_control_scope='carry').offense_control_scope, 'carry')
            load_offense_model(path, expected_control_scope='offense', allow_control_transfer=True)
        env.close()


class OpportunityRewardTests(unittest.TestCase):
    def tracker(self):
        state = live_one_timer_state()
        controller = PossessionController(control_scope='carry')
        info = dict(bench_one_timers1=7, bench_one_timer_goals1=3, bench_shots1=12, shot_player=-1)
        return OpportunityReward(state, controller, info, 1, 0.9), state, controller, info

    def test_setup_only_paid_once_and_release_requires_fresh_counter_and_receiver(self):
        tracker, _, controller, info = self.tracker()
        controller.one_timer.starts += 1
        controller.one_timer.pending = SimpleNamespace(receiver=1)
        controller.one_timer.option = SimpleNamespace(shot_value=50)
        self.assertEqual(tracker.before_frame(controller), 0.25)
        self.assertEqual(tracker.before_frame(controller), 0)
        controller.one_timer.starts += 1
        self.assertEqual(tracker.before_frame(controller), 0)
        info['shot_player'] = 1
        self.assertAlmostEqual(tracker.after_frame(info, None)[0], -0.001)
        info.update(bench_one_timers1=8, shot_player=2)
        self.assertAlmostEqual(tracker.after_frame(info, None)[0], -0.001)
        info.update(bench_one_timers1=9, shot_player=1)
        self.assertAlmostEqual(tracker.after_frame(info, None)[0], 0.499)
        self.assertAlmostEqual(tracker.after_frame(info, None)[0], -0.001)
        self.assertEqual(tracker.confirmed_releases, 1)

    def test_potential_telescopes_including_terminal_boundary(self):
        tracker, state, controller, _ = self.tracker()
        initial = tracker.potential
        with patch('nhl94_ai.env.opportunity.classic_opportunity', side_effect=[0.8, 0.2]):
            first = tracker.shape(state, controller, 1, False)
            second = tracker.shape(state, controller, 3, False)
            last = tracker.shape(state, controller, 0.25, True)
        self.assertAlmostEqual(first + 0.9 * second + 0.9**4 * last,
                               -REWARD_SPEC['potential_weight'] * initial)

    def test_unchanged_quality_does_not_farm_positive_shaping(self):
        tracker, state, controller, _ = self.tracker()
        with patch('nhl94_ai.env.opportunity.classic_opportunity', return_value=tracker.potential):
            self.assertLess(tracker.shape(state, controller, 1, False), 0)

    def test_shot_resolution_is_not_a_carrying_turnover_and_goal_keeps_priority(self):
        tracker, _, _, info = self.tracker()
        self.assertEqual(tracker.after_frame(info, 'turnover'), (-1.001, 'turnover'))
        info['bench_one_timers1'] += 1
        self.assertEqual(tracker.after_frame(info, 'turnover'), (-0.001, 'shot_ended'))
        self.assertEqual(tracker.after_frame(info, 'goal'), (1.999, 'goal'))
        self.assertEqual(tracker.summary(info)['native_one_timers'], 1)
        self.assertEqual(tracker.summary(info)['one_timer_goals'], 0)

    def test_friendly_recovery_starts_a_new_carrying_turnover_risk(self):
        tracker, _, _, info = self.tracker()
        info['bench_one_timers1'] += 1
        tracker.after_frame(info, None)
        self.assertTrue(tracker.after_shot)
        tracker.after_frame(info, None, own_skater=True)
        self.assertEqual(tracker.after_frame(info, 'turnover'), (-1.001, 'turnover'))


class OpportunityNativeBoundaryTests(unittest.TestCase):
    def env(self, scores):
        env = make_offense_env(iter(scores))
        env.task, env.attack_start, env.attack_frames, env.snapshot = 'opportunity', 1, 600, 'test'
        state = live_one_timer_state()
        state.engine.puck_owner = -1
        controller = env.controllers[0]
        controller.available = lambda _: False
        controller.one_timer = SimpleNamespace(starts=0)
        controller._last_pass_request = None
        env.views = [state, state]
        env.info.update(bench_one_timer_goals1=0, shot_player=-1)
        env.opportunity_reward = OpportunityReward(state, controller, env.info, 1, env.gamma)
        return env

    def test_discount_matches_native_goal_time_across_short_and_full_intervals(self):
        for goal_frame in (1, 2, 4, 7):
            env = self.env([(0, 0)] * (goal_frame - 1) + [(1, 0)])
            frames, total = 0, 0
            while True:
                _, reward, end, cut, info = env.step(0)
                total += env.gamma**(frames / 4) * reward
                frames += info['native_frames']
                if end or cut:
                    break
            expected = sum(env.gamma**(i / 4) * (-0.001 + (2 if i == goal_frame - 1 else 0))
                           for i in range(goal_frame))
            self.assertAlmostEqual(total, expected)
            self.assertEqual(frames, goal_frame)
            self.assertEqual(info['outcome'], 'goal')

    def test_goal_overrides_simultaneous_opponent_ownership(self):
        env = self.env([(1, 0)])
        state = env.views[0]
        state.engine.puck_owner = state.team2.skater_scnum_base()
        _, reward, end, _, info = env.step(0)
        self.assertTrue(end)
        self.assertAlmostEqual(reward, 1.999)
        self.assertEqual(info['outcome'], 'goal')


if __name__ == '__main__':
    unittest.main()
