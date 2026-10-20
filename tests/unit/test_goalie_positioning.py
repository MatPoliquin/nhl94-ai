"""Guard default selection and preserve the audited goalie-positioning rule."""
import argparse
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.agents.goalie import GoalieController, GoaliePlan
from nhl94_ai.agents.goalie_positioning import DeepGoalie
from nhl94_ai.agents.goalie_csave import CSaveGoalie
from nhl94_ai.agents.registry import add_classic_arguments
from nhl94_ai.game.constants import GameConsts as Buttons
from nhl94_ai.agents.motion import velocity
from tests.support.defense import goalie_state


class ReferenceGoalie(GoalieController):
    """Frozen positioning rule used to guard the live controller."""

    def _position(self, state, plan):
        if plan.crossing_frames is None:
            sign = 1 if state.team1.net.y > 0 else -1
            vx, vy = velocity(state.puck)
            reference = plan.receiver or (state.puck.x+8*vx, state.puck.y+8*vy)
            distance = 260-reference[1]*sign
            if distance > 10:
                plan = replace(plan, target=(max(-28, min(28, reference[0]*10/distance)), sign*250))
        return super()._position(state, plan)


class PositioningTests(unittest.TestCase):
    def test_cumulative_candidate_selection_and_incompatible_timing(self):
        parser = add_classic_arguments(argparse.ArgumentParser())
        args = parser.parse_args(['--goalie-positioning', 'deep-csave'])
        args.goalie_policy = 'selective'
        self.assertIs(type(ClassicAIV1Model(args).goalie), CSaveGoalie)
        args.goalie_save_timing = 'contact'
        with self.assertRaises(ValueError):
            ClassicAIV1Model(args)
        args.goalie_save_timing, args.goalie_policy = 'classic', 'off'
        with self.assertRaises(ValueError):
            ClassicAIV1Model(args)

    def test_C_contact_model_never_delays_A_dive(self):
        state = goalie_state(True)
        state.team1.goalie.x = 0
        plan = GoaliePlan((30, state.team1.goalie.y), 'released',
                          crossing_frames=4, crossing_x=30, height=3)
        reference, candidate = DeepGoalie('selective'), CSaveGoalie('selective')
        expected = reference._position(state, plan)
        self.assertEqual(expected[Buttons.INPUT_A], 1)
        with patch('nhl94_ai.agents.goalie_csave.contact_frame',
                   side_effect=AssertionError('C model must not evaluate a dive')):
            actual = candidate._position(state, plan)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(candidate.pending_save, reference.pending_save)
        self.assertEqual(candidate.metrics, reference.metrics)

    def test_delaying_C_does_not_start_a_save_or_lockout(self):
        state = goalie_state(True)
        state.team1.goalie.x = 0
        plan = GoaliePlan((15, state.team1.goalie.y), 'released',
                          crossing_frames=4, crossing_x=15, height=3)
        candidate = CSaveGoalie('selective')
        with patch('nhl94_ai.agents.goalie_csave.contact_frame', side_effect=(3, None)):
            action = candidate._position(state, plan)
        self.assertEqual(action[Buttons.INPUT_C], 0)
        self.assertEqual(action[Buttons.INPUT_RIGHT], 1)
        self.assertIsNone(candidate.pending_save)
        self.assertEqual(candidate.save_at, 0)
        self.assertEqual(candidate.metrics['save_requests'], 0)

    def test_default_and_explicit_factory_selection(self):
        parser = add_classic_arguments(argparse.ArgumentParser())
        self.assertIsNone(parser.parse_args([]).goalie_positioning)
        self.assertEqual(parser.parse_args(['--goalie-positioning', 'deep']).goalie_positioning, 'deep')
        self.assertIs(type(ClassicAIV1Model(SimpleNamespace(goalie_policy='selective')).goalie), CSaveGoalie)
        self.assertIs(type(ClassicAIV1Model(SimpleNamespace(
            goalie_policy='selective', goalie_positioning='classic')).goalie), GoalieController)
        self.assertIsNone(ClassicAIV1Model(SimpleNamespace(goalie_policy='off')).goalie)
        model = ClassicAIV1Model(SimpleNamespace(goalie_policy='always', goalie_positioning='deep',
                                                goalie_save_timing='contact',
                                                classic_refinements=['pass-timing']))
        self.assertIs(type(model.goalie), DeepGoalie)
        self.assertEqual(model.goalie.policy, 'always')
        self.assertEqual(model.goalie.save_timing, 'contact')
        self.assertTrue(model.goalie.pass_timing)

    def test_defaults_preserve_explicit_alternative_save_timing(self):
        model = ClassicAIV1Model(SimpleNamespace(goalie_policy='selective', goalie_save_timing='contact'))
        self.assertIs(type(model.goalie), GoalieController)
        self.assertEqual(model.goalie.save_timing, 'contact')

    def test_standard_play_enables_improved_manual_goalie_without_flags(self):
        from nhl94_ai.evaluation.play import parse_cmdline
        args = parse_cmdline(['--nn', 'ClassicAIV1', '--env', 'NHL94-Genesis-v0', '--mode', 'model_vs_game'])
        self.assertEqual(args.goalie_policy, 'selective')
        self.assertEqual(args.state, 'SabresVsMightyDucks.ManualGoalie.Start')
        self.assertIs(type(ClassicAIV1Model(args).goalie), CSaveGoalie)

    def test_play_defaults_preserve_explicit_off_and_other_game_modes(self):
        from nhl94_ai.evaluation.play import parse_cmdline
        base = ['--nn', 'ClassicAIV1', '--env', 'NHL94-Genesis-v0', '--mode', 'model_vs_game']
        for extra in (['--goalie-policy', 'off'], ['--state', 'PenguinsVsSenators.start'],
                      ['--action_type', 'HOCKEY_INTENT_DPAD'], ['--env', 'NHL941on1-Genesis-v0'],
                      ['--nn', 'MlpPolicy'], ['--mode', 'player_vs_game'], ['--mode', 'model_vs_model']):
            with self.subTest(extra=extra):
                args = parse_cmdline(base + extra)
                self.assertEqual(args.goalie_policy, 'off')
        args = parse_cmdline(base + ['--goalie-positioning', 'classic'])
        self.assertEqual(args.goalie_policy, 'selective')
        self.assertIs(type(ClassicAIV1Model(args).goalie), GoalieController)

    def test_rejects_disabled_or_replaced_controller_before_loading_model(self):
        for settings in ({'goalie_policy': 'off'}, {'goalie_decision_interval': 4},
                         {'goalie_model': '/does-not-exist.zip'}, {'goalie_planner': 'search-24'},
                         {'goalie_positioning': 'invalid'}, {'action_type': 'HOCKEY_INTENT_DPAD'}):
            args = {'goalie_policy': 'selective', 'goalie_positioning': 'deep', **settings}
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                ClassicAIV1Model(SimpleNamespace(**args))

    def test_frozen_diagnostic_and_live_actions_match_on_both_ends(self):
        for sign in (-1, 1):
            for receiver in (None, (40, sign*220), (-150, sign*245), (5, sign*255)):
                for crossing in (None, 4):
                    state = goalie_state(True)
                    state.team1.net.y = sign*264
                    state.team1.goalie.y = sign*246
                    state.puck.y, state.puck.motion_y = sign*170, sign*1.3
                    plan = GoaliePlan((15, sign*243), 'test', receiver=receiver,
                                      crossing_frames=crossing, crossing_x=15, height=4)
                    audit = ReferenceGoalie('selective')
                    live = DeepGoalie('selective')
                    np.testing.assert_array_equal(audit._position(deepcopy(state), plan),
                                                  live._position(deepcopy(state), plan))
                    self.assertEqual(audit.metrics, live.metrics)

    def test_no_target_replacement_for_crossing_or_near_goal_reference(self):
        state = goalie_state(True)
        for plan in (GoaliePlan((15, -243), 'released', crossing_frames=4),
                     GoaliePlan((15, -243), 'near-goal', receiver=(30, -255))):
            with patch.object(GoalieController, '_position') as position:
                DeepGoalie('selective')._position(state, plan)
            self.assertIs(position.call_args.args[1], plan)

    def test_live_diagnostics_publish_the_actual_position_target(self):
        state = goalie_state(True)
        state.engine.puck_owner = 6
        manager = DeepGoalie('selective')
        manager.step(state)
        self.assertEqual(manager.diagnostics['target'], manager.diagnostics['positioning_target'])
        self.assertEqual(manager.diagnostics['destination'], manager.diagnostics['target'])
        self.assertEqual(manager.diagnostics['waypoint'], manager.diagnostics['target'])


if __name__ == '__main__':
    unittest.main()
