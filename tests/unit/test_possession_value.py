"""Shared offensive utility, executable continuations and live policy wiring."""
from dataclasses import replace
import pickle
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.agents.cross_crease import CrossCreaseController, evaluate_cross_crease
from nhl94_ai.agents.decisions import classic_decision_snapshot
from nhl94_ai.agents.deke import DekeController, evaluate_deke
from nhl94_ai.agents.passing import evaluate_pass
from nhl94_ai.agents.possession import PossessionOffenseController
from nhl94_ai.agents.possession_value import (
    VALUE_MODEL, PossessionValuePlanner, coast_scene, shot_quality, transition_score,
)
from nhl94_ai.env.actions import HockeyActionController
from nhl94_ai.cli import main
from nhl94_ai.config import EnvironmentConfig
from nhl94_ai.game.constants import GameConsts as Buttons
from tests.unit.test_classic_offense import live_one_timer_state, offense_state
from tests.unit.test_cross_crease import crossing_state
from tests.unit.test_deke import deke_state


def chain_state():
    state = offense_state()
    for player in (*state.team1.players[3:], *state.team2.players):
        player.x, player.y = 100, -180
    for player, point in zip(state.team1.players, ((-80, 180), (70, 150), (-30, 220))):
        player.x, player.y = point
    state.puck.x, state.puck.y = -80, 190
    state.team2.goalie.x = 30
    return state


def finish_state():
    state = offense_state()
    for player in (*state.team1.players[1:], *state.team2.players):
        player.x, player.y = 100, -150
    state.team1.players[0].x, state.team1.players[0].y = 0, 225
    state.team1.players[0].motion_y = 1
    state.puck.x, state.puck.y = 0, 230
    state.team2.goalie.x = 40
    return state


class PossessionValueTests(unittest.TestCase):
    def setUp(self):
        self.planner = PossessionValuePlanner()
        self.offense = PossessionOffenseController()

    def test_backward_pass_gets_credit_for_a_third_player_one_timer(self):
        state = chain_state()
        option, details = evaluate_pass(state, state.team1.players[0], 1,
                                        state.team1.players[1], 'position', rank_risk=True)
        self.assertIsNotNone(option, details)
        self.assertLess(option.forward_gain, 0)
        value = self.planner.score_pass(state, option, 0)
        self.assertEqual((value.continuation, value.continuation_slot), ('one-timer', 2))
        self.planner.one_timers = False
        retained = self.planner.score_pass(state, option, 0)
        self.assertEqual(retained.continuation, 'retain')
        self.assertGreater(value.value, retained.value)

    def test_one_timer_cooldown_is_respected_at_future_reception_time(self):
        state = chain_state()
        option, _ = evaluate_pass(state, state.team1.players[0], 1,
                                  state.team1.players[1], 'position', rank_risk=True)
        self.planner.one_timer_at = 1000
        self.assertEqual(self.planner.score_pass(state, option, 0).continuation, 'retain')
        self.planner.one_timer_at = 8
        self.assertEqual(self.planner.score_pass(state, option, 0).continuation, 'one-timer')

    def test_legacy_pass_score_cannot_change_shared_value(self):
        state = chain_state()
        option, _ = evaluate_pass(state, state.team1.players[0], 1,
                                  state.team1.players[1], 'position', rank_risk=True)
        value = self.planner.score_pass(state, option, 0)
        self.assertEqual(value, self.planner.score_pass(state, replace(option, value=100000), 0))

    def test_receiver_velocity_changes_reception_and_continuation_value(self):
        values, points = [], []
        for motion in (0, 0.7):
            state = live_one_timer_state()
            state.team1.players[1].motion_y = motion
            option, _ = evaluate_pass(state, state.team1.players[0], 1,
                                      state.team1.players[1], 'position', rank_risk=True)
            points.append(option.point)
            values.append(self.planner.score_pass(state, option, 0).value)
        self.assertNotEqual(points[0], points[1])
        self.assertNotEqual(values[0], values[1])

    def test_goalie_and_defender_motion_change_shot_value(self):
        state = finish_state()
        shooter = state.team1.players[0]
        def value():
            return shot_quality(state, shooter, (0, 230), 1, 8, 4)
        clear = value()
        state.team2.goalie.motion_x = -2
        self.assertLess(value(), clear)
        state.team2.goalie.motion_x = 0
        blocker = state.team2.players[0]
        blocker.x, blocker.y, blocker.motion_x = -30, 245, 3
        moving = value()
        blocker.motion_x = 0
        self.assertLess(moving, value())

    def test_friendly_bodies_and_sticks_count_as_shot_obstacles(self):
        state = finish_state()
        shooter = state.team1.players[0]
        def value():
            return shot_quality(state, shooter, (0, 230), 1, 0, 4)
        clear = value()
        blocker = state.team1.players[1]
        blocker.x, blocker.y = 8, 246
        self.assertLess(value(), clear)
        blocker.x, blocker.stick_x, blocker.stick_y = -20, 28, 0
        self.assertLess(value(), clear)
        blocker.role = -1
        self.assertEqual(value(), clear)

    def test_transition_costs_time_and_risk_on_a_shared_scale(self):
        state = offense_state()
        fast = transition_score(state, 50, (0, 180), 8, 12)
        slow = transition_score(state, 50, (0, 180), 28, 12)
        risky = transition_score(state, 50, (0, 180), 8, -2, robustness=0.5)
        own_slot = transition_score(state, 50, (0, -200), 8, -2, robustness=0.5)
        self.assertGreater(fast.value, slow.value)
        self.assertGreater(fast.value, risky.value)
        self.assertGreater(risky.value, own_slot.value)
        self.assertEqual(fast.snapshot()['value_model'], VALUE_MODEL)

    def test_race_estimate_is_soft_but_physical_obstruction_is_still_hard(self):
        state = offense_state()
        passer, receiver = state.team1.players[:2]
        with patch('nhl94_ai.agents.passing.pressure_margin', return_value=-1):
            old, _ = evaluate_pass(state, passer, 1, receiver, 'position')
            ranked, _ = evaluate_pass(state, passer, 1, receiver, 'position', rank_risk=True)
        self.assertIsNone(old)
        self.assertIsNotNone(ranked)
        state.team2.players[0].x, state.team2.players[0].y = -40, -70
        blocked, details = evaluate_pass(state, passer, 1, receiver, 'position', rank_risk=True)
        self.assertIsNone(blocked)
        self.assertIn(details['status'], ('moving-interception', 'moving-stick-interception'))

    def test_controlled_possession_and_motion_are_required(self):
        state = offense_state()
        self.assertTrue(self.planner.available(state))
        state.team1.players[0].motion_y = None
        self.assertFalse(self.planner.available(state))
        state.team1.players[0].motion_y = 0
        state.engine.puck_owner = 1
        self.assertFalse(self.planner.available(state))

    def test_coasting_keeps_puck_offset_velocity_and_input_state(self):
        state = finish_state()
        before = pickle.dumps(state)
        future = coast_scene(state, 4)
        self.assertEqual(pickle.dumps(state), before)
        self.assertEqual(future.puck.y - future.team1.players[0].y, 5)
        self.assertEqual(future.puck.motion_y, future.team1.players[0].motion_y)
        self.assertEqual(future.puck.precise_y, float(future.puck.y))

    def test_choice_is_deterministic_pure_and_has_maximum_common_score(self):
        for state in (offense_state(), chain_state(), live_one_timer_state(away=True)):
            before = pickle.dumps(state)
            choice = self.planner.choose(state, self.offense, 0)
            self.assertEqual(choice, self.planner.choose(state, self.offense, 0))
            self.assertEqual(pickle.dumps(state), before)
            values = [row['value'] for row in self.planner.diagnostics['possession_candidates']
                      if row['status'] == 'eligible']
            self.assertEqual(choice.score.value, max(values))

    def test_execution_cadence_and_cooldowns_mask_unavailable_passes(self):
        state = live_one_timer_state()
        self.offense.pass_at = 1000
        for interval in (1, 4, 8):
            self.offense.decision_interval = interval
            choice = self.planner.choose(state, self.offense, 0, one_timer_at=1000)
            self.assertNotIn(choice.kind, ('pass', 'one-timer'))
            self.assertFalse(self.planner.diagnostics['one_timer_candidates'])
            rows = self.planner.diagnostics['candidates']
            self.assertTrue(any(row['status'] == 'pass-cooldown' for row in rows))

    def test_carry_forecast_covers_a_longer_committed_input_interval(self):
        self.offense.decision_interval = 32
        self.planner.choose(offense_state(), self.offense, 0)
        carries = [row for row in self.planner.diagnostics['possession_candidates']
                   if row['kind'] == 'carry' and row['status'] == 'eligible']
        self.assertTrue(carries)
        self.assertTrue(all(row['elapsed_frames'] >= 32 for row in carries))

    def test_optional_finishers_are_ranked_with_the_same_value_model(self):
        deke = deke_state()
        deke.team2.goalie.x = deke.team2.goalie.precise_x = 18
        deke.team2.goalie.decision_interval = 32
        for state, kind, controller, evaluator in (
                (crossing_state(), 'cross_crease', CrossCreaseController(), evaluate_cross_crease),
                (deke, 'deke', DekeController(), evaluate_deke)):
            self.planner.choose(state, self.offense, 0, finishers=((kind, controller, evaluator),))
            row = next(row for row in self.planner.diagnostics['possession_candidates'] if row['kind'] == kind)
            self.assertEqual(row['status'], 'eligible')
            self.assertEqual(row['value_model'], VALUE_MODEL)
            self.assertGreater(row['elapsed_frames'], 0)


class PossessionPolicyIntegrationTests(unittest.TestCase):
    def test_documented_play_cli_selects_the_feature_and_validates_scope(self):
        with patch('nhl94_ai.evaluation.play.run') as run:
            main(['play', '--agent', 'classic-v1', '--env', 'NHL94-Genesis-v0', '--possession-value'])
        args = run.call_args.args[0]
        self.assertTrue(args.possession_value)
        self.assertEqual(EnvironmentConfig.from_args(args).nn, 'ClassicAIV1')
        args.selfplay = True
        with self.assertRaises(ValueError):
            EnvironmentConfig.from_args(args)

    def test_flag_is_opt_in_and_rejects_incompatible_policy_combinations(self):
        self.assertIsNone(ClassicAIV1Model().possession_value)
        for flags in ({'offense_lookahead': True}, {'uncertain_carry': True}, {'chance_creation': True},
                      {'action_type': 'HOCKEY_TARGET'}, {'env': 'NHL94-Genesis-2v2-v0'}):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                ClassicAIV1Model(SimpleNamespace(possession_value=True, **flags))

    def test_selected_shot_uses_existing_lifecycle_and_both_action_schemas(self):
        for schema in ('FILTERED', 'HOCKEY_INTENT_DPAD'):
            state = finish_state()
            model = ClassicAIV1Model(SimpleNamespace(action_type=schema, possession_value=True))
            action = model.predict_frame(state)[0]
            processor = HockeyActionController(SimpleNamespace(action_type=schema, game_state=state))
            buttons = processor._process_action(action, processor._new_action_state())[0]
            self.assertEqual(model._last_decision, 'shoot')
            self.assertTrue(buttons[Buttons.INPUT_C])
            self.assertEqual(model.offense_diagnostics['possession_action'], 'shoot')

    def test_selected_pass_commits_only_first_leg_of_a_continuation(self):
        state = live_one_timer_state()
        state.team1.players[1].motion_y = 0
        model = ClassicAIV1Model(SimpleNamespace(possession_value=True))
        model.predict_frame(state)
        self.assertEqual(model._last_decision, 'position-pass')
        self.assertEqual(model.offense.pass_action.pending.receiver, 1)
        self.assertIsNone(model.one_timer.pending)
        self.assertEqual(model.offense_diagnostics['possession_score']['continuation'], 'one-timer')

    def test_selected_one_timer_preserves_receiver_and_cue_lifecycle(self):
        state = chain_state()
        state.team1.players[1].y, state.team1.players[2].y = 120, 215
        model = ClassicAIV1Model(SimpleNamespace(possession_value=True))
        action = model.predict_frame(state)[0]
        self.assertEqual(model._last_decision, 'one-timer-pass')
        self.assertEqual(model.one_timer.pending.receiver, 2)
        self.assertTrue(action[Buttons.INPUT_B])

    def test_missing_motion_uses_existing_fallback_without_stale_shared_scores(self):
        state = offense_state()
        state.team1.players[0].motion_x = None
        old, new = ClassicAIV1Model(), ClassicAIV1Model(SimpleNamespace(possession_value=True))
        np.testing.assert_array_equal(old.predict_frame(state)[0], new.predict_frame(state)[0])
        self.assertNotIn('value_model', new.offense_diagnostics)

    def test_inspector_exposes_heuristic_costs_without_mutating_agent(self):
        state = chain_state()
        model = ClassicAIV1Model(SimpleNamespace(possession_value=True))
        action = model.predict_frame(state)[0]
        before = pickle.dumps((model, state))
        snapshot = classic_decision_snapshot(model, state, action)
        self.assertEqual(pickle.dumps((model, state)), before)
        candidate = next(row for row in snapshot.candidates if row.action_id == 'pass:1')
        self.assertIn('one-timer', candidate.reason)
        self.assertEqual(candidate.scores[0].kind, 'possession')
        self.assertIsNone(candidate.probability)
        self.assertIsNone(candidate.raw_probability)
