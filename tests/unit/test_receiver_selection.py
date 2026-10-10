"""Receiver experiments preserve pass timing/category, admission and lifecycle."""
from dataclasses import asdict
import pickle
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.agents.decisions import classic_decision_snapshot
from nhl94_ai.agents.receiver_selection import INCOMPATIBLE, ReceiverSelector
from nhl94_ai.cli import main
from nhl94_ai.config import EnvironmentConfig
from nhl94_ai.evaluation.receiver_replay import configured, common_continuation
from tests.unit.test_classic_offense import offense_state, live_one_timer_state
from tests.unit.test_possession_value import chain_state, finish_state


def receivers_state():
    state = offense_state()
    state.team1.players[2].x, state.team1.players[2].y = 60, -70
    state.team2.players[0].x, state.team2.players[0].y = 0, 70
    return state


def timers_state():
    state = live_one_timer_state()
    state.team1.players[2].x, state.team1.players[2].y = 0, 220
    return state


class ReceiverSelectionTests(unittest.TestCase):
    def test_expanded_pool_adds_a_safe_backward_receiver_after_an_advance_decision(self):
        state = receivers_state()
        state.team1.players[2].x, state.team1.players[2].y = 60, -160
        model = ClassicAIV1Model(SimpleNamespace(receiver_selection='all-safe'))
        model.predict_frame(state)
        self.assertEqual(model._last_decision, 'advance-pass')
        rows = model.receiver_selector.diagnostics['candidates']
        self.assertLess(next(row for row in rows if row['slot'] == 2)['forward_gain'], 0)
        snapshot = classic_decision_snapshot(model, state, model.scheduler.action[0])
        candidate = next(row for row in snapshot.candidates if row.action_id == 'pass:2')
        self.assertEqual(candidate.status, 'eligible')
        legacy = ClassicAIV1Model(SimpleNamespace(receiver_selection='value'))
        legacy.predict_frame(state)
        self.assertNotIn(2, [row['slot'] for row in legacy.receiver_selector.diagnostics['candidates']])
        self.assertFalse(model.offense.rank_pass_risk)

    def test_receiver_value_can_credit_a_third_player_one_timer(self):
        state = chain_state()
        model = ClassicAIV1Model(SimpleNamespace(receiver_selection='all-safe'))
        selector = model.receiver_selector
        selector.offense = model.offense
        option = next(option for option in selector.options(state, 'position') if option.slot == 1)
        score = selector.score_pass(state, option, 0)
        self.assertEqual((score.continuation, score.continuation_slot), ('one-timer', 2))
        selector.one_timers = False
        self.assertEqual(selector.score_pass(state, option, 0).continuation, 'retain')

    def test_inspector_shows_the_receiver_scores_used_without_mutating_history(self):
        state = receivers_state()
        model = ClassicAIV1Model(SimpleNamespace(receiver_selection='value'))
        action = model.predict_frame(state)[0]
        before = pickle.dumps(model)
        snapshot = classic_decision_snapshot(model, state, action)
        self.assertEqual(snapshot.plan_id, 'pass:2')
        row = next(candidate for candidate in snapshot.candidates if candidate.action_id == 'pass:2')
        score = next(score.value for score in row.scores if score.kind == 'possession')
        self.assertAlmostEqual(score, 5.91)
        self.assertIn('alternative receiver', row.reason)
        self.assertEqual(before, pickle.dumps(model))

    def test_instrumentation_preserves_inputs_and_action_lifecycles(self):
        for schema in ('FILTERED', 'HOCKEY_INTENT_DPAD'):
            for interval in (1, 4, 8):
                for factory in (receivers_state, timers_state, chain_state, finish_state):
                    state = factory()
                    old = ClassicAIV1Model(SimpleNamespace(action_type=schema))
                    new = ClassicAIV1Model(SimpleNamespace(action_type=schema, receiver_selection='legacy'))
                    np.testing.assert_array_equal(old.predict_frame(state, interval), new.predict_frame(state, interval))
                    self.assertEqual(old._last_decision, new._last_decision)
                    self.assertEqual(old._last_target, new._last_target)
                    self.assertEqual(asdict(old.shot), asdict(new.shot))
                    self.assertEqual(old.one_timer.pending, new.one_timer.pending)
                    self.assertEqual(old.offense.pass_action.pending, new.offense.pass_action.pending)
                    self.assertEqual(old.offense.pass_at, new.offense.pass_at)

    def test_receiver_change_preserves_pass_time_type_and_cooldown(self):
        for schema in ('FILTERED', 'HOCKEY_INTENT_DPAD'):
            for interval in (1, 4, 8):
                old = ClassicAIV1Model(SimpleNamespace(action_type=schema))
                new = ClassicAIV1Model(SimpleNamespace(action_type=schema, receiver_selection='value'))
                state = receivers_state()
                old.predict_frame(state, interval)
                new.predict_frame(state, interval)
                self.assertEqual(new._last_decision, old._last_decision)
                self.assertEqual(new._last_pass_request['frame'], old._last_pass_request['frame'])
                self.assertEqual(new._last_pass_request['purpose'], old._last_pass_request['purpose'])
                self.assertEqual(new.offense.pass_at, old.offense.pass_at)
                self.assertEqual(old._last_pass_request['receiver'], 1)
                self.assertEqual(new._last_pass_request['receiver'], 2)

    def test_shooting_and_carrying_never_consult_receiver_ranking(self):
        for factory in (finish_state, offense_state):
            state = factory()
            if factory is offense_state:
                state.engine.puck_owner_known = False
            old, new = ClassicAIV1Model(), ClassicAIV1Model(SimpleNamespace(receiver_selection='value'))
            with patch.object(new.receiver_selector, 'choose_receiver', side_effect=AssertionError('Not a pass')):
                np.testing.assert_array_equal(old.predict_frame(state), new.predict_frame(state))
            self.assertEqual(new._last_decision, old._last_decision)

    def test_unforecastable_option_keeps_legacy_receiver(self):
        model = ClassicAIV1Model(SimpleNamespace(receiver_selection='value'))
        with patch.object(model.receiver_selector, 'score_pass', return_value=None):
            model.predict_frame(receivers_state())
        self.assertEqual(model._last_pass_request['receiver'], 1)
        self.assertEqual(model.receiver_selector.diagnostics['status'], 'unforecastable-use-legacy')

    def test_blocked_receiver_is_not_admitted_by_a_high_forecast(self):
        state = receivers_state()
        state.team2.players[0].x, state.team2.players[0].y = 30, -95
        model = ClassicAIV1Model(SimpleNamespace(receiver_selection='value'))
        model.predict_frame(state)
        self.assertEqual(model._last_pass_request['receiver'], 1)
        self.assertNotIn(2, [row['slot'] for row in model.receiver_selector.diagnostics['candidates']])
        self.assertFalse(model.offense.rank_pass_risk)

    def test_motion_changes_scores_without_mutating_observation(self):
        state = timers_state()
        values = []
        for speed in (-1, 1):
            state.team2.goalie.motion_x = speed
            before = pickle.dumps(state)
            model = ClassicAIV1Model(SimpleNamespace(receiver_selection='value'))
            model.predict_frame(state)
            rows = model.receiver_selector.diagnostics['candidates']
            values.append({row['slot']: row['value'] for row in rows})
            self.assertEqual(before, pickle.dumps(state))
        self.assertNotEqual(values[0], values[1])

    def test_replay_can_force_each_admitted_timer_but_not_an_unsafe_receiver(self):
        state = timers_state()
        history = ClassicAIV1Model(SimpleNamespace(receiver_selection='legacy'))
        for slot in (1, 2):
            branch = configured(history, slot)
            branch.predict_frame(state)
            self.assertEqual(branch._last_decision, 'one-timer-pass')
            self.assertEqual(branch.one_timer.option.slot, slot)
            self.assertEqual(branch._last_pass_request['receiver'], slot)
            followup = common_continuation(branch)
            self.assertIsNone(followup.receiver_selector)
            self.assertEqual(followup.one_timer.pending, branch.one_timer.pending)
            self.assertEqual(followup.buttons, branch.buttons)
        self.assertIsNone(history.one_timer.pending)
        with self.assertRaises(ValueError):
            configured(history, 4).predict_frame(state)

    def test_experiment_is_opt_in_and_cannot_combine_with_other_selectors(self):
        self.assertIsNone(ClassicAIV1Model().receiver_selector)
        for flag in INCOMPATIBLE:
            args = SimpleNamespace(receiver_selection='value', **{flag: ['finishing'] if flag == 'classic_refinements' else True})
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                ClassicAIV1Model(args)
        with self.assertRaises(ValueError):
            ReceiverSelector('invalid')

    def test_public_play_flag_and_environment_scope(self):
        with patch('nhl94_ai.evaluation.play.run') as run:
            main(['play', '--agent', 'classic-v1', '--env', 'NHL94-Genesis-v0', '--receiver-selection', 'value'])
        args = run.call_args.args[0]
        self.assertEqual(args.receiver_selection, 'value')
        self.assertEqual(EnvironmentConfig.from_args(args).nn, 'ClassicAIV1')
        args.selfplay = True
        with self.assertRaises(ValueError):
            EnvironmentConfig.from_args(args)
