"""Window evidence does not turn withheld forecasts or lost possession into negatives."""
from copy import deepcopy
import unittest
from unittest.mock import Mock, patch

import numpy as np

from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.evaluation.chance_creation_replay import (
    build_parser, clone_policy, confusion, dangerous_reception, forecasts, gate_reason, natural_branch, request_pass,
    sampling_strata, summarize,
)
from tests.unit.test_chance_creation import response_state
from tests.unit.test_classic_offense import offense_state


class ChanceReplayTests(unittest.TestCase):
    def test_delayed_pass_counter_does_not_label_loose_owner_as_passer(self):
        initial = offense_state()
        stages = [deepcopy(initial) for _ in range(3)]
        stages[0].engine.puck_owner = stages[1].engine.puck_owner = -254
        stages[2].engine.puck_owner = 7
        for state in stages[1:]:
            state.team1.pass_attempts += 1
            state.engine.pass_target = 1
        env = Mock()
        info = {'p1_score': 0, 'bench_clock': 299}
        env.data.lookup_all.return_value = info
        outcomes = Mock()
        outcomes.observe.return_value = False
        outcomes.summary.return_value = {}
        with patch.multiple(
                'nhl94_ai.evaluation.chance_creation_replay',
                restore_sample=Mock(return_value=(initial, initial)),
                frame_action=Mock(return_value=np.zeros(12, dtype=np.int8)),
                planner_windows=Mock(return_value=[]), crossing_touch_player=Mock(return_value=0),
                CarryOutcomes=Mock(return_value=outcomes),
                step=Mock(side_effect=[(state, info) for state in stages])):
            result = natural_branch(env, {'model': ClassicAIV1Model(), 'ram_sha256': 'same'},
                                    1, 3, enabled=True)
        event, = result['pass_events']
        self.assertNotIn('passer', event)
        self.assertEqual(event['owner_before_counter'], -254)
        self.assertEqual(event['first_possession_owner'], 7)
        self.assertFalse(event['received_requested'])

    def test_decision_observer_is_not_copied_into_gameplay_history(self):
        history = ClassicAIV1Model()
        history._tick, history._frame_remaining = 17, 2
        history._b_down, history._c_down = True, True
        history.offense.last_pass = {'owner': 0}
        history._predict_decision = lambda *_: None
        cloned = clone_policy(history)
        self.assertIs(cloned._predict_decision.__func__, ClassicAIV1Model._predict_decision)
        self.assertEqual((cloned._tick, cloned._frame_remaining, cloned._b_down, cloned._c_down),
                         (17, 2, True, True))
        cloned.offense.last_pass['owner'] = 6
        self.assertEqual(history.offense.last_pass['owner'], 0)

    def test_fresh_protocol_is_declared_and_bounded(self):
        args = build_parser().parse_args([])
        self.assertEqual((args.fresh_seed, args.fresh_seeds, args.samples_per_period), (20262801, 4, 4))
        self.assertEqual((args.horizon, args.pass_horizon), (240, 120))

    def test_gate_counts_separate_missing_windows_and_low_gain(self):
        self.assertEqual(gate_reason({'status': 'uncertified-carry'}, 10), 'uncertified-carry')
        row = {'status': 'modeled-window', 'receiver': None, 'value': -5}
        self.assertEqual(gate_reason(row, 10), 'no-window')
        row.update(receiver=1, value=20)
        self.assertEqual(gate_reason(row, 10), 'below-margin')
        row['value'] = 21
        self.assertEqual(gate_reason(row, 10), 'above-margin')
        self.assertEqual(gate_reason(row, 10, first=True), 'straight-control')

    def test_sampling_uses_prediction_strata_not_native_outcomes(self):
        rows = [{'status': 'uncertified-carry'},
                {'status': 'no-supported-response'},
                {'status': 'modeled-window', 'receiver': 1, 'value': 20}]
        self.assertEqual(sampling_strata(rows, 10), ['below-margin', 'uncertified', 'unsupported-response'])

    def test_confusion_has_all_four_cells(self):
        self.assertEqual([confusion(p, o) for p, o in ((True, True), (True, False), (False, True), (False, False))],
                         ['true-positive', 'false-positive', 'false-negative', 'true-negative'])

    def test_forecasts_do_not_mutate_shared_controller_history(self):
        state = response_state()
        history = ClassicAIV1Model()
        history._pass_at = 500
        history.offense.pass_at = 500
        before = deepcopy(vars(history.offense))
        options, status = forecasts({'state': state, 'model': history}, 1,
                                    {'target': (-24, 168), 'carry_safe': False}, 18)
        self.assertEqual(options, [])
        self.assertTrue(status['counterfactual_outside_carry_admission'])
        self.assertEqual(vars(history.offense), before)

    def test_forced_pass_preserves_prior_clocks_and_uses_actual_native_counters(self):
        state = offense_state()
        history = ClassicAIV1Model()
        history.defense.frames, history._tick = 64, 17
        state.team1.one_timer_attempts = 3
        model, action = request_pass(history, state, 1, 'one-timer')
        self.assertEqual((model.defense.frames, model._tick, model._frame_remaining), (65, 18, 3))
        self.assertEqual(model._one_timer_attempts_before, 3)
        self.assertEqual(model._one_timer_passes_before, state.team1.pass_attempts)
        self.assertEqual(model._one_timer[:2], (0, 1))
        self.assertEqual((history.defense.frames, history._tick), (64, 17))
        self.assertTrue(action[0])
        with self.assertRaisesRegex(ValueError, 'Unknown window'):
            request_pass(history, state, 1, 'unknown')

    def test_dangerous_reception_uses_native_position_and_correct_rink_end(self):
        state = offense_state()
        receiver = state.team1.players[1]
        receiver.x, receiver.y = 71, 151
        self.assertTrue(dangerous_reception(state, 1))
        receiver.x = 72
        self.assertFalse(dangerous_reception(state, 1))
        state.team1.net.y, state.team2.net.y, receiver.x, receiver.y = 264, -264, 0, -151
        self.assertTrue(dangerous_reception(state, 1))

    def test_empty_native_probes_are_not_counted_as_true_negatives(self):
        period = {'candidate_gates': {}, 'carry_rejection_reasons': {},
                  'cases': [{'sample_kind': 'fresh-stratified', 'routes': [{
                      'selector_row': {'carry_safe': False},
                      'native_route': {'retained_original_carrier': False},
                      'native_pass_probes': []}]}]}
        result = summarize([period])['target_purpose_confusion']['fresh-stratified/rejected-carry']
        self.assertEqual(result['lost_before_window'], 1)
        self.assertNotIn('true-negative', result)


if __name__ == '__main__':
    unittest.main()
