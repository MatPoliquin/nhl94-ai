"""Single-gate counterfactuals preserve other vetoes, native executors and attribution."""
from copy import deepcopy
from dataclasses import asdict
from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np

from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.agents.passing import evaluate_pass
from nhl94_ai.evaluation.one_timer_admission import AdmissionModel, configured, rejection_audit
from nhl94_ai.evaluation.one_timer_admission_replay import fork_admission
from nhl94_ai.training.one_timer_admission import candidate, evaluate, fit, reward, select
from tests.unit.test_classic_offense import live_one_timer_state, offense_state


def rejected_state(gate):
    state = live_one_timer_state()
    px, py, rx, ry = (-65, 140, -20, 145) if gate == 'window' else (-65, 180, -55, 195)
    player, receiver = state.team1.players[:2]
    player.x, player.y = px, py
    receiver.x, receiver.y, receiver.motion_y = rx, ry, 0
    state.puck.x, state.puck.y = px, py + 10
    return state


class OneTimerAdmissionTests(TestCase):
    def test_passive_audit_preserves_default_inputs_and_commitments(self):
        for state in (offense_state(), live_one_timer_state(), rejected_state('window'), rejected_state('position')):
            original, observed = ClassicAIV1Model(), AdmissionModel()
            for _ in range(12):
                np.testing.assert_array_equal(original.predict_frame(state, 4), observed.predict_frame(state, 4))
                self.assertEqual(original._last_decision, observed._last_decision)
                self.assertEqual(asdict(original.shot), asdict(observed.shot))
                self.assertEqual(original.one_timer.pending, observed.one_timer.pending)

    def test_each_gate_forces_one_existing_executor_then_restores_classic(self):
        for gate in ('window', 'position'):
            state, history = rejected_state(gate), ClassicAIV1Model()
            alternatives = rejection_audit(history, state)['alternatives']
            candidate = next(c for c in alternatives if c['gate'] == gate and c['option'].slot == 1)
            model, action = configured(history, state, candidate)
            self.assertIs(type(model), ClassicAIV1Model)
            self.assertEqual(model._last_decision, 'one-timer-pass')
            self.assertEqual(model.one_timer.pending.receiver, 1)
            self.assertTrue(action[0])
            self.assertFalse(action[8])
            self.assertIsNone(model.possession_ablation)
            self.assertIsNone(history.one_timer.pending)
            self.assertEqual(history.scheduler.frames, 0)

    def test_window_relaxation_still_checks_cue_timing(self):
        state = rejected_state('window')
        with patch('nhl94_ai.agents.passing.one_timer_contact_frame', return_value=2):
            audit = rejection_audit(ClassicAIV1Model(), state)
        row = next(r for r in audit['rows'] if r['slot'] == 1)
        self.assertEqual(row['original_status'], 'outside-one-timer-window')
        self.assertEqual(row['after_window']['status'], 'too-short-for-one-timer-cue')
        self.assertNotIn(1, [c['option'].slot for c in audit['alternatives']])

    def test_window_relaxation_still_checks_obstructions(self):
        state = rejected_state('window')
        with patch('nhl94_ai.agents.passing._swept_contact', return_value=True):
            audit = rejection_audit(ClassicAIV1Model(), state)
        row = next(r for r in audit['rows'] if r['slot'] == 1)
        self.assertEqual(row['original_status'], 'outside-one-timer-window')
        self.assertEqual(row['after_window']['status'], 'friendly-obstruction')
        self.assertNotIn(1, [c['option'].slot for c in audit['alternatives']])

    def test_cannot_remove_both_position_gates_at_once(self):
        state = rejected_state('window')
        with patch('nhl94_ai.evaluation.one_timer_admission.one_timer_position_ok', return_value=False):
            audit = rejection_audit(ClassicAIV1Model(), state)
        row = next(r for r in audit['rows'] if r['slot'] == 1)
        self.assertEqual(row['remaining_gate'], 'weaker-one-timer-position')
        self.assertNotIn(1, [c['option'].slot for c in audit['alternatives']])

    def test_counterfactual_revalidates_candidate_and_requires_fresh_b(self):
        state, model = rejected_state('position'), ClassicAIV1Model()
        candidate = rejection_audit(model, state)['alternatives'][0]
        model.buttons.b_down = True
        with self.assertRaisesRegex(ValueError, 'fresh B'):
            configured(model, state, candidate)
        model.buttons.b_down = False
        state.team1.players[1].motion_y = None
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            configured(model, state, candidate)

    def test_cooldown_and_disabled_one_timers_are_not_relaxed(self):
        state, model = rejected_state('window'), ClassicAIV1Model()
        model.one_timer.retry_at_frame = 100
        self.assertFalse(rejection_audit(model, state)['alternatives'])
        model.one_timer.retry_at_frame = 0
        model._one_timers = False
        self.assertFalse(rejection_audit(model, state)['alternatives'])

    def test_legacy_default_and_explicit_window_gate_are_identical(self):
        state = live_one_timer_state()
        args = (state, state.team1.players[0], 1, state.team1.players[1], 'one-timer')
        self.assertEqual(evaluate_pass(*args), evaluate_pass(*args, one_timer_window=True))

    def test_later_goal_after_confirmed_turnover_is_not_attack_credit(self):
        state, model = offense_state(), ClassicAIV1Model()
        state.team1.one_timer_attempts = 0
        state.engine.clock_stopped = False
        lost = deepcopy(state)
        lost.engine.puck_owner = lost.team2.skater_scnum_base()
        info = {'p1_score': 0, 'p2_score': 0, 'bench_clock': 100}
        model.predict_frame = Mock(return_value=np.zeros((1, 12), dtype=np.int8))
        with (patch('nhl94_ai.evaluation.one_timer_admission_replay.restore'),
              patch('nhl94_ai.evaluation.one_timer_admission_replay.advance'),
              patch('nhl94_ai.evaluation.one_timer_admission_replay.crossing_touch_player', return_value=-1),
              patch('nhl94_ai.evaluation.one_timer_admission_replay.read_view', side_effect=[
                  (state, info), *[(lost, info)]*4, (state, {**info, 'p1_score': 1})])):
            result = fork_admission(Mock(), b'', '', state, 1, model, np.zeros(12), 5)
        self.assertEqual(result['goals_for'], 1)
        self.assertEqual(result['attack_goal'], 0)
        self.assertEqual(result['first_opponent_possession'], 4)
        self.assertEqual(result['intended_one_timer'], 0)


class AdmissionTrainingTests(TestCase):
    @staticmethod
    def case(seed, depth, old_goal, new_goal):
        def outcome(goal):
            return dict(attack_goal=goal, goals_for=goal, goals_against=0,
                        intended_one_timer=0, intended_one_timer_goal=0)
        return {'seed': seed, 'matchup': 'example', 'frame': 100, 'baseline_decision': 'carry',
                'baseline': outcome(old_goal), 'branches': [
                    {'gate': 'position', 'option': {'slot': 1, 'shot_value': 20, 'margin': 10},
                     'features': {'contact_depth': depth}, 'outcome': outcome(new_goal)}]}

    def test_training_selects_rule_without_retuning_to_validation(self):
        cases = [self.case(seed, 180, 0, 1) for seed in range(20)]
        cases += [self.case(seed, 220, 1, 0) for seed in range(20)]
        proposals = fit(cases)
        self.assertTrue(proposals)
        rule = proposals[0]['rule']
        self.assertIsNotNone(select(cases[0], rule))
        self.assertIsNone(select(cases[-1], rule))
        validation = self.case(100, 180, 1, 0)
        self.assertEqual(evaluate([validation], rule)['totals']['paired_reward_gain'], -1)
        self.assertIsNotNone(select(validation, rule))
        self.assertFalse(fit([self.case(1, 180, 0, 1)]*40))

    def test_receiver_choice_uses_original_ranking_not_fork_outcome(self):
        case = self.case(1, 180, 0, 0)
        other = deepcopy(case['branches'][0])
        other['option'].update(slot=2, shot_value=19)
        other['outcome'].update(attack_goal=1, goals_for=1)
        case['branches'].append(other)
        self.assertEqual(candidate(case['branches'], 'position')['option']['slot'], 1)
        self.assertEqual(evaluate([case], {'gate': 'position'})['totals']['selected_attack_goals'], 0)

    def test_missing_features_keep_original_and_counterattacks_cost_value(self):
        case = self.case(1, 180, 0, 0)
        case['branches'][0]['features'] = None
        self.assertIsNone(select(case, {'gate': 'position', 'feature': 'contact_depth',
                                       'threshold': 200, 'less_equal': True}))
        outcome = case['branches'][0]['outcome']
        outcome['goals_against'] = 1
        self.assertEqual(reward(outcome), -1)
