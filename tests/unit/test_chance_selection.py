"""Native chance comparisons preserve admissibility and controller history."""
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np

from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.evaluation.chance_selection import ChanceSelection, readiness
from nhl94_ai.evaluation.chance_selection_replay import configured, fork_chance
from nhl94_ai.evaluation.chance_selection_benchmark import run_job
from nhl94_ai.evaluation import cpu_benchmark
from nhl94_ai.game.constants import GameConsts as Buttons
from tests.unit.test_classic_offense import live_one_timer_state, offense_state
from tests.unit.test_possession_value import finish_state
from nhl94_ai.training.chance_selection import evaluate, fit, select


class ChanceSelectionTests(TestCase):
    def test_observer_retains_legacy_actions_and_commitments(self):
        for factory in (finish_state, live_one_timer_state, offense_state):
            view, old = factory(), ClassicAIV1Model()
            instrumented, action, _ = configured(old, view)
            np.testing.assert_array_equal(action, old.predict_frame(view, 4)[0])
            self.assertEqual(instrumented._last_decision, old._last_decision)
            self.assertEqual(asdict(instrumented.shot), asdict(old.shot))
            self.assertEqual(instrumented.one_timer.pending, old.one_timer.pending)
            self.assertEqual(instrumented.offense.pass_action.pending, old.offense.pass_action.pending)

    def test_carry_branch_leaves_no_shot_commitment_or_persistent_override(self):
        view, history = finish_state(), ClassicAIV1Model()
        branch, action, details = configured(history, view, 'carry')
        self.assertEqual(details['baseline'], 'shoot')
        self.assertFalse(action[Buttons.INPUT_B] or action[Buttons.INPUT_C])
        self.assertFalse(branch.shot.active(branch.scheduler.decisions))
        self.assertIsNone(branch.possession_ablation)
        self.assertEqual(history.scheduler.frames, 0)
        # The single carry decision ends; original shot selection resumes.
        for _ in range(4):
            action = branch.predict_frame(view, 4)[0]
        self.assertTrue(action[Buttons.INPUT_C])
        self.assertEqual(branch._last_decision, 'shoot')

    def test_unadmitted_finish_cannot_be_forced(self):
        with self.assertRaises(ValueError):
            configured(ClassicAIV1Model(), finish_state(), 'one-timer')
        with self.assertRaises(ValueError):
            ChanceSelection(forced='invented')

    def test_missing_animation_feedback_is_distinct_from_ready(self):
        player = finish_state().team1.players[0]
        player.selection_flags = None
        self.assertEqual(readiness(player)['animation_known'], 0)
        player.selection_flags = 0x28
        self.assertEqual(readiness(player)['animation_locked'], 1)
        player.selection_flags = 8
        self.assertEqual(readiness(player)['animation_known'], 1)
        self.assertEqual(readiness(player)['animation_locked'], 0)

    def test_features_include_native_velocity_and_current_button_readiness(self):
        state, model = finish_state(), ClassicAIV1Model()
        state.team1.players[0].motion_x = -0.8
        state.team1.players[0].selection_flags = 0x28
        model.buttons.c_down = True
        _, action, details = configured(model, state)
        self.assertEqual(details['features']['vx'], -0.8)
        self.assertEqual(details['features']['animation_locked'], 1)
        self.assertEqual(details['features']['c_down'], 1)
        self.assertFalse(action[Buttons.INPUT_C])

    def test_rule_missing_features_fall_back_to_baseline(self):
        state, model = finish_state(), ClassicAIV1Model()
        model.possession_ablation = ChanceSelection(rule={
            'baseline': 'shoot', 'action': 'carry', 'feature': 'missing',
            'threshold': 0, 'less_equal': True})
        self.assertTrue(model.predict_frame(state, 4)[0, Buttons.INPUT_C])

    def test_rule_cannot_repeatedly_postpone_the_next_finish(self):
        state, model = finish_state(), ClassicAIV1Model()
        model.possession_ablation = ChanceSelection(rule={
            'baseline': 'shoot', 'action': 'carry', 'feature': 'depth',
            'threshold': 300, 'less_equal': True})
        self.assertFalse(model.predict_frame(state, 4)[0, Buttons.INPUT_C])
        for _ in range(4):
            action = model.predict_frame(state, 4)[0]
        self.assertTrue(action[Buttons.INPUT_C])
        self.assertEqual(model.possession_ablation.metrics['overrides'], 1)
        self.assertEqual(model.possession_ablation.metrics['resume-legacy'], 1)

    def test_benchmark_attaches_rule_to_controller_inside_scripted_wrapper(self):
        rule = dict(baseline='shoot', action='carry', feature='animation_locked',
                    threshold=.5, less_equal=False)

        def match(_fixture):
            agent = cpu_benchmark.create_scripted('classic-v1', SimpleNamespace(action_type='FILTERED'))
            agent.frame_skip = 4
            state = finish_state()
            state.team1.players[0].selection_flags = 0x28
            self.assertFalse(agent.predict_game_state(state)[0, Buttons.INPUT_C])
            self.assertEqual(agent.controller.possession_ablation.metrics['overrides'], 1)
            return {'goals': [0, 0], 'possession_ablation_metrics': dict(agent.possession_ablation.metrics)}

        with patch('nhl94_ai.evaluation.chance_selection_benchmark.cpu_match', side_effect=match):
            result = run_job(('sabres-ducks-manual', 1, 'candidate', rule))
        self.assertEqual(result['chance_metrics']['overrides'], 1)

    def test_continuation_goal_after_turnover_is_not_original_attack_credit(self):
        state, model = finish_state(), ClassicAIV1Model()
        state.team1.one_timer_attempts = 0
        later = deepcopy(state)
        later.engine.puck_owner = later.team2.skater_scnum_base()
        info = {'p1_score': 0, 'p2_score': 0, 'bench_clock': 100}
        goal = {**info, 'p1_score': 1}
        model.predict_frame = Mock(return_value=np.zeros((1, 12), dtype=np.int8))
        with (patch('nhl94_ai.evaluation.chance_selection_replay.restore'),
              patch('nhl94_ai.evaluation.chance_selection_replay.advance'),
              patch('nhl94_ai.evaluation.chance_selection_replay.read_view', side_effect=[
                  (state, info), (later, info), (state, goal)])):
            result = fork_chance(Mock(), b'', '', state, 1, model, np.zeros(12), 3)
        self.assertEqual(result['goals_for'], 1)
        self.assertEqual(result['attack_goal'], 0)
        self.assertEqual(result['first_opponent_possession'], 1)
        self.assertIsNone(result['first_recorded_shot'])


class ChanceTrainingTests(TestCase):
    @staticmethod
    def case(seed, value, old, new):
        def outcome(goals):
            return dict(attack_goal=goals, goals_for=goals, goals_against=0, shots=1)
        return {'seed': seed, 'baseline': 'shoot', 'features': {'animation_locked': value},
                'branches': {'shoot': outcome(old), 'carry': outcome(new)}}

    def test_fit_needs_multiple_seeds_and_keeps_rule_fixed_on_validation(self):
        rows = [self.case(seed, 1, 0, 1) for seed in range(5)]
        rows += [self.case(seed, 0, 1, 0) for seed in range(5)]
        candidates = fit(rows, min_cases=5)
        self.assertTrue(candidates)
        rule = candidates[0]['rule']
        self.assertEqual(select(rows[0], rule), 'carry')
        self.assertEqual(select(rows[-1], rule), 'shoot')
        # Adverse held-out outcomes are reported, not used to change the rule.
        heldout = [self.case(100, 1, 1, 0)]
        self.assertEqual(evaluate(heldout, rule)['totals']['paired_reward_gain'], -1)
        self.assertEqual(select(heldout[0], rule), 'carry')
        self.assertFalse(fit([self.case(1, 1, 0, 1)] * 30))

    def test_no_positive_rule_retains_classic(self):
        rows = [self.case(seed, 1, 1, 0) for seed in range(30)]
        self.assertFalse(fit(rows))
        self.assertEqual(select(rows[0], None), 'shoot')
