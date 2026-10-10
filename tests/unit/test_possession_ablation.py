"""Ablations change outer ranking without silently changing legacy admission."""
from copy import deepcopy
from dataclasses import asdict
import pickle
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.agents.possession_ablation import LegacyValueRanker, PROFILES
from nhl94_ai.cli import main
from nhl94_ai.config import EnvironmentConfig
from nhl94_ai.evaluation.possession_ablation_replay import common_continuation, configured, signatures, summarize
from tests.unit.test_classic_offense import live_one_timer_state, offense_state
from tests.unit.test_possession_value import chain_state, finish_state


class LegacyProposalTests(unittest.TestCase):
    def test_instrumentation_keeps_legacy_inputs_targets_and_lifecycles(self):
        for schema in ('FILTERED', 'HOCKEY_INTENT_DPAD'):
            for interval in (1, 4, 8):
                for factory in (offense_state, live_one_timer_state, chain_state, finish_state):
                    state = factory()
                    old = ClassicAIV1Model(SimpleNamespace(action_type=schema))
                    instrumented = ClassicAIV1Model(SimpleNamespace(action_type=schema, possession_ablation='legacy'))
                    a, b = old.predict_frame(state, interval), instrumented.predict_frame(state, interval)
                    np.testing.assert_array_equal(a, b)
                    self.assertEqual(old._last_decision, instrumented._last_decision)
                    self.assertEqual(old._last_target, instrumented._last_target)
                    self.assertEqual(asdict(old.shot), asdict(instrumented.shot))
                    self.assertEqual(old.one_timer.pending, instrumented.one_timer.pending)
                    self.assertEqual(old.offense.pass_action.pending, instrumented.offense.pass_action.pending)
                    self.assertEqual(old.offense.feint_target, instrumented.offense.feint_target)
                    self.assertEqual(old.offense.pass_at, instrumented.offense.pass_at)

    def test_continuation_switch_changes_values_but_not_candidates(self):
        for factory in (offense_state, live_one_timer_state, chain_state, finish_state):
            state = factory()
            before = pickle.dumps(state)
            policies = {}
            for profile in PROFILES:
                policy = ClassicAIV1Model(SimpleNamespace(possession_ablation=profile))
                policy.predict_frame(state)
                policies[profile] = policy
            self.assertEqual(pickle.dumps(state), before)
            for left, right in (('legacy', 'rank'), ('rank', 'continuations'), ('risk', 'continuations-risk')):
                self.assertEqual(signatures(policies[left].possession_ablation.diagnostics),
                                 signatures(policies[right].possession_ablation.diagnostics))

    def test_higher_forecast_can_defer_a_present_one_timer_without_extra_routes(self):
        state = live_one_timer_state()
        rank = ClassicAIV1Model(SimpleNamespace(possession_ablation='rank'))
        continuation = ClassicAIV1Model(SimpleNamespace(possession_ablation='continuations'))
        rank.predict_frame(state)
        continuation.predict_frame(state)
        self.assertEqual(rank._last_decision, 'one-timer-pass')
        self.assertIn(continuation._last_decision, ('carry', 'carry-breakaway'))
        a, b = rank.possession_ablation.diagnostics, continuation.possession_ablation.diagnostics
        timer_a = next(row for row in a['possession_candidates'] if row['proposal'] == 'one-timer')
        timer_b = next(row for row in b['possession_candidates'] if row['proposal'] == 'one-timer')
        self.assertEqual(timer_a, timer_b)
        self.assertGreater(b['possession_score']['value'], timer_b['value'])
        self.assertEqual(b['possession_score']['continuation'], 'one-timer')

    def test_unforecastable_proposal_retains_baseline_instead_of_being_removed(self):
        state = offense_state()
        model = ClassicAIV1Model(SimpleNamespace(possession_ablation='rank'))
        with patch.object(LegacyValueRanker, '_carry_score', return_value=None):
            model.predict_frame(state)
        self.assertEqual(model._last_decision, 'advance-pass')
        self.assertEqual(model.possession_ablation.diagnostics['status'], 'unscorable-use-legacy')
        self.assertEqual(model.possession_ablation.metrics['unscorable-fallback'], 1)

    def test_relaxed_risk_changes_estimated_gates_but_preserves_physical_blocking(self):
        state = offense_state()
        strict = ClassicAIV1Model(SimpleNamespace(possession_ablation='rank'))
        relaxed = ClassicAIV1Model(SimpleNamespace(possession_ablation='risk'))
        with patch('nhl94_ai.agents.passing.pressure_margin', return_value=-1):
            before, _ = strict.offense.passes(state, 'advance')
            after, _ = relaxed.offense.passes(state, 'advance')
        self.assertFalse(before)
        self.assertTrue(any(option.slot == 1 for option in after))
        state.team2.players[0].x, state.team2.players[0].y = -40, -70
        blocked, details = relaxed.offense.passes(state, 'advance')
        self.assertFalse(any(option.slot == 1 for option in blocked))
        self.assertIn(next(row for row in details if row['slot'] == 1)['status'],
                      ('moving-interception', 'moving-stick-interception'))

    def test_pass_gate_defaults_remain_strict_and_flags_are_independent(self):
        self.assertFalse(ClassicAIV1Model().offense.rank_pass_risk)
        for profile in PROFILES:
            model = ClassicAIV1Model(SimpleNamespace(possession_ablation=profile))
            self.assertEqual(model.offense.rank_pass_risk, profile in ('risk', 'continuations-risk'))
            self.assertEqual(model.possession_ablation.continuations, profile in ('continuations', 'continuations-risk'))

    def test_missing_motion_uses_original_policy(self):
        state = offense_state()
        state.team1.players[0].motion_x = None
        baseline = ClassicAIV1Model()
        model = ClassicAIV1Model(SimpleNamespace(possession_ablation='rank'))
        np.testing.assert_array_equal(baseline.predict_frame(state), model.predict_frame(state))
        self.assertEqual(model.possession_ablation.metrics['decisions'], 0)


class ExperimentContracts(unittest.TestCase):
    def test_other_experiments_cannot_silently_contaminate_ablation(self):
        for flag in ('possession_value', 'offense_lookahead', 'uncertain_carry', 'chance_creation',
                     'cross_crease', 'deke', 'classic_refinements'):
            args = SimpleNamespace(possession_ablation='rank', **{flag: ['finishing'] if flag == 'classic_refinements' else True})
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                ClassicAIV1Model(args)
        with self.assertRaises(ValueError):
            LegacyValueRanker('unknown')

    def test_public_play_flag_and_environment_scope(self):
        with patch('nhl94_ai.evaluation.play.run') as run:
            main(['play', '--agent', 'classic-v1', '--env', 'NHL94-Genesis-v0', '--possession-ablation', 'rank'])
        args = run.call_args.args[0]
        self.assertEqual(args.possession_ablation, 'rank')
        self.assertEqual(EnvironmentConfig.from_args(args).nn, 'ClassicAIV1')
        args.selfplay = True
        with self.assertRaises(ValueError):
            EnvironmentConfig.from_args(args)

    def test_matched_branch_configuration_preserves_history_without_mutating_original(self):
        state = live_one_timer_state()
        original = ClassicAIV1Model(SimpleNamespace(possession_ablation='legacy'))
        original.predict_frame(state)
        before = pickle.dumps(original)
        branch = configured(original, 'continuations-risk')
        self.assertEqual(branch.one_timer.pending, original.one_timer.pending)
        self.assertEqual(branch.scheduler.frames, original.scheduler.frames)
        self.assertEqual(branch.buttons, original.buttons)
        self.assertTrue(branch.offense.rank_pass_risk)
        self.assertEqual(pickle.dumps(original), before)
        followup = common_continuation(branch)
        self.assertIsNone(followup.possession_ablation)
        self.assertFalse(followup.offense.rank_pass_risk)
        self.assertEqual(followup.one_timer.pending, branch.one_timer.pending)
        self.assertEqual(followup.scheduler.frames, branch.scheduler.frames)
        self.assertEqual(followup.buttons, branch.buttons)

    def test_local_summary_counts_each_common_state_once_per_arm(self):
        branch = {'outcome': {'goals_for': 1, 'goals_against': 0, 'shots': 2, 'one_timers': 1,
                              'possession': {'ordinary_turnovers': 0}}}
        rows = [{'cases': [{'branches': {profile: deepcopy(branch) for profile in PROFILES}}]}]
        result = summarize(rows)
        self.assertTrue(all(counts == dict(cases=1, goals_for=1, goals_against=0, shots=2,
                                         one_timers=1, ordinary_turnovers=0) for counts in result.values()))
