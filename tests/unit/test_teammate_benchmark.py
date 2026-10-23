"""Matched-pair rejection and honest uncertainty/denominator regression cases."""
from copy import deepcopy
import unittest
from argparse import Namespace

import numpy as np

from nhl94_ai.evaluation.teammate_benchmark import MATCH_KEYS, compare, paired_interval, normalized_settings, outcomes, validate_report, digest_json
from nhl94_ai.evaluation.teammate_diagnostics import TeammateDiagnostics
from tests.support.offense import offense_state


def matrix():
    result = {key: 0 for key in MATCH_KEYS}
    result.update(completed=True, seconds=120, requested_periods=1, side='home', score=[1, 0],
                  shots=[3, 1], one_timer_attempts=1, one_timer_goals=0, skater_turnovers=3,
                  reception_outcomes={'receptions': 2}, teammate_diagnostics={'counts': {'danger_entries_against': 1}},
                  actions_sha256='input', final_ram_sha256='ram')
    return {'matches': {f'two-home-{seed}': {'result': {**result, 'seed': seed}, 'settings': {}} for seed in range(10)}}


class PairedBenchmarkTests(unittest.TestCase):
    def test_unchanged_is_parity_not_a_gain(self):
        report = compare(matrix(), matrix())['groups']['two-home']
        self.assertEqual(report['status'], 'pass')
        self.assertFalse(report['gain'])
        self.assertTrue(report['exact_inputs'])
        self.assertEqual(report['delta']['goal_difference']['ci95'], [0, 0])

    def test_incomplete_unmatched_and_missing_pairs_are_rejected(self):
        for field in ('initial_ram_sha256', 'lineup', 'frame_skip', 'rom_sha256', 'side', 'seconds', 'requested_periods', 'seed'):
            after = deepcopy(matrix())
            after['matches']['two-home-0']['result'][field] = 'mismatched'
            with self.subTest(field=field), self.assertRaises(ValueError):
                compare(matrix(), after)
        after = matrix()
        after['matches']['two-home-0']['result']['completed'] = False
        with self.assertRaises(ValueError):
            compare(matrix(), after)
        after = matrix()
        del after['matches']['two-home-0']
        with self.assertRaises(ValueError):
            compare(matrix(), after)

    def test_offense_gain_cannot_offset_large_defensive_loss(self):
        after = matrix()
        for row in after['matches'].values():
            row['result']['score'] = [3, 1]
        result = compare(matrix(), after)['groups']['two-home']
        self.assertEqual(result['status'], 'regression')
        self.assertFalse(result['gain'])

    def test_small_noisy_sample_is_not_declared_a_gain(self):
        interval = paired_interval([0, 0, 1, -2, 0, 3, 0, -1, 0, 1])
        self.assertLess(interval['ci95'][0], 0)
        self.assertGreater(interval['ci95'][1], 0)
        with self.assertRaises(ValueError):
            paired_interval([1])


class LiveDiagnosticsTests(unittest.TestCase):
    def test_stoppages_highlights_goalies_and_edges_have_distinct_denominators(self):
        state = offense_state()
        state.time = 100
        metrics = TeammateDiagnostics()
        diagnostics = {'controllers': [{'controller': 1, 'slot': 0, 'role': 'carrier',
                                        'decision_tick': 1, 'b_intent': 'pass'}]}
        action = np.zeros(12, dtype=np.int8)
        action[0] = 1
        metrics.observe(1, state, diagnostics, action)
        metrics.observe(2, state, diagnostics, action)
        state.engine.clock_stopped = True
        metrics.observe(3, state, diagnostics, action)
        state.engine.clock_stopped = False
        metrics.observe(4, state, diagnostics, action, highlight=True)
        state.engine.puck_owner = 5
        metrics.observe(5, state, diagnostics, np.zeros(12))
        result = metrics.summary()
        self.assertEqual(result['counts']['live_frames'], 3)
        self.assertEqual(result['live_skater_possession_share'], 2/3)
        self.assertEqual(result['live_goalie_possession_share'], 1/3)
        self.assertEqual(result['applied_edges']['b:pass'], 1)
        self.assertEqual(result['requested_b_decisions']['pass'], 1)

    def test_goalie_owner_exit_and_actual_receiver_work_without_pass_counter(self):
        state = offense_state()
        state.time = 100
        metrics = TeammateDiagnostics()
        diagnostics = {'controllers': [{'controller': 1, 'slot': 5, 'role': 'goalie-outlet'}]}
        for frame, owner in enumerate((5, 5, -1, 2), 1):
            state.engine.puck_owner = owner
            metrics.observe(frame, state, diagnostics, np.zeros(12))
        event = metrics.summary()['goalie_releases'][0]
        self.assertEqual((event['release_frame'], event['reception_frame'], event['receiver']), (3, 4, 2))
        self.assertEqual(event['outcome'], 'received')
        self.assertEqual(state.team1.pass_attempts, 0)


class RegulationContractTests(unittest.TestCase):
    def args(self, **overrides):
        values = dict(seeds=None, seed=20000, count=40, periods=3, seconds=300,
                      frame_skip=4, workers=2, variant=None, state='save')
        return Namespace(**(values | overrides))

    def test_common_regulation_and_explicit_seeds(self):
        seeds, common, variant = normalized_settings(self.args())
        self.assertEqual(len(seeds), 40)
        self.assertEqual(common['periods'], 3)
        self.assertEqual(common['seconds'], 300)
        self.assertEqual(normalized_settings(self.args(seeds=[9, 2]))[0], [9, 2])
        for variant in ({'periods': 3}, {'frame_skip': 1}, {'seconds': 300}, {'strategy': 'single'}, {'team_size': 3}):
            import json
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                normalized_settings(self.args(variant=json.dumps(variant)))
        for values in ({'count': 0}, {'seeds': [1, 1]}, {'periods': 3, 'seconds': 120}, {'frame_skip': 0}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                normalized_settings(self.args(**values))

    def test_totals_are_distinct_from_rates_and_intervals_use_actual_df(self):
        row = next(iter(matrix()['matches'].values()))['result']
        row.update(seconds=300, requested_periods=3, score=[15, 3])
        self.assertEqual(outcomes(row, 'game')['goals_for'], 15)
        self.assertEqual(outcomes(row, 'per120')['goals_for'], 2)
        from scipy.stats import t
        ordinary = paired_interval(list(range(80)))
        simultaneous = paired_interval(list(range(80)), 1 - .05/24)
        self.assertAlmostEqual(ordinary['critical'], t.ppf(.975, 79))
        self.assertGreater(simultaneous['critical'], ordinary['critical'])
        self.assertNotIn('ci95', simultaneous)

    def test_rate_display_cannot_relax_regulation_defense_gate(self):
        before, after = matrix(), matrix()
        for report in (before, after):
            for row in report['matches'].values():
                row['result'].update(seconds=300, requested_periods=3)
        for row in after['matches'].values():
            row['result']['score'] = [4, 1]
        for unit in ('game', 'per120'):
            group = compare(before, after, unit=unit)['groups']['two-home']
            self.assertEqual(group['status'], 'regression')
            self.assertFalse(group['gain'])
            self.assertEqual(group['threshold_basis'], 'regulation per900')

    def test_unknown_units_are_rejected(self):
        with self.assertRaises(ValueError):
            outcomes(next(iter(matrix()['matches'].values()))['result'], 'per900')

    def test_v2_resume_contract_rejects_policy_harness_seed_or_completion_changes(self):
        settings = dict(state='save', seconds=300, periods=3, frame_skip=4)
        report = dict(protocol='teammate-paired-v2', sources={'a': 'policy'}, harness_sources={'b': 'harness'},
                      seeds=[1], profiles=['single'], sides=['home'], settings=settings, variant={})
        contract = digest_json(report)
        report['matches'] = {'single-home-1': {'contract_sha256': contract,
            'settings': dict(settings, strategy='single', team_size=2, side='home', seed=1),
            'result': dict(completed=True, final_clock=0, final_period=3, periods=[{}, {}, {}],
                           seed=1, side='home', seconds=300, frame_skip=4, requested_periods=3, controllers=1)}}
        validate_report(report)
        for key, value in (('sources', {'a': 'new'}), ('harness_sources', {'b': 'new'}), ('seeds', [2]),
                           ('settings', settings | {'frame_skip': 1}), ('settings', settings | {'seconds': 120})):
            broken = deepcopy(report); broken[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_report(broken)
        broken = deepcopy(report); broken['matches']['single-home-1']['result']['completed'] = False
        with self.assertRaises(ValueError):
            validate_report(broken)
