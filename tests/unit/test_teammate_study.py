"""Variance-only sizing, multiplicity and zero-variance budget guards."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np

from nhl94_ai.evaluation.teammate_study import power_plan, CONFIDENCE, cross_layout
from nhl94_ai.evaluation.teammate_benchmark import MATCH_KEYS


def study_fixture(offset=0):
    result = dict.fromkeys(MATCH_KEYS, 0)
    result.update(completed=True, shots=[3, 1], one_timer_attempts=1, one_timer_goals=0, skater_turnovers=3,
                  reception_outcomes={'receptions': 2}, teammate_diagnostics={'counts': {'danger_entries_against': 1}},
                  actions_sha256='input', final_ram_sha256='ram')
    template = {'settings': {}, 'result': result}
    matches = {}
    for profile in ('single', 'two', 'three'):
        for side in ('home', 'away'):
            for seed in range(20):
                row = deepcopy(template)
                row['result'].update(seed=seed, side=side, seconds=300, requested_periods=3,
                                     score=[offset, 0], elapsed_seconds=1)
                matches[f'{profile}-{side}-{seed}'] = row
    return {'matches': matches}


class StudyPowerTests(unittest.TestCase):
    def test_zero_pilot_variance_does_not_authorize_confirmation(self):
        report = power_plan(study_fixture(), study_fixture(), study_fixture(), simulations=100)
        self.assertEqual(len(report['rows']), 24)
        self.assertAlmostEqual(report['confidence'], CONFIDENCE)
        self.assertTrue(report['zero_variance_blocks_sizing'])
        self.assertFalse(report['launch_allowed'])
        self.assertIsNone(report['confirmation_n'])

    def test_sizing_is_invariant_to_candidate_mean(self):
        base, candidate = study_fixture(), study_fixture()
        for row in candidate['matches'].values():
            seed = row['result']['seed']
            row['result']['score'] = [seed % 4, seed % 3]
            row['result']['teammate_diagnostics']['counts']['danger_entries_against'] = seed % 7
        shifted = deepcopy(candidate)
        for row in shifted['matches'].values():
            row['result']['score'][0] += 10
        a = power_plan(base, base, candidate, simulations=100)
        b = power_plan(base, base, shifted, simulations=100)
        self.assertEqual(a['rows'], b['rows'])
        self.assertEqual(a['normal_rounded_n'], b['normal_rounded_n'])
        self.assertNotIn('mean', a['rows'][0])

    def test_largest_defensive_requirement_blocks_over_budget_run(self):
        base, candidate = study_fixture(), study_fixture()
        for name, row in candidate['matches'].items():
            seed = row['result']['seed']
            row['result']['score'] = [seed % 4, seed % 3]
            row['result']['teammate_diagnostics']['counts']['danger_entries_against'] = seed % 7
            if name.startswith('three-away'):
                row['result']['teammate_diagnostics']['counts']['danger_entries_against'] = (seed % 4)*10
        report = power_plan(base, base, candidate, simulations=100)
        self.assertGreater(report['normal_rounded_n'], 400)
        self.assertEqual(report['normal_rounded_n'] % 20, 0)
        self.assertTrue(report['over_budget'])
        self.assertFalse(report['launch_allowed'])

    def test_harmful_resampled_goals_against_and_danger_do_not_count_as_power(self):
        base, candidate = study_fixture(), study_fixture()
        for row in candidate['matches'].values():
            seed = row['result']['seed']
            row['result']['score'] = [2*seed, seed]
            row['result']['teammate_diagnostics']['counts']['danger_entries_against'] = seed
        # Force a resampled study with positive GD but also harmful GA/danger.
        # The latter must fail their upper-bound guardrails, regardless of GD.
        with patch('nhl94_ai.evaluation.teammate_study.np.random.default_rng') as random:
            highest_seed_index = sorted(range(20), key=str).index(19)
            random.return_value.integers.side_effect = lambda _low, _high, shape: np.full(shape, highest_seed_index)
            report = power_plan(base, base, candidate, simulations=1)
        for row in report['rows']:
            expected = 1.0 if row['endpoint'] == 'goal_difference' else 0.0
            self.assertEqual(row['simulated_marginal_power'], expected)


class CrossLayoutAssetTests(unittest.TestCase):
    def test_original_save_mismatch_is_not_a_controller_layout_difference(self):
        report = study_fixture()
        report['seeds'] = list(range(20))
        for row in report['matches'].values():
            row['result']['state'] = 'same-named-save'
        report['matches']['two-home-0']['result']['initial_state_sha256'] = 'different-save'
        with self.assertRaises(ValueError):
            cross_layout(report)


    def test_period_goal_difference_uses_whole_seed_score_pairs(self):
        report = study_fixture(); report['seeds'] = list(range(20))
        for name, row in report['matches'].items():
            row['result']['state'] = 'same-save'
            row['result']['periods'] = [{'native_totals': {'score': [2, 1] if name.startswith('two') else [0, 0]}}
                                        for _ in range(3)]
        periods = cross_layout(report)['groups']['two-home']['periods']
        self.assertEqual([p['goal_difference']['mean'] for p in periods], [1, 1, 1])
        self.assertEqual([p['goal_difference']['n'] for p in periods], [20, 20, 20])
