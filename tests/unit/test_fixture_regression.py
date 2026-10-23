"""Matched seed/fixture coverage is required before interpreting regression scores."""
from copy import deepcopy
import unittest

from nhl94_ai.evaluation.fixture_regression import GROUPS
from nhl94_ai.evaluation.fixture_regression_compare import compare, validate_reports


PHYSICAL = {
    'penguins-senators': ([17, 15], 1), 'senators-penguins': ([17, 15], 2),
    'canadiens-nordiques': ([11, 18], 1), 'nordiques-canadiens': ([11, 18], 2),
    'sabres-ducks-manual': ([2, 0], 1), 'ducks-sabres-manual': ([2, 0], 2),
    'ducks-campbell-manual': ([0, 27], 1), 'campbell-ducks-manual': ([0, 27], 2),
}


def reports(revision):
    result = []
    for group in ('off', 'selective'):
        fixtures, rows = {}, []
        for name in GROUPS[group]:
            teams, side = PHYSICAL[name]
            fixtures[name] = {
                'matchup': name, 'teams': teams, 'side': side, 'goalie_policy': group,
                'lineup': [], 'save_sha256': 'same-save', 'rom_sha256': 'same-ROM',
                'effective_attributes': [[20] * 13 for _ in range(12)],
            }
            for seed in range(20263001, 20263021):
                rows.append({
                    'matchup': name, 'seed': seed, 'side': side, 'teams': teams,
                    'goalie_policy': group, 'lineup': [], 'initial_state_sha256': 'same-save',
                    'completed': True, 'clock_remaining': 0, 'inactive_skater_frames': 0,
                    'goals': [0, 0], 'shots': [0, 0], 'one_timers': [0, 0], 'one_timer_goals': [0, 0],
                    'offense_metrics': {'turnovers': 0, 'zone_turnovers': 0, 'goalie_contact_impulses': 0},
                    'one_timer_accounting': {'started': 0, 'ended': 0}, 'actions_sha256': 'same-input',
                })
        result.append({
            'protocol': 'nhl94-cpu-first-period-v2', 'fixture_group': group, 'revision': revision,
            'settings': {'agent': 'classic-v1', 'trials': 20, 'seed': 20263001, 'seconds': 300,
                         'frame_skip': 4, 'action_type': 'FILTERED', 'goalie_policy': group,
                         'matchups': list(GROUPS[group])},
            'sources': {'policy.py': revision}, 'matches': rows, 'starting_fixtures': fixtures,
        })
    return result


class FixtureRegressionTests(unittest.TestCase):
    def test_complete_matrix_covers_all_verified_teams_and_both_goalie_policies(self):
        result = compare(reports('old'), reports('new'))
        self.assertEqual(result['periods'], 320)
        self.assertEqual(result['paired_periods'], 160)
        self.assertEqual(len(result['by_matchup']), 8)
        self.assertEqual(set(result['by_team']),
                         {'Pittsburgh', 'Ottawa', 'Montreal', 'Quebec', 'Buffalo', 'Anaheim', 'Campbell All-Stars'})
        self.assertEqual(result['by_team']['Anaheim']['paired_periods'], 40)
        self.assertEqual(result['by_team']['Anaheim']['seed_clusters'], 20)
        self.assertEqual(result['combined']['goal_difference_seed_bootstrap_95'], [0, 0])
        self.assertEqual(result['combined']['changed_input_periods'], 0)
        self.assertFalse(result['combined']['observed_score_regression'])

    def test_observed_downside_is_visible_per_team_not_hidden_by_aggregate(self):
        old, new = reports('old'), reports('new')
        for row in new[0]['matches']:
            if row['matchup'] == 'canadiens-nordiques':
                row['goals'][1] = 1
            if row['matchup'] == 'penguins-senators':
                row['goals'][0] = 2
        result = compare(old, new)
        self.assertEqual(result['combined']['delta']['goal_difference'], 20)
        self.assertTrue(result['by_team']['Montreal']['observed_score_regression'])
        self.assertTrue(result['by_team']['Montreal']['pointwise_interval_below_zero'])
        self.assertEqual(result['by_team']['Montreal']['goal_difference_seed_bootstrap_95'], [-1, -1])

    def test_missing_duplicate_and_incomplete_trials_fail_closed(self):
        for mutation in ('missing', 'duplicate', 'incomplete', 'inactive', 'unbalanced'):
            cohort = reports('old')
            if mutation == 'missing':
                cohort[0]['matches'].pop()
            elif mutation == 'duplicate':
                cohort[0]['matches'].append(deepcopy(cohort[0]['matches'][0]))
            elif mutation == 'incomplete':
                cohort[0]['matches'][0]['completed'] = False
            elif mutation == 'inactive':
                cohort[0]['matches'][0]['inactive_skater_frames'] = 1
            else:
                cohort[0]['matches'][0]['one_timer_accounting']['started'] = 1
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_reports(cohort)

    def test_fixture_rating_changes_experiments_and_changed_sources_are_rejected(self):
        old, new = reports('old'), reports('new')
        new[0]['starting_fixtures']['canadiens-nordiques']['effective_attributes'][0][0] = 30
        with self.assertRaisesRegex(ValueError, 'ratings'):
            compare(old, new)
        for flag in ('deke', 'cross_crease', 'chance_creation', 'uncertain_carry'):
            cohort = reports('old')
            cohort[0]['settings'][flag] = True
            with self.assertRaisesRegex(ValueError, 'Experimental'):
                validate_reports(cohort)
        cohort = reports('old')
        cohort[1]['sources']['policy.py'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'sources'):
            validate_reports(cohort)

    def test_runtime_failures_are_not_fabricated_zero_scores_or_silently_omitted(self):
        old, new = reports('old'), reports('new')
        failed = new[1]['matches'].pop()
        new[1]['failures'] = [{
            'matchup': failed['matchup'], 'seed': failed['seed'],
            'error_type': 'ValueError', 'error': 'missing native goalie feedback',
        }]
        with self.assertRaisesRegex(ValueError, 'Runtime failures'):
            compare(old, new)
        result = compare(old, new, allow_failures=True)
        self.assertTrue(result['runtime_regression'])
        self.assertFalse(result['complete_matrix'])
        self.assertEqual(result['attempted_periods'], 320)
        self.assertEqual(result['periods'], 319)
        self.assertEqual(result['paired_periods'], 159)
        self.assertEqual(result['combined']['baseline']['periods'], 159)
        self.assertEqual(result['combined']['current']['periods'], 159)
        self.assertEqual(len(result['failures']['current']), 1)

    def test_selective_only_scope_cannot_include_any_auto_goalie_fixture(self):
        old, new = reports('old'), reports('new')
        result = compare([old[1]], [new[1]], selective_only=True)
        self.assertEqual(result['attempted_periods'], 160)
        self.assertEqual(result['settings']['goalie_policy'], 'selective')
        self.assertEqual(set(result['by_team']), {'Anaheim', 'Buffalo', 'Campbell All-Stars'})
        self.assertEqual(set(result['by_goalie_policy']), {'selective'})
        with self.assertRaises(ValueError):
            compare(old, new, selective_only=True)


if __name__ == '__main__':
    unittest.main()
