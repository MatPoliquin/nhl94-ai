"""Matched fixture/seed regression evidence; incomplete matrices fail closed."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from nhl94_ai.evaluation.fixture_regression import GROUPS


TEAMS = {0: 'Anaheim', 2: 'Buffalo', 11: 'Montreal', 15: 'Ottawa',
         17: 'Pittsburgh', 18: 'Quebec', 27: 'Campbell All-Stars'}


def validate_reports(reports, *, allow_failures=False, groups=('off', 'selective')):
    if len(reports) != len(groups) or {report['fixture_group'] for report in reports} != set(groups):
        raise ValueError('Each revision requires all declared fixture groups.')
    revisions = {report['revision'] for report in reports}
    if len(revisions) != 1:
        raise ValueError('Do not mix policy revisions within a cohort.')
    rows, identities, sources, failures = {}, {}, {}, {}
    seeds = set(range(20263001, 20263021))
    for report in reports:
        settings = report['settings']
        group = report['fixture_group']
        expected_settings = {'agent': 'classic-v1', 'trials': 20, 'seed': 20263001, 'seconds': 300,
                             'frame_skip': 4, 'action_type': 'FILTERED', 'goalie_policy': group}
        if (report['protocol'] != 'nhl94-cpu-first-period-v2'
                or any(settings.get(key) != value for key, value in expected_settings.items())
                or set(settings['matchups']) != set(GROUPS[group])):
            raise ValueError('Reports do not implement the declared fixture regression protocol.')
        if any(settings.get(flag, False) for flag in (
                'cross_crease', 'deke', 'uncertain_carry', 'chance_creation', 'offense_lookahead')):
            raise ValueError('Experimental tactics must stay off in the default-policy regression.')
        for name, digest in report['sources'].items():
            if name in sources and sources[name] != digest:
                raise ValueError('Policy sources differ between fixture groups.')
            sources[name] = digest
        identities.update(report['starting_fixtures'])
        expected = {(name, seed) for name in GROUPS[group] for seed in seeds}
        failed = report.get('failures', ())
        if failed and not allow_failures:
            raise ValueError('Runtime failures block a complete-period comparison; explicitly include their failed status.')
        failed_keys = {(row['matchup'], row['seed']) for row in failed}
        if len(failed_keys) != len(failed):
            raise ValueError('Duplicate runtime failure records.')
        if ({(row['matchup'], row['seed']) for row in report['matches']} | failed_keys) != expected:
            raise ValueError('Missing matchup/seed coverage.')
        for row in report['matches']:
            key = row['matchup'], row['seed']
            identity = report['starting_fixtures'][row['matchup']]
            if key in rows:
                raise ValueError('Duplicate fixture trial.')
            expected_fields = {'side': identity['side'], 'goalie_policy': group, 'teams': identity['teams'],
                               'lineup': identity['lineup'], 'initial_state_sha256': identity['save_sha256']}
            if (not row['completed'] or row['clock_remaining'] != 0 or row['inactive_skater_frames'] != 0
                    or any(row[key] != value for key, value in expected_fields.items())):
                raise ValueError('Incomplete period or incompatible physical fixture.')
            if row['one_timer_accounting']['started'] != row['one_timer_accounting']['ended']:
                raise ValueError('Unbalanced native one-timer lifecycle.')
            rows[key] = row
        for failure in failed:
            key = failure['matchup'], failure['seed']
            if key in rows or key in failures or not failure.get('error_type') or not failure.get('error'):
                raise ValueError('Ambiguous or missing runtime failure evidence.')
            failures[key] = failure
    declared_trials = sum(len(GROUPS[group]) * len(seeds) for group in groups)
    if len(rows) + len(failures) != declared_trials:
        raise ValueError(f'Each revision must account for exactly {declared_trials} declared trials.')
    return revisions.pop(), rows, identities, sources, failures


def score_summary(rows):
    result = Counter(periods=len(rows))
    for row in rows:
        ai, cpu = row['side'] - 1, 2 - row['side']
        gf, ga = row['goals'][ai], row['goals'][cpu]
        result.update({
            'goals_for': gf, 'goals_against': ga, 'goal_difference': gf - ga,
            'wins': int(gf > ga), 'draws': int(gf == ga), 'losses': int(gf < ga),
            'scoreless_periods': int(gf == 0),
            'recorded_shots': row['shots'][ai], 'opponent_shots': row['shots'][cpu],
            'native_one_timers': row['one_timers'][ai], 'native_one_timer_goals': row['one_timer_goals'][ai],
            'ordinary_turnovers': row['offense_metrics']['turnovers'],
            'zone_turnovers': row['offense_metrics']['zone_turnovers'],
            'goalie_contact_impulses': row['offense_metrics']['goalie_contact_impulses'],
        })
    return dict(result)


def paired_summary(pairs):
    before = score_summary([pair[0] for pair in pairs])
    after = score_summary([pair[1] for pair in pairs])
    clustered = defaultdict(list)
    for old, new in pairs:
        ai, cpu = new['side'] - 1, 2 - new['side']
        delta = (new['goals'][ai] - new['goals'][cpu]) - (old['goals'][ai] - old['goals'][cpu])
        clustered[new['seed']].append(delta)
    cluster_sums = np.asarray([sum(clustered[seed]) for seed in sorted(clustered)])
    cluster_counts = np.asarray([len(clustered[seed]) for seed in sorted(clustered)])
    rng = np.random.default_rng(94)
    draws = rng.integers(0, len(cluster_sums), (10000, len(cluster_sums)))
    interval = np.percentile(np.sum(cluster_sums[draws], axis=1) / np.sum(cluster_counts[draws], axis=1), (2.5, 97.5))
    delta = {key: after[key] - before[key] for key in before}
    return {
        'baseline': before, 'current': after, 'delta': delta,
        'paired_periods': len(pairs), 'seed_clusters': len(cluster_sums),
        'changed_input_periods': sum(old['actions_sha256'] != new['actions_sha256'] for old, new in pairs),
        'mean_goal_difference_change': delta['goal_difference'] / len(pairs),
        'goal_difference_seed_bootstrap_95': interval.tolist(),
        'observed_score_regression': delta['goal_difference'] < 0,
        'pointwise_interval_below_zero': bool(interval[1] < 0),
        'goalie_metrics': {
            label: dict(sum((Counter(row.get('goalie_metrics', {})) for row in rows), Counter()))
            for label, rows in (
                ('baseline', [old for old, _ in pairs]), ('current', [new for _, new in pairs]))},
    }


def compare(baseline_reports, current_reports, *, allow_failures=False, selective_only=False):
    groups = ('selective',) if selective_only else ('off', 'selective')
    old_revision, old, old_fixtures, old_sources, old_failures = validate_reports(
        baseline_reports, allow_failures=allow_failures, groups=groups)
    new_revision, new, new_fixtures, new_sources, new_failures = validate_reports(
        current_reports, allow_failures=allow_failures, groups=groups)
    if (set(old) | set(old_failures)) != (set(new) | set(new_failures)) or old_fixtures != new_fixtures:
        raise ValueError('Cannot compare different teams, saved starts, rosters, ratings, ROMs or goalie settings.')
    if old_revision == new_revision:
        raise ValueError('Regression comparison requires distinct revisions.')
    pairs = [(old[key], new[key]) for key in sorted(old.keys() & new.keys())]
    matchup_groups, team_groups, policy_groups = defaultdict(list), defaultdict(list), defaultdict(list)
    for pair in pairs:
        row = pair[1]
        matchup_groups[row['matchup']].append(pair)
        team = row['teams'][row['side'] - 1]
        if team not in TEAMS:
            raise ValueError(f'Unexpected team ID in verified fixtures: {team}')
        team_groups[TEAMS[team]].append(pair)
        policy_groups[row['goalie_policy']].append(pair)
    return {
        'protocol': 'nhl94-verified-fixture-revision-regression-v1',
        'baseline_revision': old_revision, 'current_revision': new_revision,
        'attempted_periods': len(old) + len(new) + len(old_failures) + len(new_failures),
        'periods': len(old) + len(new), 'paired_periods': len(pairs),
        'seeds': list(range(20263001, 20263021)),
        'settings': {'seconds': 300, 'frame_skip': 4, 'action_type': 'FILTERED',
                     'experimental_tactics': 'off',
                     'goalie_policy': 'selective' if selective_only else 'per-compatible-fixture'},
        'sources': {'baseline': old_sources, 'current': new_sources},
        'starting_fixtures': new_fixtures,
        'complete_matrix': not old_failures and not new_failures,
        'runtime_regression': len(new_failures) > len(old_failures),
        'failures': {'baseline': list(old_failures.values()), 'current': list(new_failures.values())},
        'cohort_completed_periods': {'baseline': len(old), 'current': len(new)},
        'cohort_all_completed_scores': {'baseline': score_summary(list(old.values())),
                                        'current': score_summary(list(new.values()))},
        'combined': paired_summary(pairs),
        'by_matchup': {name: paired_summary(values) for name, values in matchup_groups.items()},
        'by_team': {name: paired_summary(values) for name, values in team_groups.items()},
        'by_goalie_policy': {name: paired_summary(values) for name, values in policy_groups.items()},
        'bootstrap': {'unit': 'ROM seed, retaining all paired fixtures in each draw',
                      'resamples': 10000, 'seed': 94, 'intervals': 'pointwise percentile, not multiplicity adjusted'},
        'limitations': [
            'Verified installed fixtures only; this is not all-26-team coverage.',
            'First-period GF/GA and W/D/L are not full-game win rates.',
            'Fixed saved rosters/effective ratings; ROM reseeding does not regenerate saved hot/cold tables.',
            'A negative paired point estimate is observed downside even when its interval includes zero.',
            'Intervals are exploratory, pointwise and not adjusted for multiple comparisons.',
            'No equivalence/noninferiority margin was declared; absence of significance does not prove no regression.',
            'Seed bootstrapping cannot discover harmful states absent from the sampled fixture starts.',
            'Team changes also change opponents; individual rating effects are not isolated.',
            'Default tactics only; opt-in chance creation, deke, cross-crease and uncertain carry are not evaluated.',
            'Score comparisons use only pairs completed by both revisions; failures are separate, never scored as zero.',
            'If failures occur, unequal completed-period cohort totals are not a matched strength comparison.',
        ],
    }


def run(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f'Preserve the existing comparison: {output}')
    paths = [*args.baseline, *args.current]
    reports = [json.loads(Path(name).read_text(encoding='utf-8')) for name in paths]
    split = len(args.baseline)
    result = compare(reports[:split], reports[split:], allow_failures=args.include_failures,
                     selective_only=args.selective_only)
    gates = [json.loads(Path(name).read_text(encoding='utf-8')) for name in args.gates]
    policies = ('selective',) if args.selective_only else ('off', 'selective')
    if len(gates) != 2 * len(policies):
        raise ValueError('Supply baseline/current native gates for every declared goalie policy.')
    covered = set()
    for gate in gates:
        if not gate.get('cadence_gate', {}).get('passed'):
            raise ValueError('A required native cadence gate failed.')
        revision = gate['revision']
        cohort = 'baseline' if revision == result['baseline_revision'] else (
            'current' if revision == result['current_revision'] else None)
        if cohort is None or any(result['sources'][cohort].get(name) != digest
                                 for name, digest in gate['sources'].items()):
            raise ValueError('Cadence gate does not describe the measured policy revision/sources.')
        covered.add((cohort, gate['settings']['goalie_policy']))
    if covered != {(revision, policy) for revision in ('baseline', 'current') for policy in policies}:
        raise ValueError('Cadence gates do not cover both revision/goalie-policy combinations.')
    result['cadence_gates'] = [
        {'revision': gate['revision'], 'goalie_policy': gate['settings']['goalie_policy'],
         **gate['cadence_gate']} for gate in gates]
    result['reports'] = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in paths}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'combined': result['combined'], 'by_team': result['by_team']}, indent=2))
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', nargs='+', required=True)
    parser.add_argument('--current', nargs='+', required=True)
    parser.add_argument('--gates', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--include-failures', action='store_true',
                        help='Report recorded runtime failures explicitly and compare only completed pairs.')
    parser.add_argument('--selective-only', action='store_true',
                        help='Compare only BUF–ANA and ANA–Campbell with selective goalies on both revisions.')
    return parser


if __name__ == '__main__':
    run(build_parser().parse_args())
