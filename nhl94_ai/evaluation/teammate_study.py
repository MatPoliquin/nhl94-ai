"""Regulation study summaries and variance-only, fixed-budget power planning."""
import argparse
import json
import math
from pathlib import Path
import statistics

import numpy as np
from scipy.stats import norm, t

from nhl94_ai.evaluation.teammate_benchmark import MATCH_KEYS, compare, outcomes, paired_interval, validate_report

PRIMARY = {'goal_difference': .50, 'goals_against': .35, 'danger_entries_against': 1.0}
CONFIDENCE = 1 - .05/24


def load(path):
    report = json.loads(Path(path).read_text(encoding='utf-8'))
    validate_report(report)
    return report


def summarize(report):
    groups = {}
    for name, row in sorted(report['matches'].items()):
        groups.setdefault(name.rsplit('-', 1)[0], []).append(row['result'])
    summary = {}
    for key, rows in groups.items():
        metrics = {m: paired_interval([outcomes(r, 'game')[m] for r in rows]) for m in outcomes(rows[0])}
        mix = []
        for row in rows:
            events = row['pass_attacks']
            controlled = [e for e in events if e['receiver'] in e['selected_at_release']]
            cpu = [e for e in events if e['receiver'] is not None and e['receiver'] >= 0
                   and e['receiver'] not in e['selected_at_release']]
            mix.append({'seed': row['seed'], 'native_attempts': row['pass_attempts'], 'tracked': len(events),
                        'untracked_native_difference': row['pass_attempts'] - len(events),
                        'controlled_passes': len(controlled), 'cpu_passes': len(cpu),
                        'controlled_receptions': sum(e['received_frame'] is not None for e in controlled),
                        'cpu_receptions': sum(e['received_frame'] is not None for e in cpu),
                        'ambiguous_targets': len(events)-len(controlled)-len(cpu),
                        'completion_fraction': sum(e['received_frame'] is not None for e in events)/len(events) if events else None,
                        'goalie_releases': row['teammate_diagnostics']['goalie_releases']})
        periods = []
        for i in range(rows[0]['requested_periods']):
            for row in rows:
                assert sum(p['native_totals']['score'][0] for p in row['periods']) == row['score'][0]
                assert sum(p['native_totals']['shots'][0] for p in row['periods']) == row['shots'][0]
            values = [r['periods'][i]['native_totals'] for r in rows]
            periods.append({m: paired_interval([v[m] for v in values]) for m in ('pass_attempts', 'one_timer_attempts', 'one_timer_goals')} |
                           {m: paired_interval([v[field][side] for v in values]) for m, field, side in
                            (('goals_for', 'score', 0), ('goals_against', 'score', 1), ('shots_for', 'shots', 0), ('shots_against', 'shots', 1))} |
                           {'goal_difference': paired_interval([v['score'][0]-v['score'][1] for v in values])})
        summary[key] = {'n': len(rows), 'game_totals': metrics,
                        'per120_means': {m: statistics.mean(outcomes(r)[m] for r in rows) for m in metrics},
                        'periods': periods, 'pass_denominators_by_game': mix,
                        'live_possession_by_game': [{'seed': r['seed'], **{k: r['teammate_diagnostics'][k] for k in
                                                    ('counts', 'live_skater_possession_share', 'live_goalie_possession_share')}} for r in rows]}
    return summary


def cross_layout(report):
    """Seed matching across layouts is contextual, distinct from policy pairing."""
    groups = {}
    excluded = {'controllers', 'strategy', 'initial_ram_sha256'}
    # Reset controller layout changes RAM, but the original save asset must match.
    keys = [k for k in MATCH_KEYS if k not in excluded] + ['state']
    for profile in ('two', 'three'):
        for side in ('home', 'away'):
            pairs = [(report['matches'][f'single-{side}-{seed}']['result'],
                      report['matches'][f'{profile}-{side}-{seed}']['result']) for seed in report['seeds']]
            if any(any(a[k] != b[k] for k in keys) for a, b in pairs):
                raise ValueError('Unmatched cross-layout assets, ratings or settings')
            delta = {m: paired_interval([outcomes(b, 'game')[m]-outcomes(a, 'game')[m] for a, b in pairs])
                     for m in outcomes(pairs[0][0])}
            periods = [{m: paired_interval([b['periods'][i]['native_totals']['score'][index] -
                                            a['periods'][i]['native_totals']['score'][index] for a, b in pairs])
                        for m, index in (('goals_for', 0), ('goals_against', 1))} for i in range(3)]
            for i, period in enumerate(periods):
                period['goal_difference'] = paired_interval([
                    (b['periods'][i]['native_totals']['score'][0] - b['periods'][i]['native_totals']['score'][1])
                    - (a['periods'][i]['native_totals']['score'][0] - a['periods'][i]['native_totals']['score'][1])
                    for a, b in pairs])
            groups[f'{profile}-{side}'] = {'delta_team_minus_single': delta, 'periods': periods}
    return {'comparison': 'cross-layout context; only initial RAM/controller layout may differ', 'groups': groups}


def covariance(before, after):
    result = {}
    for profile in ('single', 'two', 'three'):
        for side in ('home', 'away'):
            values = [[outcomes(report['matches'][f'{profile}-{side}-{seed}']['result'], 'game')['goal_difference']
                       for seed in before['seeds']] for report in (before, after)]
            a, b = values
            cov = statistics.covariance(a, b)
            va, vb = statistics.variance(a), statistics.variance(b)
            result[f'{profile}-{side}'] = {'covariance': cov, 'correlation': cov / math.sqrt(va*vb) if va*vb else None,
                                         'paired_to_unpaired_variance': statistics.variance([y-x for x, y in zip(a,b)])/(va+vb) if va+vb else None}
    return result


def power_plan(original, default, candidate, *, simulations=5000):
    """Means never enter sizing; resample centered, inflated whole-seed vectors."""
    comparisons = [compare(ref, candidate, unit='game', confidence=CONFIDENCE) for ref in (original, default)]
    rows, columns = [], []
    critical = norm.ppf(1 - .05/(2*24)) + norm.ppf(.80)
    for reference, comparison in zip(('O', 'D'), comparisons):
        for layout, group in sorted(comparison['groups'].items()):
            if layout.startswith('single'):
                continue
            for endpoint, distance in PRIMARY.items():
                values = np.array(group['delta'][endpoint]['pairs'], dtype=float)
                sd = float(values.std(ddof=1))
                needed = math.ceil((critical*1.25*sd/distance)**2)
                rows.append({'reference': reference, 'layout': layout, 'endpoint': endpoint,
                             'pilot_sd': sd, 'planning_sd': 1.25*sd, 'distance': distance,
                             'normal_n': needed, 'zero_variance_uncertain': sd == 0})
                columns.append((values-values.mean())*1.25)
    assert len(rows) == 24
    proposed = max(80, math.ceil(max(r['normal_n'] for r in rows)/20)*20)
    credible = all(not r['zero_variance_uncertain'] for r in rows)
    # Simulation checks the rounded proposal even when over budget; no games are dispatched.
    data = np.stack(columns, axis=1)
    # GD tests a lower bound; GA/danger test upper bounds. Centered empirical
    # errors need not be symmetric, so their signs are not interchangeable.
    directions = np.array([1 if r['endpoint'] == 'goal_difference' else -1 for r in rows])
    rng = np.random.default_rng(20261022)
    successes = np.zeros(24)
    for start in range(0, simulations, 100):
        count = min(100, simulations-start)
        samples = data[rng.integers(0, len(data), (count, proposed))]
        distance = np.array([r['distance'] for r in rows])
        statistic = (directions*samples.mean(axis=1)+distance)/(samples.std(axis=1, ddof=1)/math.sqrt(proposed)+1e-100)
        successes += (statistic > t.ppf((1+CONFIDENCE)/2, proposed-1)).sum(axis=0)
    for row, successes_count in zip(rows, successes):
        row['simulated_marginal_power'] = successes_count/simulations
    adequate = all(r['simulated_marginal_power'] >= .80 for r in rows)
    # No outcome-driven increase: if approximation fails simulation, report unresolved sizing.
    allowed = credible and proposed <= 400 and adequate
    timing = [r['result']['elapsed_seconds'] for report in (original, default, candidate) for r in report['matches'].values()]
    return {'protocol': 'variance-only-25pct-inflation-whole-seed-bootstrap-v2', 'pilot_excluded_from_confirmation': True,
            'confidence': CONFIDENCE, 'rows': rows, 'normal_rounded_n': proposed, 'confirmation_n': proposed if allowed else None,
            'launch_allowed': allowed, 'simulation_replicates': simulations, 'simulation_seed': 20261022,
            'minimum_simulated_marginal_power': min(r['simulated_marginal_power'] for r in rows),
            'zero_variance_blocks_sizing': not credible, 'over_budget': proposed > 400,
            'seconds_per_game_mean_under_pilot_concurrency': statistics.mean(timing),
            'confirmation_games_if_launched': 18*proposed if allowed else 0,
            'limitations': '20 seeds, discrete outcomes and estimated variance; 80% marginal power is not joint gate power. Zero sample variance is not population equivalence.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--default', type=Path, required=True)
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    original, default = load(args.original), load(args.default)
    if args.pilot:
        result = power_plan(original, default, load(args.candidate))
    else:
        result = {'O': summarize(original), 'D': summarize(default),
                  'D_minus_O': compare(original, default, unit='game'),
                  'pairing': covariance(original, default),
                  'O_team_vs_single': cross_layout(original), 'D_team_vs_single': cross_layout(default)}
        if args.candidate:
            candidate = load(args.candidate)
            result.update(C=summarize(candidate), C_minus_D=compare(default, candidate, unit='game'),
                          C_minus_O=compare(original, candidate, unit='game'))
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
