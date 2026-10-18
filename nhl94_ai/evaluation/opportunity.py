"""Report opportunity creation separately from training reward and goals."""
from collections import Counter

import numpy as np


def opportunity_report(results, seed=94):
    summary, paired = {}, {}
    for agent, opponent in sorted({(r['agent'], r['opponent']) for r in results}):
        rows = [r for r in results if r['agent'] == agent and r['opponent'] == opponent and r['completed']]
        counts = Counter()
        for row in rows:
            counts.update({k: row['opportunity'][k] for k in (
                'setups', 'confirmed_releases', 'native_one_timers', 'one_timer_goals', 'recorded_shots',
                'movement_decisions', 'eligible_decisions')})
        summary[f'{agent}:{opponent}'] = dict(
            attempts=len(rows), totals=dict(counts), turnovers=sum(r['outcome'] == 'turnover' for r in rows),
            attempts_with_setup=sum(r['opportunity']['setups'] > 0 for r in rows),
            attempts_with_release=sum(r['opportunity']['confirmed_releases'] > 0 for r in rows),
            setup_quality_sum=sum(r['opportunity']['setup_quality_sum'] for r in rows),
            initial_openings=sum(r['opportunity']['initial_quality'] > 0 for r in rows),
            created_from_no_opening=sum(r['opportunity']['initial_quality'] == 0
                                        and r['opportunity']['setups'] > 0 for r in rows),
            mean_reward=float(np.mean([r['reward'] for r in rows])) if rows else None)
        # Timing-aligned Classic teacher is the movement comparator; unmodified
        # Classic is also reported by attempt_report's goal comparison.
        baseline = {r['snapshot']: r for r in results if r['agent'] == 'teacher'
                    and r['opponent'] == opponent and r['completed']}
        if agent == 'teacher' or not baseline:
            continue
        for metric in ('setups', 'confirmed_releases', 'goals'):
            clusters = {}
            for row in rows:
                if row['snapshot'] not in baseline:
                    continue
                ref = baseline[row['snapshot']]
                delta = (int(row['success']) - int(ref['success']) if metric == 'goals' else
                         int(row['opportunity'][metric] > 0) - int(ref['opportunity'][metric] > 0))
                clusters.setdefault((row['state'], row['seed']), []).append(delta)
            if clusters:
                sums = np.asarray([sum(v) for v in clusters.values()])
                counts = np.asarray([len(v) for v in clusters.values()])
                indices = np.random.default_rng(seed).integers(len(sums), size=(10000, len(sums)))
                samples = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
                paired[f'{agent}:{opponent}:{metric}'] = dict(
                    rate_delta=float(sums.sum() / counts.sum()), source_games=len(clusters),
                    interval95=np.quantile(samples, [0.025, 0.975]).tolist())
    return dict(task='classic-opportunity-carry', opportunities=summary, paired_vs_teacher=paired)
