"""Paired offensive-attempt results, clustered by source game."""
from collections import Counter

import numpy as np


def attempt_report(results, seed=94):
    arms = sorted({(row['agent'], row['opponent']) for row in results})
    summary, comparisons = {}, {}
    for agent, opponent in arms:
        all_rows = [r for r in results if (r['agent'], r['opponent']) == (agent, opponent)]
        rows = [r for r in all_rows if r['completed']]
        summary[f'{agent}:{opponent}'] = dict(
            attempts=len(rows), incomplete=len(all_rows) - len(rows), goals=sum(r['success'] for r in rows),
            success_rate=float(np.mean([r['success'] for r in rows])) if rows else None,
            outcomes=dict(Counter(r['outcome'] for r in rows)),
            native_frames=sum(r['attempt_frames'] for r in rows))
        baseline = {r['snapshot']: r for r in results
                    if r['agent'] == 'classic' and r['opponent'] == opponent and r['completed']}
        if agent == 'classic' or not baseline:
            continue
        clusters = {}
        for row in rows:
            if row['snapshot'] in baseline:
                key = row['state'], row['seed']
                clusters.setdefault(key, []).append(int(row['success']) - int(baseline[row['snapshot']]['success']))
        if clusters:
            sums = np.asarray([sum(v) for v in clusters.values()])
            counts = np.asarray([len(v) for v in clusters.values()])
            indices = np.random.default_rng(seed).integers(len(sums), size=(10000, len(sums)))
            samples = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
            comparisons[f'{agent}:{opponent}'] = dict(
                success_rate_delta=float(sums.sum() / counts.sum()), source_games=len(clusters),
                interval95=np.quantile(samples, [0.025, 0.975]).tolist())
    return dict(task='score-before-turnover-retreat-or-timeout', results=results,
                summary=summary, paired_vs_classic=comparisons)
