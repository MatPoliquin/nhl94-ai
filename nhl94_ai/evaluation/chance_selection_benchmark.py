"""Evaluate a frozen chance rule against Classic in complete paired periods.

An evaluation-only factory hook keeps unproven rules out of the gameplay CLI.
All match execution and metrics use the existing CPU benchmark unchanged.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from nhl94_ai.agents.registry import create_scripted
from nhl94_ai.evaluation.chance_selection import ChanceSelection
from nhl94_ai.evaluation.chance_selection_replay import source_hashes as replay_hashes
from nhl94_ai.evaluation.cpu_benchmark import cpu_match, summarize


def run_job(job):
    matchup, seed, arm, rule = job

    def factory(name, args):
        agent = create_scripted(name, args)
        if arm == 'candidate':
            agent.controller.possession_ablation = ChanceSelection(rule=rule)
        return agent

    with patch('nhl94_ai.evaluation.cpu_benchmark.create_scripted', side_effect=factory):
        row = cpu_match(('classic-v1', matchup, seed, 300, 4, 'FILTERED', 'selective'))
    row['chance_policy'] = arm
    row['chance_metrics'] = row.pop('possession_ablation_metrics')
    print(f'{arm} {matchup} {seed}: {row["goals"]}, {row["chance_metrics"]}', flush=True)
    return row


def source_hashes():
    result = replay_hashes()
    path = Path(__file__)
    result[str(path.relative_to(path.parents[2]))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rule', required=True)
    parser.add_argument('--seed', type=int, default=26001)
    parser.add_argument('--trials', type=int, default=20)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    model_bytes = Path(args.rule).read_bytes()
    model = json.loads(model_bytes)
    if (model.get('version') != 'native-chance-rule-v1' or model.get('rule') is None
            or min(args.trials, args.workers) < 1 or not 0 <= args.seed <= 2**32-args.trials):
        raise ValueError('Expected a fitted rule, positive limits and uint32 seeds')
    tested = set(range(args.seed, args.seed + args.trials))
    if tested.intersection(model['split']['training_seeds'] + model['split']['validation_seeds']):
        raise ValueError('Full-period holdout seeds must be fresh')
    sources = source_hashes()
    jobs = [(matchup, seed, arm, model['rule']) for seed in sorted(tested)
            for matchup in ('sabres-ducks-manual', 'ducks-sabres-manual')
            for arm in ('legacy', 'candidate')]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(run_job, jobs))
    if source_hashes() != sources:
        raise RuntimeError('Sources changed during chance benchmark')
    report = {'protocol': 'native-chance-rule-periods-v1', 'settings': vars(args),
              'rule_sha256': hashlib.sha256(model_bytes).hexdigest(), 'rule': model,
              'sources': sources, 'matches': rows,
              'summary': {arm: summarize([row for row in rows if row['chance_policy'] == arm])
                          for arm in ('legacy', 'candidate')}}
    data = (json.dumps(report, separators=(',', ':'))+'\n').encode()
    path = Path(args.output)
    path.write_bytes(gzip.compress(data, mtime=0) if path.suffix == '.gz' else data)


if __name__ == '__main__':
    main()
