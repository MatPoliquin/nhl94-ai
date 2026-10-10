"""Complete paired CPU periods for a frozen directional-shot candidate."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from nhl94_ai.agents.registry import create_scripted
from nhl94_ai.evaluation.cpu_benchmark import cpu_match, summarize
from nhl94_ai.evaluation.shot_direction import DirectionalShotModel, DirectionSelector
from nhl94_ai.evaluation.shot_direction_replay import source_hashes as replay_hashes


def run_job(job):
    matchup, seed, arm, model = job
    agents = []

    def factory(name, args):
        agent = create_scripted(name, args)
        if arm == 'candidate':
            agent.controller = DirectionalShotModel(args, agent.env)
            agent.controller.direction_selector = DirectionSelector(model)
        agents.append(agent)
        return agent

    with patch('nhl94_ai.evaluation.cpu_benchmark.create_scripted', side_effect=factory):
        row = cpu_match(('classic-v1', matchup, seed, 300, 4, 'FILTERED', 'selective'))
    row['direction_policy'] = arm
    row['direction_metrics'] = dict(agents[0].controller.direction_selector.metrics) if arm == 'candidate' else {}
    print(f'{arm} {matchup} {seed}: {row["goals"]}, {row["direction_metrics"]}', flush=True)
    return row


def source_hashes():
    sources = replay_hashes()
    root = Path(__file__).resolve().parents[2]
    for name in ('evaluation/shot_direction_benchmark.py', 'training/shot_direction.py'):
        path = root/'nhl94_ai'/name
        sources[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--seed', type=int, default=26301)
    parser.add_argument('--trials', type=int, default=20)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    data = Path(args.model).read_bytes()
    model = json.loads(data)
    DirectionSelector(model)
    if min(args.trials, args.workers) < 1 or not 0 <= args.seed <= 2**32-args.trials:
        raise ValueError('Use positive limits and uint32 seeds')
    seeds = set(range(args.seed, args.seed + args.trials))
    if seeds & set(model['training']['seeds'] + model['validation']['seeds']):
        raise ValueError('Complete-period holdout seeds must be fresh')
    sources = source_hashes()
    jobs = [(matchup, seed, arm, model) for seed in sorted(seeds)
            for matchup in ('sabres-ducks-manual', 'ducks-sabres-manual') for arm in ('legacy', 'candidate')]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(run_job, jobs))
    if source_hashes() != sources:
        raise RuntimeError('Sources changed during directional benchmark')
    report = {'protocol': 'native-shot-direction-periods-v1', 'settings': vars(args), 'sources': sources,
              'model': model, 'model_sha256': hashlib.sha256(data).hexdigest(), 'matches': rows,
              'summary': {arm: summarize([row for row in rows if row['direction_policy'] == arm])
                          for arm in ('legacy', 'candidate')}}
    path = Path(args.output)
    data = (json.dumps(report, separators=(',', ':'))+'\n').encode()
    path.write_bytes(gzip.compress(data, mtime=0) if path.suffix == '.gz' else data)


if __name__ == '__main__':
    main()
