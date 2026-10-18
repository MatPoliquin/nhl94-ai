"""Natural manual-goalie starts and paired, native shot-defense evaluations."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import pickle
from unittest.mock import patch

from nhl94_ai.agents import classic_v1
from nhl94_ai.agents.goalie import goalie_target
from nhl94_ai.agents.learned_goalie import decision_available
from nhl94_ai.env import factory
from nhl94_ai.evaluation import cpu_benchmark


MATCHUPS = ('sabres-ducks-manual', 'ducks-sabres-manual',
            'ducks-campbell-manual', 'campbell-ducks-manual')


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob('*.py'))}


def load_start(path):
    """Load only locally generated, explicitly selected pickle snapshots."""
    path = Path(path)
    meta = json.loads(path.with_suffix(path.suffix + '.json').read_text(encoding='utf-8'))
    if hashlib.sha256(path.read_bytes()).hexdigest() != meta['sha256']:
        raise ValueError('Goalie snapshot checksum mismatch: ' + str(path))
    with gzip.open(path, 'rb') as stream:
        return pickle.load(stream), meta


def collect_game(job):
    matchup, seed, output, maximum, *options = job
    released_only = bool(options and options[0])
    original_factory = factory.make_retro
    original_view = cpu_benchmark.cpu_view
    original_predict = classic_v1.ClassicAIV1Model.predict_frame
    holder, rows = {}, []
    frame, last_saved = 0, -1000
    state_name, side = cpu_benchmark.MATCHUPS[matchup]

    def make_env(**kwargs):
        holder['env'] = original_factory(**kwargs)
        return holder['env']

    def view_state(state, info, view_side):
        holder['root_state'] = state
        return original_view(state, info, view_side)

    def predict(model, state, *args, **kwargs):
        nonlocal frame, last_saved
        frame += 1
        manager = model.goalie
        if (len(rows) < maximum and frame - last_saved >= 24
                and manager is not None and decision_available(state, manager)):
            plan = goalie_target(state, manager.pass_windup)
            if (plan.deadline is not None and plan.deadline <= 60
                    and (not released_only or plan.crossing_frames is not None)):
                env = holder['env']
                sample = deepcopy({'root_state': holder['root_state'], 'view': state, 'model': model})
                sample['snapshot'] = env.em.get_state()
                sample['ram_sha256'] = hashlib.sha256(bytes(env.get_ram())).hexdigest()
                name = f'{matchup}-{seed}-{frame}.pkl.gz'
                path = Path(output) / name
                with gzip.open(path, 'wb') as stream:
                    pickle.dump(sample, stream, protocol=5)
                row = {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                       'matchup': matchup, 'seed': seed, 'frame': frame, 'state': state_name, 'side': side,
                       'source': f'{state_name}:{seed}', 'released_threat': plan.crossing_frames is not None,
                       'crossing_frames': plan.crossing_frames, 'deadline': plan.deadline,
                       'ram_sha256': sample['ram_sha256']}
                path.with_suffix(path.suffix + '.json').write_text(json.dumps(row, indent=2) + '\n', encoding='utf-8')
                rows.append(row)
                last_saved = frame
        return original_predict(model, state, *args, **kwargs)

    with patch.object(factory, 'make_retro', side_effect=make_env), \
            patch.object(cpu_benchmark, 'cpu_view', side_effect=view_state), \
            patch.object(classic_v1.ClassicAIV1Model, 'predict_frame', new=predict):
        period = cpu_benchmark.cpu_match(('classic-v1', matchup, seed, 300, 4, 'FILTERED', 'selective'))
    return {'matchup': matchup, 'seed': seed, 'starts': rows, 'period': period}


def collect(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / 'bank.json'
    if manifest.exists():
        raise ValueError('Refusing to replace an existing goalie bank')
    hashes = source_hashes()
    jobs = [(matchup, seed, str(output), args.maximum, getattr(args, 'released_only', False))
            for seed in range(args.seed, args.seed + args.trials) for matchup in args.matchups]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for row in executor.map(collect_game, jobs):
            results.append(row)
            print(row['matchup'], row['seed'], 'starts', len(row['starts']), flush=True)
    if hashes != source_hashes():
        raise RuntimeError('Goalie collection source changed during measurement')
    report = {'version': 1, 'kind': 'native-selected-manual-goalie-starts',
              'options': vars(args), 'source_hashes': hashes,
              'starts': [start for row in results for start in row['starts']], 'periods': results}
    manifest.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('saved', len(report['starts']), 'starts to', manifest, flush=True)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=100000)
    parser.add_argument('--trials', type=int, default=2)
    parser.add_argument('--matchups', nargs='+', choices=MATCHUPS, default=list(MATCHUPS))
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--maximum', type=int, default=24)
    parser.add_argument('--released-only', action='store_true',
                        help='Collect already released goal threats rather than pre-release situations')
    parser.add_argument('--output', required=True)
    return parser


def run(args):
    if min(args.seed, args.trials, args.workers, args.maximum) < 1:
        raise ValueError('Use positive seeds, trials, workers and start limits')
    collect(args)


if __name__ == '__main__':
    run(build_parser().parse_args())
