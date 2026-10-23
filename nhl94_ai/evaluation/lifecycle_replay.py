"""Compare complete, isolated Classic runs across a behavior-preserving refactor.

Invoke this file directly with --root so each worker imports only that revision.
Trace files contain decoded decisions and inputs, never ROM or save-state bytes.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time


def cases():
    """Fixed matrix chosen before the lifecycle extraction."""
    rows = {}

    def add(name, matchup='ducks-sabres-manual', seed=20268001, interval=4,
            schema='FILTERED', goalie='off', **options):
        rows[name] = ('classic-v1', matchup, seed, 300, interval, schema, goalie,
                      options.get('cross_crease', False), options.get('deke', False),
                      options.get('uncertain_carry', False), options.get('chance_creation', False),
                      options.get('offense_lookahead', False), options.get('refinements', ()))

    for side, matchup in (('home', 'sabres-ducks-manual'), ('away', 'ducks-sabres-manual')):
        for seed in (20268001, 20268002):
            add(f'default-{side}-{seed}', matchup, seed)
        add(f'intents-{side}', matchup, schema='HOCKEY_INTENT_DPAD')
        add(f'goalie-{side}', matchup, goalie='selective')
    add('interval-1', interval=1)
    add('interval-8', interval=8, schema='HOCKEY_INTENT_DPAD')
    add('roster', 'penguins-senators')
    add('lookahead', offense_lookahead=True)
    add('cross-crease', cross_crease=True)
    add('deke', deke=True)
    add('refinements', refinements=('pass-timing', 'carry-motion', 'finishing', 'interceptions'))
    add('chance-creation', chance_creation=True)
    add('uncertain-carry', uncertain_carry=True)
    return rows


def fingerprints(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((root / 'nhl94_ai').rglob('*.py'))}


def measure(job):
    root, output, name, fixture = job
    sys.path.insert(0, root)
    from nhl94_ai.evaluation import cpu_benchmark as cpu
    if Path(cpu.__file__).resolve() != Path(root) / 'nhl94_ai/evaluation/cpu_benchmark.py':
        raise RuntimeError('Worker imported the wrong source tree')
    original = cpu.create_scripted
    prediction_seconds = 0.0
    frames = 0
    trace_path = Path(output) / f'{name}.jsonl.gz'
    with gzip.open(trace_path, 'wt', encoding='utf-8') as trace:
        def create(*args, **kwargs):
            agent = original(*args, **kwargs)
            predict = agent.predict_game_state

            def recorded(state, deterministic=True):
                nonlocal prediction_seconds, frames
                started = time.perf_counter()
                action = predict(state, deterministic)
                prediction_seconds += time.perf_counter() - started
                frames += 1
                trace.write(json.dumps([action.tolist(), agent._last_decision,
                                        agent._last_target], separators=(',', ':')) + '\n')
                return action

            agent.predict_game_state = recorded
            return agent

        cpu.create_scripted = create
        try:
            result = cpu.cpu_match(fixture)
        finally:
            cpu.create_scripted = original
    if not result['completed']:
        raise RuntimeError(f'{name}: incomplete period')
    return name, {'fixture': fixture, 'result': result, 'prediction_seconds': prediction_seconds,
                  'predicted_frames': frames, 'trace': trace_path.name}


def compare_reports(reference, candidate):
    """Require identical trajectories and metrics; report the first input divergence."""
    before = json.loads(reference.read_text())
    after = json.loads(candidate.read_text())
    for report in (before, after):
        if report.get('protocol') != 'classic-lifecycle-replay-v1' or not report.get('cases'):
            raise ValueError('Comparison requires nonempty lifecycle replay reports')
        if not report.get('sources'):
            raise ValueError('Source provenance is missing')
    if before['cases'].keys() != after['cases'].keys():
        raise ValueError('Replay matrices differ')
    compared = []
    for name, old in before['cases'].items():
        new = after['cases'][name]
        if old['fixture'] != new['fixture']:
            raise ValueError(f'{name}: fixture settings differ')
        with gzip.open(reference.parent / old['trace'], 'rt') as left, gzip.open(
                candidate.parent / new['trace'], 'rt') as right:
            from itertools import zip_longest
            frame = 0
            for frame, (a, b) in enumerate(zip_longest(left, right), 1):
                if a != b:
                    raise ValueError(f'{name}: first divergent frame {frame}: {a!r} -> {b!r}')
            for row in (old, new):
                result = row['result']
                if (not result['completed'] or result['clock_remaining'] != 0
                        or row['predicted_frames'] != frame or result['frames'] != frame + 1):
                    raise ValueError(f'{name}: trace does not cover the complete period')
        if old['result'] != new['result']:
            keys = [key for key in old['result'] if old['result'][key] != new['result'].get(key)]
            raise ValueError(f'{name}: outcome/metric mismatch: {keys}')
        compared.append({'case': name, 'frames': new['predicted_frames'],
                         'actions_sha256': new['result']['actions_sha256'],
                         'prediction_seconds_before': old['prediction_seconds'],
                         'prediction_seconds_after': new['prediction_seconds']})
    return {'passed': True, 'cases': compared, 'reference_sources': before['sources'],
            'candidate_sources': after['sources']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', nargs='+', choices=cases())
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--compare', nargs=2, type=Path, metavar=('REFERENCE', 'CANDIDATE'))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.compare:
        report = compare_reports(*args.compare)
        args.output.parent.mkdir(parents=True, exist_ok=True)
    else:
        if args.root is None or args.workers < 1:
            parser.error('--root and positive --workers are required for measurement')
        root = args.root.resolve()
        sources = fingerprints(root)
        args.output.mkdir(parents=True)
        matrix = cases()
        selected = args.cases or list(matrix)
        jobs = [(str(root), str(args.output.resolve()), name, matrix[name]) for name in selected]
        results = {}
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for name, row in pool.map(measure, jobs):
                results[name] = row
                print(f'{name}: {row["predicted_frames"]} frames; complete', flush=True)
        if sources != fingerprints(root):
            raise RuntimeError('Sources changed during measurement')
        report = {'protocol': 'classic-lifecycle-replay-v1', 'sources': sources, 'cases': results}
        args.output = args.output / 'report.json'
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Wrote {args.output}', flush=True)


if __name__ == '__main__':
    main()
