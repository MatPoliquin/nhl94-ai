"""Paired native periods with Classic restricted to fixed observation/input rates.

The tactical interval stays at four frames in both arms. The rate cap also
restricts defense, goalie, reception and finisher feedback: no controller runs
on held frames, and every raw button stays unchanged. Only elapsed clocks and
the existing tactical countdown advance between observations.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
import math
from pathlib import Path
from nhl94_ai.evaluation.cpu_benchmark import MATCHUPS, cpu_match, summarize


def run_job(job):
    matchup, seed, repeat, seconds = job
    row = cpu_match(('classic-v1', matchup, seed, seconds, 4, 'FILTERED', 'selective'),
                    classic_control_interval=repeat)
    rate = row['control_rate']
    if (not row['completed'] or rate['controller_observations'] != math.ceil(rate['input_frames'] / repeat)
            or rate['scheduler_frames'] != rate['input_frames'] or row['frames'] != rate['input_frames'] + 1):
        raise RuntimeError('Incomplete period or inconsistent rate/clock accounting')
    side = row['side'] - 1
    print(f'repeat={repeat} {matchup} seed={seed}: '
          f'{row["goals"][side]}-{row["goals"][1-side]}, '
          f'{row["one_timer_goals"][side]} one-timer goals', flush=True)
    return row


def source_texts():
    root = Path(__file__).resolve().parents[2]
    return {str(path.relative_to(root)): path.read_text(encoding='utf-8')
            for path in sorted((root / 'nhl94_ai').rglob('*.py'))}


def portable(value):
    """Preserve diagnostic infinities without nonstandard JSON numbers."""
    if isinstance(value, dict):
        return {key: portable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [portable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=26501)
    parser.add_argument('--trials', type=int, default=10)
    parser.add_argument('--seconds', type=int, default=300)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--repeats', type=int, nargs='+', default=[1, 4])
    parser.add_argument('--matchups', choices=MATCHUPS, nargs='+', default=[
        'sabres-ducks-manual', 'ducks-sabres-manual',
        'ducks-campbell-manual', 'campbell-ducks-manual'])
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if (min(args.trials, args.seconds, args.workers, *args.repeats) < 1
            or args.seconds > 65535 or not 0 <= args.seed <= 2**32 - args.trials
            or len(set(args.matchups)) != len(args.matchups)
            or len(set(args.repeats)) != len(args.repeats)):
        raise ValueError('Use positive counts, valid clock/seeds and distinct matchups/repeats')
    sources = source_texts()
    hashes = {name: hashlib.sha256(value.encode()).hexdigest() for name, value in sources.items()}
    jobs = [(matchup, seed, repeat, args.seconds)
            for seed in range(args.seed, args.seed + args.trials)
            for matchup in args.matchups for repeat in args.repeats]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(run_job, jobs))
    if source_texts() != sources:
        raise RuntimeError('Sources changed during control-rate benchmark')
    report = {'protocol': 'classic-control-rate-v1', 'settings': vars(args),
              'semantics': {'tactical_interval': 4, 'actions': 'raw FILTERED',
                            'hold': 'all buttons; no controller feedback between observations',
                            'clocks': 'native frames continue; tactical decisions count actual calls',
                            'observation': 'existing RAM decoder; controller reads only at rate boundary',
                            'extra_latency_frames': 0, 'retuned_for_rate': False},
              'sources': hashes, 'source_text': sources, 'matches': rows,
              'summary': {str(repeat): summarize([
                  row for row in rows if row['control_rate']['repeat'] == repeat])
                          for repeat in args.repeats}}
    data = (json.dumps(portable(report), separators=(',', ':'), allow_nan=False) + '\n').encode()
    path = Path(args.output)
    path.write_bytes(gzip.compress(data, mtime=0) if path.suffix == '.gz' else data)


if __name__ == '__main__':
    main()
