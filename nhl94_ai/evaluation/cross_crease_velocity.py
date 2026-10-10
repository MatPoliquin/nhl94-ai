"""Fixed-geometry lateral/goalward velocity maps for ordinary-ROM crossings."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from nhl94_ai.env.factory import make_retro
from nhl94_ai.evaluation.benchmark import RAM
from nhl94_ai.evaluation.cross_crease_probe import ShotTiming, snapshot_start, summarize, trial
from nhl94_ai.game.ram import validate_rom_seed
from nhl94_ai.tasks.cross_crease_setup import GOALIE_LEVELS, set_crossing_velocity


LATERAL_SPEEDS = (0, 0.5, 1, 1.25, 1.5, 1.75, 2, 2.5, 3)
GOALWARD_SPEEDS = (-0.5, -0.25, 0, 0.25, 0.5)


def validate_grid(args):
    validate_rom_seed(args.seed)
    if args.seeds < 1 or args.seed + args.seeds > 2**32 or args.workers < 1:
        raise ValueError('Use a positive uint32 seed count and positive worker count.')
    for values, low, high, name in (
        (args.lateral, 0, 3, 'lateral'), (args.goalward, -1, 1, 'goalward'),
    ):
        if (not values or len(set(values)) != len(values)
                or any(not math.isfinite(value) or not low <= value <= high for value in values)):
            raise ValueError(f'{name} grid must have unique finite values in {low}..{high}.')
    if not 20 <= args.width <= 110 or not 160 <= args.depth <= 230:
        raise ValueError('Use bounded fixed width/depth for an isolated finishing grid.')
    if 'stationary' in args.policies:
        raise ValueError('Stationary control would discard the swept independent velocities.')


def safe_summary(rows):
    summary = summarize(rows)
    summary['safe_goals'] = sum(row['goal'] and row['goalie_contact_impulses'] == 0 for row in rows)
    summary['contact_trials'] = sum(row['goalie_contact_impulses'] > 0 for row in rows)
    summary['crossing_attempts'] = sum(row['controller_metrics'].get('attempts', 0) > 0 for row in rows)
    summary['crossing_goals'] = sum(row['goal'] and row['controller_metrics'].get('attempts', 0) > 0 for row in rows)
    summary['all_observed_trials_scored_safely'] = bool(rows) and summary['safe_goals'] == len(rows)
    return summary


def _fixture(options):
    args, side, direction, seed = options
    env = make_retro(game='NHL94-Genesis-v0', state='PenguinsVsSenators.DefenseZone', num_players=1)
    rows = []
    try:
        for name, (address, kind) in RAM.items():
            env.data.set_variable(name, {'address': address, 'type': kind})
        base, slots = snapshot_start(env, scenario='near', seed=seed, side=side,
                                     direction=direction, handedness=seed % 2)
        timing = ShotTiming(args.press_frame, args.hold_frames, args.aim)
        geometry = None
        for goalward in args.goalward:
            for lateral in args.lateral:
                env.em.set_state(base)
                actual = set_crossing_velocity(
                    env, slots[0], direction=direction, lateral=lateral, goalward=goalward,
                    width=args.width, depth=args.depth)
                snapshot = env.em.get_state()
                for rating in args.goalies:
                    ram_hash = None
                    for policy in args.policies:
                        row = trial(env, snapshot, carrier=slots[0], goalie=slots[1], inactive=slots[2],
                                    level=GOALIE_LEVELS[rating], direction=direction,
                                    policy=policy, timing=timing, trace=args.trace)
                        initial = row['initial']
                        observed = tuple(initial['player_velocity'])
                        if not np.allclose(actual, observed, rtol=0, atol=1e-12):
                            raise AssertionError(('Initial velocity differs from requested raw encoding', actual, observed))
                        positions = tuple(initial['player']), tuple(initial['puck']), tuple(initial['goalie'])
                        if geometry is not None and positions != geometry:
                            raise AssertionError(('Velocity sweep changed initial geometry', geometry, positions))
                        geometry = positions
                        if ram_hash is not None and row['initial_ram_sha256'] != ram_hash:
                            raise AssertionError('Policies did not start from identical exposed RAM.')
                        ram_hash = row['initial_ram_sha256']
                        row.update(side=side, direction=direction, seed=seed, handedness=seed % 2,
                                   goalie_rating=rating, lateral=lateral, goalward=goalward)
                        rows.append(row)
    finally:
        env.close()
    return rows


def summarize_grid(rows, args):
    return [
        {'lateral': lateral, 'goalward': goalward,
         'results': {
             rating: {
                 policy: safe_summary([row for row in rows if row['lateral'] == lateral
                                       and row['goalward'] == goalward and row['goalie_rating'] == rating
                                       and row['policy'] == policy])
                 for policy in args.policies}
             for rating in args.goalies}}
        for goalward in args.goalward for lateral in args.lateral
    ]


def _sources():
    root = Path(__file__).resolve().parents[2]
    names = (
        'evaluation/cross_crease_velocity.py', 'evaluation/cross_crease_probe.py',
        'evaluation/benchmark.py', 'tasks/cross_crease_setup.py', 'tasks/defense_setup.py',
        'game/ram.py', 'game/state.py', 'agents/classic_v1.py', 'agents/cross_crease.py',
        'agents/goalie.py', 'agents/offense.py', 'agents/motion.py', 'agents/defense.py',
        'agents/passing.py', 'env/factory.py', 'env/target_control.py',
    )
    return {str((root / 'nhl94_ai' / name).relative_to(root)):
            hashlib.sha256((root / 'nhl94_ai' / name).read_bytes()).hexdigest() for name in names}


def run(args):
    validate_grid(args)
    ShotTiming(args.press_frame, args.hold_frames, args.aim)
    sources = _sources()
    fixtures = [(args, side, direction, seed) for side in (1, 2) for direction in (-1, 1)
                for seed in range(args.seed, args.seed + args.seeds)]
    if args.workers == 1:
        batches = [_fixture(fixture) for fixture in fixtures]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            batches = list(pool.map(_fixture, fixtures))
    if _sources() != sources:
        raise RuntimeError('Sources changed while the velocity grid was running; report is not source-frozen.')
    rows = [row for batch in batches for row in batch]
    report = {
        'protocol': 'isolated-cross-crease-velocity-v1', 'settings': vars(args), 'sources': sources,
        'coordinates': {'lateral': 'positive toward crossing, rink units/emulator frame',
                        'goalward': 'positive toward attacking goal, rink units/emulator frame',
                        'raw_quantization': 'round(velocity * 65536 / 17)',
                        'geometry': 'fixed carrier/puck current and previous positions before first input'},
        'limitations': [
            'A velocity map at one fixed geometry, not universal speed limits or match strength.',
            'All initial headings are lateral; goalward drift is varied independently of facing.',
            'Some incoming impulses exceed the carrier normal speed cap; they decay through native friction.',
            'Classic may reposition before C; forced held attempts start with the exact swept momentum.',
            'Safe-goal counts require an attributed goal and no native goalie-contact impulse.',
            'All-observed success is a sample result, not a guaranteed scoring probability.',
        ],
        'summary': {rating: {policy: safe_summary(
            [row for row in rows if row['goalie_rating'] == rating and row['policy'] == policy])
                            for policy in args.policies} for rating in args.goalies},
        'grid': summarize_grid(rows, args), 'trials': rows,
    }
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = (json.dumps(report, separators=(',', ':')) + '\n').encode('utf-8')
        path.write_bytes(gzip.compress(data, mtime=0) if path.suffix == '.gz' else data)
    for rating in args.goalies:
        for policy in args.policies:
            print(f'{rating}/{policy}: safe goals / trials, by goalward row and lateral column')
            print('vx:', ', '.join(str(value) for value in args.lateral))
            for goalward in args.goalward:
                cells = [cell for cell in report['grid'] if cell['goalward'] == goalward]
                values = [f"{cell['results'][rating][policy]['safe_goals']}/"
                          f"{cell['results'][rating][policy]['attempts']}"
                          f"{'!' if cell['results'][rating][policy]['contact_trials'] else ''}" for cell in cells]
                print(f'vy={goalward}: ' + ' '.join(values))
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lateral', type=float, nargs='+', default=list(LATERAL_SPEEDS))
    parser.add_argument('--goalward', type=float, nargs='+', default=list(GOALWARD_SPEEDS))
    parser.add_argument('--width', type=float, default=35)
    parser.add_argument('--depth', type=float, default=216)
    parser.add_argument('--goalies', nargs='+', choices=GOALIE_LEVELS, default=list(GOALIE_LEVELS))
    parser.add_argument('--policies', nargs='+', choices=('tap', 'held', 'classic'), default=['tap', 'held', 'classic'])
    parser.add_argument('--seed', type=int, default=83000)
    parser.add_argument('--seeds', type=int, default=2)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--press-frame', type=int, default=0)
    parser.add_argument('--hold-frames', type=int, default=24)
    parser.add_argument('--aim', type=int, choices=(-1, 1), default=1)
    parser.add_argument('--trace', action='store_true')
    parser.add_argument('--output')
    return parser


if __name__ == '__main__':
    run(build_parser().parse_args())
