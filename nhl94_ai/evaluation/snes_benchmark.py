"""Paired seeded SNES benchmarks for two-policy comparisons.

Every (policy, side, seed) cell plays one independent first period with the ROM's
RNG state pinned by the seed (see seed_rng in snes_session), so the per-seed pair
shares the exact same random stream and outcome differences reflect the policy,
not the draw. Re-running a cell reproduces it exactly.

Per the stage-7 protocol: 300-clock-second first periods, four-frame tactical
decisions, 20 seeds per side -> 80 periods for a two-policy comparison. Periods
shortened with --max-frames are labeled fixed-save trials, not independent
samples.
"""
import argparse
import json
from pathlib import Path

from nhl94_ai.evaluation.human_recording import write_json
from nhl94_ai.evaluation.snes_session import classic_options, play_snes_period


def run_cell(options, side, seed, *, frame_skip=4, side_overrides=None, max_frames=None):
    job = {'side': side, 'human': False, 'goalie_policy': options['goalie_policy'],
           'frame_skip': frame_skip, 'classic_options': options, 'seed': seed}
    if side_overrides:
        job['classic_options_by_side'] = side_overrides
    if max_frames is not None:
        job['max_frames'] = max_frames
    return play_snes_period(job)


def compare(options_a, options_b, seeds, *, frame_skip=4, max_frames=None):
    """Paired first periods: each seed plays A-at-home vs B-at-away and the mirror."""
    policies = {'a': options_a, 'b': options_b}
    base = {name: value for name, value in options_a.items() if name != 'model_args'}
    rows = []
    for seed in seeds:
        for home_policy in ('a', 'b'):
            away_policy = 'b' if home_policy == 'a' else 'a'
            overrides = {side: policies[policy]['model_args']
                         for side, policy in (('home', home_policy), ('away', away_policy))}
            row = run_cell(base, 'home', seed, frame_skip=frame_skip,
                           side_overrides=overrides, max_frames=max_frames)
            row.update(home_policy=home_policy, away_policy=away_policy, seed=seed)
            rows.append(row)
    return rows


def summarize(rows):
    summary = {'periods': len(rows)}
    for label in ('a', 'b'):
        goals = [row['goals'][0 if row['home_policy'] == label else 1] for row in rows]
        shots = [row['shots'][0 if row['home_policy'] == label else 1] for row in rows]
        against = [row['goals'][1 if row['home_policy'] == label else 0] for row in rows]
        summary[label] = {
            'periods': len(goals),
            'goals': sum(goals), 'goals_against': sum(against), 'shots': sum(shots),
            'goal_diff': sum(goals) - sum(against),
            'mean_goals': sum(goals) / max(1, len(goals)),
            'seed_spread': [max(goals) - min(goals), max(shots) - min(shots)] if goals else None,
        }
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='New report directory; existing paths are never overwritten')
    parser.add_argument('--policy-a', help='Classic JSON settings for policy A')
    parser.add_argument('--policy-b', help='Classic JSON settings for policy B (default: same as A)')
    parser.add_argument('--seeds', type=int, default=20, help='Paired seeds per side (default: 20)')
    parser.add_argument('--seed-start', type=int, default=1, help='First seed value (default: 1)')
    parser.add_argument('--frame-skip', type=int, default=4, help='Classic tactical decision interval')
    parser.add_argument('--max-frames', type=int,
                        help='Cap period length for smoke runs; results are then fixed-save trials')
    return parser


def _policy_options(config_path):
    from types import SimpleNamespace
    args = SimpleNamespace(classic_config=config_path)
    options = classic_options(args)
    # Resolved model args for the per-side override path in play_snes_period.
    model_args = {name: value for name, value in options.items()
                  if name not in ('env', 'goalie_policy', 'goalie_positioning')}
    return {**options, 'model_args': model_args}


def run(args):
    if args.seeds < 1 or args.frame_skip < 1:
        raise ValueError('Use positive seeds/frame-skip')
    options_a = _policy_options(args.policy_a)
    options_b = _policy_options(args.policy_b or args.policy_a)
    path = Path(args.output).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=False)
    seeds = range(args.seed_start, args.seed_start + args.seeds)
    rows = compare(options_a, options_b, seeds, frame_skip=args.frame_skip,
                   max_frames=args.max_frames)
    report = {'schema': 'nhl94-snes-benchmark-v1', 'policy_a': args.policy_a,
              'policy_b': args.policy_b or args.policy_a, 'seeds': list(seeds),
              'frame_skip': args.frame_skip, 'max_frames': args.max_frames,
              'fixed_save_trials': args.max_frames is not None,
              'rows': rows, 'summary': summarize(rows)}
    write_json(path / 'benchmark.json', report)
    print(json.dumps(report['summary'], indent=2))
    return report
