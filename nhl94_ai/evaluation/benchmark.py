"""Paired, seeded scripted-agent matches using goals rather than shaped reward.

Each trial is one first period of full-team NHL94, with the teams swapped for
the return fixture. The ROM RNG is seeded explicitly; Gym's seed alone does
not change the emulator save state. RAM corrections below are applied equally
to every controller, without changing its tactics or the installed integration.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import copy
import hashlib
import inspect
import json
from pathlib import Path
from time import perf_counter_ns
from types import SimpleNamespace

import numpy as np

from nhl94_ai.agents.registry import ALIASES, create_scripted
from nhl94_ai.game.state import NHL94GameState
from nhl94_ai.game.ram import ROM_RNG_ADDRESS
from nhl94_ai.evaluation.pass_outcomes import PassOutcomes


# NHL94 Genesis RAM symbols: random, SCnum[], teamselect[], clock, shots.
RAM = {
    'bench_rng': (ROM_RNG_ADDRESS, '>u4'),
    'bench_control1': (0xFFC320, '>i2'), 'bench_control2': (0xFFC322, '>i2'),
    'bench_team1': (0xFFC328, '>u2'), 'bench_team2': (0xFFC32A, '>u2'),
    'bench_clock': (0xFFC468, '>u2'),
    'bench_shots1': (0xFFC6CE, '>u2'), 'bench_shots2': (0xFFCA32, '>u2'),
    'bench_one_timers1': (0xFFCA2A, '>u2'), 'bench_one_timers2': (0xFFCD8E, '>u2'),
    'bench_one_timer_goals1': (0xFFCA2C, '>u2'), 'bench_one_timer_goals2': (0xFFCD90, '>u2'),
}

# Benchmark-only ablation of the same controller, without the pass/one-timer branch.
AGENTS = [*ALIASES, 'classic-v1-direct']


def make_agent(name):
    args = SimpleNamespace(action_type='FILTERED', one_timers=name != 'classic-v1-direct')
    return create_scripted('classic-v1' if name == 'classic-v1-direct' else name, args)


def _position(info, side, slot):
    index = slot - (side - 1) * 6
    if not 0 <= index <= 5:
        return 9999, 9999
    prefix = f'g{side}_' if index == 5 else f'p{side}_' + (f'{index + 1}_' if index else '')
    return info[prefix + 'x'], info[prefix + 'y']


def update_state(state, info, env=None):
    """Resolve both teams' possession/control from slots, not misaddressed stars."""
    if env is not None:
        from nhl94_ai.game.ram import pass_geometry_info
        info = pass_geometry_info(env, info)
    info = dict(info)
    owner = info['puck_owner']
    for side, team in enumerate((state.team1, state.team2), 1):
        full = 'fullstar_' if side == 1 else 'p2_fullstar_'
        info[full + 'x'], info[full + 'y'] = _position(info, side, owner)
        empty = f'p{side}_emptystar_'
        slot = info[f'bench_control{side}']
        info[empty + 'x'], info[empty + 'y'] = _position(info, side, slot)
        if team.owns_scnum(slot):
            team.control = 0 if slot == team.goalie_scnum() else slot - team.skater_scnum_base() + 1
        info[f'p{side}_shots'] = info[f'bench_shots{side}']
    state.BeginFrame(info, [0] * 6)
    # Avoid ambiguous stars when two players happen to occupy the same point.
    for side, team in enumerate((state.team1, state.team2), 1):
        slot = info[f'bench_control{side}']
        if team.owns_scnum(slot):
            team.control = 0 if slot == team.goalie_scnum() else slot - team.skater_scnum_base() + 1
        team.goalie_haspuck = owner == team.goalie_scnum()
        team.player_haspuck = team.owns_scnum(owner) and not team.goalie_haspuck


def away_view(state):
    """Swap team references while retaining physical coordinates and goal ends."""
    view = copy(state)
    view.team1, view.team2 = state.team2, state.team1
    return view


def match(fixture):
    from nhl94_ai.env.factory import make_retro
    home, away, seed, seconds, frame_skip = fixture
    env = make_retro(game='NHL94-Genesis-v0', num_players=2)
    try:
        for name, (address, kind) in RAM.items():
            env.data.set_variable(name, {'address': address, 'type': kind})
        agents = [make_agent(name) for name in (home, away)]
        for agent in agents:
            agent.frame_skip = frame_skip
        _, info = env.reset(seed=seed)
        env.data.set_value('bench_rng', seed)
        env.data.set_value('bench_clock', seconds)
        _, _, terminated, truncated, info = env.step(np.zeros(24, dtype=np.int8))
        if [info['bench_team1'], info['bench_team2']] != [1, 2] or info['period'] != 0:
            raise ValueError('Benchmark requires a first-period save with controllers on opposing teams.')
        state, frames = NHL94GameState(5), 1
        durations, calls = [0, 0], [0, 0]
        setups = [0, 0]
        passes = [PassOutcomes(1), PassOutcomes(2)]
        digest = hashlib.sha256()
        budget = seconds * 120 + 6000  # Also bounds stoppages/replays with a frozen clock.
        while info['bench_clock'] > 0 and not (terminated or truncated) and frames < budget:
            update_state(state, info, env)
            actions = []
            for index, view in enumerate((state, away_view(state))):
                start = perf_counter_ns()
                before_tick = agents[index]._tick
                actions.extend(agents[index].predict_game_state(view)[0])
                durations[index] += perf_counter_ns() - start
                calls[index] += 1
                new_pass = agents[index]._tick != before_tick and agents[index]._last_decision == 'one-timer-pass'
                setups[index] += new_pass
                request = agents[index]._last_pass_request
                if request and request['frame'] != getattr(passes[index], 'last_request_frame', None):
                    passes[index].start(frames, info, request['passer'], request['receiver'], request['purpose'])
                    passes[index].last_request_frame = request['frame']
            buttons = np.asarray(actions, dtype=np.int8)
            digest.update(buttons.tobytes())
            _, _, terminated, truncated, info = env.step(buttons)
            frames += 1
            for tracker in passes:
                tracker.observe(frames, info)
        for tracker in passes:
            tracker.finish(frames, info, 'period_ended' if info['bench_clock'] == 0 else 'trial_incomplete')
        return {
            'agents': [home, away], 'seed': seed,
            'goals': [info['p1_score'], info['p2_score']],
            'shots': [info['bench_shots1'], info['bench_shots2']],
            'one_timer_setups': setups,
            'pass_outcomes': [tracker.summary() for tracker in passes],
            'pass_events': [tracker.events for tracker in passes],
            'one_timers': [info['bench_one_timers1'], info['bench_one_timers2']],
            'one_timer_goals': [info['bench_one_timer_goals1'], info['bench_one_timer_goals2']],
            'clock_remaining': info['bench_clock'], 'frames': frames,
            'completed': info['bench_clock'] == 0,
            'terminated': bool(terminated), 'truncated': bool(truncated),
            'decision_ns': durations, 'decisions': calls,
            'actions_sha256': digest.hexdigest(),
            'initial_state_sha256': hashlib.sha256(env.initial_state).hexdigest(),
        }
    finally:
        env.close()


def summarize(results, candidate, opponents):
    summary = {}
    for opponent in opponents:
        rows = [r for r in results if opponent in r['agents'] and r['completed']]
        scores = [(r['goals'][r['agents'].index(candidate)], r['goals'][r['agents'].index(opponent)]) for r in rows]
        timings = {}
        for name in (candidate, opponent):
            nanoseconds = sum(r['decision_ns'][r['agents'].index(name)] for r in rows)
            calls = sum(r['decisions'][r['agents'].index(name)] for r in rows)
            timings[name] = round(nanoseconds / max(calls, 1) / 1000, 2)
        summary[opponent] = {
            'games': len(rows), 'wins': sum(a > b for a, b in scores),
            'draws': sum(a == b for a, b in scores), 'losses': sum(a < b for a, b in scores),
            'goals_for': sum(a for a, _ in scores), 'goals_against': sum(b for _, b in scores),
            'mean_decision_us': timings,
        }
        for field in ('one_timer_setups', 'one_timers', 'one_timer_goals'):
            summary[opponent][field] = {
                name: sum(r.get(field, [0, 0])[r['agents'].index(name)] for r in rows)
                for name in (candidate, opponent)
            }
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--agent', choices=AGENTS, default='classic-v1')
    parser.add_argument('--opponents', choices=AGENTS, nargs='+', default=['classic-v1-direct'])
    parser.add_argument('--seed', type=int, default=2000)
    parser.add_argument('--pairs', type=int, default=20, help='Seeds per opponent; each seed plays both sides')
    parser.add_argument('--seconds', type=int, default=300, help='Game-clock seconds per first-period trial')
    parser.add_argument('--frame-skip', type=int, default=4)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--output', default=None)
    return parser


def run(args):
    if min(args.pairs, args.seconds, args.frame_skip, args.workers) < 1 or args.seconds > 65535:
        raise ValueError('Pairs, seconds, frame-skip and workers must be positive; seconds must fit uint16.')
    if args.seed < 0 or args.seed + args.pairs > 2**32:
        raise ValueError('ROM seeds must fit uint32.')
    candidate = ALIASES.get(args.agent, args.agent)
    opponents = [ALIASES.get(name, name) for name in args.opponents]
    if candidate in opponents or len(set(opponents)) != len(opponents):
        raise ValueError('Choose distinct opponents different from the candidate.')
    fixtures = [(home, away, seed, args.seconds, args.frame_skip)
                for seed in range(args.seed, args.seed + args.pairs)
                for opponent in args.opponents
                for home, away in ((args.agent, opponent), (opponent, args.agent))]
    sources = {}
    for name in (args.agent, *args.opponents):
        model = make_agent(name).controller
        for cls in type(model).__mro__:
            if cls is object:
                continue
            path = Path(inspect.getfile(cls))
            sources[path.name] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                  'lines': len(path.read_text(encoding='utf-8').splitlines())}
    root = Path(__file__).resolve().parents[1]
    for name in ('game/state.py', 'game/ram.py', 'game/geometry.py', 'env/factory.py',
                 'agents/base.py', 'agents/defense.py', 'agents/motion.py', 'agents/offense.py',
                 'agents/passing.py', 'evaluation/pass_outcomes.py'):
        path = root / name
        sources[name] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                         'lines': len(path.read_text(encoding='utf-8').splitlines())}
    protocol_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if args.workers == 1:
        results = [match(fixture) for fixture in fixtures]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(match, fixtures))
    from importlib.metadata import version
    report = {'protocol': 'nhl94-paired-first-period-v2', 'protocol_sha256': protocol_sha256,
              'game': 'NHL94-Genesis-v0',
              'settings': vars(args), 'sources': sources,
              'versions': {name: version(name) for name in ('nhl94-ai', 'stable-retro', 'numpy')},
              'summary': summarize(results, args.agent, args.opponents), 'matches': results}
    if args.output:
        path = Path(args.output).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report['summary'], indent=2))
    if not all(r['completed'] for r in results):
        raise RuntimeError('Incomplete matches: benchmark failed; do not treat partial scores as results.')
    return report
