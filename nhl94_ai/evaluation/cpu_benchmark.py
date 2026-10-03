"""Seeded first-period trials against NHL94's built-in CPU, with real rosters."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from nhl94_ai.agents.registry import ALIASES, add_classic_arguments, create_scripted
from nhl94_ai.env.actions import HockeyActionController
from nhl94_ai.evaluation.benchmark import RAM, away_view, update_state
from nhl94_ai.game.state import NHL94GameState
from nhl94_ai.game.ram import pass_geometry_info, restore_away_control, select_cpu_side
from nhl94_ai.evaluation.pass_outcomes import PassOutcomes
from nhl94_ai.evaluation.offense_metrics import OffenseMetrics


MATCHUPS = {
    'penguins-senators': ('PenguinsVsSenators.start', 1),
    'senators-penguins': ('PenguinsVsSenators.start', 2),
    'nordiques-canadiens': ('CanadiensVsNordiques.start', 2),
}
DEFAULT_MATCHUPS = list(MATCHUPS)
MATCHUPS.update({
    'sabres-ducks-manual': ('SabresVsMightyDucks.ManualGoalie.Start', 1),
    'ducks-sabres-manual': ('SabresVsMightyDucks.ManualGoalie.Start', 2),
    'ducks-campbell-manual': ('MightyDucksVsAllStarCampbell.ManualGoalie.Start', 1),
    'campbell-ducks-manual': ('MightyDucksVsAllStarCampbell.ManualGoalie.Start', 2),
})
CPU_RAM = {
    **RAM,
    'home_team': (0xFFC330, '>u2'), 'away_team': (0xFFC332, '>u2'),
    'home_roster': (0xFFC6EC, '>u4'), 'away_roster': (0xFFCA50, '>u4'),
    **{f'cpu_{slot}_{field}': (0xFFB04A + slot * 0x80 + offset, '|u1')
       for slot in range(12)
       for field, offset in (('flags', 0x62), ('roster', 0x66), ('accuracy', 0x6D))},
}


def select_side(data, info, side):
    """Transfer the single joystick; the other team must run the game's CPU.

    Mirrors restorep1's pfjoycon/pfna flag changes. Leaving a second human
    controller assigned with zero input would not test the built-in opponent.
    """
    if (info['bench_team1'], info['bench_team2'], info['period']) != (1, 0, 0):
        raise ValueError('CPU benchmark requires a first-period, home-only joystick save.')
    select_cpu_side(data, info, side, controller_prefix='bench', player_prefix='cpu', slots=6)


def lineup(info, rom):
    """Resolve names from live roster indices, not save-state filenames."""
    players = []
    for slot in range(12):
        address = info['home_roster' if slot < 6 else 'away_roster']
        address += int.from_bytes(rom[address:address + 2], 'big')
        for _ in range(info[f'cpu_{slot}_roster']):
            address += int.from_bytes(rom[address:address + 2], 'big') + 8
        size = int.from_bytes(rom[address:address + 2], 'big')
        if not 2 < size < 64 or address + size >= len(rom):
            raise ValueError('Invalid ROM roster pointer/name.')
        name = rom[address + 2:address + size].rstrip(b'\0').decode('ascii')
        players.append({'slot': slot, 'name': name,
                        'shot_accuracy': info[f'cpu_{slot}_accuracy'] if slot % 6 != 5 else None})
    return players


def cpu_view(state, info, side):
    if side == 1:
        # Same inference used in normal play, including any stale star readings.
        state.BeginFrame(info, [0] * 6)
        return state
    corrected = dict(info, bench_control1=-1, bench_control2=info['bench_control1'])
    update_state(state, corrected)
    return away_view(state)


def cpu_match(fixture):
    from nhl94_ai.env.factory import make_retro
    from stable_retro.data import get_romfile_path
    agent_name, matchup, seed, seconds, frame_skip, schema, *options = fixture
    goalie_policy = options[0] if options else 'off'
    cross_crease = bool(options[1]) if len(options) > 1 else False
    if goalie_policy != 'off' and schema != 'FILTERED':
        raise ValueError('Manual goalie CPU trials require FILTERED buttons')
    state_name, side = MATCHUPS[matchup]
    env = make_retro(game='NHL94-Genesis-v0', state=state_name, num_players=1, goalie_policy=goalie_policy)
    try:
        for name, (address, kind) in CPU_RAM.items():
            env.data.set_variable(name, {'address': address, 'type': kind})
        env.reset(seed=seed)
        env.data.update_ram()
        select_side(env.data, env.data.lookup_all(), side)
        env.data.set_value('bench_rng', seed)
        env.data.set_value('bench_clock', seconds)
        *_, info = env.step(np.zeros(12, dtype=np.int8))
        starting_lineup = lineup(info, Path(get_romfile_path('NHL94-Genesis-v0')).read_bytes())
        if (info['bench_team1'], info['bench_team2']) != (side, 0):
            raise ValueError('CPU opponent still has a joystick assigned.')
        agent = create_scripted(agent_name, SimpleNamespace(
            action_type=schema, goalie_policy=goalie_policy, cross_crease=cross_crease))
        agent.frame_skip = frame_skip
        state, frames = NHL94GameState(5), 1
        context = SimpleNamespace(action_type=schema, game_state=state)
        processor = HockeyActionController(context)
        macro = processor._new_action_state()
        decisions, digest = Counter(), hashlib.sha256()
        defense_frames, defense_requests = Counter(), Counter()
        controlled_recoveries = 0
        passes = PassOutcomes(side)
        offense = OffenseMetrics()
        budget = seconds * 120 + 6000
        while info['bench_clock'] > 0 and frames < budget:
            info = pass_geometry_info(env, info)
            view = cpu_view(state, info, side)
            offense.observe(frames, view)
            before_tick = agent._tick
            action = agent.predict_game_state(view)[0]
            if agent._tick != before_tick:
                decisions[agent._last_decision] += 1
            request = agent._last_pass_request
            if request and request['frame'] != getattr(passes, 'last_request_frame', None):
                passes.start(frames, info, request['passer'], request['receiver'], request['purpose'])
                passes.last_request_frame = request['frame']
            diagnostics = agent.defense_diagnostics
            if diagnostics:
                defense_frames[diagnostics['decision']] += 1
                if diagnostics['mode'].endswith('-request'):
                    defense_requests[diagnostics['mode']] += 1
            if schema == 'HOCKEY_INTENT_DPAD':
                context.game_state = view
                buttons = processor._process_action(action, macro)[0]
            else:
                buttons = action
            digest.update(np.asarray(buttons, dtype=np.int8).tobytes())
            before_owner = info['puck_owner']
            *_, info = env.step(buttons)
            if side == 2:
                info = restore_away_control(env.data, info, controller_prefix='bench', player_prefix='cpu', slots=6)
            if diagnostics and before_owner != info['puck_owner'] == diagnostics['acting_slot']:
                controlled_recoveries += 1
            frames += 1
            passes.observe(frames, info)
        passes.finish(frames, info, 'period_ended' if info['bench_clock'] == 0 else 'trial_incomplete')
        return {
            'agent': agent_name, 'matchup': matchup, 'side': side, 'seed': seed,
            'goals': [info['p1_score'], info['p2_score']],
            'shots': [info['bench_shots1'], info['bench_shots2']],
            'one_timers': [info['bench_one_timers1'], info['bench_one_timers2']],
            'one_timer_goals': [info['bench_one_timer_goals1'], info['bench_one_timer_goals2']],
            'decisions': dict(decisions), 'frames': frames,
            'defense_frames': dict(defense_frames), 'defense_requests': dict(defense_requests),
            'controlled_recoveries': controlled_recoveries,
            'pass_outcomes': passes.summary(), 'pass_events': passes.events,
            'offense_metrics': offense.summary(), 'zone_entries': offense.entries,
            'clock_remaining': info['bench_clock'], 'completed': info['bench_clock'] == 0,
            'teams': [info['home_team'], info['away_team']], 'lineup': starting_lineup,
            'actions_sha256': digest.hexdigest(),
            'initial_state_sha256': hashlib.sha256(env.initial_state).hexdigest(),
            'goalie_policy': goalie_policy,
            'goalie_metrics': dict(agent.goalie.metrics) if agent.goalie else {},
            'cross_crease': cross_crease,
            'cross_crease_metrics': dict(agent.cross_crease.metrics) if agent.cross_crease else {},
            'cross_crease_events': agent.cross_crease.events if agent.cross_crease else [],
        }
    finally:
        env.close()


def summarize(results):
    summary = {}
    for matchup in dict.fromkeys(row['matchup'] for row in results):
        rows = [r for r in results if r['matchup'] == matchup and r['completed']]
        scores = [(r['goals'][r['side'] - 1], r['goals'][2 - r['side']]) for r in rows]
        summary[matchup] = {
            'periods': len(rows), 'wins': sum(a > b for a, b in scores),
            'draws': sum(a == b for a, b in scores), 'losses': sum(a < b for a, b in scores),
            'goals_for': sum(a for a, _ in scores), 'goals_against': sum(b for _, b in scores),
            'one_timer_setups': sum(r['decisions'].get('one-timer-pass', 0) for r in rows),
            'one_timers': sum(r['one_timers'][r['side'] - 1] for r in rows),
            'one_timer_goals': sum(r['one_timer_goals'][r['side'] - 1] for r in rows),
            'periods_without_one_timers': sum(r['one_timers'][r['side'] - 1] == 0 for r in rows),
        }
        outcomes = Counter()
        for row in rows:
            outcomes.update(row.get('pass_outcomes', {}).get('outcomes', {}))
        summary[matchup]['pass_outcomes'] = dict(outcomes)
        summary[matchup]['wrong_recipient'] = sum(r.get('pass_outcomes', {}).get('wrong_recipient', 0) for r in rows)
        metrics = Counter()
        for row in rows:
            metrics.update(row.get('offense_metrics', {}))
        summary[matchup]['offense_metrics'] = dict(metrics)
        goalie_metrics = Counter()
        for row in rows:
            goalie_metrics.update(row.get('goalie_metrics', {}))
        summary[matchup]['goalie_metrics'] = dict(goalie_metrics)
        crossing_metrics = Counter()
        for row in rows:
            crossing_metrics.update(row.get('cross_crease_metrics', {}))
        summary[matchup]['cross_crease_metrics'] = dict(crossing_metrics)
        summary[matchup]['passes_by_purpose'] = {
            purpose: dict(Counter(event['outcome'] for row in rows for event in row.get('pass_events', [])
                                  if event.get('purpose', 'one-timer') == purpose))
            for purpose in sorted({event.get('purpose', 'one-timer') for row in rows
                                   for event in row.get('pass_events', [])})
        }
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--agent', choices=ALIASES, default='classic-v1')
    parser.add_argument('--matchups', choices=MATCHUPS, nargs='+', default=DEFAULT_MATCHUPS)
    parser.add_argument('--goalie-policy', choices=['off', 'selective', 'always'], default='off')
    parser.add_argument('--trials', type=int, default=20)
    parser.add_argument('--seed', type=int, default=8000)
    parser.add_argument('--seconds', type=int, default=300)
    parser.add_argument('--frame-skip', type=int, default=4)
    parser.add_argument('--action-type', choices=['FILTERED', 'HOCKEY_INTENT_DPAD'], default='FILTERED')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--output')
    return add_classic_arguments(parser)


def run(args):
    if (min(args.trials, args.seconds, args.frame_skip, args.workers) < 1
            or args.seconds > 65535 or args.seed < 0 or args.seed + args.trials > 2**32):
        raise ValueError('Use positive counts, a uint16 clock, and uint32 ROM seeds.')
    if len(set(args.matchups)) != len(args.matchups):
        raise ValueError('Choose distinct matchups.')
    if args.goalie_policy != 'off' and args.action_type != 'FILTERED':
        raise ValueError('Manual goalie CPU trials require FILTERED buttons')
    fixtures = [(args.agent, matchup, seed, args.seconds, args.frame_skip, args.action_type, args.goalie_policy)
                for matchup in args.matchups for seed in range(args.seed, args.seed + args.trials)]
    if getattr(args, 'cross_crease', False):
        fixtures = [(*fixture, True) for fixture in fixtures]
    controller = create_scripted(args.agent, SimpleNamespace(action_type=args.action_type)).controller
    files = {Path(__file__), Path(inspect.getfile(PassOutcomes)), Path(inspect.getfile(OffenseMetrics)),
             Path(inspect.getfile(HockeyActionController)), Path(inspect.getfile(NHL94GameState)),
             Path(inspect.getfile(update_state))}
    files.update(Path(inspect.getfile(cls)) for cls in type(controller).__mro__ if cls is not object)
    root = Path(__file__).resolve().parents[2]
    files.update(root / 'nhl94_ai' / name for name in (
        'game/ram.py', 'game/geometry.py', 'env/factory.py', 'env/target_control.py',
        'agents/base.py', 'agents/defense.py', 'agents/motion.py', 'agents/offense.py', 'agents/passing.py',
        'agents/goalie.py', 'agents/cross_crease.py', 'agents/registry.py'))
    sources = {str(path.relative_to(root)):
               hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    if args.workers == 1:
        results = [cpu_match(fixture) for fixture in fixtures]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(cpu_match, fixtures))
    from importlib.metadata import version
    report = {
        'protocol': 'nhl94-cpu-first-period-v2', 'settings': vars(args), 'sources': sources,
        'versions': {name: version(name) for name in ('nhl94-ai', 'stable-retro', 'numpy')},
        'limitations': ['Fixed starting rosters, first periods only; not full-game win rates.',
                        'Away trials transfer the joystick to the away team and correct RAM control observations.',
                        'Defense uses authoritative control slots; offensive inference is unchanged.',
                        'Defense reacts each frame; frame-skip controls the offensive decision interval.',
                        'Team changes also change opponents; this does not isolate any individual rating effect.'],
        'summary': summarize(results), 'matches': results,
    }
    if args.output:
        path = Path(args.output).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report['summary'], indent=2))
    if not all(row['completed'] for row in results):
        raise RuntimeError('Incomplete CPU periods; partial scores must not count as results.')
    return report
