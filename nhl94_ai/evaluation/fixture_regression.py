"""Run the same verified CPU fixtures from an explicitly isolated policy revision.

Invoke this file directly when selecting another revision root, so package
imports resolve there before loading any NHL94 code.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch


GROUPS = {
    'off': ('penguins-senators', 'senators-penguins', 'canadiens-nordiques', 'nordiques-canadiens'),
    'selective': ('sabres-ducks-manual', 'ducks-sabres-manual', 'ducks-campbell-manual', 'campbell-ducks-manual'),
    'standard-gate': ('sabres-ducks-manual', 'ducks-sabres-manual'),
}


def fixture_identity(cpu, matchup, policy, seed):
    import numpy as np
    from stable_retro.data import get_romfile_path
    from nhl94_ai.env.factory import make_retro
    from nhl94_ai.game.ram import pass_geometry_info
    from nhl94_ai.game.state import NHL94GameState
    from nhl94_ai.agents.goalie import GoalieController
    state_name, side = cpu.MATCHUPS[matchup]
    env = make_retro(game='NHL94-Genesis-v0', state=state_name, num_players=1, goalie_policy=policy)
    try:
        for name, (address, kind) in cpu.CPU_RAM.items():
            env.data.set_variable(name, {'address': address, 'type': kind})
        env.reset(seed=seed)
        env.data.update_ram()
        cpu.select_side(env.data, env.data.lookup_all(), side)
        env.data.set_value('bench_rng', seed)
        env.data.set_value('bench_clock', 300)
        *_, info = env.step(np.zeros(12, dtype=np.int8))
        view = cpu.cpu_view(NHL94GameState(5), pass_geometry_info(env, info), side)
        if policy != 'off':
            GoalieController.validate(view)
        if (info['bench_team1'], info['bench_team2'], info['period']) != (side, 0, 0):
            raise RuntimeError('Fixture is not a first-period, single-joystick CPU start.')
        roles = [env.data.memory.extract(0xFFB04A + slot * 0x80 + 0x34, '>i2') for slot in range(12)]
        if any(role <= 0 for slot, role in enumerate(roles) if slot % 6 != 5) or (roles[5], roles[11]) != (0, 0):
            raise RuntimeError('Fixture contains inactive actors or incorrectly identified goalies.')
        return {
            'matchup': matchup, 'state': state_name, 'side': side, 'goalie_policy': policy,
            'teams': [info['home_team'], info['away_team']], 'roles': roles,
            'save_sha256': hashlib.sha256(env.initial_state).hexdigest(),
            'rom_sha256': hashlib.sha256(Path(get_romfile_path('NHL94-Genesis-v0')).read_bytes()).hexdigest(),
            'goalie_modes': list(view.engine.goalie_modes) if view.engine.goalie_modes is not None else None,
            'lineup': cpu.lineup(info, Path(get_romfile_path('NHL94-Genesis-v0')).read_bytes()),
            'effective_attributes': [
                [env.data.memory.extract(0xFFB04A + slot * 0x80 + offset, '|u1')
                 for offset in range(0x67, 0x74)] for slot in range(12)],
        }
    finally:
        env.close()


def cadence_gate(cpu, report, seed, goalie_policy):
    """Compare a full CPU trial with independent explicit native-frame dispatch."""
    original = cpu.create_scripted
    class NativeDispatch:
        def __init__(self, controller):
            self.controller = controller
            self.frame_skip = 4

        def __getattr__(self, name):
            return getattr(self.controller, name)

        def predict_game_state(self, state):
            return self.controller.predict_frame(state, self.frame_skip)

    def direct(*args, **kwargs):
        return NativeDispatch(original(*args, **kwargs).controller)
    fixture = ('classic-v1', 'ducks-sabres-manual', seed, 300, 4, 'FILTERED', goalie_policy)
    with patch.object(cpu, 'create_scripted', side_effect=direct):
        repeated = cpu.cpu_match(fixture)
    expected = next(row for row in report['matches'] if row['matchup'] == fixture[1] and row['seed'] == seed)
    keys = ('completed', 'frames', 'clock_remaining', 'actions_sha256', 'initial_state_sha256',
            'teams', 'lineup', 'goals', 'shots', 'one_timers', 'one_timer_goals')
    if any(repeated[key] != expected[key] for key in keys):
        raise RuntimeError('Native dispatch gate differs from the complete production benchmark prefix.')
    return {'passed': True, 'seed': seed, 'matchup': fixture[1],
            'frames': repeated['frames'], 'actions_sha256': repeated['actions_sha256']}


def run(args):
    if min(args.trials, args.workers) < 1 or args.seed < 0 or args.seed + args.trials > 2**32:
        raise ValueError('Use positive trial/worker counts and uint32 ROM seeds.')
    if args.gate and args.group == 'off':
        raise ValueError('The cadence gate requires a group containing the standard Ducks/Sabres fixture.')
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f'Preserve the existing regression report: {output}')
    runner_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    root = Path(args.revision_root).resolve()
    sys.path.insert(0, str(root))
    from nhl94_ai.evaluation import cpu_benchmark as cpu
    if Path(cpu.__file__).resolve() != root / 'nhl94_ai/evaluation/cpu_benchmark.py':
        raise RuntimeError('Wrong policy package imported; invoke this runner as a file with the isolated revision root.')
    cpu.MATCHUPS['canadiens-nordiques'] = ('CanadiensVsNordiques.start', 1)
    policy = 'selective' if args.group == 'selective' else 'off'
    settings = cpu.build_parser().parse_args([
        '--agent', 'classic-v1', '--matchups', *GROUPS[args.group], '--trials', str(args.trials),
        '--seed', str(args.seed), '--seconds', '300', '--frame-skip', '4',
        '--action-type', 'FILTERED', '--goalie-policy', policy, '--workers', str(args.workers)])
    revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    identities = {name: fixture_identity(cpu, name, policy, args.seed) for name in settings.matchups}
    report = cpu.run(settings)
    keys = {(name, seed) for name in settings.matchups for seed in range(args.seed, args.seed + args.trials)}
    if {(row['matchup'], row['seed']) for row in report['matches']} != keys or len(report['matches']) != len(keys):
        raise RuntimeError('Duplicate or missing trials in the declared fixture matrix.')
    for row in report['matches']:
        identity = identities[row['matchup']]
        if not row['completed'] or row['clock_remaining'] != 0 or row.get('inactive_skater_frames', 0) != 0:
            raise RuntimeError('Incomplete period or inactive actors in fixture regression.')
        if row['teams'] != identity['teams'] or row['initial_state_sha256'] != identity['save_sha256']:
            raise RuntimeError('Physical teams or save state changed during the trial.')
        if row['lineup'] != identity['lineup']:
            raise RuntimeError('A trial does not use the declared starting lineup/effective shot ratings.')
    report['cadence_gate'] = cadence_gate(cpu, report, args.seed, policy) if args.gate else None
    for name, digest in report['sources'].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Policy source changed during the reference measurement: {name}')
    report['revision'] = revision
    report['fixture_group'] = args.group
    report['starting_fixtures'] = identities
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != runner_hash:
        raise RuntimeError('Regression runner changed during measurement.')
    report['runner_sha256'] = runner_hash
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f'Completed {len(report["matches"])} periods at {revision[:7]}, group {args.group}; '
          f'cadence gate: {report["cadence_gate"]}')
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision-root', required=True)
    parser.add_argument('--group', choices=GROUPS, required=True)
    parser.add_argument('--trials', type=int, default=20)
    parser.add_argument('--seed', type=int, default=20263001)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--gate', action='store_true')
    parser.add_argument('--output', required=True)
    return parser


if __name__ == '__main__':
    run(build_parser().parse_args())
