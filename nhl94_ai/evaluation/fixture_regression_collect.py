"""Complete every declared trial, recording known runtime failures without scores."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import traceback
from unittest.mock import patch

from nhl94_ai.evaluation import cpu_benchmark as cpu
from nhl94_ai.evaluation.fixture_regression import GROUPS, fixture_identity


def trial(fixture):
    from nhl94_ai.agents.goalie import GoalieController
    original = GoalieController.validate
    diagnostic = {}
    def checked(state):
        try:
            return original(state)
        except ValueError:
            goalie = state.team1.goalie
            required = ('role', 'selection_flags', 'live_state_flags', 'live_anim', 'assignment',
                        'speed', 'agility', 'weight', 'energy', 'motion_x', 'motion_y', 'passing', 'stick')
            diagnostic.update({
                'missing_goalie_fields': [field for field in required if getattr(goalie, field) is None],
                'goalie_fields': {field: getattr(goalie, field) for field in required},
                'puck_owner': state.engine.puck_owner, 'control': state.team1.defense_control,
                'goalie_modes': state.engine.goalie_modes, 'controller_teams': state.engine.controller_teams,
                'missing_puck_fields': [field for field in ('height', 'motion_x', 'motion_y')
                                        if getattr(state.puck, field) is None],
            })
            raise
    with patch.object(GoalieController, 'validate', side_effect=checked):
        try:
            return {'status': 'completed', 'match': cpu.cpu_match(fixture)}
        except (ValueError, RuntimeError) as error:
            failure = {
                'matchup': fixture[1], 'seed': fixture[2], 'side': cpu.MATCHUPS[fixture[1]][1],
                'goalie_policy': fixture[6], 'error_type': type(error).__name__, 'error': str(error),
                'traceback': traceback.format_exc(), 'diagnostic': diagnostic,
            }
            print(f'FAILED {fixture[1]} seed {fixture[2]}: {error}', flush=True)
            return {'status': 'failed', 'failure': failure}


def run(args):
    root = Path(__file__).resolve().parents[2]
    template = json.loads(Path(args.template).read_text(encoding='utf-8'))
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    if revision != template['revision'] or args.group != 'selective':
        raise ValueError('Failure-aware recovery must describe the same current revision and selective group.')
    sources = template['sources']
    for name, digest in sources.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Do not change the measured policy to recover failed trials: {name}')
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f'Preserve the existing collection: {output}')
    settings = cpu.build_parser().parse_args([
        '--agent', 'classic-v1', '--matchups', *GROUPS[args.group], '--trials', '20', '--seed', '20263001',
        '--seconds', '300', '--frame-skip', '4', '--action-type', 'FILTERED',
        '--goalie-policy', 'selective', '--workers', str(args.workers)])
    identities = {name: fixture_identity(cpu, name, 'selective', 20263001) for name in settings.matchups}
    fixtures = [('classic-v1', name, seed, 300, 4, 'FILTERED', 'selective')
                for name in settings.matchups for seed in range(20263001, 20263021)]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(trial, fixture) for fixture in fixtures]
        for future in as_completed(futures):
            results.append(future.result())
            print(f'Finished {len(results)}/{len(fixtures)} declared trials', flush=True)
    matches = sorted((result['match'] for result in results if result['status'] == 'completed'),
                     key=lambda row: (row['matchup'], row['seed']))
    failures = sorted((result['failure'] for result in results if result['status'] == 'failed'),
                      key=lambda row: (row['matchup'], row['seed']))
    for name, digest in sources.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Measured source changed during failed-trial recovery: {name}')
    for row in matches:
        identity = identities[row['matchup']]
        if (not row['completed'] or row['clock_remaining'] != 0 or row['inactive_skater_frames'] != 0
                or row['lineup'] != identity['lineup'] or row['initial_state_sha256'] != identity['save_sha256']):
            raise RuntimeError('Recovered period did not satisfy the declared physical fixture.')
    report = {
        'protocol': 'nhl94-cpu-first-period-v2', 'settings': vars(settings), 'revision': revision,
        'fixture_group': args.group, 'sources': sources, 'starting_fixtures': identities,
        'matches': matches, 'failures': failures, 'attempted_trials': len(results),
        'summary': cpu.summarize(matches), 'recovered_after_failure': True,
        'collector_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'limitations': ['Failed trials have no scores and must not be counted as completed periods.',
                        'The original policy was not modified to recover trial results.'],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f'Completed declared matrix: {len(matches)} complete periods, {len(failures)} runtime failures.', flush=True)
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--template', required=True)
    parser.add_argument('--group', choices=['selective'], default='selective')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--output', required=True)
    return parser


if __name__ == '__main__':
    run(build_parser().parse_args())
