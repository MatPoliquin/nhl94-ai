"""Fail closed unless native replay cadence and source provenance match a CPU benchmark."""
import argparse
import hashlib
import json
from pathlib import Path

from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.evaluation.carry_replay import DECISION_INTERVAL, replay_seed, source_hashes


REQUIRED_SOURCES = {
    f'nhl94_ai/{name}' for name in (
        'agents/base.py', 'agents/classic_v1.py', 'agents/carry.py', 'agents/offense.py', 'agents/possession.py',
        'agents/passing.py', 'agents/receiving.py', 'agents/skating.py', 'agents/finishing.py', 'env/actions.py', 'env/factory.py',
        'evaluation/benchmark.py', 'evaluation/cpu_benchmark.py', 'game/state.py', 'game/ram.py',
    )
}
MATCHUPS = {1: 'sabres-ducks-manual', 2: 'ducks-sabres-manual'}


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def validate_sources(sources, root):
    if not isinstance(sources, dict) or not sources:
        raise ValueError('Benchmark must record nonempty source fingerprints.')
    root = Path(root).resolve()
    for name, expected in sources.items():
        if not isinstance(name, str) or not isinstance(expected, str):
            raise ValueError('Source fingerprints must map relative paths to SHA256 strings.')
        path = (root / name).resolve()
        if Path(name).is_absolute() or not path.is_relative_to(root):
            raise ValueError(f'Source path must stay inside the repository: {name}')
        if len(expected) != 64 or any(char not in '0123456789abcdef' for char in expected):
            raise ValueError(f'Invalid SHA256 fingerprint for {name}.')
        if not path.is_file():
            raise ValueError(f'Measured source is missing: {name}')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Source fingerprint mismatch: {name}. Regenerate the current default reference.')


def validate_benchmark(report, root):
    """Require the standard completed default-policy fixture, not an experimental reference."""
    if not isinstance(report, dict) or report.get('protocol') != 'nhl94-cpu-first-period-v2':
        raise ValueError('Gate requires a nhl94-cpu-first-period-v2 benchmark report.')
    settings = report.get('settings')
    required = {
        'agent': 'classic-v1', 'seconds': 300, 'frame_skip': DECISION_INTERVAL,
        'action_type': 'FILTERED', 'goalie_policy': 'off',
        'cross_crease': False, 'deke': False, 'uncertain_carry': False,
    }
    if not isinstance(settings, dict):
        raise ValueError('Benchmark settings are missing.')
    for name, expected in required.items():
        if name not in settings or settings[name] != expected:
            raise ValueError(f'Benchmark requires {name}={expected!r}.')
    if settings.get('chance_creation', False):
        raise ValueError('Benchmark requires chance_creation=False.')
    if settings.get('offense_lookahead', False):
        raise ValueError('Benchmark requires offense_lookahead=False.')
    if settings.get('classic_refinements'):
        raise ValueError('Default replay gate requires classic_refinements to be disabled.')
    if (not isinstance(settings.get('matchups'), list)
            or len(settings['matchups']) != 2 or set(settings['matchups']) != set(MATCHUPS.values())):
        raise ValueError('Benchmark must contain both standard Ducks/Sabres matchups.')
    seed, trials = settings.get('seed'), settings.get('trials')
    if (not _integer(seed) or not _integer(trials) or trials < 1
            or seed < 0 or seed + trials > 2**32):
        raise ValueError('Benchmark requires positive trial counts and uint32 ROM seeds.')
    sources = report.get('sources')
    validate_sources(sources, root)
    missing = REQUIRED_SOURCES - sources.keys()
    if missing:
        raise ValueError(f'Benchmark source coverage is incomplete: {", ".join(sorted(missing))}')
    rows = report.get('matches')
    if not isinstance(rows, list) or len(rows) != 2 * trials:
        raise ValueError('Benchmark has incomplete or duplicate trial coverage.')
    matches = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('Benchmark matches must be objects.')
        side, match_seed = row.get('side'), row.get('seed')
        if not _integer(side) or not _integer(match_seed) or side not in MATCHUPS:
            raise ValueError('Benchmark match has an invalid physical side or seed.')
        key = side, match_seed
        if key in matches or row.get('matchup') != MATCHUPS[side]:
            raise ValueError('Benchmark has duplicate trials or inconsistent physical matchups.')
        if row.get('completed') is not True or row.get('clock_remaining') != 0:
            raise ValueError(f'Benchmark period is incomplete: side {side}, seed {match_seed}.')
        if not _integer(row.get('frames')) or row['frames'] < 2:
            raise ValueError('Benchmark match must record its complete native frame count.')
        digest = row.get('actions_sha256')
        if not isinstance(digest, str) or len(digest) != 64 or any(
                char not in '0123456789abcdef' for char in digest):
            raise ValueError('Benchmark match must record a valid applied-action SHA256.')
        matches[key] = row
    if set(matches) != {(side, value) for side in MATCHUPS for value in range(seed, seed + trials)}:
        raise ValueError('Benchmark seed/side coverage does not match its settings.')
    return matches


def validate_prefix(prefix, expected):
    if (prefix.get('status') != 'prefix-only' or prefix.get('cases') != []
            or prefix.get('missing_strata') != []):
        raise ValueError('Gate must measure an unbranched default replay prefix.')
    if prefix.get('seed') != expected['seed'] or prefix.get('clock_remaining') != 0:
        raise ValueError('Native replay seed is wrong or its period did not complete.')
    if (prefix.get('benchmark_frames') != expected['frames']
            or prefix.get('applied_action_frames') != expected['frames'] - 1
            or prefix.get('searched_frames') != expected['frames'] - 1):
        raise ValueError('Native replay frame count differs from the complete CPU benchmark.')
    if prefix.get('actions_sha256') != expected['actions_sha256']:
        raise ValueError('Native replay applied actions differ from the production CPU benchmark.')


def run(args, *, root=None):
    root = Path(__file__).resolve().parents[2] if root is None else Path(root)
    benchmark = json.loads(Path(args.benchmark).read_text(encoding='utf-8'))
    matches = validate_benchmark(benchmark, root)
    seed = benchmark['settings']['seed'] if args.seed is None else args.seed
    if (2, seed) not in matches:
        raise ValueError(f'Benchmark contains no standard away default trial for seed {seed}.')
    expected = matches[2, seed]
    sources = {**source_hashes(), **benchmark['sources']}
    gate_name = 'nhl94_ai/evaluation/carry_replay_gate.py'
    sources[gate_name] = hashlib.sha256((root / gate_name).read_bytes()).hexdigest()
    validate_sources(sources, root)
    if args.output:
        output = Path(args.output).resolve()
        if output == Path(args.benchmark).resolve() or output in {
                (root / name).resolve() for name in sources}:
            raise ValueError('Gate output must not overwrite its benchmark reference or measured sources.')
    prefix = replay_seed(seed, ClassicAIV1Model, search_frames=300 * 120 + 6000,
                         horizon=1, min_depth=140, strata=())
    validate_sources(sources, root)
    validate_prefix(prefix, expected)
    report = {
        'protocol': 'nhl94-carry-replay-gate-v1', 'passed': True,
        'benchmark_report': Path(args.benchmark).name,
        'fixture': {'seed': seed, 'side': 2, 'matchup': MATCHUPS[2],
                    'seconds': 300, 'frame_skip': DECISION_INTERVAL, 'action_type': 'FILTERED',
                    'goalie_policy': 'off'},
        'frames': expected['frames'], 'actions_sha256': prefix['actions_sha256'],
        'sources': sources,
        'checks': {'benchmark_complete': True, 'current_sources_match': True,
                   'sources_unchanged_during_gate': True, 'replay_period_complete': True,
                   'frame_count_identical': True, 'applied_actions_identical': True},
        'limitations': [
            'One complete away default period proves caller parity for this fixture, not playing strength.',
            'This gate does not certify alternative policies, sampled defender reach or shot outcome interpretation.',
        ],
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f'PASS: current sources and complete four-frame replay match seed {seed} '
          f'({expected["frames"]} native frames).')
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', type=Path, required=True,
                        help='Current completed default 300-second Ducks/Sabres CPU benchmark JSON.')
    parser.add_argument('--seed', type=int, help='Away seed from that report; defaults to its first seed.')
    parser.add_argument('--output', type=Path, help='Write a gate result only after every check passes.')
    return parser


if __name__ == '__main__':
    run(build_parser().parse_args())
