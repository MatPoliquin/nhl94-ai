"""Strict paired comparison for the Ducks/Sabres agent-refactor protocol.

This tool only reads reports and manifests; it never imports or runs a policy.
Behavior mode establishes comparability and reports changes, not acceptance.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


PROTOCOL = 'nhl94-cpu-first-period-v2'
COMPARISON_PROTOCOL = 'agents-refactor-comparison-v1'
SIDES = {'sabres-ducks-manual': 1, 'ducks-sabres-manual': 2}
CORE_PROFILES = {'native-1': 1, 'native-4': 4}
CORE_OPTIONS = {
    'goalie_policy': 'selective', 'goalie_positioning': 'deep-csave',
    'goalie_planner': None, 'retention_model': None, 'goalie_save_timing': 'classic',
    'goalie_model': None, 'goalie_decision_interval': None, 'classic_control_interval': 0,
    'rebound_recovery': False, 'one_timer_execution': None, 'reception_control': False,
    'shot_placement': None, 'receiver_selection': None, 'possession_ablation': None,
    'possession_value': False, 'classic_refinements': [], 'offense_lookahead': False,
    'cross_crease': False, 'opportunistic_cross_crease': False, 'deke': False,
    'uncertain_carry': False, 'chance_creation': False, 'teammate_support_options': False,
}
# All other match fields, including event/metric payloads, remain gameplay evidence.
DIAGNOSTICS = {'decisions'}
ASSET_FIELDS = ('shot_placement_sha256', 'goalie_model_sha256', 'retention_model_sha256')
METRICS = ('pass_outcomes', 'ordinary_pass_metrics', 'one_timer_metrics',
           'goalie_metrics', 'goalie_context_metrics', 'carrier_defense_metrics',
           'offense_metrics', 'carry_metrics', 'retention_metrics', 'reception_metrics')


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _seeds(tier):
    _require(tier in ('full', 'check'), 'Unknown coverage tier')
    return set(range(8000, 8020 if tier == 'full' else 8002))


def _settings(report):
    # These two fields describe output location and the explicitly checked subset.
    return {key: value for key, value in report['settings'].items()
            if key not in ('output', 'trials')}


def _manifest(directory):
    result = _read(Path(directory) / 'run-manifest.json')
    for name in ('revision', 'sources', 'assets', 'dependencies', 'python', 'machine'):
        _require(result.get(name), f'{directory}: missing manifest {name}')
    names = {Path(name).name for name in result['assets']}
    _require({'rom.md', 'SabresVsMightyDucks.ManualGoalie.Start.state'} <= names,
             f'{directory}: missing ROM/save-state hashes')
    _require(all(isinstance(value, str) and len(value) == 64 for value in result['assets'].values()),
             f'{directory}: invalid asset hashes')
    return result


def _validate(report, manifest, cadence, seeds, *, core):
    _require(report.get('protocol') == PROTOCOL, 'Wrong benchmark protocol')
    settings = report['settings']
    required = {'agent': 'classic-v1', 'seed': 8000, 'seconds': 300,
                'frame_skip': cadence, 'workers': 1, 'trials': len(seeds)}
    if core:
        required.update(CORE_OPTIONS, action_type='FILTERED')
    for key, value in required.items():
        _require(settings.get(key) == value, f'Wrong profile setting: {key}')
    _require(settings.get('matchups') == list(SIDES), 'Wrong matchup matrix')
    _require(report.get('sources'), 'Missing report source fingerprints')
    for name, digest in report['sources'].items():
        _require(manifest['sources'].get(name) == digest, f'Source/manifest mismatch: {name}')
    versions = report.get('versions', {})
    _require(set(versions) >= {'nhl94-ai', 'stable-retro', 'numpy'}, 'Missing dependency versions')
    installed = {name.lower().replace('_', '-'): version
                 for name, version in manifest['dependencies'].items()}
    for name, version in versions.items():
        _require(installed.get(name.lower().replace('_', '-')) == version,
                 f'Dependency/manifest mismatch: {name}')
    for name in ASSET_FIELDS:
        _require(name in report, f'Missing artifact fingerprint: {name}')
        if report[name] is not None:
            _require(report[name] in manifest['assets'].values(), f'Unmanifested artifact: {name}')
    expected = {(matchup, seed) for matchup in SIDES for seed in seeds}
    indexed = {}
    for row in report['matches']:
        key = row['matchup'], row['seed']
        _require(key in expected, f'Unexpected trial: {key}')
        _require(key not in indexed, f'Duplicate trial: {key}')
        _require(not row.get('error') and row.get('completed') is True and row.get('clock_remaining') == 0
                 and isinstance(row.get('frames'), int) and row['frames'] > 1,
                 f'Incomplete trial: {key}')
        _require(row.get('side') == SIDES[key[0]], f'Wrong controller side: {key}')
        _require(row.get('agent') == settings['agent'], f'Wrong trial agent: {key}')
        _require(row.get('teams') == [2, 0] and len(row.get('lineup', [])) == 12,
                 f'Wrong teams/lineup: {key}')
        _require({player.get('slot') for player in row['lineup']} == set(range(12))
                 and all(player.get('name') for player in row['lineup']), f'Invalid lineup: {key}')
        _require(len(row.get('actions_sha256', '')) == 64
                 and len(row.get('initial_state_sha256', '')) == 64, f'Missing trial hashes: {key}')
        for field in ('goals', 'shots', 'one_timers', 'one_timer_goals'):
            _require(len(row.get(field, [])) == 2, f'Missing home/away {field}: {key}')
        for name in CORE_OPTIONS.keys() - {'teammate_support_options'}:
            _require(name in row and row[name] == settings[name], f'Trial/profile mismatch: {name}, {key}')
        accounting = row.get('one_timer_accounting', {})
        _require('started' in accounting and accounting['started'] == accounting.get('ended'),
                 f'Unbalanced lifecycle: {key}')
        indexed[key] = row
    _require(set(indexed) == expected, f'Missing trials: {sorted(expected - indexed.keys())}')
    return indexed


def _differences(before, after, fields):
    return {key: {'before': before.get(key), 'after': after.get(key)} for key in sorted(fields)
            if key not in before or key not in after or before[key] != after[key]}


def _sum_metrics(rows, field):
    total = Counter()

    def add(value, path=''):
        if isinstance(value, dict):
            for key, child in value.items():
                add(child, f'{path}.{key}' if path else key)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            total[path] += value

    for row in rows:
        add(row.get(field, {}))
    return dict(total)


def summarize(rows):
    scores = [(row['goals'][row['side'] - 1], row['goals'][2 - row['side']]) for row in rows]
    result = {'completed': len(rows), 'wins': sum(a > b for a, b in scores),
              'draws': sum(a == b for a, b in scores), 'losses': sum(a < b for a, b in scores),
              'goals_for': sum(a for a, _ in scores), 'goals_against': sum(b for _, b in scores),
              'frames': sum(row['frames'] for row in rows),
              'actions_sha256': {str(row['seed']): row['actions_sha256'] for row in rows}}
    for field in ('shots', 'one_timers', 'one_timer_goals'):
        result[field + '_for'] = sum(row[field][row['side'] - 1] for row in rows)
        result[field + '_against'] = sum(row[field][2 - row['side']] for row in rows)
    result['metrics'] = {field: _sum_metrics(rows, field) for field in METRICS}
    return result


def compare(reference, candidate, *, tier='full', mode='exact', profiles=None, reference_tier='full'):
    """Require the entire declared matrix, selecting checkpoint seeds explicitly."""
    _require(mode in ('exact', 'behavior'), 'Unknown comparison mode')
    reference, candidate = Path(reference), Path(candidate)
    core = profiles is None
    profiles = CORE_PROFILES if core else profiles
    _require(bool(profiles), 'An empty profile matrix cannot pass')
    before_manifest, after_manifest = _manifest(reference), _manifest(candidate)
    for field in ('assets', 'dependencies', 'python', 'machine'):
        _require(before_manifest[field] == after_manifest[field], f'Manifest mismatch: {field}')
    selected = _seeds(tier)
    _require(selected <= _seeds(reference_tier), 'Reference lacks required seed coverage')
    if core:
        for directory in (reference, candidate):
            _require({p.stem for p in directory.glob('native-*.json')} == set(profiles),
                     f'Unexpected or missing cadence reports: {directory}')
    paired, summaries, source_changes = [], [], {}
    for profile, cadence in profiles.items():
        old_report, new_report = (_read(directory / f'{profile}.json')
                                  for directory in (reference, candidate))
        old = _validate(old_report, before_manifest, cadence, _seeds(reference_tier), core=core)
        new = _validate(new_report, after_manifest, cadence, selected, core=core)
        _require(_settings(old_report) == _settings(new_report), f'Effective settings differ: {profile}')
        for field in ('protocol', 'versions', *ASSET_FIELDS):
            _require(old_report[field] == new_report[field], f'Report metadata differs: {field}')
        source_changes[profile] = _differences(old_report['sources'], new_report['sources'],
                                               old_report['sources'].keys() | new_report['sources'].keys())
        for key, row in new.items():
            original = old[key]
            for field in ('side', 'lineup', 'teams', 'initial_state_sha256'):
                _require(original[field] == row[field], f'Trial provenance differs: {key}, {field}')
            fields = original.keys() | row.keys()
            paired.append({'profile': profile, 'cadence': cadence, 'matchup': key[0], 'seed': key[1],
                           'action_hash_match': original['actions_sha256'] == row['actions_sha256'],
                           'gameplay': _differences(original, row, fields - DIAGNOSTICS),
                           'diagnostics': _differences(original, row, fields & DIAGNOSTICS)})
        for matchup in SIDES:
            before_rows = [old[matchup, seed] for seed in sorted(selected)]
            after_rows = [new[matchup, seed] for seed in sorted(selected)]
            a, b = summarize(before_rows), summarize(after_rows)
            summaries.append({'profile': profile, 'cadence': cadence, 'matchup': matchup,
                              'before': a, 'after': b,
                              'goal_difference_delta': (b['goals_for'] - b['goals_against']) -
                                                       (a['goals_for'] - a['goals_against']),
                              'action_hash_matches': sum(x['actions_sha256'] == y['actions_sha256']
                                                         for x, y in zip(before_rows, after_rows))})
    exact = all(not row['gameplay'] and not row['diagnostics'] for row in paired)
    return {'protocol': COMPARISON_PROTOCOL, 'comparator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'mode': mode, 'tier': tier, 'reference': str(reference), 'candidate': str(candidate),
            'passed': exact if mode == 'exact' else None, 'exact': exact,
            'requires_behavior_review': mode == 'behavior', 'paired': paired, 'summaries': summaries,
            'source_changes': source_changes,
            'timing': {'before': before_manifest.get('timings_seconds'),
                       'after': after_manifest.get('timings_seconds')}}


def readable(report):
    lines = [f"{report['mode']} / {report['tier']}: exact={report['exact']}",
             'Profile | Agent team | Completed | W/D/L | GF/GA | GD delta | Hash matches']
    for row in report['summaries']:
        result = row['after']
        lines.append(f"{row['profile']} | {row['matchup'].split('-')[0]} | {result['completed']} | "
                     f"{result['wins']}/{result['draws']}/{result['losses']} | "
                     f"{result['goals_for']}/{result['goals_against']} | "
                     f"{row['goal_difference_delta']:+d} | {row['action_hash_matches']}/{result['completed']}")
    lines.append(f"Changed trials: {sum(bool(r['gameplay'] or r['diagnostics']) for r in report['paired'])}")
    if report['requires_behavior_review']:
        lines.append('Behavior changes require focused reproduction, divergence review, and the plan escalation rule.')
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--tier', choices=('full', 'check'), default='full')
    parser.add_argument('--reference-tier', choices=('full', 'check'), default='full')
    parser.add_argument('--mode', choices=('exact', 'behavior'), default='exact')
    parser.add_argument('--profiles', nargs='+', help='Optional report stems and cadences, e.g. retention:4')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists() and not args.output.with_suffix('.txt').exists(), 'Refusing to overwrite evidence')
    profiles = dict((name, int(cadence)) for name, cadence in
                    (item.rsplit(':', 1) for item in args.profiles)) if args.profiles else None
    report = compare(args.reference, args.candidate, tier=args.tier, mode=args.mode,
                     profiles=profiles, reference_tier=args.reference_tier)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    args.output.with_suffix('.txt').write_text(readable(report), encoding='utf-8')
    print(readable(report), end='')
    if report['passed'] is False:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
