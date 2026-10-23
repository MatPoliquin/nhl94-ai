"""Immutable-source paired teammate matrix and bounded performance screening.

Run: python -m nhl94_ai.evaluation.teammate_benchmark run --source SNAPSHOT
     --output NEW_DIRECTORY [--count 10 --workers 12]
Compare: ... compare --before BASE --after CANDIDATE --output comparison.json
Snapshots must contain nhl94_ai; run each implementation in a fresh interpreter.
Existing completed jobs are verified and reused, never overwritten.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout, redirect_stderr
import gzip
import hashlib
import json
import math
import os
import importlib.metadata
from pathlib import Path
import shutil
import statistics
import subprocess
import sys


PROFILES = {'single': {'strategy': 'single', 'team_size': 2},
            'two': {'strategy': 'coordinated', 'team_size': 2},
            'three': {'strategy': 'coordinated', 'team_size': 3}}
for _size, _name in ((2, 'two'), (3, 'three')):
    PROFILES[_name + '-independent'] = {'strategy': 'independent', 'team_size': _size}
    PROFILES[_name + '-pressure-off'] = {'strategy': 'coordinated', 'team_size': _size, 'pressure': False}
    PROFILES[_name + '-support'] = {'strategy': 'coordinated', 'team_size': _size, 'support_options': True}

MATCH_KEYS = ('seed', 'side', 'controllers', 'seconds', 'requested_periods', 'frame_skip',
              'strategy', 'goalies', 'opponent', 'teams', 'lineup', 'initial_ratings', 'rom_sha256',
              'initial_state_sha256', 'initial_ram_sha256')


def source_hashes(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((root / 'nhl94_ai').rglob('*.py'))}


def snapshot(root, destination):
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copytree(root / 'nhl94_ai', destination / 'nhl94_ai', ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('pyproject.toml', 'README.md'):
        shutil.copy2(root / name, destination / name)
    (destination / 'source-manifest.json').write_text(json.dumps(source_hashes(destination), indent=2) + '\n')


def measure(job):
    root, output, name, settings, trace, contract_hash = job
    # Workers are spawned by a fresh interpreter with this source first on sys.path.
    from nhl94_ai.evaluation.teammates import play_match
    output = Path(output)
    target = output / (name + '.json')
    if target.exists():
        row = json.loads(target.read_text())
        if row['settings'] != settings or row.get('contract_sha256') != contract_hash or not row['result']['completed']:
            raise ValueError('Existing job is incompatible or incomplete: ' + name)
        return name, row
    trace_path = output / (name + '.jsonl') if trace else None
    with (output / (name + '.log')).open('a') as log, redirect_stdout(log), redirect_stderr(log):
        result = play_match(**settings, trace=trace_path, event_trace=output / (name + '.events.jsonl.gz'))
    row = {'settings': settings, 'contract_sha256': contract_hash, 'result': result}
    target.write_text(json.dumps(row, indent=2) + '\n')
    if trace_path:
        with trace_path.open('rb') as stream, gzip.open(str(trace_path) + '.gz', 'wb') as compressed:
            shutil.copyfileobj(stream, compressed)
        trace_path.unlink()
    if not result['completed']:
        raise RuntimeError('Incomplete match: ' + name)
    return name, row


POLICY_OPTIONS = {'support_options', 'pressure', 'offpuck', 'intent_arbitration',
                  'defensive_support', 'primary_policy', 'role_balance', 'transient_receiving'}


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def runtime_contract():
    import stable_retro
    import stable_retro._retro as native
    core = Path(stable_retro.core_path()) / 'genesis_plus_gx_libretro.so'
    return {'python': sys.version, 'executable': sys.executable,
            'native_sha256': hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
            'core_sha256': hashlib.sha256(core.read_bytes()).hexdigest(),
            'dependencies': {name: importlib.metadata.version(name) for name in
                             ('numpy', 'scipy', 'stable-retro', 'gymnasium')}}


def normalized_settings(args):
    seeds = args.seeds if args.seeds is not None else list(range(args.seed, args.seed + args.count))
    if not seeds or len(seeds) != len(set(seeds)) or any(seed < 0 or seed > 0xffffffff for seed in seeds):
        raise ValueError('Seeds must be a nonempty unique list of ROM uint32 seeds')
    if args.periods not in (1, 3) or not 1 <= args.seconds <= 300 or args.periods == 3 and args.seconds != 300:
        raise ValueError('Use one short period or three 300-second regulation periods')
    if args.frame_skip < 1 or args.workers < 1:
        raise ValueError('Cadence and workers must be positive')
    variant = json.loads(args.variant) if args.variant else {}
    if not isinstance(variant, dict) or set(variant) - POLICY_OPTIONS:
        raise ValueError('Variants may contain only team policy options; match settings are common to single and team')
    return seeds, {'state': args.state, 'seconds': args.seconds, 'periods': args.periods,
                   'frame_skip': args.frame_skip}, variant


def run_matrix(args):
    root, output, harness = args.source.resolve(), args.output.resolve(), args.harness.resolve()
    seeds, common, variant = normalized_settings(args)
    hashes, harness_hashes = source_hashes(root), source_hashes(harness)
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / 'manifest.json'
    contract = {'protocol': 'teammate-paired-v2', 'sources': hashes, 'source': str(root),
                'harness_sources': harness_hashes, 'harness': str(harness), 'runtime': runtime_contract(),
                'seeds': seeds, 'profiles': args.profiles, 'sides': args.sides,
                'settings': common, 'variant': variant, 'trace_seed': seeds[0] if args.trace else None}
    if manifest.exists() and json.loads(manifest.read_text()) != contract:
        raise ValueError('Output already belongs to a different matrix')
    manifest.write_text(json.dumps(contract, indent=2) + '\n')
    contract_hash = digest_json(contract)
    jobs = []
    for profile in args.profiles:
        for side in args.sides:
            for seed in seeds:
                settings = dict(PROFILES[profile], **common, side=side, seed=seed)
                if profile != 'single':
                    settings.update(variant)
                jobs.append((str(root), str(output), f'{profile}-{side}-{seed}', settings,
                             args.trace and seed == seeds[0], contract_hash))
    matches = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(measure, job) for job in jobs]):
            name, row = future.result()
            matches[name] = row
            print(name, row['result']['score'], flush=True)
    if source_hashes(root) != hashes or source_hashes(harness) != harness_hashes:
        raise RuntimeError('Policy or harness changed during matrix')
    from nhl94_ai.evaluation.teammates import source_hash
    if {row['result']['source_sha256'] for row in matches.values()} != {source_hash()}:
        raise ValueError('Mixed policy versions in results')
    report = {**contract, 'matches': matches}
    validate_report(report)
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


def validate_report(report):
    if report.get('protocol') != 'teammate-paired-v2':
        return  # Historical v1 comparisons remain supported, without v2 provenance claims.
    contract = {key: value for key, value in report.items() if key != 'matches'}
    expected = {f'{profile}-{side}-{seed}' for profile in report['profiles']
                for side in report['sides'] for seed in report['seeds']}
    if set(report['matches']) != expected:
        raise ValueError('Incomplete matrix')
    expected_hash = digest_json(contract)
    for name, row in report['matches'].items():
        result = row['result']
        profile, side, seed = name.rsplit('-', 2)
        settings = dict(PROFILES[profile], **report['settings'], side=side, seed=int(seed))
        if profile != 'single':
            settings.update(report['variant'])
        if row.get('contract_sha256') != expected_hash or row['settings'] != settings:
            raise ValueError('Job policy/harness/settings contract mismatch: ' + name)
        if (not result['completed'] or result['final_clock'] != 0  # pylint: disable=too-many-boolean-expressions
                or result['final_period'] != settings['periods']
                or len(result['periods']) != settings['periods']
                or any(result[k] != settings[k] for k in ('seed', 'side', 'seconds', 'frame_skip'))
                or result['requested_periods'] != settings['periods']
                or result['controllers'] != (1 if profile == 'single' else settings['team_size'])):
            raise ValueError('Incomplete or misconfigured match: ' + name)


def paired_interval(values, confidence=0.95):
    """Two-sided Student-t interval with uncapped degrees of freedom."""
    from scipy.stats import t
    if len(values) < 2 or not 0 < confidence < 1:
        raise ValueError('At least two pairs and confidence strictly between zero and one required')
    critical = float(t.ppf((1 + confidence) / 2, len(values) - 1))
    mean, sd = statistics.mean(values), statistics.stdev(values)
    radius = critical * sd / math.sqrt(len(values))
    result = {'n': len(values), 'mean': mean, 'sd': sd, 'confidence': confidence,
              'critical': critical, 'interval': [mean - radius, mean + radius], 'pairs': values}
    if confidence == 0.95:
        result['ci95'] = result['interval']
    return result


def outcomes(row, unit='per120'):
    if unit not in ('game', 'per120'):
        raise ValueError('Outcome unit must be game or per120')
    factor = 1 if unit == 'game' else 120 / (row['seconds'] * row['requested_periods'])
    counts = row['teammate_diagnostics']['counts']
    return {key: value * factor for key, value in {
        'goals_for': row['score'][0], 'goals_against': row['score'][1],
        'goal_difference': row['score'][0] - row['score'][1],
        'shots_for': row['shots'][0], 'shots_against': row['shots'][1],
        'one_timer_attempts': row['one_timer_attempts'], 'one_timer_goals': row['one_timer_goals'],
        'receptions': row['reception_outcomes']['receptions'], 'turnovers': row['skater_turnovers'],
        'danger_entries_against': counts.get('danger_entries_against', 0)}.items()}


def compare(before, after, *, confidence=0.95, unit='per120'):
    validate_report(before)
    validate_report(after)
    if before.get('protocol') == 'teammate-paired-v2' and any(before[k] != after.get(k) for k in ('harness_sources', 'runtime', 'settings', 'seeds')):
        raise ValueError('Measurement harness/runtime/settings/seeds differ')
    if set(before['matches']) != set(after['matches']):
        raise ValueError('Seed/profile sets differ')
    groups = {}
    for name, new in sorted(after['matches'].items()):
        old = before['matches'][name]
        a, b = old['result'], new['result']
        if not a['completed'] or not b['completed'] or any(a[key] != b[key] for key in MATCH_KEYS):
            raise ValueError('Incomplete or unmatched pair: ' + name)
        profile = name.rsplit('-', 2)[0]
        key = profile + '-' + b['side']
        groups.setdefault(key, []).append((a, b))
    result = {}
    for key, pairs in groups.items():
        durations = {a['seconds'] * a['requested_periods'] for a, _ in pairs}
        if len(durations) != 1:
            raise ValueError('Duration differs between seeds: ' + key)
        duration = durations.pop()
        # Changing the display unit must never relax the decision thresholds.
        # Historical short screens retain their per120 margins; regulation uses
        # the separately registered per900 margins, not the historical screen.
        basis = 900 if duration == 900 else 120
        scale = duration / basis if unit == 'game' else 120 / basis
        margins = {'goal_difference': -.50 * scale, 'goals_against': .35 * scale,
                   'danger_entries_against': 1.0 * scale}
        delta = {metric: paired_interval([outcomes(b, unit)[metric] - outcomes(a, unit)[metric] for a, b in pairs], confidence)
                 for metric in outcomes(pairs[0][0])}
        gd, ga, danger = (delta[metric]['interval'] for metric in
                          ('goal_difference', 'goals_against', 'danger_entries_against'))
        accepted = gd[0] > margins['goal_difference'] and ga[1] < margins['goals_against'] and danger[1] < margins['danger_entries_against']
        regressed = gd[1] < margins['goal_difference'] or ga[0] > margins['goals_against'] or danger[0] > margins['danger_entries_against']
        result[key] = {'delta': delta, 'status': 'pass' if accepted else 'regression' if regressed else 'inconclusive',
                       'thresholds_in_display_unit': margins,
                       'threshold_basis': 'regulation per900' if basis == 900 else 'historical per120 screen; not regulation acceptance',
                       'gain': accepted and gd[0] > 0,
                       'exact_inputs': all(a['actions_sha256'] == b['actions_sha256'] for a, b in pairs),
                       'exact_final_ram': all(a['final_ram_sha256'] == b['final_ram_sha256'] for a, b in pairs),
                       'before_means': {m: statistics.mean(outcomes(a, unit)[m] for a, _ in pairs) for m in delta},
                       'after_means': {m: statistics.mean(outcomes(b, unit)[m] for _, b in pairs) for m in delta}}
    return {'protocol': 'paired-t-v2', 'groups': result, 'confidence': confidence, 'unit': unit,
            'extension': 'none; seed count governed by registered study'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    snap = commands.add_parser('snapshot')
    snap.add_argument('--source', type=Path, default=Path.cwd())
    snap.add_argument('--output', type=Path, required=True)
    run = commands.add_parser('run')
    run.add_argument('--source', type=Path, required=True)
    run.add_argument('--output', type=Path, required=True)
    run.add_argument('--seed', type=int, default=9400)
    run.add_argument('--count', type=int, default=10)
    run.add_argument('--seeds', nargs='+', type=int)
    run.add_argument('--periods', type=int, choices=(1, 3), default=1)
    run.add_argument('--frame-skip', type=int, default=4)
    run.add_argument('--state', default='SabresVsMightyDucks.ManualGoalie.Start')
    run.add_argument('--sides', nargs='+', choices=('home', 'away'), default=['home', 'away'])
    run.add_argument('--harness', type=Path, default=Path(__file__).resolve().parents[2])
    run.add_argument('--seconds', type=int, default=120)
    run.add_argument('--workers', type=int, default=12)
    run.add_argument('--profiles', nargs='+', choices=tuple(PROFILES), default=['single', 'two', 'three'])
    run.add_argument('--trace', action='store_true')
    run.add_argument('--variant', help='JSON play_match options; never applied to single')
    run.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    comparison = commands.add_parser('compare')
    comparison.add_argument('--before', type=Path, required=True)
    comparison.add_argument('--after', type=Path, required=True)
    comparison.add_argument('--output', type=Path, required=True)
    comparison.add_argument('--confidence', type=float, default=0.95)
    comparison.add_argument('--unit', choices=('game', 'per120'), default='game')
    args = parser.parse_args()
    if args.command == 'snapshot':
        snapshot(args.source.resolve(), args.output.resolve())
    elif args.command == 'compare':
        report = compare(json.loads(args.before.read_text()), json.loads(args.after.read_text()), confidence=args.confidence, unit=args.unit)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    elif args.worker:
        run_matrix(args)
    else:
        normalized_settings(args)
        # Load policy package from its immutable snapshot; pin ALL measurement modules separately.
        command = ('import sys; sys.path.insert(0, sys.argv.pop(1)); '
                   'import nhl94_ai.evaluation; nhl94_ai.evaluation.__path__.insert(0, sys.argv.pop(1)); '
                   'from nhl94_ai.evaluation.teammate_benchmark import main; main()')
        env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        subprocess.run([sys.executable, '-c', command, str(args.source.resolve()),
                        str(args.harness.resolve() / 'nhl94_ai/evaluation'), *sys.argv[1:],
                        '--harness', str(args.harness.resolve()), '--worker'], check=True, env=env)



if __name__ == '__main__':
    main()
