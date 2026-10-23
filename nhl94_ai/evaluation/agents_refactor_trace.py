"""Replay one saved CPU trial with identical, external action instrumentation.

Invoke this file directly with --root to trace a frozen or candidate checkout.
The saved uninstrumented match must reproduce exactly before its trace is usable.
Tracing is deliberately outside elapsed-time benchmark measurements.
"""
import argparse
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict, is_dataclass
import gzip
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch


PROTOCOL = 'agents-refactor-native-trace-v1'


def _json(value):
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, 'tolist'):
        return value.tolist()
    raise TypeError(f'Unsupported trace evidence: {type(value).__name__}')


def _input(state):
    result = {'engine': vars(state.engine), 'puck': vars(state.puck), 'time': state.time,
              'shooting': state.is_shooting, 'shootout': state.is_shootout_active}
    for name in ('team1', 'team2'):
        team = getattr(state, name)
        result[name] = {field: getattr(team, field) for field in (
            'controller', 'control', 'defense_control', 'defense_goalie', 'pass_attempts', 'one_timer_attempts')}
        result[name].update(net=vars(team.net), stats=vars(team.stats), goalie=vars(team.goalie),
                            players=[vars(player) for player in team.players])
    return deepcopy(result)


def verify_result(measured, saved, frames, applied_hash):
    """Compare the native result in the same JSON representation as CPU reports."""
    measured = json.loads(json.dumps(measured, default=_json))
    if measured != saved:
        fields = sorted(key for key in measured.keys() | saved.keys()
                        if measured.get(key) != saved.get(key))
        raise RuntimeError(f'Instrumented trial does not reproduce saved report: {fields}')
    if frames + 1 != measured['frames'] or applied_hash != measured['actions_sha256']:
        raise RuntimeError('Applied trace does not reproduce the saved action hash/frame count')
    return measured


def measure(root, expected_report, matchup, seed, output):
    root, output = root.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(output)
    saved = json.loads(expected_report.read_text())
    selected = [row for row in saved['matches'] if (row['matchup'], row['seed']) == (matchup, seed)]
    if len(selected) != 1 or not selected[0]['completed']:
        raise ValueError('Expected report must contain exactly one complete matching trial')
    for name, digest in saved['sources'].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'The source does not match the saved report: {name}')
    sys.path.insert(0, str(root))
    import numpy as np
    from nhl94_ai.evaluation import cpu_benchmark as cpu
    from nhl94_ai.env import factory
    if Path(cpu.__file__).resolve() != root / 'nhl94_ai/evaluation/cpu_benchmark.py':
        raise RuntimeError('Trace imported the wrong source root')
    output.mkdir(parents=True)
    pending = None
    frames = 0
    applied_hash = hashlib.sha256()
    make_env, create = factory.make_retro, cpu.create_scripted
    with gzip.open(output / 'frames.jsonl.gz', 'wt', encoding='utf-8') as trace:
        def traced_env(*args, **kwargs):
            env = make_env(*args, **kwargs)
            step = env.step

            def applied(action):
                nonlocal pending, frames
                # The neutral initialization step precedes the policy timeline.
                if pending is not None:
                    frames += 1
                    buttons = np.asarray(action, dtype=np.int8)
                    applied_hash.update(buttons.tobytes())
                    trace.write(json.dumps(dict(pending, frame=frames, applied=buttons.tolist()),
                                           default=_json, separators=(',', ':')) + '\n')
                    pending = None
                return step(action)

            env.step = applied
            return env

        def traced_agent(*args, **kwargs):
            agent = create(*args, **kwargs)
            predict = agent.predict_game_state

            def observed(state, deterministic=True):
                nonlocal pending
                inputs = _input(state)
                action = predict(state, deterministic)
                pending = {'input': inputs, 'requested': action.tolist(),
                           'decision': agent._last_decision, 'target': agent._last_target,
                           'offense': deepcopy(agent.offense_diagnostics),
                           'defense': deepcopy(agent.defense_diagnostics),
                           'goalie': deepcopy(agent.goalie_diagnostics)}
                return action

            agent.predict_game_state = observed
            return agent

        args = SimpleNamespace(**dict(saved['settings'], matchups=[matchup], seed=seed,
                                      trials=1, workers=1, output=None))
        with patch.object(factory, 'make_retro', traced_env), patch.object(cpu, 'create_scripted', traced_agent), \
                (output / 'runner.stdout.log').open('w') as stdout, redirect_stdout(stdout):
            measured = cpu.run(args)['matches'][0]
    (output / 'measured-result.json').write_text(json.dumps(measured, default=_json, indent=2) + '\n')
    measured = verify_result(measured, selected[0], frames, applied_hash.hexdigest())
    report = {'protocol': PROTOCOL, 'source_root': str(root), 'sources': saved['sources'],
              'instrumentation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'expected_report': str(expected_report.resolve()), 'settings': saved['settings'],
              'result': measured, 'trace': 'frames.jsonl.gz', 'traced_frames': frames,
              'uninstrumented_match_reproduced': True}
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def _rows(path, report):
    digest = hashlib.sha256()
    count = 0
    with gzip.open(path.parent / report['trace'], 'rt', encoding='utf-8') as stream:
        for count, line in enumerate(stream, 1):
            row = json.loads(line)
            if row['frame'] != count or len(row['applied']) != 12 or any(v not in (0, 1) for v in row['applied']):
                raise ValueError('Invalid native trace frame/buttons')
            digest.update(bytes(row['applied']))
            yield row
    if (count != report['traced_frames'] or count + 1 != report['result']['frames']
            or digest.hexdigest() != report['result']['actions_sha256']):
        raise ValueError('Trace is incomplete or its applied action hash is invalid')


def compare(reference, candidate):
    before, after = (json.loads(path.read_text()) for path in (reference, candidate))
    for report in (before, after):
        if (report.get('protocol') != PROTOCOL or not report.get('uninstrumented_match_reproduced')
                or not report['result']['completed'] or report['result']['clock_remaining'] != 0):
            raise ValueError('Trace must reproduce a complete uninstrumented trial')
    if before['instrumentation_sha256'] != after['instrumentation_sha256']:
        raise ValueError('Trace instrumentation differs')
    for field in ('matchup', 'seed', 'side', 'lineup', 'initial_state_sha256'):
        if before['result'][field] != after['result'][field]:
            raise ValueError(f'Trace trial provenance differs: {field}')
    settings = [{key: value for key, value in report['settings'].items() if key != 'output'}
                for report in (before, after)]
    if settings[0] != settings[1]:
        raise ValueError('Trace effective settings differ')
    first_action = first_input = first_diagnostic = None
    for old, new in zip_longest(_rows(reference, before), _rows(candidate, after)):
        if first_action is None and (old is None or new is None or old['applied'] != new['applied']):
            first_action = {'frame': (old or new)['frame'], 'before': old, 'after': new}
        if old is not None and new is not None:
            if first_input is None and old['input'] != new['input']:
                first_input = old['frame']
            keys = (old.keys() | new.keys()) - {'applied', 'input', 'frame'}
            if first_diagnostic is None and any(old.get(key) != new.get(key) for key in keys):
                first_diagnostic = old['frame']
    return {'actions_identical': first_action is None, 'first_action_difference': first_action,
            'first_input_difference_frame': first_input, 'first_diagnostic_difference_frame': first_diagnostic,
            'before_result': before['result'], 'after_result': after['result'],
            'instrumentation_sha256': before['instrumentation_sha256']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--expected-report', type=Path)
    parser.add_argument('--matchup', choices=('sabres-ducks-manual', 'ducks-sabres-manual'))
    parser.add_argument('--seed', type=int)
    parser.add_argument('--compare', nargs=2, type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.compare:
        report = compare(*args.compare)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    else:
        if None in (args.root, args.expected_report, args.matchup, args.seed):
            parser.error('Measurement requires root, expected report, matchup, and seed')
        measure(args.root, args.expected_report, args.matchup, args.seed, args.output)
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
