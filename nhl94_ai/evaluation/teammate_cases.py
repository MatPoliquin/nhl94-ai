"""Bounded complete-history forks from predeclared native pass failure cases.

Run in a fresh interpreter with the selected policy package and pinned evaluator.
Pickles are private local artifacts and must only be loaded from trusted captures.
"""
import argparse
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import random
from types import SimpleNamespace

import numpy as np

from nhl94_ai.agents.teammates import TeammateCoordinator
from nhl94_ai.env.actions import filter_menu_actions
from nhl94_ai.evaluation.teammates import DEFAULT_STATE, make_match_env, prepare_match
from nhl94_ai.evaluation.teammate_metrics import ReceptionOutcomes
from nhl94_ai.evaluation.teammate_diagnostics import TeammateDiagnostics
from nhl94_ai.evaluation.cpu_benchmark import CPU_RAM


def arguments(settings):
    return SimpleNamespace(env='NHL94-Genesis-v0', action_type='FILTERED', goalie_policy='off',
        hyperparams_dict={'frame_skip': settings.get('frame_skip', 4)}, side=settings['side'],
        team_size=settings['team_size'], mode='team_vs_game',
        **{'teammate_' + key: settings.get(key, 'baseline' if key=='primary_policy' else key=='pressure') for key in
           ('support_options', 'pressure', 'offpuck', 'intent_arbitration', 'defensive_support', 'primary_policy', 'role_balance')})


def capture(env, agent, info):
    return {'emulator': env.em.get_state(), 'agent': deepcopy(agent), 'info': deepcopy(info),
            'python_random': random.getstate(), 'numpy_random': np.random.get_state(),
            'ram_sha256': hashlib.sha256(env.get_ram().tobytes()).hexdigest()}


def restore(env, snapshot):
    assert env.em.set_state(snapshot['emulator'])
    env.data.update_ram()
    assert hashlib.sha256(env.get_ram().tobytes()).hexdigest() == snapshot['ram_sha256']
    random.setstate(snapshot['python_random']); np.random.set_state(snapshot['numpy_random'])
    return deepcopy(snapshot['agent']), deepcopy(snapshot['info'])


def branch(env, snapshot, mode, horizon=240):
    agent, info = restore(env, snapshot)
    if mode == 'disable-arbitration':
        agent.intent_arbitration = False
    elif mode == 'preserve-transient-commitment':
        original = agent.receiving.finish
        def finish(frame, reason):
            if (reason == 'receiver-selection-lost' and agent.receiving.pending is not None
                    and agent.receiving.pending.phase == 'flight'
                    and any(s.team1.controlled_scnum() < 0 for s in agent.states)):
                return
            original(frame, reason)
        agent.receiving.finish = finish
    elif mode == 'disable-movement-before-release':
        agent.offpuck = False
    elif mode == 'coast-flight':
        original = agent.receiving.receiver_action
        def coast(*args, **kwargs):
            result = original(*args, **kwargs)
            if result is not None and agent.receiving.pending is not None and agent.receiving.pending.phase == 'flight':
                action, target, role = result
                action = action.copy(); action[4:8] = 0
                return action, target, role
            return result
        agent.receiving.receiver_action = coast
    elif mode == 'refresh-receiver-controller':
        original = agent.receiving.observe
        def observe(state, slots, frame):
            original(state, slots, frame)
            plan = agent.receiving.pending
            if plan is not None and plan.receiver in slots:
                plan.receiver_controller = slots.index(plan.receiver) + 1
        agent.receiving.observe = observe
    metrics, receptions = TeammateDiagnostics(), ReceptionOutcomes()
    rows, digest = [], hashlib.sha256()
    initial_scores = [info[f'p{i}_score'] for i in (agent.side, 3-agent.side)]
    initial_shots = [info[f'p{i}_shots'] for i in (agent.side, 3-agent.side)]
    for _ in range(horizon):
        info = agent.controllers[0].platform.frame_info(env, info)
        output = agent.act(info)
        action = filter_menu_actions(env.data, output.action)
        diagnostics = output.diagnostics['teammates']; state = agent.states[0]
        highlight = bool(info.get('goalie_clock_flags', 0) & 0x10)
        slots = [r['slot'] for r in diagnostics['controllers']]
        receptions.observe(agent.frames, state, slots, highlight=highlight)
        metrics.observe(agent.frames, state, diagnostics, action, highlight=highlight)
        plan = agent.receiving.pending
        mismatch = plan is not None and plan.receiver_controller is not None and slots[plan.receiver_controller-1] != plan.receiver
        rows.append({'frame': agent.frames, 'owner': info['puck_owner'], 'target': info.get('pass_target'),
                     'slots': slots, 'pads': action.tolist(), 'roles': [r['role'] for r in diagnostics['controllers']],
                     'sources': [r.get('input_source') for r in diagnostics['controllers']],
                     'commitment': plan.snapshot() if plan else None, 'receiver_controller_mismatch': bool(mismatch),
                     'events': deepcopy(diagnostics.get('events', [])),
                     'requests': [r.get('b_intent') for r in diagnostics['controllers']],
                     'arbitration': [r.get('b_arbitration') for r in diagnostics['controllers']]})
        digest.update(action.astype(np.int8).tobytes())
        *_, info = env.step(action)
    receptions.finish(agent.frames+1, 'case-horizon')
    return {'actions_sha256': digest.hexdigest(), 'final_ram_sha256': hashlib.sha256(env.get_ram().tobytes()).hexdigest(),
            'score': [info[f'p{i}_score']-v for i,v in zip((agent.side,3-agent.side),initial_scores)],
            'shots_raw_delta': [info[f'p{i}_shots']-v for i,v in zip((agent.side,3-agent.side),initial_shots)],
            'passes': receptions.events, 'measurements': metrics.summary(), 'rows': rows}


def run(selection, group, output):
    cases = [c for c in selection['cases'] if c['id'].startswith(group+'-')]
    assert cases
    settings = cases[0]['settings']; args = arguments(settings)
    output.mkdir(parents=True, exist_ok=False)
    captured = {}
    with make_match_env(settings.get('state', DEFAULT_STATE), team_size=args.team_size) as env:
        for name, (address, kind) in CPU_RAM.items():
            env.data.set_variable(name, {'address': address, 'type': kind})
        info = prepare_match(env, seed=settings['seed'], seconds=settings['seconds'], side=1 if args.side=='home' else 2,
                             team_size=args.team_size)
        agent = TeammateCoordinator(args)
        digest = hashlib.sha256()
        for frame in range(1, cases[0]['reference_frames']+1):
            info = agent.controllers[0].platform.frame_info(env, info)
            for case in cases:
                if frame == case['capture_frame']:
                    snapshot = capture(env, agent, info)
                    captured[case['id']] = snapshot
                    (output/(case['id']+'.pickle.gz')).write_bytes(gzip.compress(pickle.dumps(snapshot)))
            action = filter_menu_actions(env.data, agent.act(info).action)
            digest.update(action.astype(np.int8).tobytes())
            *_, info = env.step(action)
        assert digest.hexdigest() == cases[0]['reference_actions_sha256'], 'Native caller must reproduce entire historical input trace'
        results = {}
        for case in cases:
            snapshot = captured[case['id']]
            modes = (('default', 'repeat-default', 'disable-arbitration') if group.startswith('5a') else
                     ('default', 'repeat-default', 'coast-flight', 'refresh-receiver-controller', 'disable-movement-before-release', 'preserve-transient-commitment'))
            results[case['id']] = {'selection': case, 'branches': {mode: branch(env, snapshot, mode) for mode in modes}}
            a,b = (results[case['id']]['branches'][mode] for mode in ('default','repeat-default'))
            assert a == b, 'Complete-history branch replay failed'
        from nhl94_ai.evaluation.teammates import source_hash
        provenance = {'policy_sha256': source_hash(), 'case_runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        (output/'report.json').write_text(json.dumps({'group': group, 'provenance': provenance, 'prefix_parity': True, 'cases': results},indent=2)+'\n')
    print(group, len(results), 'cases; exact prefix and complete-history replay')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--group', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.selection.read_text()), args.group, args.output)


if __name__ == '__main__':
    main()
