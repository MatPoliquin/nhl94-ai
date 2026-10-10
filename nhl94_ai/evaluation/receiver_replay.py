"""Fork every admitted recipient from sampled legacy pass decisions.

Restores emulator/RNG, decoded state and controller history. Every branch passes
at the same time and continues with legacy tactics after that initial choice.
Samples precede outcome observation; no ROM/snapshot bytes are saved.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.agents.receiver_selection import ReceiverSelector
from nhl94_ai.env.factory import make_retro
from nhl94_ai.evaluation.cpu_benchmark import CPU_RAM, MATCHUPS, select_side
from nhl94_ai.evaluation.possession_ablation_replay import observation, source_hashes as agent_hashes
from nhl94_ai.evaluation.refinement_replay import advance, fork, read_view, restore
from nhl94_ai.game.state import NHL94GameState


def configured(history, slot=None):
    model = deepcopy(history)
    model.receiver_selector = ReceiverSelector('legacy')
    model.receiver_selector.expand_pool = history.receiver_selector.expand_pool
    model.receiver_selector.forced_slot = slot
    return model


def common_continuation(policy):
    model = deepcopy(policy)
    model.receiver_selector = None
    model.args = deepcopy(model.args)
    model.args.receiver_selection = None
    return model


def source_hashes():
    sources = agent_hashes()
    root = Path(__file__).resolve().parents[2]
    for name in ('receiver_replay.py', 'pass_outcomes.py'):
        path = root / 'nhl94_ai/evaluation' / name
        sources[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return sources


def run_job(job):
    matchup, seed, horizon, limit, spacing, goalie_policy, all_safe = job
    save, side = MATCHUPS[matchup]
    env = make_retro(game='NHL94-Genesis-v0', state=save, num_players=1, goalie_policy=goalie_policy)
    cases, counts, last_sample = [], Counter(), -spacing
    try:
        for name, (address, kind) in CPU_RAM.items():
            env.data.set_variable(name, {'address': address, 'type': kind})
        env.reset(seed=seed)
        env.data.update_ram()
        select_side(env.data, env.data.lookup_all(), side)
        env.data.set_value('bench_rng', seed)
        env.data.set_value('bench_clock', 300)
        env.step(np.zeros(12, dtype=np.int8))
        model = ClassicAIV1Model(SimpleNamespace(
            action_type='FILTERED', goalie_policy=goalie_policy, receiver_selection='legacy'))
        model.receiver_selector.expand_pool = all_safe
        state, frame, digest = NHL94GameState(5), 1, hashlib.sha256()
        while frame < 42000:
            view, info = read_view(env, state, side)
            if info['bench_clock'] == 0:
                break
            eligible = (model.scheduler.remaining == 0 and not view.engine.clock_stopped
                        and ReceiverSelector.available(view))
            before = deepcopy(model) if eligible else None
            before_count = model.receiver_selector.metrics['passes']
            action = model.predict_frame(view, frame_skip=4)[0]
            if model.receiver_selector.metrics['passes'] > before_count:
                details = model.receiver_selector.diagnostics
                counts['passes'] += 1
                counts[details['purpose'] + ':passes'] += 1
                counts['value-disagreements'] += details['value_slot'] != details['baseline_slot']
                multiple = len(details['candidates']) > 1
                counts['multi-receiver'] += multiple
                if multiple and len(cases) < limit and frame - last_sample >= spacing:
                    assert before is not None, 'Pass selection occurred outside a captured decision'
                    snapshot = env.em.get_state()
                    ram_digest = hashlib.sha256(env.get_ram().tobytes()).hexdigest()
                    case = {'frame': frame, 'ram_sha256': ram_digest, 'observation': observation(view),
                            'selection': deepcopy(details), 'branches': []}
                    for candidate in details['candidates']:
                        slot = candidate['slot']
                        branch = configured(before, slot)
                        buttons = branch.predict_frame(view, frame_skip=4)[0]
                        assert branch._last_decision == model._last_decision
                        assert branch._last_pass_request['frame'] == model._last_pass_request['frame']
                        assert branch._last_pass_request['receiver'] == slot
                        if slot == details['baseline_slot']:
                            assert np.array_equal(buttons, action)
                        result = fork(env, snapshot, ram_digest, state, side,
                                      common_continuation(branch), buttons, horizon, observe_passes=True)
                        case['branches'].append({'slot': slot, 'buttons': buttons.tolist(),
                                                 'decision': branch._last_decision,
                                                 'target': branch._last_target, 'outcome': result})
                    if not cases:
                        repeated = fork(env, snapshot, ram_digest, state, side,
                                        common_continuation(model), action, horizon, observe_passes=True)
                        control = next(branch for branch in case['branches'] if branch['slot'] == details['baseline_slot'])
                        assert repeated == control['outcome'], 'Nondeterministic receiver fork'
                    restore(env, snapshot, ram_digest)
                    cases.append(case)
                    last_sample = frame
            digest.update(np.asarray(action, dtype=np.int8).tobytes())
            advance(env, action, side)
            frame += 1
        _, info = read_view(env, state, side)
        if info['bench_clock'] != 0:
            raise RuntimeError('Incomplete baseline period')
        print(f'{matchup} seed {seed}: {len(cases)} cases, {dict(counts)}', flush=True)
        return {'matchup': matchup, 'side': side, 'seed': seed, 'frames': frame, 'completed': True,
                'goals': [info['p1_score'], info['p2_score']], 'actions_sha256': digest.hexdigest(),
                'counts': dict(counts), 'cases': cases}
    finally:
        env.close()


def summarize(rows):
    totals = {profile: Counter() for profile in ('legacy', 'value')}
    for row in rows:
        for case in row['cases']:
            for profile, counts in totals.items():
                slot = case['selection']['baseline_slot' if profile == 'legacy' else 'value_slot']
                outcome = next(branch['outcome'] for branch in case['branches'] if branch['slot'] == slot)
                counts.update(cases=1, goals_for=outcome['goals_for'], goals_against=outcome['goals_against'],
                              shots=outcome['shots'], one_timers=outcome['one_timers'],
                              ordinary_turnovers=outcome['possession']['ordinary_turnovers'])
                if outcome['pass_events']:
                    event = outcome['pass_events'][0]
                    counts['pass:' + event['outcome']] += 1
                    counts['wrong-recipient'] += event['actual'] is not None and event['actual'] != event['intended']
    return {profile: dict(counts) for profile, counts in totals.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=25041)
    parser.add_argument('--trials', type=int, default=2)
    parser.add_argument('--horizon', type=int, default=240)
    parser.add_argument('--limit', type=int, default=24)
    parser.add_argument('--spacing', type=int, default=60)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--goalie-policy', choices=('off', 'selective'), default='selective')
    parser.add_argument('--all-safe', action='store_true',
                        help='Observe and fork all physically admitted ordinary receivers, including setup passes')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if min(args.trials, args.horizon, args.limit, args.spacing, args.workers) < 1 or not 0 <= args.seed < 2**32 - args.trials:
        raise ValueError('Use positive limits and valid uint32 seeds')
    sources = source_hashes()
    jobs = [(matchup, seed, args.horizon, args.limit, args.spacing, args.goalie_policy, args.all_safe)
            for matchup in ('sabres-ducks-manual', 'ducks-sabres-manual')
            for seed in range(args.seed, args.seed + args.trials)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(run_job, jobs))
    if source_hashes() != sources:
        raise RuntimeError('Sources changed during replay; discard results')
    report = {'protocol': 'receiver-all-options-common-continuation-v1', 'settings': vars(args),
              'sources': sources, 'matches': rows, 'summary': summarize(rows),
              'limitations': ['First spaced multi-receiver passes on baseline trajectories, not an unbiased sample.',
                              'Only the initial receiver changes; later decisions use the legacy policy.',
                              'Finite-horizon outcomes do not establish full-game strength.']}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()
