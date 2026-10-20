"""Compare ordinary-pass execution from identical native requests and histories."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import numpy as np

from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.evaluation.cpu_benchmark import MATCHUPS
from nhl94_ai.evaluation.replay_support import (
    advance, observation, prepare_period, read_view, replay_source_hashes, restore, write_replay_report,
)
from nhl94_ai.env.factory import make_retro
from nhl94_ai.game.state import NHL94GameState


def fork_reception(env, snapshot, digest, state, side, history, corrected, horizon):
    restore(env, snapshot, digest)
    model, decoded = deepcopy(history), deepcopy(state)
    model.offense.pass_action.stick_control = corrected
    request = model.offense.pass_action.pending
    intended, passer = request.receiver, request.passer
    view, _ = read_view(env, decoded, side)
    flight, trace, reason = False, [], 'bounded-horizon'
    for elapsed in range(horizon):
        action = model.scheduler.action[0] if elapsed == 0 else model.predict_frame(view, frame_skip=4)[0]
        advance(env, action, side)
        view, info = read_view(env, decoded, side)
        owner = view.engine.puck_owner
        flight |= owner < 0
        new_possession = owner >= 0 and (flight or owner != passer)
        trace.append({'frame': elapsed + 1, 'puck': (view.puck.x, view.puck.y), 'owner': owner,
                      'control': view.team1.defense_control, 'decision': model._last_decision,
                      'buttons': np.asarray(action, dtype=int).tolist()})
        if new_possession or model.offense.pass_action.pending is None or info['bench_clock'] == 0:
            event = model.offense.pass_action.last_pass or {}
            reason = 'new-possession' if new_possession else event.get('outcome', 'period-ended')
            break
    return {'intended_reception': bool(new_possession and owner == intended),
            'team_reception': bool(new_possession and view.team1.owns_scnum(owner)),
            'intercepted': bool(new_possession and view.team2.owns_scnum(owner)),
            'owner': owner, 'frames': elapsed + 1, 'end_reason': reason,
            'metrics': dict(model.offense.pass_action.reception_metrics), 'trace': trace}


def source_hashes():
    return replay_source_hashes('evaluation/reception_replay.py')


def run_job(job):
    matchup, seed, limit, horizon = job
    save, side = MATCHUPS[matchup]
    env = make_retro(game='NHL94-Genesis-v0', state=save, num_players=1, goalie_policy='selective')
    cases = []
    try:
        prepare_period(env, seed, side)
        model = ClassicAIV1Model(SimpleNamespace(action_type='FILTERED', goalie_policy='selective'))
        state, frame, digest, starts = NHL94GameState(5), 1, hashlib.sha256(), 0
        while frame < 42000:
            view, info = read_view(env, state, side)
            if info['bench_clock'] == 0:
                break
            action = model.predict_frame(view, frame_skip=4)[0]
            request = model.offense.pass_action.pending
            if request is not None and request.frame == model.scheduler.frames:
                starts += 1
                if len(cases) < limit:
                    snapshot = env.em.get_state()
                    ram_digest = hashlib.sha256(env.get_ram().tobytes()).hexdigest()
                    branches = {name: fork_reception(env, snapshot, ram_digest, state, side, model, enabled, horizon)
                                for name, enabled in (('legacy', False), ('stick', True))}
                    if not cases:
                        assert branches['legacy'] == fork_reception(
                            env, snapshot, ram_digest, state, side, model, False, horizon), 'Nondeterministic restore'
                    restore(env, snapshot, ram_digest)
                    cases.append({'frame': frame, 'ram_sha256': ram_digest, 'request': request.snapshot(),
                                  'observation': observation(view), 'branches': branches})
            digest.update(np.asarray(action, dtype=np.int8).tobytes())
            advance(env, action, side)
            frame += 1
        _, info = read_view(env, state, side)
        if info['bench_clock'] != 0:
            raise RuntimeError('Incomplete reception collection period')
        print(f'{matchup} {seed}: {len(cases)}/{starts} ordinary passes', flush=True)
        return {'matchup': matchup, 'side': side, 'seed': seed, 'frames': frame, 'completed': True,
                'actions_sha256': digest.hexdigest(), 'goals': [info['p1_score'], info['p2_score']],
                'starts': starts, 'cases': cases}
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=25501)
    parser.add_argument('--trials', type=int, default=10)
    parser.add_argument('--limit', type=int, default=20)
    parser.add_argument('--horizon', type=int, default=120)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if min(args.trials, args.limit, args.horizon, args.workers) < 1 or not 0 <= args.seed <= 2**32-args.trials:
        raise ValueError('Use positive limits and valid uint32 seeds')
    sources = source_hashes()
    jobs = [(matchup, seed, args.limit, args.horizon)
            for matchup in ('sabres-ducks-manual', 'ducks-sabres-manual')
            for seed in range(args.seed, args.seed + args.trials)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(run_job, jobs))
    if source_hashes() != sources:
        raise RuntimeError('Sources changed during reception replay')
    totals = {name: dict(sum((Counter({key: case['branches'][name][key]
                                      for key in ('intended_reception', 'team_reception', 'intercepted')})
                              for row in rows for case in row['cases']), Counter()))
              for name in ('legacy', 'stick')}
    report = {'protocol': 'native-reception-execution-v1', 'settings': vars(args), 'sources': sources,
              'totals': totals, 'matches': rows,
              'limitations': ['First ordinary pass requests per legacy period; selected before outcomes.',
                              'Same native state, RNG and complete controller history for both branches.',
                              'Ends on new possession, pass lifecycle ending, period end or horizon.',
                              'Reception outcomes only; no claim about subsequent goals or full-game strength.']}
    write_replay_report(args.output, report)
    print(json.dumps(totals))


if __name__ == '__main__':
    main()
