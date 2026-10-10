"""Collect native primary-shot outcomes for aim/hold alternatives at legacy shots.

Every candidate starts at the same normal-shot decision with identical ROM RNG
and controller history. Additional B/C inputs are suppressed after the requested
hold. Stop on primary-shot resolution, new possession, or a bounded timeout so
later attacks cannot label this attempt.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.agents.shot_placement import extend_hold, shot_actions, shot_features
from nhl94_ai.env.factory import make_retro
from nhl94_ai.evaluation.carry_outcomes import CarryOutcomes
from nhl94_ai.evaluation.cpu_benchmark import CPU_RAM, MATCHUPS, select_side
from nhl94_ai.evaluation.possession_ablation_replay import observation, source_hashes as agent_hashes
from nhl94_ai.evaluation.refinement_replay import advance, read_view, restore
from nhl94_ai.game.constants import GameConsts as Buttons
from nhl94_ai.game.state import NHL94GameState
from nhl94_ai.tasks.cross_crease_setup import crossing_touch_player


def placed(history, side, hold):
    model = deepcopy(history)
    model.shot.side = side
    model.shot.hold_until_frame = model.scheduler.frames + hold
    extend_hold(model, hold)
    model._last_target = side * 13, model._last_target[1]
    action = model.scheduler.action.copy()
    action[0, Buttons.INPUT_UP:Buttons.INPUT_C] = 0
    model.shot.aim(action[0])
    model.scheduler.action = action
    return model


def fork_shot(env, snapshot, digest, state, side, history, aim, hold, horizon):
    restore(env, snapshot, digest)
    model, decoded = placed(history, aim, hold), deepcopy(state)
    view, start = read_view(env, decoded, side)
    tracker = CarryOutcomes(view, start[f'p{side}_score'], crossing_touch_player(env))
    action = model.scheduler.action[0]
    held, reason, trace, flight = 0, 'bounded-horizon-no-goal', [], False
    shooter = view.engine.puck_owner
    for elapsed in range(horizon):
        if elapsed:
            action = model.predict_frame(view, frame_skip=4)[0]
        action = action.copy()
        action[Buttons.INPUT_B] = 0
        if elapsed >= hold:
            action[Buttons.INPUT_C] = 0
        held += bool(action[Buttons.INPUT_C])
        advance(env, action, side)
        view, info = read_view(env, decoded, side)
        ended = tracker.observe(elapsed + 1, view, info[f'p{side}_score'], crossing_touch_player(env))
        owner = view.engine.puck_owner
        flight |= owner < 0
        new_possession = owner >= 0 and (flight or owner != shooter)
        if elapsed < 24 or elapsed % 8 == 0:
            trace.append({'frame': elapsed + 1, 'puck': (view.puck.x, view.puck.y),
                          'owner': view.engine.puck_owner, 'held_c': bool(action[Buttons.INPUT_C])})
        if ended or new_possession or info['bench_clock'] == 0:
            reason = tracker.end_reason or ('new-possession' if new_possession else 'period-ended')
            break
    outcome = tracker.summary()
    assert held == min(hold, elapsed + 1), 'Shot replay did not execute the requested C hold'
    return {'goal': int(info[f'p{side}_score'] > start[f'p{side}_score']),
            'resolved': True,
            'end_reason': reason, 'held_c_frames': held, 'frames': tracker.elapsed,
            'native': outcome, 'trace': trace}


def source_hashes():
    sources = agent_hashes()
    root = Path(__file__).resolve().parents[2]
    path = Path(__file__)
    sources[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return sources


def run_job(job):
    matchup, seed, horizon, limit = job
    save, side = MATCHUPS[matchup]
    env = make_retro(game='NHL94-Genesis-v0', state=save, num_players=1, goalie_policy='selective')
    cases, counts = [], Counter()
    try:
        for name, (address, kind) in CPU_RAM.items():
            env.data.set_variable(name, {'address': address, 'type': kind})
        env.reset(seed=seed)
        env.data.update_ram()
        select_side(env.data, env.data.lookup_all(), side)
        env.data.set_value('bench_rng', seed)
        env.data.set_value('bench_clock', 300)
        env.step(np.zeros(12, dtype=np.int8))
        model = ClassicAIV1Model(SimpleNamespace(action_type='FILTERED', goalie_policy='selective'))
        state, frame, digest = NHL94GameState(5), 1, hashlib.sha256()
        while frame < 42000:
            view, info = read_view(env, state, side)
            if info['bench_clock'] == 0:
                break
            action = model.predict_frame(view, frame_skip=4)[0]
            started = (model._last_decision == 'shoot' and action[Buttons.INPUT_C]
                       and model.shot.hold_until_frame == model.scheduler.frames + 4)
            if started:
                counts['normal-shot-starts'] += 1
                choices = shot_actions(view, 4)
                features = [shot_features(view, aim, hold) for aim, hold in choices]
                if any(row is None for row in features):
                    counts['unforecastable'] += 1
                elif len(cases) < limit:
                    snapshot = env.em.get_state()
                    ram_digest = hashlib.sha256(env.get_ram().tobytes()).hexdigest()
                    case = {'frame': frame, 'ram_sha256': ram_digest, 'observation': observation(view),
                            'baseline': choices[0], 'branches': []}
                    for (aim, hold), values in zip(choices, features):
                        result = fork_shot(env, snapshot, ram_digest, state, side, model, aim, hold, horizon)
                        case['branches'].append({'aim': aim, 'hold': hold, 'features': values, 'outcome': result})
                    if not cases:
                        repeated = fork_shot(env, snapshot, ram_digest, state, side, model, *choices[0], horizon)
                        assert repeated == case['branches'][0]['outcome'], 'Non-deterministic shot restore'
                    restore(env, snapshot, ram_digest)
                    cases.append(case)
            digest.update(np.asarray(action, dtype=np.int8).tobytes())
            advance(env, action, side)
            frame += 1
        _, info = read_view(env, state, side)
        if info['bench_clock'] != 0:
            raise RuntimeError('Incomplete baseline shot-collection period')
        print(f'{matchup} {seed}: {len(cases)} shot states, {dict(counts)}', flush=True)
        return {'matchup': matchup, 'side': side, 'seed': seed, 'frames': frame,
                'completed': True, 'goals': [info['p1_score'], info['p2_score']],
                'actions_sha256': digest.hexdigest(), 'counts': dict(counts), 'cases': cases}
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=25301)
    parser.add_argument('--trials', type=int, default=40)
    parser.add_argument('--horizon', type=int, default=120)
    parser.add_argument('--limit', type=int, default=12)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if min(args.trials, args.horizon, args.limit, args.workers) < 1 or not 0 <= args.seed <= 2**32-args.trials:
        raise ValueError('Use positive limits and valid uint32 seeds')
    sources = source_hashes()
    jobs = [(matchup, seed, args.horizon, args.limit)
            for matchup in ('sabres-ducks-manual', 'ducks-sabres-manual')
            for seed in range(args.seed, args.seed + args.trials)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(run_job, jobs))
    if source_hashes() != sources:
        raise RuntimeError('Sources changed during shot collection')
    report = {'protocol': 'native-primary-shot-alternatives-v3', 'settings': vars(args),
              'sources': sources, 'matches': rows,
              'limitations': ['Legacy normal-shot starts only; not all possible shooting states.',
                              'Goal by the horizon or new possession; additional B/C inputs suppressed after the initial hold.',
                              'Later attacks and rebound recoveries are not credited; timeout is a bounded-return zero.',
                              'Fixed two-roster benchmark; split complete seeds, never individual alternative rows.']}
    data = (json.dumps(report, separators=(',', ':'), allow_nan=False) + '\n').encode()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(gzip.compress(data, mtime=0) if output.suffix == '.gz' else data)
    print('Collected', sum(len(row['cases']) for row in rows), 'shot states')


if __name__ == '__main__':
    main()
