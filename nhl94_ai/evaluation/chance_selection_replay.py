"""Compare admitted finishes and a four-frame carry on restored native states.

Sample chances before observing outcomes. Every branch changes one decision,
then uses the original controller, including its shot/pass commitments. Goals
span a bounded continuation; an additional label stops credit at opponent
possession. Neither label is an independent full-game value estimate.
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
from nhl94_ai.agents.defense import controlled_slot
from nhl94_ai.env.factory import make_retro
from nhl94_ai.evaluation.chance_selection import ChanceSelection, readiness
from nhl94_ai.evaluation.cpu_benchmark import CPU_RAM, MATCHUPS, select_side
from nhl94_ai.evaluation.possession_ablation_replay import observation, source_hashes as agent_hashes
from nhl94_ai.evaluation.refinement_replay import advance, read_view, restore
from nhl94_ai.game.constants import GameConsts as Buttons
from nhl94_ai.game.state import NHL94GameState


def configured(history, state, alternative=None):
    model = deepcopy(history)
    probe = ChanceSelection(forced=alternative)
    model.possession_ablation = probe
    action = model.predict_frame(state, frame_skip=4)[0]
    details = deepcopy(probe.diagnostics.get('chance'))
    model.possession_ablation = None
    return model, action, details


def fork_chance(env, snapshot, digest, state, side, policy, action, horizon):
    restore(env, snapshot, digest)
    model, decoded = deepcopy(policy), deepcopy(state)
    view, start = read_view(env, decoded, side)
    owner = view.engine.puck_owner
    shots_before, attempts_before = view.team1.stats.shots, view.team1.one_timer_attempts
    actions, trace = hashlib.sha256(), []
    loss, unlock, first_shot, first_timer = None, None, None, None
    for elapsed in range(horizon):
        if elapsed:
            action = model.predict_frame(view, frame_skip=4)[0]
        player = view.team1.get_player_by_scnum(owner)
        ready = readiness(player)
        if unlock is None and ready['animation_known'] and not ready['animation_locked']:
            unlock = elapsed
        if elapsed < 40 or elapsed % 8 == 0:
            trace.append({'frame': elapsed, 'owner': view.engine.puck_owner,
                          'buttons': np.asarray(action, dtype=int).tolist(),
                          'decision': model._last_decision, 'carrier': ready,
                          'puck': (view.puck.x, view.puck.y),
                          'shot_player': view.engine.shot_player})
        actions.update(np.asarray(action, dtype=np.int8).tobytes())
        advance(env, action, side)
        view, info = read_view(env, decoded, side)
        if loss is None and view.team2.owns_scnum(view.engine.puck_owner):
            loss = elapsed + 1
        if first_shot is None and view.team1.stats.shots > shots_before:
            first_shot = elapsed + 1
        if first_timer is None and view.team1.one_timer_attempts > attempts_before:
            first_timer = elapsed + 1
        if info['bench_clock'] == 0 or any(info[f'p{s}_score'] != start[f'p{s}_score'] for s in (1, 2)):
            break
    goals = info[f'p{side}_score'] - start[f'p{side}_score']
    return {'goals_for': goals, 'goals_against': info[f'p{3-side}_score'] - start[f'p{3-side}_score'],
            'attack_goal': goals if loss is None else 0, 'first_opponent_possession': loss,
            'first_unlock': unlock, 'first_recorded_shot': first_shot, 'first_one_timer': first_timer,
            'shots': view.team1.stats.shots - shots_before,
            'one_timers': view.team1.one_timer_attempts - attempts_before,
            'frames': elapsed + 1, 'actions_sha256': actions.hexdigest(), 'trace': trace}


def source_hashes():
    sources = agent_hashes()
    root = Path(__file__).resolve().parents[2]
    for name in ('chance_selection.py', 'chance_selection_replay.py'):
        path = Path(__file__).with_name(name)
        sources[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return sources


def run_job(job):
    matchup, seed, horizon, spacing = job
    save, side = MATCHUPS[matchup]
    env = make_retro(game='NHL94-Genesis-v0', state=save, num_players=1, goalie_policy='selective')
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
        model = ClassicAIV1Model(SimpleNamespace(action_type='FILTERED', goalie_policy='selective'))
        state, frame, digest = NHL94GameState(5), 1, hashlib.sha256()
        while frame < 42000:
            view, info = read_view(env, state, side)
            if info['bench_clock'] == 0:
                break
            player = view.team1.get_player_by_scnum(view.engine.puck_owner)
            eligible = (player is not None and player is not view.team1.goalie
                        and view.engine.puck_owner == controlled_slot(view.team1)
                        and not view.engine.clock_stopped and player.passing is not None
                        and (model.scheduler.remaining == 0 or model.scheduler.was_defending)
                        and model.one_timer.pending is None and model.offense.pass_action.pending is None
                        and not model.shot.active(model.scheduler.decisions + 1))
            before = deepcopy(model) if eligible else None
            action = model.predict_frame(view, frame_skip=4)[0]
            if before is not None and model._last_decision in ('shoot', 'one-timer-pass', 'pass-button-release'):
                baseline, control, details = configured(before, view)
                assert np.array_equal(control, action), 'Instrumentation changed Classic input'
                assert baseline._last_decision == model._last_decision
                if details is not None:
                    counts['chances'] += 1
                if details is not None and frame - last_sample >= spacing:
                    snapshot = env.em.get_state()
                    ram_digest = hashlib.sha256(env.get_ram().tobytes()).hexdigest()
                    branches = {}
                    for alternative in details['alternatives']:
                        branch, first, check = configured(before, view, alternative)
                        assert check['features'] == details['features']
                        branches[alternative] = fork_chance(
                            env, snapshot, ram_digest, state, side, branch, first, horizon)
                    original = fork_chance(env, snapshot, ram_digest, state, side, model, action, horizon)
                    assert original == branches[details['baseline']], 'Legacy continuation changed'
                    if not cases:
                        assert original == fork_chance(
                            env, snapshot, ram_digest, state, side, model, action, horizon), 'Unstable restore'
                    restore(env, snapshot, ram_digest)
                    cases.append({'frame': frame, 'ram_sha256': ram_digest, **details,
                                  'observation': observation(view), 'branches': branches})
                    last_sample = frame
            digest.update(np.asarray(action, dtype=np.int8).tobytes())
            advance(env, action, side)
            frame += 1
        _, info = read_view(env, state, side)
        if info['bench_clock'] != 0:
            raise RuntimeError('Incomplete chance collection period')
        print(f'{matchup} {seed}: {len(cases)} cases, {dict(counts)}', flush=True)
        return {'matchup': matchup, 'side': side, 'seed': seed, 'frames': frame,
                'completed': True, 'actions_sha256': digest.hexdigest(),
                'goals': [info['p1_score'], info['p2_score']],
                'shots': [info['bench_shots1'], info['bench_shots2']], 'counts': dict(counts), 'cases': cases}
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=25901)
    parser.add_argument('--trials', type=int, default=20)
    parser.add_argument('--horizon', type=int, default=240)
    parser.add_argument('--spacing', type=int, default=60)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if min(args.trials, args.horizon, args.spacing, args.workers) < 1 or not 0 <= args.seed <= 2**32-args.trials:
        raise ValueError('Use positive limits and uint32 seeds')
    sources = source_hashes()
    jobs = [(matchup, seed, args.horizon, args.spacing)
            for matchup in ('sabres-ducks-manual', 'ducks-sabres-manual')
            for seed in range(args.seed, args.seed + args.trials)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(run_job, jobs))
    if source_hashes() != sources:
        raise RuntimeError('Sources changed during chance collection')
    report = {'protocol': 'native-chance-selection-v1', 'settings': vars(args),
              'sources': sources, 'matches': rows,
              'limitations': ['First spaced legacy finish decisions, including locked/button-held starts.',
                              'Only admitted normal shots/one-timers and one-decision carry are compared.',
                              'Identical emulator/RNG/controller history; all later decisions use Classic.',
                              'Goals include later attacks in 240 frames; attack_goal excludes credit after opponent possession.',
                              'Overlapping cases are correlated; split complete seeds for learning.',
                              'Two fixed rosters and first periods; not game-wide expected goals.']}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(report, separators=(',', ':'), allow_nan=False)+'\n').encode()
    path.write_bytes(gzip.compress(data, mtime=0) if path.suffix == '.gz' else data)
    print(json.dumps({'periods': len(rows), 'cases': sum(len(row['cases']) for row in rows)}))


if __name__ == '__main__':
    main()
