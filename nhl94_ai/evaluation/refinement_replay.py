"""Diagnose Classic refinements on a verified default trajectory and native forks.

All forks restore emulator state (including RNG), controller history and decoded
RAM state. The first spaced divergences are sampled, not selected by outcome.
Each branch retains its selected policy for the stated horizon. These are local
development probes, not independent full-game strength measurements.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.agents.carry import body_clearance, carry_path
from nhl94_ai.agents.defense import controlled_slot
from nhl94_ai.agents.finishing import normal_finish
from nhl94_ai.agents.motion import velocity
from nhl94_ai.agents.passing import evaluate_pass
from nhl94_ai.agents.skating import grounded_step
from nhl94_ai.env.factory import make_retro
from nhl94_ai.evaluation.carry_replay import source_hashes
from nhl94_ai.evaluation.carry_outcomes import CarryOutcomes
from nhl94_ai.evaluation.cpu_benchmark import CPU_RAM, MATCHUPS, cpu_view, select_side
from nhl94_ai.game.ram import pass_geometry_info, restore_away_control
from nhl94_ai.game.state import NHL94GameState
from nhl94_ai.tasks.cross_crease_setup import crossing_touch_player


REFINEMENTS = ('pass-timing', 'carry-motion', 'finishing', 'interceptions')


def configured(history, refinement):
    model = deepcopy(history)
    model.refinements = frozenset((refinement,))
    model.offense.pass_timing = refinement == 'pass-timing'
    model.offense.carry_motion = refinement == 'carry-motion'
    model.defense.verify_interceptions = refinement == 'interceptions'
    return model


def read_view(env, state, side):
    env.data.update_ram()
    info = pass_geometry_info(env, env.data.lookup_all())
    return cpu_view(state, info, side), info


def advance(env, action, side):
    *_, info = env.step(action)
    if side == 2:
        restore_away_control(env.data, info, controller_prefix='bench', player_prefix='cpu', slots=6)


def restore(env, snapshot, digest):
    env.em.set_state(snapshot)
    env.data.update_ram()
    assert hashlib.sha256(env.get_ram().tobytes()).hexdigest() == digest


def policy_details(model):
    return {'decision': model._last_decision, 'target': model._last_target,
            'offense': deepcopy(model.offense_diagnostics),
            'defense': deepcopy(model.defense_diagnostics)}


def carry_probe(state, player, target, baseline, alternate):
    path = carry_path(player, target)
    return {
        'legacy_route': baseline.offense._estimated_carry(state, player, target),
        'native_route': alternate.offense._estimated_carry(state, player, target),
        'initial_clearance': body_clearance(state, player, [player]),
        'native_bounded': path is not None,
        'native_clearance_by_frame': [body_clearance(state, player, path[:frame + 1])
                                      for frame in range(len(path))] if path else None,
    }


def interception_probe(state, before, baseline):
    plan = baseline.defense.plan
    slot = baseline.defense.desired_slot
    player = state.team1.get_player_by_scnum(slot)
    if plan.mode not in ('recover-safe', 'intercept-pass') or player is None:
        return None
    controller = deepcopy(before.defense)
    controller.frames = baseline.defense.frames
    trace = []

    def observed_step(*args, **kwargs):
        future = grounded_step(*args, **kwargs)
        trace.append(future)
        return future

    with patch('nhl94_ai.agents.defense.grounded_step', side_effect=observed_step):
        arrival = controller._skating_arrival(state, slot, player, plan.target, plan.puck_arrival)
    return {'slot': slot, 'target': plan.target, 'deadline': plan.puck_arrival,
            'verified_arrival': arrival, 'puck_velocity': velocity(state.puck),
            'initial_player_velocity': velocity(player),
            'switch_delay': controller._switch_delay(state, slot),
            'trace': [{'frame': index + 1, 'point': (future.x, future.y),
                       'distance': math.dist((future.x, future.y), plan.target),
                       'relative_speed': math.dist(velocity(future), velocity(state.puck))}
                      for index, future in enumerate(trace)]}


def fork(env, snapshot, digest, state, side, policy, action, horizon):
    restore(env, snapshot, digest)
    model, decoded = deepcopy(policy), deepcopy(state)
    view, start = read_view(env, decoded, side)
    first_recovery, loss, first_shot, first_release = None, None, None, None
    outcomes = CarryOutcomes(view, start[f'p{side}_score'], crossing_touch_player(env))
    trace = []
    for elapsed in range(horizon):
        if elapsed:
            action = model.predict_frame(view, frame_skip=4)[0]
        before = view.engine.puck_owner
        advance(env, action, side)
        view, info = read_view(env, decoded, side)
        outcomes.observe(elapsed + 1, view, info[f'p{side}_score'], crossing_touch_player(env))
        if first_release is None and any(e['release_frame'] == elapsed + 1 for e in outcomes.shot_events):
            first_release = {'frame': elapsed + 1, 'puck': (view.puck.x, view.puck.y),
                             'velocity': velocity(view.puck), 'height': view.puck.height,
                             'goalie': (view.team2.goalie.x, view.team2.goalie.y),
                             'shooter': view.engine.shot_player}
        owner = view.engine.puck_owner
        if view.team1.owns_scnum(owner) and first_recovery is None:
            first_recovery = elapsed + 1
        if view.team2.owns_scnum(owner) and loss is None:
            loss = elapsed + 1
        if info[f'bench_shots{side}'] > start[f'bench_shots{side}'] and first_shot is None:
            first_shot = {'frame': elapsed + 1, 'puck': (view.puck.x, view.puck.y),
                          'velocity': (view.puck.motion_x, view.puck.motion_y),
                          'goalie': (view.team2.goalie.x, view.team2.goalie.y),
                          'shooter': view.engine.shot_player}
        if elapsed < 20 or before != owner or elapsed % 12 == 0:
            trace.append({'frame': elapsed + 1, 'owner': owner,
                          'puck': (view.puck.x, view.puck.y),
                          'decision': model._last_decision, 'target': model._last_target})
        if info['bench_clock'] == 0 or any(info[f'p{s}_score'] != start[f'p{s}_score'] for s in (1, 2)):
            break
    return {'frames': elapsed + 1,
            'goals_for': info[f'p{side}_score'] - start[f'p{side}_score'],
            'goals_against': info[f'p{3-side}_score'] - start[f'p{3-side}_score'],
            'shots': info[f'bench_shots{side}'] - start[f'bench_shots{side}'],
            'opponent_shots': info[f'bench_shots{3-side}'] - start[f'bench_shots{3-side}'],
            'one_timers': info[f'bench_one_timers{side}'] - start[f'bench_one_timers{side}'],
            'recovery_frame': first_recovery, 'loss_frame': loss, 'first_recorded_shot': first_shot,
            'first_release': first_release, 'possession': outcomes.summary(),
            'final_owner': view.engine.puck_owner, 'trace': trace}


def run_job(job):
    matchup, seed, horizon, limit, spacing = job
    save, side = MATCHUPS[matchup]
    env = make_retro(game='NHL94-Genesis-v0', state=save, num_players=1)
    counts, sampled, last_sample = Counter(), Counter(), Counter()
    cases, passes, shots = [], [], []
    try:
        for name, (address, kind) in CPU_RAM.items():
            env.data.set_variable(name, {'address': address, 'type': kind})
        env.reset(seed=seed)
        env.data.update_ram()
        select_side(env.data, env.data.lookup_all(), side)
        env.data.set_value('bench_rng', seed)
        env.data.set_value('bench_clock', 300)
        env.step(np.zeros(12, dtype=np.int8))
        model = ClassicAIV1Model(SimpleNamespace(action_type='FILTERED'))
        state, frame, digest = NHL94GameState(5), 1, hashlib.sha256()
        while frame < 42000:
            view, info = read_view(env, state, side)
            if info['bench_clock'] == 0:
                break
            owner = view.engine.puck_owner
            player = view.team1.get_player_by_scnum(owner)
            offense = (player is not None and player is not view.team1.goalie
                       and owner == controlled_slot(view.team1) and not view.engine.clock_stopped
                       and (model._frame_remaining <= 0 or model._was_defending)
                       and model._one_timer is None and model._tick >= model._shot_until)
            defense = owner < 0 and frame % 4 == 0 and not view.engine.clock_stopped
            before = deepcopy(model) if offense or defense else None
            action = model.predict_frame(view, frame_skip=4)[0]
            request = model._last_pass_request
            if request and request['frame'] == model.defense.frames and player is not None:
                index = request['receiver'] - view.team1.skater_scnum_base()
                purpose = 'one-timer' if 'one-timer' in request['purpose'] else 'outlet' if player is view.team1.goalie else (
                    'advance' if request['purpose'] == 'advance-pass' else 'position')
                details = evaluate_pass(view, player, index, view.team1.players[index], purpose,
                                        release_prediction=True)[1]
                passes.append({'frame': frame, 'request': dict(request), 'revised': details})
            if offense and model._last_decision == 'shoot':
                choice = normal_finish(view, player, 4)
                shots.append({'frame': frame, 'baseline_side': model._shot_side,
                              'choice': asdict(choice) if choice else None,
                              'player': (player.x, player.y), 'puck': (view.puck.x, view.puck.y),
                              'goalie': (view.team2.goalie.x, view.team2.goalie.y),
                              'goalie_velocity': (view.team2.goalie.motion_x, view.team2.goalie.motion_y)})
            if before is not None:
                for refinement in REFINEMENTS[:3] if offense else REFINEMENTS[3:]:
                    alternate = configured(before, refinement)
                    alternate_action = alternate.predict_frame(view, frame_skip=4)[0]
                    changed = (not np.array_equal(action, alternate_action)
                               or (refinement == 'finishing' and model._last_decision == 'shoot'
                                   and (model._shot_side, model._shot_hold_until) != (
                                       alternate._shot_side, alternate._shot_hold_until)))
                    counts[f'{refinement}:examined'] += 1
                    if not changed:
                        continue
                    counts[f'{refinement}:changed'] += 1
                    counts[f'{refinement}:{model._last_decision}->{alternate._last_decision}'] += 1
                    if sampled[refinement] >= limit or frame - last_sample[refinement] < spacing:
                        continue
                    snapshot = env.em.get_state()
                    ram_digest = hashlib.sha256(env.get_ram().tobytes()).hexdigest()
                    case = {'frame': frame, 'refinement': refinement, 'owner': owner,
                            'ram_sha256': ram_digest, 'puck': (view.puck.x, view.puck.y),
                            'player': (player.x, player.y) if player else None,
                            'baseline': policy_details(model), 'revised': policy_details(alternate),
                            'buttons': {'baseline': action.tolist(), 'revised': alternate_action.tolist()},
                            'outcomes': {}}
                    if refinement == 'carry-motion':
                        case['carry_probe'] = carry_probe(view, player, model._last_target, model, alternate)
                    if refinement == 'interceptions':
                        case['interception_probe'] = interception_probe(view, before, model)
                    for label, policy, buttons in (('baseline', model, action), ('revised', alternate, alternate_action)):
                        case['outcomes'][label] = fork(
                            env, snapshot, ram_digest, state, side, policy, buttons, horizon)
                    if refinement == 'finishing' and model._last_decision == alternate._last_decision == 'shoot':
                        aim, hold = deepcopy(model), deepcopy(model)
                        aim._shot_side = alternate._shot_side
                        hold._shot_hold_until = alternate._shot_hold_until
                        for label, policy in (('aim-only', aim), ('hold-only', hold)):
                            case['outcomes'][label] = fork(
                                env, snapshot, ram_digest, state, side, policy, action, horizon)
                    restore(env, snapshot, ram_digest)
                    cases.append(case)
                    sampled[refinement] += 1
                    last_sample[refinement] = frame
            digest.update(np.asarray(action, dtype=np.int8).tobytes())
            advance(env, action, side)
            frame += 1
        _, info = read_view(env, state, side)
        return {'matchup': matchup, 'seed': seed, 'side': side, 'frames': frame,
                'goals': [info['p1_score'], info['p2_score']], 'clock': info['bench_clock'],
                'actions_sha256': digest.hexdigest(), 'counts': dict(counts),
                'passes': passes, 'shots': shots, 'cases': cases}
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=20266001)
    parser.add_argument('--trials', type=int, default=4)
    parser.add_argument('--horizon', type=int, default=180)
    parser.add_argument('--limit', type=int, default=6)
    parser.add_argument('--spacing', type=int, default=180)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    jobs = [(matchup, seed, args.horizon, args.limit, args.spacing)
            for matchup in ('sabres-ducks-manual', 'ducks-sabres-manual')
            for seed in range(args.seed, args.seed + args.trials)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(run_job, jobs))
    report = {'settings': vars(args), 'sources': source_hashes(), 'matches': rows,
              'probe_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('Completed', len(rows), 'default periods and', sum(len(r['cases']) for r in rows), 'matched-state cases.')


if __name__ == '__main__':
    main()
