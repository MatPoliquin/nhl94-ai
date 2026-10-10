"""Compare replanning after each observed same-shooter rebound recovery.

Original trajectories are restored after every fork. Collection selects the
first recovery within each active normal-shot commitment before its outcome is
known. All subsequent play in both branches uses the unchanged Classic policy.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.env.factory import make_retro
from nhl94_ai.evaluation.cpu_benchmark import CPU_RAM, MATCHUPS, select_side
from nhl94_ai.evaluation.possession_ablation_replay import observation, source_hashes as agent_hashes
from nhl94_ai.evaluation.refinement_replay import advance, read_view, restore
from nhl94_ai.game.state import NHL94GameState
from nhl94_ai.tasks.cross_crease_setup import crossing_touch_player


def trace_state(env, state, model, action, frame, slot):
    player = state.team1.get_player_by_scnum(slot)
    return {'frame': frame, 'owner': state.engine.puck_owner, 'control': state.team1.defense_control,
            'puck': (state.puck.x, state.puck.y, state.puck.height),
            'puck_velocity': (state.puck.motion_x, state.puck.motion_y, state.puck.motion_z),
            'last_touch': crossing_touch_player(env), 'shot_player': state.engine.shot_player,
            'shots': state.team1.stats.shots, 'one_timers': state.team1.one_timer_attempts,
            'clock_stopped': state.engine.clock_stopped,
            'shooter': {name: getattr(player, name) for name in (
                'x', 'y', 'motion_x', 'motion_y', 'selection_flags', 'live_state_flags',
                'live_anim', 'live_anim_frame', 'animation_timer', 'stick_x', 'stick_y')},
            'decision': model._last_decision, 'decision_count': model.scheduler.decisions,
            'buttons': np.asarray(action, dtype=int).tolist(), 'shot': asdict(model.shot)}


def fork_rebound(env, snapshot, digest, decoded, side, history, enabled, horizon, *, wait_for_animation=True):
    restore(env, snapshot, digest)
    model, state = deepcopy(history), deepcopy(decoded)
    view, start = read_view(env, state, side)
    assert model.shot.same_shooter_recovery(view, model.scheduler.decisions)
    model.shot.rebound_recovery = enabled
    model.shot.rebound_wait_for_animation = wait_for_animation
    slot, recoveries = model.shot.slot, model.shot.recoveries
    original_until = model.shot.until_decision
    ready_at_start = model.shot.recovery_ready(view)
    trace, first_replan, first_shot, first_possession_end = [], None, None, None
    attempts_before = view.team1.one_timer_attempts
    for elapsed in range(horizon):
        decisions_before = model.scheduler.decisions
        action = model.predict_frame(view, frame_skip=4)[0]
        if elapsed == 0:
            assert model.shot.recoveries == recoveries + int(enabled and ready_at_start)
        if (model.shot.recoveries > recoveries or model.shot.until_decision != original_until
                or not model.shot.active(model.scheduler.decisions)):
            model.shot.rebound_recovery = False
        if (first_replan is None and model.scheduler.decisions != decisions_before
                and model._last_decision != 'shot-follow-through'):
            first_replan = {'frame': elapsed, 'decision': model._last_decision}
        if elapsed < 100 or elapsed % 8 == 0:
            trace.append(trace_state(env, view, model, action, elapsed, slot))
        advance(env, action, side)
        view, info = read_view(env, state, side)
        if first_possession_end is None and view.engine.puck_owner != slot:
            first_possession_end = {'frame': elapsed + 1, 'owner': view.engine.puck_owner}
        if first_shot is None and info[f'bench_shots{side}'] > start[f'bench_shots{side}']:
            first_shot = {'frame': elapsed + 1, 'shooter': view.engine.shot_player}
        if info['bench_clock'] == 0:
            break
    return {'first_replan': first_replan, 'first_recorded_shot': first_shot,
            'recovery_replans': model.shot.recoveries - recoveries,
            'first_possession_end': first_possession_end,
            'goals_for': info[f'p{side}_score'] - start[f'p{side}_score'],
            'goals_against': info[f'p{3-side}_score'] - start[f'p{3-side}_score'],
            'recorded_shots': info[f'bench_shots{side}'] - start[f'bench_shots{side}'],
            'native_one_timers': view.team1.one_timer_attempts - attempts_before,
            'frames': elapsed + 1, 'trace': trace}


def source_hashes():
    sources = agent_hashes()
    root, path = Path(__file__).resolve().parents[2], Path(__file__)
    sources[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return sources


def run_job(job):
    matchup, seed, horizon = job
    save, side = MATCHUPS[matchup]
    env = make_retro(game='NHL94-Genesis-v0', state=save, num_players=1, goalie_policy='selective')
    cases, seen = [], set()
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
            shot = model.shot
            key = shot.until_decision, shot.slot, shot.shots_before
            candidate = shot.same_shooter_recovery(view, model.scheduler.decisions) and key not in seen
            if candidate:
                seen.add(key)
                snapshot = env.em.get_state()
                ram_digest = hashlib.sha256(env.get_ram().tobytes()).hexdigest()
                branches = {name: fork_rebound(env, snapshot, ram_digest, state, side, model, enabled, horizon,
                                               wait_for_animation=wait)
                            for name, enabled, wait in (('legacy', False, True), ('immediate', True, False),
                                                       ('ready', True, True))}
                if not cases:
                    assert branches['legacy'] == fork_rebound(
                        env, snapshot, ram_digest, state, side, model, False, horizon), 'Nondeterministic restore'
                restore(env, snapshot, ram_digest)
                cases.append({'frame': frame, 'ram_sha256': ram_digest, 'shot': asdict(shot),
                              'observation': observation(view), 'branches': branches})
            action = model.predict_frame(view, frame_skip=4)[0]
            if candidate:
                assert action.tolist() == cases[-1]['branches']['legacy']['trace'][0]['buttons']
            digest.update(np.asarray(action, dtype=np.int8).tobytes())
            advance(env, action, side)
            frame += 1
        _, info = read_view(env, state, side)
        if info['bench_clock'] != 0:
            raise RuntimeError('Incomplete rebound collection period')
        print(f'{matchup} {seed}: {len(cases)} same-shooter recoveries', flush=True)
        return {'matchup': matchup, 'side': side, 'seed': seed, 'frames': frame, 'completed': True,
                'actions_sha256': digest.hexdigest(), 'goals': [info['p1_score'], info['p2_score']],
                'shots': [info['bench_shots1'], info['bench_shots2']], 'cases': cases}
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=25701)
    parser.add_argument('--trials', type=int, default=20)
    parser.add_argument('--horizon', type=int, default=240)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if min(args.trials, args.horizon, args.workers) < 1 or not 0 <= args.seed <= 2**32-args.trials:
        raise ValueError('Use positive limits and uint32 seeds')
    sources = source_hashes()
    jobs = [(matchup, seed, args.horizon)
            for matchup in ('sabres-ducks-manual', 'ducks-sabres-manual')
            for seed in range(args.seed, args.seed + args.trials)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(run_job, jobs))
    if source_hashes() != sources:
        raise RuntimeError('Sources changed during rebound replay')
    totals = {}
    for variant in ('legacy', 'immediate', 'ready'):
        branches = [c['branches'][variant] for m in rows for c in m['cases']]
        totals[variant] = {'cases': len(branches),
                          **{key: sum(b[key] for b in branches) for key in (
                              'goals_for', 'goals_against', 'recorded_shots', 'native_one_timers', 'recovery_replans')}}
    report = {'protocol': 'native-same-shooter-rebound-v2', 'settings': vars(args), 'sources': sources,
              'totals': totals, 'matches': rows,
              'limitations': ['First observed same-shooter recovery per active normal-shot commitment, before outcome selection.',
                              'An earlier loose-puck observation with increased shot count and matching/unknown shooter is required.',
                              'The ready branch can wait for animation unlock within the original commitment; later commitments use legacy behavior.',
                              'Goals/shots cover a bounded continuation, including later attacks; overlapping cases are correlated.',
                              'Fixed two-roster first periods; slow recoveries after follow-through expires are outside the experiment.']}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(report, separators=(',', ':'), allow_nan=False)+'\n').encode()
    path.write_bytes(gzip.compress(data, mtime=0) if path.suffix == '.gz' else data)
    print(json.dumps(totals))


if __name__ == '__main__':
    main()
