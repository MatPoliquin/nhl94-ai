"""Replay nine directional inputs at Classic's existing normal-shot starts.

C hold and shot selection are unchanged. The first shot is attributed separately
from a bounded attack that includes friendly rebounds and later Classic actions.
No emulator or ROM bytes are written to disk.
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
from nhl94_ai.env.factory import make_retro
from nhl94_ai.evaluation.carry_outcomes import CarryOutcomes
from nhl94_ai.evaluation.cpu_benchmark import CPU_RAM, MATCHUPS, select_side
from nhl94_ai.evaluation.possession_ablation_replay import observation, source_hashes as agent_hashes
from nhl94_ai.evaluation.refinement_replay import advance, read_view, restore
from nhl94_ai.evaluation.shot_direction import PADS, direction_features, legacy_pad, placed
from nhl94_ai.game.constants import GameConsts as Buttons
from nhl94_ai.game.state import NHL94GameState
from nhl94_ai.tasks.cross_crease_setup import crossing_touch_player


def fork_direction(env, snapshot, digest, state, side, history, pad, horizon):
    restore(env, snapshot, digest)
    model = deepcopy(history) if pad is None else placed(history, pad)
    decoded = deepcopy(state)
    view, start = read_view(env, decoded, side)
    tracker = CarryOutcomes(view, start[f'p{side}_score'], crossing_touch_player(env))
    key = model.shot.slot, model.shot.hold_until_frame
    trace, actions = [], hashlib.sha256()
    primary, release, abandoned = None, None, None
    opponent_run = stopped_run = 0
    attack_end = 'horizon'
    for elapsed in range(horizon):
        action = model.scheduler.action[0].copy() if elapsed == 0 else model.predict_frame(view, frame_skip=4)[0]
        if primary is None and abandoned is None and (
                (model.shot.slot, model.shot.hold_until_frame) != key
                or model.one_timer.pending is not None or model.offense.pass_action.pending is not None):
            abandoned = 'next-action-before-primary-release'
        if elapsed == 0:
            assert action[Buttons.INPUT_C] and not action[Buttons.INPUT_B]
            assert model.shot.hold_until_frame == history.shot.hold_until_frame
            assert model.shot.side == history.shot.side, 'Changed follow-through side'
        actions.update(np.asarray(action, dtype=np.int8).tobytes())
        advance(env, action, side)
        view, info = read_view(env, decoded, side)
        tracker.observe(elapsed + 1, view, info[f'p{side}_score'], crossing_touch_player(env))
        if primary is None and abandoned is None and tracker.shot_events:
            event = tracker.shot_events[0]
            if event['shooter'] == key[0]:
                primary = event
                release = {'frame': event['release_frame'], 'observed_frame': elapsed + 1,
                           'native_direction': env.data.memory.extract(0xFFBEDC, '>i2'),
                           'point': (view.puck.x, view.puck.y, view.puck.height),
                           'velocity': (view.puck.motion_x, view.puck.motion_y, view.puck.motion_z),
                           'evidence': event['evidence']}
            else:
                abandoned = 'different-shooter'
        player = view.team1.get_player_by_scnum(key[0])
        if elapsed < 35 or elapsed % 12 == 0:
            trace.append({'frame': elapsed + 1, 'owner': view.engine.puck_owner,
                          'puck': (view.puck.x, view.puck.y, view.puck.height),
                          'puck_velocity': (view.puck.motion_x, view.puck.motion_y, view.puck.motion_z),
                          'shooter_animation': (player.live_anim, player.live_anim_frame),
                          'shooter_flags': player.selection_flags,
                          'buttons': np.asarray(action, dtype=int).tolist(), 'decision': model._last_decision})
        opponent_run = opponent_run + 1 if view.team2.owns_scnum(view.engine.puck_owner) else 0
        stopped_run = stopped_run + 1 if view.engine.clock_stopped else 0
        if info[f'p{side}_score'] > start[f'p{side}_score']:
            attack_end = 'goal'
        elif info[f'p{3-side}_score'] > start[f'p{3-side}_score']:
            attack_end = 'opponent-goal'
        elif opponent_run >= 4:
            attack_end = 'opponent-possession'
        elif stopped_run >= 4:
            attack_end = 'stoppage'
        elif info['bench_clock'] == 0:
            attack_end = 'period-ended'
        if attack_end != 'horizon':
            break
    native = tracker.summary()
    return {'goal': int(info[f'p{side}_score'] > start[f'p{side}_score']),
            'goals_against': info[f'p{3-side}_score'] - start[f'p{3-side}_score'],
            'primary_goal': int(primary is not None and primary['outcome'] == 'goal'),
            'primary_end': primary['outcome'] if primary is not None else abandoned or 'no-observed-release',
            'release': release, 'attack_end': attack_end, 'native': native,
            'frames': elapsed + 1, 'actions_sha256': actions.hexdigest(), 'trace': trace}


def source_hashes():
    sources = agent_hashes()
    root = Path(__file__).resolve().parents[2]
    for name in ('shot_direction.py', 'shot_direction_replay.py', 'carry_outcomes.py'):
        path = Path(__file__).with_name(name)
        sources[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    path = root/'nhl94_ai/training/shot_placement.py'
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
                baseline = legacy_pad(view)
                pads = [baseline, *(pad for pad in PADS if pad != baseline)]
                features = [direction_features(view, pad) for pad in pads]
                if any(row is None for row in features):
                    counts['missing-features'] += 1
                if len(cases) < limit:
                    snapshot = env.em.get_state()
                    ram_digest = hashlib.sha256(env.get_ram().tobytes()).hexdigest()
                    branches = [{'pad': pad, 'features': values,
                                 'outcome': fork_direction(env, snapshot, ram_digest, state, side, model, pad, horizon)}
                                for pad, values in zip(pads, features)]
                    original = fork_direction(env, snapshot, ram_digest, state, side, model, None, horizon)
                    assert branches[0]['outcome'] == original, 'Directional baseline changed original continuation'
                    if not cases:
                        assert original == fork_direction(env, snapshot, ram_digest, state, side, model, None, horizon)
                    restore(env, snapshot, ram_digest)
                    cases.append({'frame': frame, 'ram_sha256': ram_digest,
                                  'observation': observation(view), 'baseline': baseline, 'branches': branches})
            digest.update(np.asarray(action, dtype=np.int8).tobytes())
            advance(env, action, side)
            frame += 1
        _, info = read_view(env, state, side)
        if info['bench_clock'] != 0:
            raise RuntimeError('Incomplete directional-shot collection period')
        print(f'{matchup} {seed}: {len(cases)} shot starts, {dict(counts)}', flush=True)
        return {'matchup': matchup, 'side': side, 'seed': seed, 'frames': frame,
                'completed': True, 'actions_sha256': digest.hexdigest(),
                'goals': [info['p1_score'], info['p2_score']],
                'shots': [info['bench_shots1'], info['bench_shots2']], 'counts': dict(counts), 'cases': cases}
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=26101)
    parser.add_argument('--trials', type=int, default=20)
    parser.add_argument('--horizon', type=int, default=240)
    parser.add_argument('--limit', type=int, default=12)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if min(args.trials, args.horizon, args.limit, args.workers) < 1 or not 0 <= args.seed <= 2**32-args.trials:
        raise ValueError('Use positive limits and uint32 seeds')
    sources = source_hashes()
    jobs = [(matchup, seed, args.horizon, args.limit)
            for matchup in ('sabres-ducks-manual', 'ducks-sabres-manual')
            for seed in range(args.seed, args.seed + args.trials)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(run_job, jobs))
    if source_hashes() != sources:
        raise RuntimeError('Sources changed during directional shot collection')
    report = {'protocol': 'native-shot-direction-v1', 'settings': vars(args), 'sources': sources, 'matches': rows,
              'limitations': ['First normal-shot starts per baseline period, collected before outcomes.',
                              'Nine screen-direction inputs; four-frame C hold and original follow-through side.',
                              'Every baseline fork equals the unmodified controller; first control repeated per period.',
                              'Only original aim changes; subsequent shots/passes use Classic.',
                              'Primary goal separately attributed; attack goal includes rebounds and subsequent finishes.',
                              'Attack ends at goal, four opponent-ownership or stopped-clock frames, period end, or horizon.',
                              'Overlapping cases are correlated; split complete seeds and both assignments together.']}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(report, separators=(',', ':'), allow_nan=False)+'\n').encode()
    path.write_bytes(gzip.compress(data, mtime=0) if path.suffix == '.gz' else data)
    print('Collected', sum(len(row['cases']) for row in rows), 'normal-shot starts')


if __name__ == '__main__':
    main()
