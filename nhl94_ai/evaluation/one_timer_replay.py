"""Trace committed one-timers and compare cue execution in restored native states.

All starts are collected before knowing outcomes, including successful controls.
Branches change only the original sequence; subsequent play uses Classic.
"""
import argparse
from collections import Counter
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
from nhl94_ai.env.actions import HOCKEY_PASS_PRESS_FRAMES
from nhl94_ai.env.factory import make_retro
from nhl94_ai.evaluation.cpu_benchmark import CPU_RAM, MATCHUPS, select_side
from nhl94_ai.evaluation.possession_ablation_replay import observation, source_hashes as agent_hashes
from nhl94_ai.evaluation.refinement_replay import advance, read_view, restore
from nhl94_ai.game.constants import GameConsts as Buttons
from nhl94_ai.game.state import NHL94GameState
from nhl94_ai.tasks.cross_crease_setup import crossing_touch_player


class FrameCueModel(ClassicAIV1Model):  # pylint: disable=abstract-method
    """Negative control: preserve the four-frame pass, then refresh cues per frame."""
    def _scheduled_frame(self, state, deterministic):
        request = self.one_timer.pending
        if (request is not None and not self._defending(state)
                and self.scheduler.frames + 1 - request.started >= HOCKEY_PASS_PRESS_FRAMES):
            self._prepare_exclusive_frame()
            step = self._continue_one_timer(state)
            return self._encode(*step) if step is not None else self._neutral('one-timer-ended')
        return super()._scheduled_frame(state, deterministic)


def configured(history, variant):
    model = deepcopy(history)
    if variant in ('early-cue', 'release-retry'):
        model.one_timer.execution = variant
    elif variant == 'frame-cue':
        model.__class__ = FrameCueModel
    elif variant != 'legacy':
        raise ValueError('Unknown one-timer replay variant')
    return model


def trace_state(env, view, request, action, decision, elapsed):
    receiver = view.team1.get_player_by_scnum(request.receiver)
    passer = view.team1.get_player_by_scnum(request.passer)
    base = 0xFFB04A + request.receiver * 0x80
    return {'frame': elapsed, 'owner': view.engine.puck_owner, 'control': view.team1.defense_control,
            'puck': (view.puck.x, view.puck.y, view.puck.height),
            'puck_velocity': (view.puck.motion_x, view.puck.motion_y, view.puck.motion_z),
            'pass_target': view.engine.pass_target, 'shot_player': view.engine.shot_player,
            'passes': view.team1.pass_attempts, 'last_touch': crossing_touch_player(env),
            'sflags': view.engine.sflags,
            'attempts': view.team1.one_timer_attempts, 'clock_stopped': view.engine.clock_stopped,
            'passer': {name: getattr(passer, name) for name in (
                'x', 'y', 'selection_flags', 'live_state_flags', 'live_anim', 'live_anim_frame',
                'animation_timer', 'is_falling')},
            'receiver': {name: getattr(receiver, name) for name in (
                'x', 'y', 'motion_x', 'motion_y', 'stick_x', 'stick_y', 'is_one_timer',
                'is_falling', 'live_anim', 'live_anim_frame', 'selection_flags', 'live_state_flags',
                'assignment', 'decision_timer', 'facing', 'animation_timer')},
            'receiver_no_puck': env.data.memory.extract(base + 0x5E, '|i1'),
            'buttons': np.asarray(action, dtype=int).tolist(), 'decision': decision}


def fork_timer(env, snapshot, digest, state, side, history, variant, horizon):
    restore(env, snapshot, digest)
    model, decoded = configured(history, variant), deepcopy(state)
    request = deepcopy(model.one_timer.pending)
    metrics_before = Counter(model.one_timer.metrics)
    view, start = read_view(env, decoded, side)
    trace, ended, first_activation, first_attempt, first_possession = [], None, None, None, None
    loose, cues = False, []
    was_c = history.buttons.c_down
    for elapsed in range(horizon):
        action = model.scheduler.action[0].copy() if elapsed == 0 else model.predict_frame(view, frame_skip=4)[0]
        if elapsed == 0:
            assert np.array_equal(action, history.scheduler.action[0]), 'Replay changed the original pass input'
            assert action[Buttons.INPUT_B] and not action[Buttons.INPUT_C]
        if ended is None and model.one_timer.pending is None:
            changes = Counter(model.one_timer.metrics) - metrics_before
            assert sum(changes.values()) == 1, changes
            ended = {'reason': next(iter(changes)), 'frame': elapsed}
            # Keep execution-only branches from changing later one-timer sequences.
            model.__class__ = ClassicAIV1Model
            model.one_timer.execution = None
        if action[Buttons.INPUT_C] and not was_c and ended is None:
            cues.append(elapsed)
        was_c = bool(action[Buttons.INPUT_C])
        if elapsed < 100 or elapsed % 8 == 0:
            trace.append(trace_state(env, view, request, action, model._last_decision, elapsed))
        advance(env, action, side)
        view, info = read_view(env, decoded, side)
        receiver = view.team1.get_player_by_scnum(request.receiver)
        if ended is None and receiver.is_one_timer and first_activation is None:
            first_activation = elapsed + 1
        if (ended is None and view.engine.shot_player == request.receiver
                and view.team1.one_timer_attempts > request.attempts_before and first_attempt is None):
            first_attempt = {'frame': elapsed + 1, 'shooter': view.engine.shot_player}
        owner = view.engine.puck_owner
        loose |= owner < 0
        if first_possession is None and owner >= 0 and (loose or owner != request.passer):
            first_possession = {'frame': elapsed + 1, 'owner': owner}
        if info['bench_clock'] == 0:
            break
    return {'primary_end': ended, 'activation_frame': first_activation, 'first_attempt': first_attempt,
            'first_possession': first_possession, 'cue_edges': cues,
            'goals_for': info[f'p{side}_score'] - start[f'p{side}_score'],
            'goals_against': info[f'p{3-side}_score'] - start[f'p{3-side}_score'],
            'continuation_native_attempts': view.team1.one_timer_attempts - request.attempts_before,
            'frames': elapsed + 1, 'trace': trace}


def source_hashes():
    sources = agent_hashes()
    root, path = Path(__file__).resolve().parents[2], Path(__file__)
    sources[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return sources


def run_job(job):
    matchup, seed, horizon, variants = job
    save, side = MATCHUPS[matchup]
    env = make_retro(game='NHL94-Genesis-v0', state=save, num_players=1, goalie_policy='selective')
    cases = []
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
            request = model.one_timer.pending
            if request is not None and request.started == model.scheduler.frames:
                snapshot = env.em.get_state()
                ram_digest = hashlib.sha256(env.get_ram().tobytes()).hexdigest()
                branches = {variant: fork_timer(env, snapshot, ram_digest, state, side, model, variant, horizon)
                            for variant in variants}
                if not cases:
                    assert branches['legacy'] == fork_timer(
                        env, snapshot, ram_digest, state, side, model, 'legacy', horizon), 'Nondeterministic restore'
                restore(env, snapshot, ram_digest)
                cases.append({'frame': frame, 'ram_sha256': ram_digest, 'request': asdict(request),
                              'observation': observation(view), 'branches': branches})
            digest.update(np.asarray(action, dtype=np.int8).tobytes())
            advance(env, action, side)
            frame += 1
        _, info = read_view(env, state, side)
        if info['bench_clock'] != 0:
            raise RuntimeError('Incomplete one-timer collection period')
        print(f'{matchup} {seed}: {len(cases)} one-timer starts, {model.one_timer.metrics}', flush=True)
        return {'matchup': matchup, 'side': side, 'seed': seed, 'frames': frame, 'completed': True,
                'actions_sha256': digest.hexdigest(), 'goals': [info['p1_score'], info['p2_score']],
                'one_timer_metrics': model.one_timer.metrics, 'cases': cases}
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=25441)
    parser.add_argument('--trials', type=int, default=20)
    parser.add_argument('--horizon', type=int, default=240)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--variants', nargs='+', choices=('legacy', 'frame-cue', 'early-cue', 'release-retry'),
                        default=['legacy', 'frame-cue', 'early-cue', 'release-retry'])
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if ('legacy' not in args.variants or len(args.variants) != len(set(args.variants))
            or min(args.trials, args.horizon, args.workers) < 1 or not 0 <= args.seed <= 2**32-args.trials):
        raise ValueError('Use distinct variants including legacy, positive limits and uint32 seeds')
    sources = source_hashes()
    jobs = [(matchup, seed, args.horizon, args.variants)
            for matchup in ('sabres-ducks-manual', 'ducks-sabres-manual')
            for seed in range(args.seed, args.seed + args.trials)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(run_job, jobs))
    if source_hashes() != sources:
        raise RuntimeError('Sources changed during one-timer replay')
    totals = {}
    for variant in args.variants:
        branches = [c['branches'][variant] for r in rows for c in r['cases']]
        totals[variant] = {'cases': len(branches), 'primary_end': dict(Counter(
            b['primary_end']['reason'] if b['primary_end'] else 'unresolved' for b in branches)),
                          'activated': sum(b['activation_frame'] is not None for b in branches),
                          'goals_for': sum(b['goals_for'] for b in branches),
                          'goals_against': sum(b['goals_against'] for b in branches)}
    report = {'protocol': 'native-one-timer-execution-v2', 'settings': vars(args), 'sources': sources,
              'totals': totals, 'matches': rows,
              'limitations': ['Every original one-timer start, including successful controls; no post-outcome selection.',
                              'Identical native state/RNG and complete controller history at each branch start.',
                              'Only initial sequence execution changes; later play uses Classic.',
                              'Activation/first_attempt refer only to the initial pending sequence and requested receiver.',
                              'Bounded continuation goals include rebounds/later attacks; overlapping cases are correlated.']}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(report, separators=(',', ':'), allow_nan=False)+'\n').encode()
    path.write_bytes(gzip.compress(data, mtime=0) if path.suffix == '.gz' else data)
    print(json.dumps(totals))


if __name__ == '__main__':
    main()
