"""Replay one-timers rejected by exactly one positional heuristic on Classic trajectories."""
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

from nhl94_ai.agents.defense import controlled_slot
from nhl94_ai.env.factory import make_retro
from nhl94_ai.evaluation.carry_outcomes import CarryOutcomes
from nhl94_ai.evaluation.cpu_benchmark import CPU_RAM, MATCHUPS, select_side
from nhl94_ai.evaluation.one_timer_admission import AdmissionModel, GATES, configured, original_controller
from nhl94_ai.evaluation.pass_outcomes import PassOutcomes
from nhl94_ai.evaluation.possession_ablation_replay import observation, source_hashes as agent_hashes
from nhl94_ai.evaluation.refinement_replay import advance, read_view, restore
from nhl94_ai.game.state import NHL94GameState
from nhl94_ai.tasks.cross_crease_setup import crossing_touch_player


def pending_key(model):
    request = model.one_timer.pending
    return (request.passer, request.receiver, request.started) if request is not None else None


def fork_admission(env, snapshot, digest, state, side, history, action, horizon):
    restore(env, snapshot, digest)
    model, decoded = original_controller(history), deepcopy(state)
    view, start = read_view(env, decoded, side)
    outcomes = CarryOutcomes(view, start[f'p{side}_score'], crossing_touch_player(env))
    passes = PassOutcomes(side)
    request = model._last_pass_request
    if request and request['frame'] == model.scheduler.frames:
        passes.start(0, start, request['passer'], request['receiver'], request['purpose'])
    original_key, original_shot = pending_key(model), None
    actions, trace = hashlib.sha256(), []
    loss, opponent_run, stopped_run, attack_stopped = None, 0, 0, None
    end = 'horizon'
    for elapsed in range(horizon):
        if elapsed:
            action = model.predict_frame(view, frame_skip=4)[0]
        still_original = original_key is not None and pending_key(model) == original_key
        attempts = view.team1.one_timer_attempts
        actions.update(np.asarray(action, dtype=np.int8).tobytes())
        advance(env, action, side)
        view, info = read_view(env, decoded, side)
        passes.observe(elapsed + 1, info)
        outcomes.observe(elapsed + 1, view, info[f'p{side}_score'], crossing_touch_player(env))
        if (still_original and original_shot is None and view.team1.one_timer_attempts > attempts
                and view.engine.shot_player == original_key[1]):
            original_shot = outcomes.shot_events[-1]
        opponent_run = opponent_run + 1 if view.team2.owns_scnum(view.engine.puck_owner) else 0
        stopped_run = stopped_run + 1 if view.engine.clock_stopped else 0
        if opponent_run >= 4 and loss is None:
            loss = elapsed + 1
        if stopped_run >= 4 and attack_stopped is None:
            attack_stopped = elapsed + 1
        if elapsed < 32 or elapsed % 12 == 0:
            trace.append({'frame': elapsed + 1, 'owner': view.engine.puck_owner,
                          'puck': (view.puck.x, view.puck.y, view.puck.height),
                          'puck_velocity': (view.puck.motion_x, view.puck.motion_y),
                          'buttons': np.asarray(action, dtype=int).tolist(), 'decision': model._last_decision,
                          'shot_player': view.engine.shot_player})
        if any(info[f'p{s}_score'] != start[f'p{s}_score'] for s in (1, 2)):
            end = 'goal'
            break
        if info['bench_clock'] == 0:
            end = 'period-ended'
            break
    if passes.pending is not None:
        passes.finish(elapsed + 1, info, 'unresolved-at-horizon')
    native = outcomes.summary()
    goals = info[f'p{side}_score'] - start[f'p{side}_score']
    return {'goals_for': goals, 'goals_against': info[f'p{3-side}_score'] - start[f'p{3-side}_score'],
            'attack_goal': int(goals > 0 and loss is None and attack_stopped is None),
            'first_opponent_possession': loss, 'first_stoppage': attack_stopped,
            'intended_one_timer': int(original_shot is not None),
            'intended_one_timer_goal': int(original_shot is not None and original_shot['outcome'] == 'goal'),
            'intended_one_timer_event': original_shot,
            'initial_pass_events': passes.events, 'native': native, 'end': end,
            'frames': elapsed + 1, 'actions_sha256': actions.hexdigest(), 'trace': trace}


def source_hashes():
    result = agent_hashes()
    root = Path(__file__).resolve().parents[2]
    for name in ('one_timer_admission.py', 'one_timer_admission_replay.py', 'pass_outcomes.py', 'chance_selection.py'):
        path = root/'nhl94_ai/evaluation'/name
        result[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    path = root/'nhl94_ai/training/shot_placement.py'
    result[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def run_job(job):
    matchup, seed, horizon, limit, spacing = job
    save, side = MATCHUPS[matchup]
    env = make_retro(game='NHL94-Genesis-v0', state=save, num_players=1, goalie_policy='selective')
    cases, counts, sampled = [], Counter(), Counter()
    last_sample = dict.fromkeys(GATES, -spacing)
    try:
        for name, (address, kind) in CPU_RAM.items():
            env.data.set_variable(name, {'address': address, 'type': kind})
        env.reset(seed=seed)
        env.data.update_ram()
        select_side(env.data, env.data.lookup_all(), side)
        env.data.set_value('bench_rng', seed)
        env.data.set_value('bench_clock', 300)
        env.step(np.zeros(12, dtype=np.int8))
        model = AdmissionModel(SimpleNamespace(action_type='FILTERED', goalie_policy='selective'))
        state, frame, actions = NHL94GameState(5), 1, hashlib.sha256()
        while frame < 42000:
            view, info = read_view(env, state, side)
            if info['bench_clock'] == 0:
                break
            eligible = (view.engine.puck_owner == controlled_slot(view.team1) and not view.engine.clock_stopped
                        and (model.scheduler.remaining == 0 or model.scheduler.was_defending)
                        and model.one_timer.pending is None and model.offense.pass_action.pending is None
                        and not model.shot.active(model.scheduler.decisions + 1))
            sampleable = [gate for gate in GATES if sampled[gate] < limit and frame - last_sample[gate] >= spacing]
            before = original_controller(model) if eligible and sampleable else None
            action = model.predict_frame(view, frame_skip=4)[0]
            audit = model.admission_audit
            if audit is not None:
                counts['attack-decisions'] += 1
                counts['audit:' + audit['status']] += 1
                counts.update('original:' + row['original_status'] for row in audit['rows'])
                counts.update('after-window:' + row.get('remaining_gate', row['after_window']['status'])
                              for row in audit['rows'] if 'after_window' in row)
                counts.update('single-gate:' + row['gate'] for row in audit['alternatives'])
                candidates = audit['alternatives'] if audit['status'] == 'audited' else []
                gates = {row['gate'] for row in candidates}
                if candidates:
                    counts['decisions-with-alternative'] += 1
                    counts['baseline:' + model._last_decision] += 1
                if before is not None and gates.intersection(sampleable):
                    baseline, first = configured(before, view)
                    assert np.array_equal(first, action) and baseline._last_decision == model._last_decision
                    snapshot = env.em.get_state()
                    digest = hashlib.sha256(env.get_ram().tobytes()).hexdigest()
                    original = fork_admission(env, snapshot, digest, state, side, baseline, first, horizon)
                    assert original == fork_admission(env, snapshot, digest, state, side, model, action, horizon)
                    if not cases:
                        assert original == fork_admission(env, snapshot, digest, state, side, baseline, first, horizon)
                    branches = []
                    for candidate in candidates:
                        branch, buttons = configured(before, view, candidate)
                        result = fork_admission(env, snapshot, digest, state, side, branch, buttons, horizon)
                        branches.append({'gate': candidate['gate'], 'option': asdict(candidate['option']),
                                         'features': candidate['features'], 'outcome': result})
                    restore(env, snapshot, digest)
                    cases.append({'frame': frame, 'ram_sha256': digest, 'baseline_decision': model._last_decision,
                                  'baseline': original, 'observation': observation(view),
                                  'rejections': audit['rows'], 'branches': branches})
                    for gate in gates:
                        last_sample[gate] = frame
                        sampled[gate] += 1
            actions.update(np.asarray(action, dtype=np.int8).tobytes())
            advance(env, action, side)
            frame += 1
        _, info = read_view(env, state, side)
        if info['bench_clock'] != 0:
            raise RuntimeError('Incomplete admission audit period')
        print(f'{matchup} {seed}: {len(cases)} cases, {dict(sampled)}, {dict(counts)}', flush=True)
        return {'matchup': matchup, 'side': side, 'seed': seed, 'frames': frame, 'completed': True,
                'actions_sha256': actions.hexdigest(), 'goals': [info['p1_score'], info['p2_score']],
                'shots': [info['bench_shots1'], info['bench_shots2']], 'counts': dict(counts), 'cases': cases}
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=26101)
    parser.add_argument('--trials', type=int, default=20)
    parser.add_argument('--horizon', type=int, default=360)
    parser.add_argument('--limit', type=int, default=12)
    parser.add_argument('--spacing', type=int, default=60)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--matchups', nargs='+', choices=list(MATCHUPS),
                        default=['sabres-ducks-manual', 'ducks-sabres-manual'])
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if min(args.trials, args.horizon, args.limit, args.spacing, args.workers) < 1 or not 0 <= args.seed <= 2**32-args.trials:
        raise ValueError('Use positive limits and uint32 seeds')
    sources = source_hashes()
    jobs = [(matchup, seed, args.horizon, args.limit, args.spacing) for seed in range(args.seed, args.seed + args.trials)
            for matchup in args.matchups]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        matches = list(pool.map(run_job, jobs))
    if source_hashes() != sources:
        raise RuntimeError('Sources changed during admission collection')
    report = {'protocol': 'native-one-timer-admission-v1', 'settings': vars(args), 'sources': sources, 'matches': matches,
              'limitations': ['Union of first spaced eligible states per gate, capped per period, on legacy trajectories.',
                              'Each alternative passes every original gate except exactly one positional heuristic.',
                              'Counterfactual forces that pass for one decision, including over original shot priority.',
                              'Fresh B edge, original cue/recipient executor; subsequent decisions use Classic.',
                              'Attack credit stops at four consecutive opponent or stopped-clock frames; continuation runs to horizon, period end or first goal.',
                              'Counterattack goals conceded count even after losing the initial attack.',
                              'Cases can overlap and are correlated; split complete seeds and both assignments together.']}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(report, separators=(',', ':'), allow_nan=False)+'\n').encode()
    path.write_bytes(gzip.compress(data, mtime=0) if path.suffix == '.gz' else data)
    print('Collected', sum(len(m['cases']) for m in matches), 'one-timer admission cases')


if __name__ == '__main__':
    main()
