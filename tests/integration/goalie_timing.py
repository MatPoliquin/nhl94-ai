"""Replay natural CPU goalie opportunities against an archived policy."""
import argparse
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

from nhl94_ai.agents import classic_v1
from nhl94_ai.agents.goalie import GoalieController, _receiver_contact, goalie_arrival, goalie_steer
from nhl94_ai.env import factory
from nhl94_ai.evaluation.cpu_benchmark import CPU_RAM, MATCHUPS, cpu_match, cpu_view
from nhl94_ai.game.constants import GameConsts as Buttons
from nhl94_ai.game.ram import pass_geometry_info, restore_away_control
from nhl94_ai.game.state import NHL94GameState


class CapturedEnough(Exception):
    pass


def baseline_controller(root):
    name = 'nhl94_ai.agents._goalie_timing_baseline'
    spec = importlib.util.spec_from_file_location(name, Path(root) / 'nhl94_ai/agents/goalie.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def capture(matchup, seed, baseline):
    original_factory, original_predict = factory.make_retro, classic_v1.ClassicAIV1Model.predict_frame
    holder, samples = {}, []
    frame = 0
    last_attempts = None

    def make_env(**kwargs):
        env = original_factory(**kwargs)
        holder['env'] = env
        return env

    def predict(model, state, *args, **kwargs):
        nonlocal frame, last_attempts
        frame += 1
        goalie = state.team1.goalie
        manager = model.goalie
        selected = (state.team1.defense_control == state.team1.defense_goalie
                    and goalie.selection_flags & 8 and not goalie.live_state_flags & 4)
        available = not goalie.unavailable & 2 and not goalie.selection_flags & 0x20
        incoming, plan = _receiver_contact(state, None), baseline.goalie_target(state)
        attempts = state.team2.one_timer_attempts
        release = (last_attempts is not None and attempts > last_attempts
                   and state.team2.owns_scnum(state.engine.shot_player)
                   and state.engine.last_puck_player == state.engine.shot_player)
        evidence = ({'attempts_before': last_attempts, 'attempts_after': attempts,
                     'shot_player': state.engine.shot_player,
                     'last_puck_player': state.engine.last_puck_player} if release else None)
        last_attempts = attempts
        opportunity = (plan.crossing_frames is not None and plan.crossing_frames <= 12
                       and abs(plan.crossing_x - goalie.x) <= 18)
        pass_first = opportunity and incoming is not None and incoming[0] < plan.crossing_frames
        align_first = (opportunity and abs(plan.crossing_x - goalie.x) > 8 and plan.height <= 15
                       and goalie_arrival(goalie, plan.target) + 1 <= plan.crossing_frames
                       and goalie_steer(goalie, plan.target)[4:8].any())
        ready = selected and available and manager.frames >= manager.save_at and manager.pending_save is None
        kind = ('native-one-timer' if release else
                'receiver-before-goalie' if pass_first else 'align-before-save')
        opportunity = selected and release or ready and (pass_first or align_first)
        if opportunity and sum(sample['kind'] == kind for sample in samples) < 2:
            samples.append({'matchup': matchup, 'seed': seed, 'frame': frame,
                            'kind': kind, 'one_timer_release': evidence,
                            'crossing_frames': plan.crossing_frames,
                            'snapshot': holder['env'].em.get_state(), 'manager': deepcopy(manager),
                            'state': deepcopy(state)})
            if len(samples) >= 4:
                raise CapturedEnough()
        return original_predict(model, state, *args, **kwargs)

    with patch.object(factory, 'make_retro', side_effect=make_env), \
            patch.object(classic_v1, 'GoalieController', baseline.GoalieController), \
            patch.object(classic_v1.ClassicAIV1Model, 'predict_frame', new=predict):
        try:
            cpu_match(('classic-v1', matchup, seed, 300, 4, 'FILTERED', 'selective'))
        except CapturedEnough:
            pass
    return samples


def replay(sample, fixed):
    state_name, side = MATCHUPS[sample['matchup']]
    env = factory.make_retro(game='NHL94-Genesis-v0', state=state_name, num_players=1,
                             goalie_policy='selective')
    try:
        for name, (address, kind) in CPU_RAM.items():
            env.data.set_variable(name, {'address': address, 'type': kind})
        env.reset(seed=sample['seed'])
        env.em.set_state(sample['snapshot'])
        env.data.update_ram()
        info = env.data.lookup_all()
        opponent = 3 - side
        before = info[f'p{opponent}_score']
        attempts = info[f'bench_one_timers{opponent}']
        manager = deepcopy(sample['manager'])
        if fixed:
            updated = GoalieController('selective')
            for name, value in vars(manager).items():
                if hasattr(updated, name):
                    setattr(updated, name, deepcopy(value))
            manager = updated
        state, trace = NHL94GameState(5), []
        for frame in range(80):
            info = pass_geometry_info(env, info)
            view = cpu_view(state, info, side)
            memory = bytes(env.get_ram())
            with patch.object(env.data, 'set_value', side_effect=AssertionError('Policy attempted a RAM write')):
                action = manager.step(view)
            assert bytes(env.get_ram()) == memory, 'Policy mutated gameplay RAM instead of pressing buttons'
            if action is None:
                action = np.zeros(12, dtype=np.int8)
            assert action.shape == (12,) and np.isin(action, (0, 1)).all(), 'Invalid button action'
            trace.append({
                'frame': frame, 'goalie': [view.team1.goalie.x, view.team1.goalie.y],
                'velocity': [view.team1.goalie.motion_x, view.team1.goalie.motion_y],
                'puck': [view.puck.x, view.puck.y, view.puck.height],
                'owner': view.engine.puck_owner, 'mode': manager.phase,
                'animation': view.team1.goalie.live_anim,
                'locked': bool(view.team1.goalie.unavailable & 2
                               or view.team1.goalie.selection_flags & 0x20),
                'cpu_fallback': bool(view.team1.goalie.live_state_flags & 4),
                'target': manager.diagnostics['destination'], 'buttons': action.tolist(),
                'one_timer_attempts': info[f'bench_one_timers{opponent}'] - attempts,
                'goals_against': info[f'p{opponent}_score'] - before,
            })
            if info[f'p{opponent}_score'] > before or view.engine.clock_stopped:
                break
            *_, info = env.step(action)
            if side == 2:
                info = restore_away_control(env.data, info, controller_prefix='bench',
                                           player_prefix='cpu', slots=6)
        return {'fixed': fixed, 'trace': trace, 'metrics': dict(manager.metrics)}
    finally:
        env.close()


def check_replays(results):
    alignments = [row for row in results if row['kind'] == 'align-before-save']
    releases = [row for row in results if row['one_timer_release'] is not None]
    assert alignments, 'No natural reachable alignment captured.'
    assert releases, 'No native one-timer release with shooter attribution captured.'
    movement_improved = False
    for row in results:
        before, after = row['before']['trace'], row['after']['trace']
        assert before[0]['goalie'] == after[0]['goalie'] and before[0]['puck'] == after[0]['puck']
        for frame in after:
            if frame['locked'] or frame['cpu_fallback']:
                assert not any(frame['buttons']), 'Animation lock/CPU fallback was overridden'
        if row['kind'] == 'align-before-save':
            assert before[0]['buttons'][Buttons.INPUT_C], 'Baseline did not request the early save'
            assert not after[0]['buttons'][Buttons.INPUT_C] and any(after[0]['buttons'][4:8])
            target = before[0]['target'][0]
            window = min(len(before), len(after), math.ceil(row['crossing_frames']) + 1)
            movement_improved |= (min(abs(frame['goalie'][0] - target) for frame in after[:window])
                                  < min(abs(frame['goalie'][0] - target) for frame in before[:window]))
        evidence = row['one_timer_release']
        if evidence is not None:
            assert evidence['attempts_after'] > evidence['attempts_before']
            assert evidence['shot_player'] == evidence['last_puck_player']
    assert movement_improved, 'Button changes did not improve any actual pre-contact alignment'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root', required=True)
    parser.add_argument('--output')
    args = parser.parse_args()
    baseline = baseline_controller(args.baseline_root)
    results = []
    for matchup, seed in (('ducks-sabres-manual', 20261204), ('ducks-campbell-manual', 20261211),
                          ('campbell-ducks-manual', 20261214), ('ducks-campbell-manual', 20261203)):
        samples = capture(matchup, seed, baseline)
        print('Natural goalie snapshots:', matchup, seed, len(samples), flush=True)
        for sample in samples:
            before, after = replay(sample, False), replay(sample, True)
            result = {key: sample[key] for key in ('matchup', 'seed', 'frame', 'kind',
                                                 'one_timer_release', 'crossing_frames')}
            result.update(before=before, after=after)
            results.append(result)
            print('Replay:', matchup, seed, sample['frame'],
                  'goals', before['trace'][-1]['goals_against'], after['trace'][-1]['goals_against'],
                  'one-timers', before['trace'][-1]['one_timer_attempts'],
                  after['trace'][-1]['one_timer_attempts'], flush=True)
    check_replays(results)
    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
