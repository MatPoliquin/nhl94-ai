"""Native release-position regressions; run explicitly with the NHL94 ROM."""
from dataclasses import replace
from functools import partial
from types import SimpleNamespace

import numpy as np

from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.agents.motion import VELOCITY_SCALE
from nhl94_ai.env.actions import HockeyActionController
from nhl94_ai.env.factory import build_single_nhl94_env
from nhl94_ai.evaluation.benchmark import away_view
from nhl94_ai.evaluation.play import parse_cmdline
from nhl94_ai.game.constants import GameConsts as Buttons
from nhl94_ai.game.ram import select_cpu_side
from nhl94_ai.tasks.defense_setup import _place_object, _rebuild_object_order, _reset_player
from nhl94_ai.tasks.registry import TASKS, get_task, register_task
from nhl94_ai.training.datasets import get_game_state
from tests.integration.classic_offense import _feint_setup, _offense_setup, _one_timer_cut_setup


def _shot_setup(env, game, *, side, depth, speed):
    _feint_setup(env, game)
    env.data.update_ram()
    select_cpu_side(env.data, env.data.lookup_all(), side)
    memory = env.data.memory
    actual = memory.extract(0xFFC320, '>i2')
    sign = 1 if side == 1 else -1
    for slot in (*range(5), *range(6, 11)):
        _place_object(memory, slot, (100, -sign * 180))
    role = memory.extract(0xFFB04A + actual * 0x80 + 0x34, '>i2')
    assignment = 1 if role in (1, 2) else 6 if role == 4 else 4
    _reset_player(memory, actual, (40, sign * depth), 0 if side == 1 else 4,
                  assignment, True, (0x10, 0x11))
    memory.assign(0xFFB7AA, '>i2', actual)
    memory.assign(0xFFBEDA, '>i2', actual)
    memory.assign(0xFFBEE0, '>i2', -1)
    base = 0xFFB04A + actual * 0x80
    memory.assign(base + 0x2A, '>i2', round(sign * speed / VELOCITY_SCALE))
    memory.assign(base + 0x6C, '|u1', 20)
    _place_object(memory, 14, (40, sign * (depth + 12)))
    _place_object(memory, 11 if side == 1 else 5, (-50, sign * 260))
    _rebuild_object_order(memory)


def shot_release_guard():
    name = 'ClassicShotReleaseProbe'
    for side in (1, 2):
        for depth, speed in ((225, 0), (245, 1), (250, 0)):
            register_task(name, replace(
                get_task('DefenseZone'),
                initialize=partial(_shot_setup, side=side, depth=depth, speed=speed),
                reward=lambda _: 0.0, done=lambda _: False))
            args = parse_cmdline([
                '--mode=model_vs_game', '--nn=ClassicAIV1', '--env=NHL94-Genesis-v0',
                '--state=PenguinsVsSenators.DefenseZone', f'--rf={name}', '--action_type=FILTERED',
            ])
            env = build_single_nhl94_env(args, {'clip_reward': False}, use_frame_skip=False)
            try:
                for schema in ('FILTERED', 'HOCKEY_INTENT_DPAD'):
                    for guarded in (False, True):
                        env.reset(seed=7)
                        model = ClassicAIV1Model(SimpleNamespace(action_type=schema, one_timers=False))
                        context = SimpleNamespace(action_type=schema)
                        processor = HockeyActionController(context)
                        macro = processor._new_action_state()
                        actual = env.unwrapped.data.memory.extract(0xFFC320, '>i2')
                        released = False
                        before_release_y = None
                        for frame in range(64):
                            state = get_game_state(env)
                            view = state if side == 1 else away_view(state)
                            context.game_state = view
                            if guarded:
                                action = model.predict_frame(view)[0]
                                if frame == 0:
                                    assert (model._last_decision == 'shoot') == (depth == 225), (
                                        side, depth, schema, model.offense_diagnostics)
                                if schema == 'HOCKEY_INTENT_DPAD':
                                    action = processor._process_action(action, macro)[0]
                            else:
                                action = np.zeros(12, dtype=np.int8)
                                action[Buttons.INPUT_RIGHT] = 1
                                action[Buttons.INPUT_C] = int(frame < 4)
                            if (view.engine.puck_owner < 0 and view.engine.shot_player == actual
                                    and view.has_released_shot):
                                sign = 1 if side == 1 else -1
                                behind = view.puck.y * sign >= view.team2.net.y * sign
                                assert not behind if guarded or depth == 225 else behind, (
                                    side, depth, schema, guarded, frame, view.puck.y)
                                assert before_release_y is not None
                                origin_behind = before_release_y * sign >= view.team2.net.y * sign
                                assert not origin_behind if guarded or depth == 225 else origin_behind
                                released = True
                                break
                            if view.is_shooting:
                                before_release_y = view.puck.y
                            env.step(action)
                        assert released or guarded and depth != 225, (side, depth, schema, guarded)
                    print(f'PASS: {schema}, AI side {side}, depth {depth}, speed {speed}: '
                          'native release geometry is guarded at interval 4')
            finally:
                env.close()
                TASKS.pop(name)


def _cue_setup(env, game):
    _one_timer_cut_setup(env, game)
    memory = env.data.memory
    actual = memory.extract(0xFFC320, '>i2')
    receiver, blocker = [slot for slot in range(5) if slot != actual][:2]
    _place_object(memory, receiver, (35, 205))
    _place_object(memory, blocker, (-100, -180))
    _place_object(memory, 11, (-50, 260))
    _rebuild_object_order(memory)


def earliest_contact_cue():
    name = 'ClassicCueProbe'
    register_task(name, replace(get_task('DefenseZone'), initialize=_cue_setup,
                               reward=lambda _: 0.0, done=lambda _: False))
    args = parse_cmdline([
        '--mode=model_vs_game', '--nn=ClassicAIV1', '--env=NHL94-Genesis-v0',
        '--state=PenguinsVsSenators.DefenseZone', f'--rf={name}', '--action_type=FILTERED',
    ])
    env = build_single_nhl94_env(args, {'clip_reward': False}, use_frame_skip=False)
    try:
        for schema in ('FILTERED', 'HOCKEY_INTENT_DPAD'):
            for interval in (4, 8):
                env.reset(seed=7)
                model = ClassicAIV1Model(SimpleNamespace(action_type=schema))
                context = SimpleNamespace(action_type=schema)
                processor = HockeyActionController(context)
                macro = processor._new_action_state()
                for frame in range(32):
                    state = get_game_state(env)
                    context.game_state = state
                    action = model.predict_frame(state, interval)[0]
                    if frame == 0 and interval == 8:
                        candidate = model.offense.diagnostics['one_timer_candidates'][0]
                        assert candidate['status'] == 'too-short-for-one-timer-cue', candidate
                        assert candidate['first_contact_frame'] <= candidate['cue_frame']
                        assert candidate['flight_frames'] + candidate['release_frames'] > candidate['cue_frame']
                    if interval == 8:
                        assert model._last_decision != 'one-timer-pass'
                    if schema == 'HOCKEY_INTENT_DPAD':
                        action = processor._process_action(action, macro)[0]
                    env.step(action)
                attempts = get_game_state(env).team1.one_timer_attempts
                assert attempts == (1 if interval == 4 else 0), (schema, interval, attempts)
                print(f'PASS: {schema}, interval {interval}: earliest contact gates the native short pass')
    finally:
        env.close()
        TASKS.pop(name)


def _inactive_setup(env, game):
    _offense_setup(env, game)
    memory = env.data.memory
    actual = memory.extract(0xFFC320, '>i2')
    blocker = [slot for slot in range(5) if slot != actual][1]
    _place_object(memory, blocker, (-160, -30))
    memory.assign(0xFFB04A + blocker * 0x80 + 0x34, '>i2', -1)
    memory.assign(0xFFB04A + blocker * 0x80 + 0x63, '|u1', 4)
    _rebuild_object_order(memory)


def inactive_blocker():
    name = 'ClassicInactiveProbe'
    register_task(name, replace(get_task('DefenseZone'), initialize=_inactive_setup,
                               reward=lambda _: 0.0, done=lambda _: False))
    args = parse_cmdline([
        '--mode=model_vs_game', '--nn=ClassicAIV1', '--env=NHL94-Genesis-v0',
        '--state=PenguinsVsSenators.DefenseZone', f'--rf={name}', '--action_type=FILTERED',
    ])
    env = build_single_nhl94_env(args, {'clip_reward': False}, use_frame_skip=False)
    try:
        env.reset(seed=7)
        state = get_game_state(env)
        model = ClassicAIV1Model(args)
        receiver = next(slot for slot in range(5) if slot != state.engine.puck_owner)
        before = state.team1.pass_attempts
        observed = None
        for frame in range(90):
            state = get_game_state(env)
            action = model.predict_frame(state)[0]
            if frame == 0:
                assert model._last_decision == 'advance-pass', model.offense_diagnostics
                assert any(p.role is not None and p.role < 0 for p in state.team1.players)
            if observed is None and state.team1.pass_attempts > before:
                observed = state.engine.pass_target
                assert observed == receiver
            if observed == receiver and state.engine.puck_owner == receiver:
                print('PASS: an inactive native skater is not a phantom advancement-pass blocker')
                break
            env.step(action)
        else:
            raise AssertionError('No native reception past inactive blocker')
    finally:
        env.close()
        TASKS.pop(name)


if __name__ == '__main__':
    shot_release_guard()
    earliest_contact_cue()
    inactive_blocker()
