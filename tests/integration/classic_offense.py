"""Isolated ROM progression passes, with actual recipient and reception evidence."""
from dataclasses import replace
from functools import partial
import math

import numpy as np

from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.agents.motion import VELOCITY_SCALE
from nhl94_ai.agents.offense import FEINT_FRAMES
from nhl94_ai.env.factory import build_single_nhl94_env
from nhl94_ai.env.intents import HOCKEY_INTENT_PASS_START
from nhl94_ai.evaluation.play import parse_cmdline
from nhl94_ai.game.constants import GameConsts as Buttons
from nhl94_ai.tasks.defense_setup import _place_object, _rebuild_object_order, _reset_player
from nhl94_ai.tasks.registry import TASKS, get_task, register_task
from nhl94_ai.training.datasets import get_game_state


def _offense_setup(env, game):
    get_task('DefenseZone').initialize(env, game)
    memory = env.data.memory
    actual = memory.extract(0xFFC320, '>i2')
    opponent_carrier = memory.extract(0xFFB7AA, '>i2')
    others = [slot for slot in range(5) if slot != actual]
    positions = {actual: (0, -130), **dict(zip(others, ((-80, -30), (100, -210), (110, -240), (70, -220))))}
    positions.update(dict(zip(range(6, 11), ((40, -75), (110, 80), (110, 140), (90, 180), (100, 220)))))
    for slot, point in positions.items():
        role = memory.extract(0xFFB04A + slot * 0x80 + 0x34, '>i2')
        assignment = 1 if role in (1, 2) else (6 if role == 4 else 4)
        temporary = (0x10, 0x11) if slot == actual else ((0x11,) if slot == opponent_carrier else ())
        _reset_player(memory, slot, point, 0, assignment, slot == actual, temporary)
        memory.assign(0xFFB04A + slot * 0x80 + 0x71, '|u1', 30)
    memory.assign(0xFFB04A + actual * 0x80 + 0x6E, '|u1', 20)
    memory.assign(0xFFB7AA, '>i2', actual)
    memory.assign(0xFFBEDA, '>i2', actual)
    memory.assign(0xFFBEE0, '>i2', -1)
    _place_object(memory, 14, (0, -120))
    _rebuild_object_order(memory)


def _reception_setup(env, game, *, receiver_energy):
    _offense_setup(env, game)
    memory = env.data.memory
    actual = memory.extract(0xFFC320, '>i2')
    receiver = next(slot for slot in range(5) if slot != actual)
    memory.assign(0xFFB04A + actual * 0x80 + 0x6E, '|u1', 30)
    base = 0xFFB04A + receiver * 0x80
    memory.assign(base + 0x71, '|u1', 20)
    roster = memory.extract(base + 0x66, '|u1')
    memory.assign(0xFFC700 + roster * 2, '>u2', receiver_energy)


def advancement_pass(*, receiver_energy=4096):
    name = 'ClassicAdvancementProbe'
    register_task(name, replace(get_task('DefenseZone'),
                               initialize=partial(_reception_setup, receiver_energy=receiver_energy),
                               reward=lambda _: 0.0, done=lambda _: False))
    try:
        for schema in ('FILTERED', 'HOCKEY_INTENT_DPAD'):
            args = parse_cmdline([
                '--mode=model_vs_game', '--nn=ClassicAIV1', '--env=NHL94-Genesis-v0',
                '--state=PenguinsVsSenators.DefenseZone', f'--rf={name}', f'--action_type={schema}',
            ])
            env = build_single_nhl94_env(args, {'clip_reward': False}, use_frame_skip=False)
            try:
                env.reset(seed=7)
                state = get_game_state(env)
                actual = state.team1.defense_control
                receiver = next(slot for slot in range(5) if slot != actual)
                before = state.team1.pass_attempts
                model = ClassicAIV1Model(args)
                observed = None
                incoming_speed, incoming_energy = None, None
                for frame in range(90):
                    state = get_game_state(env)
                    if observed is None and state.team1.pass_attempts > before:
                        assert state.engine.pass_target == receiver, (schema, state.engine.pass_target, receiver)
                        observed = state.engine.pass_target
                    if observed is not None and state.engine.puck_owner == receiver:
                        player = state.team1.get_player_by_scnum(receiver)
                        assert player.y > -75
                        assert state.team1.defense_control == receiver, (schema, state.team1.defense_control, receiver)
                        assert incoming_speed is not None and incoming_speed <= 13000 + 350 * player.stick
                        if receiver_energy < 4096:
                            assert incoming_energy <= receiver_energy
                            assert incoming_speed > 13000 + 350 * player.stick * incoming_energy / 4096
                        break
                    if observed is not None and state.engine.puck_owner < 0:
                        incoming_speed = math.hypot(state.puck.motion_x, state.puck.motion_y) / VELOCITY_SCALE
                        incoming_energy = state.team1.get_player_by_scnum(receiver).energy
                    action = model.predict_frame(state)[0]
                    if frame == 0:
                        assert state.team1.get_player_by_scnum(receiver).energy == receiver_energy
                        assert state.team1.get_player_by_scnum(receiver).stick == 20
                        assert model._last_decision == 'advance-pass', model.offense_diagnostics
                        assert model.offense.pass_action.pending.receiver == receiver
                    assert not model.defense_diagnostics, (schema, frame, model.defense_diagnostics)
                    if schema == 'FILTERED':
                        assert not action[Buttons.INPUT_C], (schema, frame, action)
                    else:
                        assert action[0] == 0 or action[0] >= HOCKEY_INTENT_PASS_START, (schema, action)
                        assert not action[-1], (schema, frame, action)
                    env.step(action)
                else:
                    raise AssertionError((schema, 'No observed advanced reception', model.offense_diagnostics))
                print(f'PASS: {schema} ROM selects receiver {receiver}, completes an advanced reception '
                      f'in {frame} frames at energy {receiver_energy}, without defensive switching or one-timer C')
            finally:
                env.close()
    finally:
        TASKS.pop(name)


def _feint_setup(env, game):
    _offense_setup(env, game)
    memory = env.data.memory
    actual = memory.extract(0xFFC320, '>i2')
    _place_object(memory, actual, (0, 180))
    for slot in range(5):
        if slot != actual:
            _place_object(memory, slot, (100, -180))
    for slot in range(6, 11):
        _place_object(memory, slot, (-8, 210) if slot == 6 else (110, -180))
    for slot in (actual, *range(6, 11)):
        base = 0xFFB04A + slot * 0x80
        for offset, value in ((0x67, 64), (0x68, 20), (0x69, 20), (0x6D, 20)):
            memory.assign(base + offset, '|u1', value)
        roster = memory.extract(base + 0x66, '|u1')
        memory.assign(0xFFC700 + (slot // 6) * 0x364 + roster * 2, '>u2', 4096)
    _place_object(memory, 11, (-18, 250))
    _place_object(memory, 14, (0, 190))
    _rebuild_object_order(memory)


def purposeful_feint():
    name = 'ClassicFeintProbe'
    register_task(name, replace(get_task('DefenseZone'), initialize=_feint_setup,
                               reward=lambda _: 0.0, done=lambda _: False))
    try:
        args = parse_cmdline([
            '--mode=model_vs_game', '--nn=ClassicAIV1', '--env=NHL94-Genesis-v0',
            '--state=PenguinsVsSenators.DefenseZone', f'--rf={name}', '--action_type=FILTERED',
        ])
        env = build_single_nhl94_env(args, {'clip_reward': False}, use_frame_skip=False)
        try:
            env.reset(seed=7)
            model = ClassicAIV1Model(args)
            state = get_game_state(env)
            actual, before_x = state.team1.defense_control, state.team1.get_controlled_player().x
            for frame in range(FEINT_FRAMES):
                action = model.predict_frame(get_game_state(env))[0]
                if frame == 0:
                    assert model._last_decision == 'feint', model.offense_diagnostics
                    assert model.offense_diagnostics['reason'] == 'open shooting lane'
                assert not action[Buttons.INPUT_B] and not action[Buttons.INPUT_C]
                env.step(action)
            player = get_game_state(env).team1.get_player_by_scnum(actual)
            assert player.x > before_x, (before_x, player.x)
            assert model.offense.feint_at > model.scheduler.frames
            print('PASS: purposeful cut changes actual carrier direction without pass/shot input; bounded cooldown')
        finally:
            env.close()
    finally:
        TASKS.pop(name)


def _one_timer_cut_setup(env, game):
    _feint_setup(env, game)
    memory = env.data.memory
    actual = memory.extract(0xFFC320, '>i2')
    receiver, blocker = [slot for slot in range(5) if slot != actual][:2]
    for slot, point in ((receiver, (60, 220)), (blocker, (10, 206)),
                        (6, (70, 245)), (11, (0, 220))):
        _place_object(memory, slot, point)
    memory.assign(0xFFB04A + blocker * 0x80 + 0x63, '|u1', 4)
    memory.assign(0xFFB04A + actual * 0x80 + 0x54, '>u2', 1)
    _place_object(memory, 14, (0, 190))
    _rebuild_object_order(memory)


def one_timer_setup_cut():
    name = 'ClassicOneTimerCutProbe'
    register_task(name, replace(get_task('DefenseZone'), initialize=_one_timer_cut_setup,
                               reward=lambda _: 0.0, done=lambda _: False))
    try:
        for schema in ('FILTERED', 'HOCKEY_INTENT_DPAD'):
            args = parse_cmdline([
                '--mode=model_vs_game', '--nn=ClassicAIV1', '--env=NHL94-Genesis-v0',
                '--state=PenguinsVsSenators.DefenseZone', f'--rf={name}', f'--action_type={schema}',
            ])
            env = build_single_nhl94_env(args, {'clip_reward': False}, use_frame_skip=False)
            try:
                env.reset(seed=7)
                state = get_game_state(env)
                before = state.team1.one_timer_attempts
                passer = state.engine.puck_owner
                memory = env.unwrapped.data.memory
                base = 0xFFB04A + passer * 0x80
                initial_pose = tuple(memory.extract(base + offset, '>i4') for offset in (0, 0x14, 0x54))
                model = ClassicAIV1Model(args)
                requested = None
                for frame in range(100):
                    state = get_game_state(env)
                    action = model.predict_frame(state)[0]
                    if frame == 0:
                        assert model._last_decision == 'one-timer-setup', model.offense_diagnostics
                    if model._last_decision == 'one-timer-setup':
                        assert model._last_target == model.offense.feint_target
                    if model._last_decision == 'one-timer-pass':
                        requested = model.one_timer.pending.receiver
                        pose = tuple(memory.extract(base + offset, '>i4') for offset in (0, 0x14, 0x54))
                        assert pose != initial_pose, 'Setup must change native position or facing'
                    if state.team1.one_timer_attempts > before:
                        assert requested is not None and state.engine.shot_player == requested
                        assert model.one_timer.pending is None
                        assert model.one_timer.metrics.get('shot-released', 0) == 1
                        print(f'PASS: {schema} executes a coherent setup cut, selects receiver {requested}, '
                              f'and completes on native one-timer release in {frame} frames')
                        break
                    env.step(action)
                else:
                    raise AssertionError((schema, 'No native shot after setup cut', model.offense_diagnostics))
            finally:
                env.close()
    finally:
        TASKS.pop(name)


if __name__ == '__main__':
    advancement_pass()
    advancement_pass(receiver_energy=1024)
    purposeful_feint()
    one_timer_setup_cut()
