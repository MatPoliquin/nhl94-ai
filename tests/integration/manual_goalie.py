"""Native B handoffs, goalie movement, C saves, A dives, catches and outlets.

Use the supplied manual-goalie save. Isolated puck/contact fixtures change only
this emulator instance after B has selected the goalie; production never forces
goalie selection or changes settings.
"""
import json
import os
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from nhl94_ai.agents.goalie import DIVE_ANIMATION, SAVE_ANIMATIONS, GoalieController
from nhl94_ai.agents.registry import create_scripted
from nhl94_ai.cli import main
from nhl94_ai.config import load_hyperparams, resolve_hyperparams_for_model
from nhl94_ai.env.factory import make_retro
from nhl94_ai.evaluation.benchmark import away_view
from nhl94_ai.evaluation.play import NHL94Player
from nhl94_ai.game.constants import GameConsts as Buttons
from nhl94_ai.game.ram import pass_geometry_info, restore_away_control, select_cpu_side
from nhl94_ai.game.state import NHL94GameState
from nhl94_ai.tasks.defense_setup import _facing, _place_object, _rebuild_object_order, _reset_player


STATE = 'SabresVsMightyDucks.ManualGoalie.Start'


def feedback(env, side=1):
    env.data.update_ram()
    info = pass_geometry_info(env, env.data.lookup_all())
    state = NHL94GameState(5)
    state.BeginFrame(info, [0] * 6)
    return (away_view(state) if side == 2 else state), info


def advance(env, action, side=1):
    env.step(action)
    if side == 2:
        restore_away_control(env.data, env.data.lookup_all())
    return feedback(env, side)


def restore(env, snapshot):
    env.em.set_state(snapshot)
    return feedback(env)


def takeover(env, side=1):
    env.reset(seed=7)
    env.data.update_ram()
    select_cpu_side(env.data, env.data.lookup_all(), side)
    state, info = feedback(env, side)
    assert state.engine.goalie_modes == (0, 0)
    assert state.engine.controller_teams == (side, 0)
    agent = create_scripted('classic-v1', SimpleNamespace(action_type='FILTERED', goalie_policy='always'))
    agent.frame_skip = 4
    for frame in range(5000):
        action = agent.predict_game_state(state)[0]
        state, info = advance(env, action, side)
        goalie = state.team1.goalie
        ready = (state.team1.defense_control == state.team1.defense_goalie
                 and not goalie.unavailable & 2 and not goalie.selection_flags & 0x20
                 and not goalie.live_state_flags & 4)
        if (agent.goalie.metrics['takeovers'] and agent.goalie.phase == 'positioning'
                and ready and state.engine.puck_owner != state.team1.defense_goalie):
            assert goalie.selection_flags & 8
            assert agent.goalie.metrics['handoff_timeouts'] == 0
            print(f'PASS: {"home" if side == 1 else "away"} B takeover confirmed at frame {frame}; '
                  f'{dict(agent.goalie.metrics)}')
            return env.em.get_state()
    raise AssertionError(('No native manual goalie takeover', side, agent.goalie_diagnostics, info))


def shot_fixture(env, *, x=8, y=-205, goalie_x=0, raw_vy=-14000):
    memory = env.data.memory
    _place_object(memory, 5, (goalie_x, -246))
    for slot in range(12):
        if slot not in (5, 11):
            _place_object(memory, slot, (-100 + (slot % 6) * 40, 40 if slot < 6 else 110))
    _place_object(memory, 14, (x, y))
    memory.assign(0xFFB74A + 0x2A, '>i2', raw_vy)
    memory.assign(0xFFB7AA, '>i2', -256)
    memory.assign(0xFFBEE0, '>i2', -1)
    memory.assign(0xFFBED8, '>i2', -1)
    memory.assign(0xFFC2EC, '>u2', memory.extract(0xFFC2EC, '>u2') & ~0x0C00)
    memory.assign(0xFFC2EE, '>u2', memory.extract(0xFFC2EE, '>u2') & ~0x1000)
    _rebuild_object_order(memory)
    return feedback(env)


def movement_and_buttons(env, snapshot):
    restore(env, snapshot)
    before, _ = feedback(env)
    action = np.zeros(12, dtype=np.int8)
    action[Buttons.INPUT_RIGHT] = 1
    for _ in range(12):
        state, _ = advance(env, action)
    assert state.team1.defense_control == 5
    assert state.team1.goalie.x > before.team1.goalie.x
    assert state.team1.goalie.motion_x > 0
    neutral = np.zeros(12, dtype=np.int8)
    for _ in range(8):
        state, _ = advance(env, neutral)
    assert abs(state.team1.goalie.motion_x) < 0.05
    print('PASS: D-pad moves the selected goalie; neutral uses strong goalie braking')

    restore(env, snapshot)
    state, _ = shot_fixture(env, x=24, y=-217, goalie_x=-2, raw_vy=-20000)
    manager = GoalieController('selective')
    animations = []
    for _ in range(12):
        state, _ = advance(env, manager.step(state))
        animations.append(state.team1.goalie.live_anim)
    assert DIVE_ANIMATION in animations, (animations, manager.diagnostics)
    assert manager.metrics['dive_animations'] == 1
    assert manager.metrics['save_requests'] == 0
    print('PASS: exclusive A + direction requests a ROM-confirmed dive, not a skater boost')


def save_and_outlet(env, snapshot):
    restore(env, snapshot)
    state, initial = shot_fixture(env)
    manager = GoalieController('selective')
    seen, contact = [], False
    catch_snapshot = None
    for _ in range(70):
        state, info = advance(env, manager.step(state))
        seen.append(state.team1.goalie.live_anim)
        contact |= state.engine.puck_owner == 5 or state.puck.motion_y > 0
        if state.engine.puck_owner == 5:
            catch_snapshot = env.em.get_state()
            break
    assert any(animation in SAVE_ANIMATIONS for animation in seen), (seen, manager.diagnostics)
    assert manager.metrics['save_animations'] == 1
    assert contact and info['p2_score'] == initial['p2_score'], (seen, info['p2_score'])
    print('PASS: C starts a ROM save animation; the incoming puck contacts the goalie without a goal')

    if catch_snapshot is None:
        restore(env, snapshot)
        state, _ = shot_fixture(env, x=0, y=-223, raw_vy=-2000)
        manager = GoalieController('selective')
        for _ in range(80):
            state, _ = advance(env, manager.step(state))
            if state.engine.puck_owner == 5:
                catch_snapshot = env.em.get_state()
                break
    assert catch_snapshot is not None, ('No controlled low-speed catch', manager.diagnostics)
    print('PASS: live puck ownership confirms a controlled goalie catch')

    memory = env.data.memory
    for slot, point in enumerate(((0, -155), (100, 120), (-100, 120), (100, 220), (-100, 220))):
        role = memory.extract(0xFFB04A + slot * 0x80 + 0x34, '>i2')
        assignment = 1 if role in (1, 2) else 6 if role == 4 else 4
        _reset_player(memory, slot, point, _facing(point, (state.team1.goalie.x, state.team1.goalie.y)),
                      assignment, False, (0x11,) if slot == 0 else ())
    _rebuild_object_order(memory)
    state, _ = feedback(env)
    for _ in range(220):
        action = manager.step(state)
        assert action is not None, (manager.diagnostics, state.engine.puck_owner)
        state, _ = advance(env, action)
        if manager.metrics['outlet_receptions']:
            break
    assert manager.metrics['outlets_launched'] == 1, manager.diagnostics
    assert manager.metrics['outlet_receptions'] == 1, manager.diagnostics
    assert state.team1.defense_control in range(5)
    assert state.engine.puck_owner == state.team1.defense_control
    print(f'PASS: safe B outlet has a fresh pass counter and actual skater reception: {dict(manager.metrics)}')
    return catch_snapshot


def clear_and_return(env, snapshot, catch_snapshot):
    restore(env, catch_snapshot)
    memory = env.data.memory
    for slot in range(5):
        _place_object(memory, slot, (-100 + slot * 40, 180))
        base = 0xFFB04A + slot * 0x80
        memory.assign(base + 0x63, '|u1', memory.extract(base + 0x63, '|u1') | 4)
    _rebuild_object_order(memory)
    state, _ = feedback(env)
    manager = GoalieController('selective')
    requested = released = False
    for _ in range(220):
        action = manager.step(state)
        assert action is not None, (manager.diagnostics, state.engine.puck_owner)
        requested |= bool(action[Buttons.INPUT_A])
        state, _ = advance(env, action)
        if requested and state.engine.puck_owner != 5:
            released = state.puck.motion_y > 0 or (state.puck.motion_z or 0) > 0
            break
    assert requested and released, manager.diagnostics
    print('PASS: no safe outlet leads to a bounded A clear; A is not treated as freeze')

    restore(env, snapshot)
    state, _ = shot_fixture(env, x=60, y=40, raw_vy=0)
    manager = GoalieController('selective')
    for _ in range(20):
        state, _ = advance(env, manager.step(state))
        if manager.metrics['returns']:
            break
    assert state.team1.defense_control in range(5), manager.diagnostics
    assert manager.metrics['returns'] == 1
    print('PASS: short B press/release returns to a feedback-confirmed skater')


class Finished(Exception):
    """Bound a watched command after its first confirmed takeover."""


def watched(args):
    args.hyperparams_dict = resolve_hyperparams_for_model(load_hyperparams(args.hyperparams), args.nn)
    player = NHL94Player(args, None, need_display=True)
    try:
        player.display_frame_interval = 0
        shown_step = player.display_env.step
        frames = 0

        def capture(actions):
            nonlocal frames
            result = shown_step(actions)
            frames += 1
            model = player.ai_sys.models[1].controller
            assert model.goalie is not None
            assert frames < 6000, ('Watched goalie takeover never completed', args.side, model.goalie_diagnostics)
            if model.goalie.metrics['takeovers'] and model.goalie.phase == 'positioning':
                info = result[3][0]
                assert info['defense_control1'] == (5 if args.side == 'home' else 11)
                assert model.goalie_diagnostics['target'] is not None
                raise Finished
            return result

        with patch.object(player.display_env, 'step', side_effect=capture):
            try:
                player.play(continuous=False, need_reset=False)
            except Finished:
                pass
        print(f'PASS: actual watched --side {args.side} --goalie-policy selective command confirms takeover')
    finally:
        player.close()


def run():
    env = make_retro(game='NHL94-Genesis-v0', state=STATE, num_players=1, goalie_policy='always')
    try:
        home = takeover(env)
        movement_and_buttons(env, home)
        catch = save_and_outlet(env, home)
        clear_and_return(env, home, catch)
        takeover(env, 2)
    finally:
        env.close()
    ordinary = make_retro(game='NHL94-Genesis-v0', num_players=1)
    try:
        ordinary.reset(seed=7)
        state, _ = feedback(ordinary)
        with_mode = state.engine.goalie_modes
        assert any(with_mode), with_mode
        try:
            GoalieController('selective').step(state)
        except ValueError as error:
            assert 'Manual goalie settings' in str(error)
        else:
            raise AssertionError('Automatic save silently accepted manual goalie policy')
        print('PASS: automatic-goalie save fails explicitly rather than silently enabling a policy')
    finally:
        ordinary.close()
    for side in ('home', 'away'):
        with patch.dict(os.environ, {'SDL_VIDEODRIVER': 'dummy', 'SDL_AUDIODRIVER': 'dummy'}), \
                patch('nhl94_ai.evaluation.play.run', side_effect=watched):
            main(['play', '--agent', 'classic-v1', '--env', 'NHL94-Genesis-v0', '--mode', 'model_vs_game',
                  '--state', STATE, '--side', side, '--goalie-policy', 'selective',
                  '--max_playback_speed', '1.0'])
    print(json.dumps({'manual_goalie_checks': 'passed', 'state': STATE}))


if __name__ == '__main__':
    run()
