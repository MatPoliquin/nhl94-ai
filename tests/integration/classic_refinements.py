"""Native regression checks for Classic release timing and shot execution."""
from types import SimpleNamespace

import numpy as np

from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.agents.motion import VELOCITY_SCALE
from nhl94_ai.agents.passing import pass_launch_view, selected_receiver
from nhl94_ai.env.actions import HockeyActionController
from nhl94_ai.env.factory import make_retro
from nhl94_ai.game.constants import GameConsts as Buttons
from nhl94_ai.game.ram import pass_geometry_info
from nhl94_ai.game.state import NHL94GameState
from nhl94_ai.tasks.defense_setup import _place_object, _rebuild_object_order
from tests.integration.classic_offense import _offense_setup


def _read(env, state):
    env.data.update_ram()
    state.BeginFrame(pass_geometry_info(env, env.data.lookup_all()), [0] * 6)
    return state


def recipient_timing():
    env = make_retro(game='NHL94-Genesis-v0', state='PenguinsVsSenators.DefenseZone', num_players=1)
    try:
        env.reset(seed=7)
        _offense_setup(env, 'NHL94-Genesis-v0')
        snapshot = env.em.get_state()
        for points, motions, expected in (
                (((70, -38), (32, -26)), ((-1.477020263671875, -0.3789825439453125),
                                        (1.2803955078125, -0.597137451171875)), 0),
                (((32, -32), (51, -45)), ((-0.4293060302734375, -0.830596923828125),
                                        (-1.34368896484375, -0.850830078125)), 2)):
            env.em.set_state(snapshot)
            memory = env.data.memory
            assert memory.extract(0xFFC320, '>i2') == 1
            positions = {1: (0, -100), 0: points[0], 2: points[1], 3: (-110, -180), 4: (110, -210)}
            positions.update({slot: (-110 + (slot - 6) * 45, 170) for slot in range(6, 11)})
            for slot, point in positions.items():
                _place_object(memory, slot, point)
            for slot, motion in zip((0, 2), motions):
                for offset, component in zip((0x28, 0x2A), motion):
                    memory.assign(0xFFB04A + slot * 0x80 + offset, '>i2', round(component / VELOCITY_SCALE))
            _place_object(memory, 14, (0, -90))
            _rebuild_object_order(memory)
            state = _read(env, NHL94GameState(5))
            before = state.team1.pass_attempts
            old = selected_receiver(state.team1, state.puck, 1, 1)
            team, puck = pass_launch_view(state, state.team1.players[1])
            predicted = selected_receiver(team, puck, 1, 1)
            assert old != expected and predicted == expected, (old, predicted, expected)
            action = np.zeros(12, dtype=np.int8)
            action[Buttons.INPUT_B] = action[Buttons.INPUT_UP] = action[Buttons.INPUT_RIGHT] = 1
            env.step(action)
            _read(env, state)
            assert state.team1.pass_attempts == before
            assert (state.puck.x, state.puck.y) == (puck.x, puck.y)
            env.step(action)
            _read(env, state)
            assert state.team1.pass_attempts == before + 1
            assert state.engine.pass_target == predicted, (predicted, state.engine.pass_target)
            print(f'PASS: release-time recipient {predicted} matches native selection; current-position estimate was {old}')
    finally:
        env.close()


def aimed_release():
    for schema in ('FILTERED', 'HOCKEY_INTENT_DPAD'):
        env = make_retro(game='NHL94-Genesis-v0', state='PenguinsVsSenators.DefenseZone', num_players=1)
        try:
            env.reset(seed=7)
            _offense_setup(env, 'NHL94-Genesis-v0')
            memory = env.data.memory
            owner = memory.extract(0xFFC320, '>i2')
            for slot in (*range(5), *range(6, 11)):
                _place_object(memory, slot, ((slot % 5 - 2) * 40, -150 if slot < 5 else -70))
            _place_object(memory, owner, (0, 225))
            _place_object(memory, 11, (30, 250))
            _place_object(memory, 14, (0, 230))
            memory.assign(0xFFB04A + owner * 0x80 + 0x54, '>u4', 0)
            memory.assign(0xFFB04A + owner * 0x80 + 0x6C, '|u1', 20)
            _rebuild_object_order(memory)
            state = _read(env, NHL94GameState(5))
            model = ClassicAIV1Model(SimpleNamespace(
                action_type=schema, one_timers=False, classic_refinements=['finishing']))
            processor = HockeyActionController(SimpleNamespace(action_type=schema, game_state=state))
            macro = processor._new_action_state()
            shots = state.team1.stats.shots
            held, release = 0, {}
            for frame in range(48):
                _read(env, state)
                if state.team1.stats.shots > shots:
                    assert state.engine.shot_player == owner
                    assert held == release['hold_frames'], (schema, held, release)
                    assert memory.extract(0xFFBEDC, '>i2') == (2 if release['side'] > 0 else 6)
                    print(f'PASS: {schema} emits {held}-frame C hold and native aimed shot by frame {frame}')
                    break
                action = model.predict_frame(state)[0]
                if frame == 0:
                    assert model._last_decision == 'shoot', model._last_decision
                    release = model.offense_diagnostics['normal_finish']
                buttons = processor._process_action(action, macro)[0]
                held += bool(buttons[Buttons.INPUT_C])
                env.step(buttons)
            else:
                raise AssertionError((schema, 'No native shot release'))
        finally:
            env.close()


if __name__ == '__main__':
    recipient_timing()
    aimed_release()
