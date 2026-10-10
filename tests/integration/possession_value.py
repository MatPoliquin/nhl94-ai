"""Native shared-value choices through both public action encodings and cadences."""
from types import SimpleNamespace

from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.agents.motion import VELOCITY_SCALE
from nhl94_ai.agents.possession_value import VALUE_MODEL
from nhl94_ai.env.actions import HockeyActionController
from nhl94_ai.env.factory import make_retro
from nhl94_ai.game.constants import GameConsts as Buttons
from nhl94_ai.game.state import NHL94GameState
from nhl94_ai.tasks.defense_setup import _place_object, _rebuild_object_order
from tests.integration.classic_offense import _offense_setup
from tests.integration.classic_refinements import _read


def native_choice(scenario, schema, interval, refinements=()):
    env = make_retro(game='NHL94-Genesis-v0', state='PenguinsVsSenators.DefenseZone', num_players=1)
    try:
        env.reset(seed=7)
        _offense_setup(env, 'NHL94-Genesis-v0')
        memory = env.data.memory
        owner = memory.extract(0xFFC320, '>i2')
        others = [slot for slot in range(5) if slot != owner]
        for slot in (*range(5), *range(6, 11)):
            _place_object(memory, slot, ((slot % 5 - 2) * 40, -180))
        if scenario == 'shoot':
            _place_object(memory, owner, (0, 225))
            _place_object(memory, 14, (0, 230))
            memory.assign(0xFFB04A + owner * 0x80 + 0x2A, '>i2', round(1 / VELOCITY_SCALE))
        else:
            for slot, point in zip((owner, *others[:2]), ((-80, 180), (70, 150), (-30, 220))):
                _place_object(memory, slot, point)
            _place_object(memory, 14, (-80, 190))
        _place_object(memory, 11, (40, 250))
        _rebuild_object_order(memory)
        state = _read(env, NHL94GameState(5))
        model = ClassicAIV1Model(SimpleNamespace(
            action_type=schema, possession_value=True, classic_refinements=refinements))
        processor = HockeyActionController(SimpleNamespace(action_type=schema, game_state=state))
        macro = processor._new_action_state()
        shots, passes = state.team1.stats.shots, state.team1.pass_attempts
        desired, held, observed_pass, expected_hold = None, 0, False, interval
        for frame in range(90):
            _read(env, state)
            if scenario == 'shoot' and state.team1.stats.shots > shots:
                assert state.engine.shot_player == owner
                assert held == expected_hold, (schema, interval, held, expected_hold)
                break
            if scenario == 'position-pass' and state.team1.pass_attempts > passes:
                if not observed_pass:
                    assert state.engine.pass_target == desired, (schema, interval, frame, desired, state.engine.pass_target)
                    observed_pass = True
                if state.engine.puck_owner == desired:
                    assert state.team1.defense_control == desired
                    break
            action = model.predict_frame(state, interval)[0]
            if frame == 0:
                assert model._last_decision == scenario, (scenario, schema, interval, model.offense_diagnostics)
                assert model.offense_diagnostics['value_model'] == VALUE_MODEL
                desired = model.offense_diagnostics['desired_slot']
                expected_hold = model.offense_diagnostics.get('normal_finish', {}).get('hold_frames', interval)
            buttons = processor._process_action(action, macro)[0]
            held += bool(buttons[Buttons.INPUT_C])
            env.step(buttons)
        else:
            raise AssertionError((scenario, schema, interval, 'No native outcome', model.offense_diagnostics))
        print(f'PASS: possession value {scenario}, {schema}, interval {interval}, '
              f'refinements {list(refinements)}, native outcome in {frame} frames')
    finally:
        env.close()


if __name__ == '__main__':
    for encoding in ('FILTERED', 'HOCKEY_INTENT_DPAD'):
        for cadence in (1, 4, 8):
            for kind in ('shoot', 'position-pass'):
                native_choice(kind, encoding, cadence)
        native_choice('shoot', encoding, 4, ('finishing',))
