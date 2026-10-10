"""Every admitted receiver executes through both action encodings and cadences."""
from types import SimpleNamespace

from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.env.actions import HockeyActionController
from nhl94_ai.env.factory import make_retro
from nhl94_ai.game.state import NHL94GameState
from nhl94_ai.tasks.defense_setup import _place_object, _rebuild_object_order
from tests.integration.classic_offense import _offense_setup
from tests.integration.classic_refinements import _read


def native_receiver(schema, interval, recipient, *, point=(70, -50), expect_reception=True, stick_control=False):
    env = make_retro(game='NHL94-Genesis-v0', state='PenguinsVsSenators.DefenseZone', num_players=1)
    try:
        env.reset(seed=7)
        _offense_setup(env, 'NHL94-Genesis-v0')
        memory = env.data.memory
        owner = memory.extract(0xFFC320, '>i2')
        others = [slot for slot in range(5) if slot != owner]
        for slot, position in zip((owner, *others), ((0, -130), (-80, -30), point, (110, -240), (70, -220))):
            _place_object(memory, slot, position)
        for slot, position in zip(range(6, 11), ((0, 70), (110, 80), (110, 140), (90, 180), (100, 220))):
            _place_object(memory, slot, position)
        _place_object(memory, 14, (0, -120))
        _rebuild_object_order(memory)
        state = _read(env, NHL94GameState(5))
        model = ClassicAIV1Model(SimpleNamespace(action_type=schema, receiver_selection='legacy'))
        model.offense.pass_action.stick_control = stick_control
        model.receiver_selector.forced_slot = desired = others[recipient]
        processor = HockeyActionController(SimpleNamespace(action_type=schema, game_state=state))
        macro = processor._new_action_state()
        passes, launched = state.team1.pass_attempts, False
        for frame in range(90):
            _read(env, state)
            if state.team1.pass_attempts > passes:
                if not launched:
                    assert state.engine.pass_target == desired, (schema, interval, desired, state.engine.pass_target)
                    launched = True
                if state.engine.puck_owner == desired:
                    assert expect_reception, 'The documented miss has changed; update its characterization'
                    assert state.team1.defense_control == desired
                    break
            action = model.predict_frame(state, interval)[0]
            if model._last_decision == 'receive-pass':
                assert state.team1.defense_control == desired
                assert not model.buttons.b_down and not model.buttons.c_down
            if frame == 0:
                assert model._last_decision == 'advance-pass', model.offense_diagnostics
                assert model._last_pass_request['receiver'] == desired
                assert len(model.receiver_selector.diagnostics['candidates']) == 2
                model.receiver_selector = None
            env.step(processor._process_action(action, macro)[0])
        else:
            assert launched and not expect_reception, (schema, interval, recipient, 'No native reception')
            assert state.team2.owns_scnum(state.engine.puck_owner)
        outcome = 'reception' if expect_reception else 'admitted pass miss (model limitation)'
        print(f'PASS: receiver {desired}, {schema}, interval {interval}, {outcome} in {frame} frames')
    finally:
        env.close()


if __name__ == '__main__':
    for encoding in ('FILTERED', 'HOCKEY_INTENT_DPAD'):
        for cadence in (1, 4, 8):
            for option in (0, 1):
                native_receiver(encoding, cadence, option)
    native_receiver('FILTERED', 4, 1, point=(60, -70), expect_reception=False)
