"""Native aim/hold execution through buttons and intents, including long holds."""
from types import SimpleNamespace

from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.env.actions import HockeyActionController
from nhl94_ai.env.factory import make_retro
from nhl94_ai.game.constants import GameConsts as Buttons
from nhl94_ai.game.state import NHL94GameState
from nhl94_ai.evaluation.shot_placement_replay import placed
from nhl94_ai.tasks.defense_setup import _place_object, _rebuild_object_order
from tests.integration.classic_offense import _offense_setup
from tests.integration.classic_refinements import _read


def run(schema, interval, side, hold, *, replay=False):
    env = make_retro(game='NHL94-Genesis-v0', state='PenguinsVsSenators.DefenseZone', num_players=1)
    try:
        env.reset(seed=7)
        _offense_setup(env, 'NHL94-Genesis-v0')
        memory = env.data.memory
        owner = memory.extract(0xFFC320, '>i2')
        for slot in (*range(5), *range(6, 11)):
            _place_object(memory, slot, (100, -180))
        _place_object(memory, owner, (0, 225))
        _place_object(memory, 14, (0, 230))
        _place_object(memory, 11, (40, 250))
        _rebuild_object_order(memory)
        state = _read(env, NHL94GameState(5))
        model = ClassicAIV1Model(SimpleNamespace(action_type=schema))
        if not replay:
            model.shot_placement = SimpleNamespace(choose=lambda *_: (side, hold), diagnostics={})
        processor = HockeyActionController(SimpleNamespace(action_type=schema, game_state=state))
        macro = processor._new_action_state()
        held, released = [], False
        for frame in range(hold + 30):
            _read(env, state)
            action = model.predict_frame(state, interval)[0]
            if replay and frame == 0:
                assert schema == 'FILTERED'
                model = placed(model, side, hold)
                action = model.scheduler.action[0]
            buttons = processor._process_action(action, macro)[0]
            if frame < hold + 2:
                held.append(bool(buttons[Buttons.INPUT_C]))
            if frame == 0:
                assert model._last_decision == 'shoot', model.offense_diagnostics
                assert bool(buttons[Buttons.INPUT_LEFT]) == (side == -1)
                assert bool(buttons[Buttons.INPUT_RIGHT]) == (side == 1)
            env.step(buttons)
            _read(env, state)
            released |= state.engine.puck_owner < 0 and state.engine.shot_player == owner
        assert held == [True]*hold + [False]*2, (schema, interval, side, hold, held)
        assert released, (schema, interval, side, hold, 'No native shot release')
        print(f'PASS: {schema}, interval {interval}, aim {side}, hold {hold}, replay {replay}, native shot released')
    finally:
        env.close()


if __name__ == '__main__':
    for encoding in ('FILTERED', 'HOCKEY_INTENT_DPAD'):
        for aim in (-1, 0, 1):
            for duration in (1, 4, 8, 12):
                run(encoding, 4, aim, duration)
        for cadence in (1, 8):
            run(encoding, cadence, 0, 12)
    for aim in (-1, 0, 1):
        for duration in (1, 4, 8, 12):
            run('FILTERED', 4, aim, duration, replay=True)
