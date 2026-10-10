"""Explicit ordinary-ROM calibration of deke-only carrier movement."""
from copy import copy

import numpy as np

from nhl94_ai.agents.skating import grounded_step
from nhl94_ai.env.factory import make_retro
from nhl94_ai.evaluation.benchmark import RAM, away_view, update_state
from nhl94_ai.game.constants import GameConsts as Buttons
from nhl94_ai.game.state import NHL94GameState
from tests.integration.deke import _setup


def native_motion():
    env = make_retro(game='NHL94-Genesis-v0', state='PenguinsVsSenators.DefenseZone', num_players=1)
    try:
        for name, (address, kind) in RAM.items():
            env.data.set_variable(name, {'address': address, 'type': kind})
        maximum = [0.0, 0.0, 0.0]
        for side in (1, 2):
            for pad in ((1, 0), (-1, 0), (1, 1), (-1, -1), (0, 0), (0, -1), (0, 1)):
                env.reset(seed=71000)
                _setup(env, side=side, direction=1, scenario='clear', seed=71000,
                       handedness=0, jitter=0)
                *_, info = env.step(np.zeros(12, dtype=np.int8))
                state = NHL94GameState(5)
                update_state(state, info, env)
                view = state if side == 1 else away_view(state)
                slot = view.engine.puck_owner
                forecast = copy(view.team1.get_player_by_scnum(slot))
                action = np.zeros(12, dtype=np.int8)
                if pad[0]:
                    action[Buttons.INPUT_RIGHT if pad[0] > 0 else Buttons.INPUT_LEFT] = 1
                if pad[1]:
                    action[Buttons.INPUT_UP if pad[1] > 0 else Buttons.INPUT_DOWN] = 1
                for frame in range(1, 25):
                    forecast = grounded_step(forecast, pad)
                    *_, info = env.step(action)
                    update_state(state, info, env)
                    view = state if side == 1 else away_view(state)
                    actual = view.team1.get_player_by_scnum(slot)
                    assert view.engine.puck_owner == slot, (side, pad, frame, 'carrier lost puck')
                    if frame not in (1, 4, 8, 12, 24):
                        continue
                    errors = (
                        max(abs(forecast.precise_x - actual.precise_x), abs(forecast.precise_y - actual.precise_y)),
                        max(abs(forecast.motion_x - actual.motion_x), abs(forecast.motion_y - actual.motion_y)),
                        min(abs(forecast.facing_phase - actual.facing_phase),
                            8 - abs(forecast.facing_phase - actual.facing_phase)),
                    )
                    maximum = [max(previous, error) for previous, error in zip(maximum, errors)]
                    assert errors[0] <= 1 / 65536, (side, pad, frame, errors, vars(actual), vars(forecast))
                    assert errors[1] <= 17 / 65536, (side, pad, frame, errors)
                    assert errors[2] <= 1 / 65536, (side, pad, frame, errors)
        print(f'PASS: 14 native carrier traces at 1/4/8/12/24 frames; max position/velocity/phase errors {maximum}')
        return maximum
    finally:
        env.close()


if __name__ == '__main__':
    native_motion()
