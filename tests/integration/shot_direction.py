"""Native nine-way aim at both attacking ends and with both handedness values."""
from copy import deepcopy
from types import SimpleNamespace

from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.env.factory import make_retro
from nhl94_ai.evaluation.cpu_benchmark import CPU_RAM
from nhl94_ai.evaluation.cross_crease_probe import snapshot_start
from nhl94_ai.evaluation.refinement_replay import advance, read_view
from nhl94_ai.evaluation.shot_direction import PADS, placed
from nhl94_ai.game.constants import GameConsts as Buttons
from nhl94_ai.game.state import NHL94GameState
from nhl94_ai.tasks.cross_crease_setup import set_crossing_velocity
from nhl94_ai.tasks.defense_setup import _place_object, _rebuild_object_order


def run():
    env = make_retro(game='NHL94-Genesis-v0', state='PenguinsVsSenators.DefenseZone', num_players=1)
    try:
        for name, (address, kind) in CPU_RAM.items():
            env.data.set_variable(name, {'address': address, 'type': kind})
        for side in (1, 2):
            for hand in (0, 1):
                _, (owner, goalie, _) = snapshot_start(
                    env, scenario='near', seed=27000, side=side, direction=1, handedness=hand)
                set_crossing_velocity(env, owner, direction=1, lateral=0, goalward=0, width=35, depth=225)
                _place_object(env.data.memory, goalie, (70, 250 if side == 1 else -250))
                _rebuild_object_order(env.data.memory)
                snapshot = env.em.get_state()
                state = NHL94GameState(5)
                view, _ = read_view(env, state, side)
                model = ClassicAIV1Model(SimpleNamespace(action_type='FILTERED', one_timers=False))
                model.predict_frame(view, 4)
                assert model._last_decision == 'shoot', (side, hand, model._last_decision)
                releases = {}
                for pad in PADS:
                    env.em.set_state(snapshot)
                    env.data.update_ram()
                    branch, decoded = placed(model, pad), deepcopy(state)
                    held = []
                    for frame in range(48):
                        view, _ = read_view(env, decoded, side)
                        action = branch.scheduler.action[0] if frame == 0 else branch.predict_frame(view, 4)[0]
                        held.append(bool(action[Buttons.INPUT_C]))
                        advance(env, action, side)
                        view, _ = read_view(env, decoded, side)
                        if view.engine.puck_owner < 0 and view.engine.shot_player == owner:
                            native = env.data.memory.extract(0xFFBEDC, '>i2')
                            expected = {(0, 1): 0, (1, 1): 1, (1, 0): 2, (1, -1): 3,
                                        (0, -1): 4, (-1, -1): 5, (-1, 0): 6, (-1, 1): 7, (0, 0): 8}[pad]
                            assert native == expected, (side, hand, pad, native)
                            assert held[:4] == [True]*4 and not any(held[4:]), held
                            releases[pad] = view.puck.motion_z
                            print(f'PASS: end {side}, hand {hand}, pad {pad}, native {native}, vz {view.puck.motion_z:.4f}')
                            break
                    else:
                        raise AssertionError((side, hand, pad, 'No native release'))
                for x in (-1, 0, 1):
                    assert releases[x, 1] > releases[x, 0] > releases[x, -1], (side, hand, x, releases)
    finally:
        env.close()


if __name__ == '__main__':
    run()
