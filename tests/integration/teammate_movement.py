"""Native no-burst support forecast and faceoff ownership calibration."""
import json
from pathlib import Path
from copy import copy

import numpy as np

from nhl94_ai.agents.teammates import TeammateCoordinator
from nhl94_ai.agents.teammate_faceoff import faceoff_center
from nhl94_ai.agents.teammate_offpuck import movement_action
from nhl94_ai.agents.teammate_support import support_forecast
from nhl94_ai.evaluation.teammates import DEFAULT_STATE, make_match_env, prepare_match
from nhl94_ai.game.ram import teammate_controller_info
from nhl94_ai.game.state import NHL94GameState
from nhl94_ai.game.skating import grounded_step
from tests.support.teammates import team_args


def faceoffs():
    results = []
    for side in (1, 2):
        for size in (2, 3):
            for controller in range(size):
                with make_match_env(DEFAULT_STATE, team_size=size) as env:
                    info = prepare_match(env, seed=94, seconds=120, side=side, team_size=size)
                    # Reset fixture: permute existing selected slots, without changing any players.
                    slots = [info[f'defense_control{i+1}'] for i in range(size)]
                    center = (side - 1)*6 + 2
                    old = slots.index(center)
                    slots[old], slots[controller] = slots[controller], slots[old]
                    for index, slot in enumerate(slots):
                        env.data.memory.assign(0xFFC320 + 2*index, '>i2', slot)
                    env.data.update_ram()
                    info = env.data.lookup_all()
                    agent = TeammateCoordinator(team_args(side='home' if side == 1 else 'away', team_size=size))
                    edges, previous = [0]*size, np.zeros(12*size)
                    animation = set()
                    for frame in range(240):
                        info = agent.controllers[0].platform.frame_info(env, info)
                        result = agent.act(info)
                        if agent.states[0].engine.clock_stopped:
                            for index in range(size):
                                edges[index] += bool(result.action[index*12] and not previous[index*12])
                                if index != controller:
                                    assert not result.action[index*12:(index+1)*12].any()
                            participant = agent.states[0].team1.get_player_by_scnum(center)
                            animation.add(participant.live_anim)
                        else:
                            break
                        previous = result.action.copy()
                        *_, info = env.step(result.action)
                    assert edges[controller] > 0 and sum(edges) == edges[controller], (side, size, controller, edges)
                    assert not agent.states[0].engine.clock_stopped, 'draw did not drop within 240 frames'
                    results.append({'side': side, 'size': size, 'controller': controller+1,
                                    'edges': edges, 'animations': sorted(animation), 'drop_frame': frame})
    return results


def movement():
    results = []
    for side in (1, 2):
        with make_match_env(DEFAULT_STATE) as env:
            info = prepare_match(env, seed=94, seconds=120, side=side)
            agent = TeammateCoordinator(team_args(side='home' if side == 1 else 'away'))
            snapshot = None
            # Reach a clean native off-puck support window; do not reset actor physics.
            for _ in range(2000):
                info = agent.controllers[0].platform.frame_info(env, info)
                result = agent.act(info)
                state = agent.states[1]
                player = state.team1.get_player_by_scnum(state.team1.controlled_scnum())
                if (not state.engine.clock_stopped and player is not None and player.role > 0  # pylint: disable=too-many-boolean-expressions
                        and state.engine.puck_owner != state.team1.controlled_scnum()
                        and not player.is_falling and not player.unavailable & 6
                        and abs(player.x) < 85 and abs(player.y) < 220
                        and all((player.x-p.x)**2+(player.y-p.y)**2 > 45**2
                                for t in (state.team1, state.team2) for p in [*t.players, t.goalie] if p is not player)):
                    snapshot = env.em.get_state()
                    break
                *_, info = env.step(result.action)
            assert snapshot is not None, 'no clean movement window'
            for coast in (False, True):
                for dx, dy in ((50, 40), (-50, -40), (0, 0)):
                    assert env.em.set_state(snapshot)
                    env.data.update_ram()
                    info = env.data.lookup_all()
                    state = NHL94GameState(5)
                    if side == 2:
                        state.Flip()
                    state.BeginFrame(teammate_controller_info(info, 2, side), [0]*6)
                    slot = state.team1.controlled_scnum()
                    player = state.team1.get_player_by_scnum(slot)
                    target = (player.x+dx, player.y+dy)
                    predicted = support_forecast(state, slot, target, coast=coast).team1.get_player_by_scnum(slot)
                    observed_updates = []
                    for _ in range(12):
                        before = copy(player)
                        action = np.zeros(24, dtype=np.int8)
                        action[12:], _ = movement_action(player, target, coast=coast)
                        assert not action[20]
                        *_, info = env.step(action)
                        state.BeginFrame(teammate_controller_info(info, 2, side), TeammateCoordinator._game_action(action[12:]))
                        player = state.team1.get_player_by_scnum(slot)
                        if coast:
                            alternatives = [before]
                            for _update in range(3):
                                alternatives.append(grounded_step(alternatives[-1], (0, 0)))
                            errors = [max(abs(p.precise_x-player.precise_x), abs(p.precise_y-player.precise_y))
                                      for p in alternatives]
                            observed_updates.append(errors.index(min(errors)) if min(errors) < 1/65536 else None)
                    error = max(abs(predicted.precise_x-player.precise_x), abs(predicted.precise_y-player.precise_y))
                    # Retain failures against the predeclared one-unit threshold.
                    # Native video frames can contain zero or two grounded updates.
                    results.append({'side': side, 'coast': coast, 'target': target,
                                    'position_error': error, 'within_one_unit': error < 1,
                                    'observed_coast_updates_per_frame': observed_updates})
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--require-forecast', action='store_true', help='Fail if any window exceeds the declared one-unit gate')
    args = parser.parse_args()
    report = {'faceoffs': faceoffs(), 'movement': movement()}
    report['forecast_gate_passed'] = all(row['within_one_unit'] for row in report['movement'])
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print('PASS', len(report['faceoffs']), 'faceoff layouts;', len(report['movement']),
          '12-frame movement windows; forecast gate:', report['forecast_gate_passed'])
    if args.require_forecast:
        assert report['forecast_gate_passed'], 'Native movement remains an uncertified approximation'
