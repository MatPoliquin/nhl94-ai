"""Fixed native forecast windows, every priced horizon, with native loop clocks."""
import argparse
from copy import copy
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from nhl94_ai.agents.teammates import TeammateCoordinator
from nhl94_ai.agents.teammate_offpuck import movement_action
from nhl94_ai.agents.teammate_support import support_forecast
from nhl94_ai.evaluation.teammates import DEFAULT_STATE, make_match_env, prepare_match, source_hash
from nhl94_ai.game.ram import teammate_controller_info
from nhl94_ai.game.state import NHL94GameState
from tests.support.teammates import team_args
from tests.integration.teammate_movement import movement


def native_clock(env):
    return {key: env.data.memory.extract(address, '>u2') for key, address in
            (('vcount', 0xFFB03E), ('oldvcount', 0xFFB040))}


def state_view(info, side, size, action):
    state = NHL94GameState(5)
    if side == 2:
        state.Flip()
    state.BeginFrame(teammate_controller_info(info, 2, side, team_size=size),
                     TeammateCoordinator._game_action(action[12:24]))
    return state


def calibrate(output):
    output.mkdir(parents=True, exist_ok=False)
    policy_hash = source_hash()
    runner_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    results = []
    for side in (1, 2):
        for size in (2, 3):
            with make_match_env(DEFAULT_STATE, team_size=size) as env:
                info = prepare_match(env, seed=94, seconds=120, side=side, team_size=size)
                agent = TeammateCoordinator(team_args(side='home' if side==1 else 'away', team_size=size))
                snapshots = []
                for frame in range(1, 721):
                    info = agent.controllers[0].platform.frame_info(env, info)
                    if frame in (180, 360, 720):
                        snapshots.append((frame, env.em.get_state(), info.copy()))
                    *_, info = env.step(agent.act(info).action)
                for frame, snapshot, info0 in snapshots:
                    name = f'side{side}-size{size}-frame{frame}'
                    (output/(name+'.state.gz')).write_bytes(gzip.compress(snapshot))
                    for coast in (False, True):
                        for dx, dy in ((50,40),(-50,-40),(0,0)):
                            assert env.em.set_state(snapshot)
                            env.data.update_ram(); info=info0.copy()
                            action=np.zeros(12*size,dtype=np.int8)
                            state=state_view(info,side,size,action)
                            slot=state.team1.controlled_scnum(); player=state.team1.get_player_by_scnum(slot)
                            row={'snapshot':name, 'side':side, 'size':size,'frame':frame,'slot':slot,'coast':coast,
                                 'snapshot_sha256':hashlib.sha256(snapshot).hexdigest(), 'offset':[dx,dy]}
                            if player is None:
                                results.append(row | {'unsupported':'transient-selection','gate_passed':False});continue
                            target=(player.x+dx,player.y+dy)
                            clearance=min(((player.x-p.x)**2+(player.y-p.y)**2)**.5 for team in (state.team1,state.team2)
                                          for p in [*team.players,team.goalie] if p is not player)
                            row.update(initial_animation=[player.live_anim,player.live_anim_frame], initial_velocity=[player.motion_x,player.motion_y],
                                       initial_clearance=clearance, initial_unavailable=player.unavailable)
                            try:
                                forecast=[support_forecast(state,slot,target,coast=coast,frames=h).team1.get_player_by_scnum(slot)
                                          for h in range(1,13)]
                            except RuntimeError as error:
                                results.append(row | {'unsupported':str(error), 'gate_passed':False})
                                continue
                            model=copy(player); horizons=[]; previous_clock=native_clock(env)
                            for h in range(1,13):
                                action[12:24],_=movement_action(player,target,coast=coast)
                                model_action,_=movement_action(model,target,coast=coast)
                                action_equal=bool(np.array_equal(model_action,action[12:24]))
                                *_,info=env.step(action)
                                state=state_view(info,side,size,action); player=state.team1.get_player_by_scnum(slot)
                                clock=native_clock(env)
                                predicted=forecast[h-1]
                                error=max(abs(predicted.precise_x-player.precise_x),abs(predicted.precise_y-player.precise_y))
                                near=min(((player.x-p.x)**2+(player.y-p.y)**2)**.5 for team in (state.team1,state.team2)
                                         for p in [*team.players,team.goalie] if p is not player)
                                horizons.append({'horizon':h,'position_error':error,'action_agreement':action_equal,
                                                 'clock_before':previous_clock,'clock_after':clock,
                                                 'oldvcount_delta':(clock['oldvcount']-previous_clock['oldvcount'])%65536,
                                                 'animation':[player.live_anim,player.live_anim_frame], 'clearance':near,
                                                 'native_position':[player.precise_x,player.precise_y],
                                                 'predicted_position':[predicted.precise_x,predicted.precise_y]})
                                model=predicted; previous_clock=clock
                            row.update(horizons=horizons, gate_passed=all(h['position_error']<=1 and h['action_agreement'] for h in horizons),
                                       collision_exposure=any(h['clearance']<24 for h in horizons))
                            results.append(row)
    historical=movement()
    assert source_hash() == policy_hash, 'Source changed during calibration'
    report={'policy_sha256':policy_hash, 'runner_sha256':runner_hash, 'selection':'seed94; both sides; 2/3 bots; frames180,360,720; three fixed targets; coast/steer; no outcome-based exclusions',
            'timebase':'DoGameFrame samples vcount-oldvcount in D7; updateplayers integrates once with 17*D7 and accelerates with D7. Video observation can interrupt the object loop.',
            'historical_windows':historical,'windows':results,
            'gate_passed':all(r['gate_passed'] for r in results) and all(r['within_one_unit'] for r in historical),
            'supported_pricing_domain':'Uncertified: future native loop boundaries and within-loop actor phase unavailable; no new movement promotion.'}
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(len(results)+len(historical),'fixed windows; forecast gate',report['gate_passed'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    calibrate(parser.parse_args().output)
