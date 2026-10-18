"""Compare independent goalie/free-puck forecasts with native emulator steps.

Only these isolated fixtures write gameplay state. Production planning reads
observations and emits buttons. Contact and wall events end free-flight tests.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from nhl94_ai.agents.goalie_physics import KeeperMotion, PuckFlight, PADS, contact_margin
from nhl94_ai.env.factory import make_retro
from nhl94_ai.game.constants import GameConsts as Buttons
from tests.integration.manual_goalie import STATE, advance, feedback, restore, shot_fixture, takeover


def buttons(pad, save=False):
    result = np.zeros(12, dtype=np.int8)
    if pad[0]:
        result[Buttons.INPUT_RIGHT if pad[0] > 0 else Buttons.INPUT_LEFT] = 1
    if pad[1]:
        result[Buttons.INPUT_UP if pad[1] > 0 else Buttons.INPUT_DOWN] = 1
    result[Buttons.INPUT_C] = save
    return result


def run(output):
    env = make_retro(game='NHL94-Genesis-v0', state=STATE, num_players=1, goalie_policy='always')
    rows = []
    try:
        snapshot = takeover(env)
        # Real goalie ratings, plus controlled motion/energy extremes. Each
        # trajectory retains its own fixed attributes for the forecast horizon.
        for ratings in ((64, 20, 20, 4096), (88, 8, 10, 2048), (48, 28, 29, 3072)):
            for pad in PADS:
                for initial_vx in (-6000, 0, 6000):
                    restore(env, snapshot)
                    shot_fixture(env, x=100, y=-150, raw_vy=1000)
                    memory = env.data.memory
                    base = 0xFFB04A + 5 * 0x80
                    for offset, value in zip((0x67, 0x68, 0x69), ratings[:3]):
                        memory.assign(base + offset, '|u1', value)
                    roster = memory.extract(base + 0x66, '|u1')
                    memory.assign(0xFFC700 + roster * 2, '>u2', ratings[3])
                    memory.assign(0xFFC2FC, '|u1', memory.extract(0xFFC2FC, '|u1') & ~16)
                    memory.assign(base + 0x28, '>i2', initial_vx)
                    # Finish the emulated CPU's in-progress update after fixture
                    # writes; registers may still reference pre-injection RAM.
                    state, _ = advance(env, buttons(pad))
                    projected = KeeperMotion.observe(state.team1.goalie)
                    errors = []
                    for frame in range(1, 49):
                        projected = projected.step(pad)
                        state, _ = advance(env, buttons(pad))
                        actual = KeeperMotion.observe(state.team1.goalie)
                        # Stop before leaving the independently modeled crease.
                        if (abs(actual.x >> 16) > 34 or not 234 <= abs(actual.y >> 16) <= 254
                                or state.team1.goalie.live_state_flags & 4
                                or state.team1.goalie.unavailable & 6
                                or state.team1.defense_control != 5):
                            break
                        errors.append({'frame': frame,
                            'position_error': max(abs(actual.x-projected.x), abs(actual.y-projected.y))/65536,
                            'velocity_error': max(abs(actual.vx-projected.vx), abs(actual.vy-projected.vy)),
                            'actual': asdict(actual), 'predicted': asdict(projected)})
                    rows.append({'kind': 'movement', 'ratings': ratings, 'pad': pad,
                                 'initial_vx': initial_vx, 'frames': errors})
        for initial_vx in (-6000, 0, 6000):
            restore(env, snapshot)
            shot_fixture(env, x=0, y=-160, raw_vy=-1000)
            env.data.memory.assign(0xFFB04A+5*0x80+0x28, '>i2', initial_vx)
            state, _ = advance(env, buttons((0, 0)))
            projected = KeeperMotion.observe(state.team1.goalie)
            errors = []
            for frame in range(1, 13):
                projected = projected.step(save=frame == 1)
                state, _ = advance(env, buttons((0, 0), save=frame == 1))
                actual = KeeperMotion.observe(state.team1.goalie)
                errors.append({'frame': frame,
                    'position_error': max(abs(actual.x-projected.x),abs(actual.y-projected.y))/65536,
                    'velocity_error': max(abs(actual.vx-projected.vx),abs(actual.vy-projected.vy)),
                    'animation': state.team1.goalie.live_anim,
                    'actual': asdict(actual), 'predicted': asdict(projected)})
            rows.append({'kind': 'save-motion', 'initial_vx': initial_vx, 'frames': errors})
        for one_timer in (False, True):
            for x in (-19, -10, 0, 10, 19):
                for z, vz in ((0, 0), (4, 2200), (18, -1500)):
                    restore(env, snapshot)
                    shot_fixture(env, x=x, y=-190, raw_vy=-15000)
                    memory = env.data.memory
                    memory.assign(0xFFB74A + 0x18, '>i4', z * 65536)
                    memory.assign(0xFFB74A + 0x2C, '>i2', vz)
                    memory.assign(0xFFBF7A, '>i2', -1)
                    flags = memory.extract(0xFFC2F8, '>u2')
                    memory.assign(0xFFC2F8, '>u2', (flags & ~0x200) | (0x200 if one_timer else 0))
                    state, _ = advance(env, buttons((0, 0)))
                    assert state.engine.native_one_timer_shot == one_timer
                    projected = PuckFlight.observe(state.puck)
                    predicted_contact, actual_contact, errors = None, None, []
                    forecast, forecast_keeper = projected, KeeperMotion.observe(state.team1.goalie)
                    for tick in range(1, 25):
                        forecast, forecast_keeper = forecast.step(), forecast_keeper.step()
                        if contact_margin(forecast_keeper, forecast, one_timer=one_timer) >= 0:
                            predicted_contact = tick
                            break
                        if abs(forecast.y >> 16) >= 261:
                            break
                    for frame in range(1, 25):
                        projected = projected.step()
                        before = state
                        state, _ = advance(env, buttons((0, 0)))
                        actual = PuckFlight.observe(state.puck)
                        keeper = KeeperMotion.observe(state.team1.goalie)
                        changed = state.engine.puck_owner >= 0 or memory.extract(0xFFBF7A, '>i2') == 5
                        if changed:
                            actual_contact = frame
                            break
                        if abs(state.puck.y) >= 258:
                            break
                        errors.append({'frame': frame,
                            'position_error': max(abs(actual.x-projected.x), abs(actual.y-projected.y),
                                                  abs(actual.z-projected.z))/65536,
                            'velocity_error': max(abs(actual.vx-projected.vx), abs(actual.vy-projected.vy),
                                                  abs(actual.vz-projected.vz)),
                            'actual': asdict(actual), 'predicted': asdict(projected)})
                    rows.append({'kind': 'flight', 'one_timer': one_timer, 'x': x, 'z': z, 'vz': vz,
                                 'frames': errors, 'predicted_contact': predicted_contact,
                                 'actual_contact': actual_contact})
    finally:
        env.close()
    report = {'protocol': 'classic-independent-goalie-physics-v1', 'cases': rows,
              'source': 'Classic and local NHL94 ROM disassembly; no opponent source',
              'sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
                  Path(__file__), Path('nhl94_ai/agents/goalie_physics.py'))}}
    for kind in ('movement', 'save-motion', 'flight'):
        errors = [f for row in rows if row['kind'] == kind for f in row['frames']]
        report[kind] = {'frames': len(errors),
            'max_position_error': max(f['position_error'] for f in errors),
            'max_velocity_error': max(f['velocity_error'] for f in errors),
            'exact_frames': sum(f['position_error'] == f['velocity_error'] == 0 for f in errors)}
        report[kind]['by_horizon'] = {str(horizon): {
            'frames': sum(f['frame'] <= horizon for f in errors),
            'max_position_error': max(f['position_error'] for f in errors if f['frame'] <= horizon)}
            for horizon in (12, 24, 48)}
    flights = [row for row in rows if row['kind'] == 'flight']
    report['contact'] = {'cases': len(flights), 'event_matches': sum(
        (row['predicted_contact'] is None) == (row['actual_contact'] is None) for row in flights),
        'within_one_frame': sum(row['predicted_contact'] is not None and row['actual_contact'] is not None
            and abs(row['predicted_contact']-row['actual_contact']) <= 1 for row in flights)}
    Path(output).write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('movement', 'save-motion', 'flight')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    run(parser.parse_args().output)
