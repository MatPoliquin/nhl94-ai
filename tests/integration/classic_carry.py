"""Explicit native checks for ordinary carry steering and receiving lookahead."""
import argparse
from dataclasses import replace
from functools import partial
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from nhl94_ai.agents.carry import BURST_LOCK_FRAMES, CARRY_FRAMES, carry_path, forecast_carry
from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.env.factory import build_single_nhl94_env, make_retro
from nhl94_ai.env.intents import HOCKEY_INTENT_NOOP
from nhl94_ai.evaluation.benchmark import RAM, away_view, update_state
from nhl94_ai.game.state import NHL94GameState
from nhl94_ai.game.ram import _animation_frames
from nhl94_ai.game.constants import GameConsts as Buttons
from nhl94_ai.agents.motion import VELOCITY_SCALE
from nhl94_ai.evaluation.play import parse_cmdline
from nhl94_ai.tasks.cross_crease_setup import (
    OBJECT_BASE, OBJECT_STRIDE, CrossingTraffic, keep_traffic_idle, prepare_isolated_crossing,
)
from nhl94_ai.tasks.defense_setup import _place_object, _rebuild_object_order, _reset_player
from nhl94_ai.tasks.registry import TASKS, get_task, register_task
from nhl94_ai.training.datasets import get_game_state
from tests.integration.classic_offense import _offense_setup


def native_carry_motion():
    env = make_retro(game='NHL94-Genesis-v0', state='PenguinsVsSenators.DefenseZone', num_players=1)
    try:
        for name, (address, kind) in RAM.items():
            env.data.set_variable(name, {'address': address, 'type': kind})
        maximum = [0.0, 0.0, 0.0]
        traces = 0
        for side in (1, 2):
            for interval in (1, 4, 10):
                for agility, speed, weight, energy in ((5, 5, 96, 1024), (30, 30, 32, 4096)):
                    env.reset(seed=94001)
                    slot, _, inactive = prepare_isolated_crossing(
                        env, side=side, width=80, depth=160, speed=4500)
                    memory = env.data.memory
                    base = OBJECT_BASE + slot * OBJECT_STRIDE
                    for offset, value in ((0x67, weight), (0x68, agility), (0x69, speed)):
                        memory.assign(base + offset, '|u1', value)
                    roster = memory.extract(base + 0x66, '|u1')
                    memory.assign(0xFFC700 + (slot // 6) * 0x364 + roster * 2, '>u2', energy)
                    *_, info = env.step(np.zeros(12, dtype=np.int8))
                    state = NHL94GameState(5)
                    update_state(state, info, env)
                    view = state if side == 1 else away_view(state)
                    carrier = view.team1.get_player_by_scnum(slot)
                    sign = 1 if side == 1 else -1
                    target = carrier.x + 12, carrier.y + sign * 24
                    path = carry_path(carrier, target, decision_interval=interval)
                    assert path is not None
                    controller = ClassicAIV1Model(SimpleNamespace(action_type='FILTERED', one_timers=False))
                    for elapsed in range(CARRY_FRAMES):
                        view = state if side == 1 else away_view(state)
                        if elapsed % interval == 0:
                            action = np.zeros(12, dtype=np.int8)
                            controller._steer(action, view.team1.get_player_by_scnum(slot), *target)
                            action = controller._encode(action, HOCKEY_INTENT_NOOP)[0]
                        *_, info = env.step(action)
                        update_state(state, info, env)
                        view = state if side == 1 else away_view(state)
                        actual, predicted = view.team1.get_player_by_scnum(slot), path[elapsed + 1]
                        assert view.engine.puck_owner == slot, (side, interval, elapsed, 'lost possession')
                        errors = (
                            max(abs(actual.precise_x - predicted.precise_x),
                                abs(actual.precise_y - predicted.precise_y)),
                            max(abs(actual.motion_x - predicted.motion_x),
                                abs(actual.motion_y - predicted.motion_y)),
                            min(abs(actual.facing_phase - predicted.facing_phase),
                                8 - abs(actual.facing_phase - predicted.facing_phase)),
                        )
                        maximum = [max(previous, error) for previous, error in zip(maximum, errors)]
                        assert max(errors) <= 17 / 65536, (side, interval, agility, elapsed, errors)
                        assert all(memory.extract(OBJECT_BASE + other * OBJECT_STRIDE + 0x34, '>i2') == -1
                                   for other in inactive)
                    traces += 1
        print(f'PASS: {traces} ordinary carry traces, both ends and 1/4/10-frame input cadence; '
              f'maximum position/velocity/facing errors {maximum}')
        return maximum
    finally:
        env.close()


def _wing_setup(env, game, *, pursuing=False):
    _offense_setup(env, game)
    memory = env.data.memory
    carrier = memory.extract(0xFFC320, '>i2')
    receiver = next(slot for slot in range(5) if slot != carrier)
    for slot in range(12):
        if slot not in (carrier, receiver, 6, 11):
            _place_object(memory, slot, (-100 + (slot % 5) * 50, -120 - (slot // 5) * 50))
            memory.assign(OBJECT_BASE + slot * OBJECT_STRIDE + 0x34, '>i2', -1)
    _place_object(memory, carrier, (80, 145))
    role = memory.extract(OBJECT_BASE + receiver * OBJECT_STRIDE + 0x34, '>i2')
    _reset_player(memory, receiver, (115, 180), 6, 0, False)
    memory.assign(OBJECT_BASE + receiver * OBJECT_STRIDE + 0x34, '>i2', role)
    memory.assign(OBJECT_BASE + receiver * OBJECT_STRIDE + 0x28, '>i2', -round(1.5 / VELOCITY_SCALE))
    for slot in (carrier, receiver, 6):
        base = OBJECT_BASE + slot * OBJECT_STRIDE
        for offset, value in ((0x67, 64), (0x68, 20), (0x69, 20), (0x6D, 20), (0x71, 30)):
            memory.assign(base + offset, '|u1', value)
        roster = memory.extract(base + 0x66, '|u1')
        memory.assign(0xFFC700 + (slot // 6) * 0x364 + roster * 2, '>u2', 4096)
    for offset, value in ((0x67, 32), (0x68, 30), (0x69, 30)):
        memory.assign(OBJECT_BASE + receiver * OBJECT_STRIDE + offset, '|u1', value)
    _reset_player(memory, 6, (0, 230), 0, 3 if pursuing else 0, False, (0x11,) if pursuing else ())
    memory.assign(OBJECT_BASE + 6 * OBJECT_STRIDE + 0x28, '>i2', -round(2 / VELOCITY_SCALE))
    _place_object(memory, 11, (-15, 250))
    _place_object(memory, 14, (80, 155))
    _rebuild_object_order(memory)


def native_wing_reception(*, active_defender=False, uncertain_carry=False):
    name = 'ClassicCarryContinuationProbe'
    rows = []
    register_task(name, replace(get_task('DefenseZone'), initialize=partial(_wing_setup, pursuing=active_defender),
                               reward=lambda _: 0.0, done=lambda _: False))
    try:
        for schema in ('FILTERED', 'HOCKEY_INTENT_DPAD'):
            args = parse_cmdline([
                '--mode=model_vs_game', '--nn=ClassicAIV1', '--env=NHL94-Genesis-v0',
                '--state=PenguinsVsSenators.DefenseZone', f'--rf={name}', f'--action_type={schema}',
            ])
            args.one_timers = False
            args.offense_lookahead = True
            args.uncertain_carry = uncertain_carry
            env = build_single_nhl94_env(args, {'clip_reward': False}, use_frame_skip=False)
            try:
                env.reset(seed=94001)
                state = get_game_state(env)
                carrier = state.engine.puck_owner
                receiver = next(slot for slot in range(5) if slot != carrier)
                friendly = CrossingTraffic(receiver, 'friendly-initial', (115, 180), False, False)
                opponent = CrossingTraffic(6, 'opponent-initial', (0, 230), True, active_defender)
                model = ClassicAIV1Model(args)
                before = state.team1.pass_attempts
                shots_before = state.team1.stats.shots
                observed = None
                received = None
                for elapsed in range(192):
                    state = get_game_state(env)
                    action = model.predict_frame(state)[0]
                    if elapsed == 0:
                        assert model._last_decision == 'position-pass', model.offense_diagnostics
                        option = model.offense_diagnostics['receiver']
                        values = {'position': option['continuation_position_value'],
                                  'finish': option['continuation_finish_value'],
                                  'opportunity': option['continuation_value']}
                        assert option['shot_value'] == 0, option
                        assert option['continuation_position_value'] > 0, option
                        assert option['continuation_finish_value'] == 0, option
                        assert option['continuation_value'] > model.offense_diagnostics['retained_value'] + 12, option
                        defender_start = state.team2.players[0].precise_x, state.team2.players[0].precise_y
                    if observed is None and state.team1.pass_attempts > before:
                        assert state.engine.pass_target == receiver
                        observed = state.engine.pass_target
                    if received is None and state.engine.puck_owner == receiver:
                        assert state.team1.defense_control == receiver
                        assert model.offense.pending is None
                        assert model.offense.last_pass['outcome'] == 'received'
                        assert observed == receiver
                        assert model._last_decision not in ('pass-release', 'pass-flight')
                        received = elapsed
                        reception_position = state.team1.get_player_by_scnum(receiver).x
                    if active_defender and received is not None and elapsed >= received + CARRY_FRAMES:
                        defender = state.team2.players[0]
                        assert state.engine.puck_owner == receiver
                        assert math.dist(defender_start, (defender.precise_x, defender.precise_y)) > 1
                        print(f'PASS: {schema} receives at frame {received} and retains possession '
                              f'through its {CARRY_FRAMES}-frame continuation against active CPU pursuit')
                        rows.append({'schema': schema, 'reception_frame': received,
                                     'safe_through_frame': elapsed, 'values': values})
                        break
                    if state.team1.stats.shots > shots_before:
                        assert received is not None
                        assert state.team1.get_player_by_scnum(receiver).x < reception_position
                        if state.engine.shot_player != receiver:
                            assert state.engine.shot_player == carrier
                            assert model.offense.last_pass['passer'] == receiver
                            assert model.offense.last_pass['actual_receiver'] == carrier
                            assert model.offense.last_pass['outcome'] == 'received'
                        print(f'PASS: {schema} selects a wide receiver for a short carry, '
                              f'receives at frame {received}, replans, carries inside and '
                              f'records a verified receiver/return-pass shot at frame {elapsed}')
                        rows.append({'schema': schema, 'reception_frame': received,
                                     'recorded_shot_frame': elapsed, 'shooter': state.engine.shot_player,
                                     'initial_receiver': receiver, 'values': values})
                        break
                    if schema == 'FILTERED' and received is None:
                        assert not action[Buttons.INPUT_C], (elapsed, action)
                    if received is None:
                        keep_traffic_idle(env.unwrapped, friendly)
                    keep_traffic_idle(env.unwrapped, opponent)
                    env.step(action)
                else:
                    raise AssertionError((schema, 'No native receive-carry-shot sequence', model.offense_diagnostics))
            finally:
                env.close()
    finally:
        TASKS.pop(name)
    return rows


def native_active_pressure():
    """Observe real nearest-puck CPU pursuit; never hold the defender idle."""
    env = make_retro(game='NHL94-Genesis-v0', state='PenguinsVsSenators.DefenseZone', num_players=1)
    rows = []
    try:
        for name, (address, kind) in RAM.items():
            env.data.set_variable(name, {'address': address, 'type': kind})
        animations = _animation_frames('NHL94-Genesis-v0', 0xC5E)
        assert all(sum(abs(duration) for _, duration in frames) >= BURST_LOCK_FRAMES
                   for frames in animations)
        scenarios = {'head-on': (0, 45), 'flank': (35, 15), 'trailing': (-30, -30), 'remote': (100, -80)}
        for side in (1, 2):
            for profile, (agility, speed, weight, energy) in {
                    'low': (5, 5, 96, 1024), 'high': (30, 30, 32, 4096)}.items():
                for scenario, (dx, dy) in scenarios.items():
                    env.reset(seed=94021)
                    slot, _, inactive = prepare_isolated_crossing(
                        env, side=side, width=55, depth=170, speed=4500)
                    memory = env.data.memory
                    sign = 1 if side == 1 else -1
                    defender = 6 if side == 1 else 0
                    base = OBJECT_BASE + defender * OBJECT_STRIDE
                    point = -55 + dx, sign * (170 + dy)
                    heading = round(math.atan2(-dx, -sign * dy) * 4 / math.pi) % 8
                    _reset_player(memory, defender, point, heading, 3, False, (0x11,))
                    memory.assign(base + 0x34, '>i2', 3)
                    for offset, value in ((0x67, weight), (0x68, agility), (0x69, speed),
                                          (0x6B, 7), (0x71, 20), (0x75, 20)):
                        memory.assign(base + offset, '|u1', value)
                    roster = memory.extract(base + 0x66, '|u1')
                    memory.assign(0xFFC700 + (defender // 6) * 0x364 + roster * 2, '>u2', energy)
                    _rebuild_object_order(memory)
                    *_, info = env.step(np.zeros(12, dtype=np.int8))
                    state = NHL94GameState(5)
                    update_state(state, info, env)
                    view = state if side == 1 else away_view(state)
                    player = view.team1.get_player_by_scnum(slot)
                    opponent = view.team2.get_player_by_scnum(defender)
                    assert not opponent.selection_flags & 8
                    start = opponent.precise_x, opponent.precise_y
                    heading_start = opponent.facing_phase
                    target = player.x + 12, player.y + sign * 24
                    _, forecast = forecast_carry(view, player, target, assess_uncertainty=True)
                    model = ClassicAIV1Model(SimpleNamespace(action_type='FILTERED', one_timers=False))
                    contact, lost = None, None
                    movement = 0.0
                    turning = 0.0
                    checking = bool(opponent.selection_flags & 0x20)
                    for elapsed in range(1, CARRY_FRAMES + 1):
                        view = state if side == 1 else away_view(state)
                        if (elapsed - 1) % 4 == 0:
                            action = np.zeros(12, dtype=np.int8)
                            model._steer(action, view.team1.get_player_by_scnum(slot), *target)
                        *_, info = env.step(action)
                        update_state(state, info, env)
                        view = state if side == 1 else away_view(state)
                        player, opponent = view.team1.get_player_by_scnum(slot), view.team2.get_player_by_scnum(defender)
                        movement = max(movement, math.dist(start, (opponent.precise_x, opponent.precise_y)))
                        difference = abs(opponent.facing_phase - heading_start)
                        turning = max(turning, min(difference, 8 - difference))
                        checking |= bool(opponent.selection_flags & 0x20)
                        if math.dist((player.x, player.y), (opponent.x, opponent.y)) <= 16 and contact is None:
                            contact = elapsed
                        if view.engine.puck_owner != slot and lost is None:
                            lost = elapsed
                        assert all(memory.extract(OBJECT_BASE + other * OBJECT_STRIDE + 0x34, '>i2') == -1
                                   for other in inactive if other != defender)
                    if forecast['carry_safe']:
                        assert contact is None and lost is None, (side, profile, scenario, forecast, contact, lost)
                    assert movement > 0 or turning > 0 or checking, (
                        side, profile, scenario, 'No observed CPU response')
                    rows.append({'side': side, 'profile': profile, 'scenario': scenario,
                                 'safe': forecast['carry_safe'], 'pressure': forecast['carry_pressure'],
                                 'viable': forecast['carry_viable'], 'risk': forecast['carry_risk'],
                                 'status': forecast['carry_status'],
                                 'defender_movement': movement, 'body_contact_frame': contact,
                                 'defender_turning': turning, 'checking_animation': checking,
                                 'possession_loss_frame': lost})
        assert any(row['body_contact_frame'] is not None or row['possession_loss_frame'] is not None for row in rows)
        assert any(row['safe'] for row in rows)
        assert any(not row['safe'] and row['viable'] and row['possession_loss_frame'] is None for row in rows)
        print(f'PASS: {len(rows)} active CPU pressure cases, both ends and skating profiles; '
              f'{sum(row["safe"] for row in rows)} certified routes, no observed false-safe route')
        return rows
    finally:
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output')
    arguments = parser.parse_args()
    errors = native_carry_motion()
    default_wing = native_wing_reception()
    wing = native_wing_reception(uncertain_carry=True)
    active_wing = native_wing_reception(active_defender=True, uncertain_carry=True)
    pressure = native_active_pressure()
    if arguments.output:
        root = Path(__file__).resolve().parents[2]
        sources = ('agents/carry.py', 'agents/offense.py', 'agents/classic_v1.py',
                   'agents/passing.py', 'game/state.py', 'agents/skating.py')
        report = {'motion_maximum_errors': errors, 'active_pressure': pressure,
                  'default_wing': default_wing, 'idle_wing': wing, 'active_wing': active_wing,
                  'sources': {name: hashlib.sha256((root / 'nhl94_ai' / name).read_bytes()).hexdigest()
                              for name in sources}}
        report['sources']['tests/integration/classic_carry.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        Path(arguments.output).write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
