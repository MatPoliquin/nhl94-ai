"""ROM telemetry and reactive-defense parity between interval and per-frame callers."""
import hashlib
import json
from dataclasses import replace

import numpy as np

from nhl94_ai.agents.base import AgentInput, configure_scripted_frames
from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.agents.registry import create_scripted
from nhl94_ai.config import load_hyperparams, resolve_hyperparams_for_model
from nhl94_ai.env.factory import build_single_nhl94_env, make_retro
from nhl94_ai.evaluation.play import parse_cmdline
from nhl94_ai.game.specs import GAMES
from nhl94_ai.game.state import NHL94GameState
from nhl94_ai.tasks.defense_setup import _place_object, _rebuild_object_order
from nhl94_ai.tasks.registry import TASKS, get_task, register_task
from nhl94_ai.training.datasets import get_game_state


def telemetry():
    for game, spec in GAMES.items():
        env = make_retro(game=game, num_players=1)
        try:
            env.reset(seed=7)
            *_, info = env.step(np.zeros(12, dtype=np.int8))
            state = NHL94GameState(spec.skaters_per_team)
            state.BeginFrame(info, [0] * 6)
            assert state.puck.motion_x is not None and state.puck.height is not None
            for team in (state.team1, state.team2):
                slot = team.skater_scnum_base() + spec.skaters_per_team
                assert env.data.memory.extract(0xFFB04A + slot * 0x80 + 0x34, '>i2') == 0
                assert team.defense_goalie == slot
                for index, player in enumerate(team.players):
                    assert player.role > 0
                    assert 0 <= player.speed <= 30 and 0 <= player.agility <= 30
                    assert 0 <= player.energy <= 4096
                    assert player.motion_x is not None and player.motion_y is not None
                    player_slot = team.skater_scnum_base() + index
                    assert player.selection_flags == info[f'defense_{player_slot}_flags']
            assert state.team1.defense_control == info['defense_control1']
            print(f'PASS: {game} reads real skater motion, attributes, energy and goalie slots')
        finally:
            env.close()


def capture(schema, skip):
    args = parse_cmdline(['--mode=model_vs_game', '--nn=ClassicAIV1', '--env=NHL94-Genesis-v0',
                         '--rf=PostPlay', f'--action_type={schema}'])
    params = resolve_hyperparams_for_model(load_hyperparams(args.hyperparams), args.nn)
    params['frame_skip'] = 4
    env = build_single_nhl94_env(args, params, use_frame_skip=skip, use_sticky_action=False)
    try:
        agent = create_scripted('classic-v1', args)
        configure_scripted_frames(agent, env, record=True)
        agent.frame_skip = 4
        observation, _ = env.reset(seed=7)
        digest = hashlib.sha256()
        for _ in range(64 // (4 if skip else 1)):
            result = agent.act(AgentInput(get_game_state(env), observation))
            before = np.asarray(observation).copy()
            observation, reward, terminated, truncated, info = env.step(result.action)
            samples = info['scripted_frames'] if skip else [
                (before, result.action, reward, terminated or truncated, result.diagnostics)]
            for obs, action, value, done, details in samples:
                digest.update(np.asarray(obs, dtype='<f8').tobytes())
                digest.update(np.asarray(action, dtype=np.int8).tobytes())
                digest.update(json.dumps([value, done, details], sort_keys=True).encode())
            assert not (terminated or truncated)
        return digest.hexdigest()
    finally:
        env.close()


def _switch_setup(env, game):
    get_task('DefenseZone').initialize(env, game)
    memory = env.data.memory
    actual = memory.extract(0xFFC320, '>i2')
    carrier = memory.extract(0xFFB7AA, '>i2')
    others = [slot for slot in range(5) if slot != actual]
    _place_object(memory, actual, (100, 0))
    for slot, position in zip(others, ((20, -143), (0, -167), (-90, 80), (90, 80))):
        _place_object(memory, slot, position)
    for slot in range(6, 11):
        _place_object(memory, slot, (0, -140) if slot == carrier else (-100 + (slot - 6) * 45, 100))
    _place_object(memory, 14, (0, -140))
    _rebuild_object_order(memory)


def useful_switch():
    """Use buttons, not controller RAM writes, to obtain a useful non-ideal skater."""
    name = 'ClassicSwitchProbe'
    register_task(name, replace(get_task('DefenseZone'), initialize=_switch_setup,
                               reward=lambda _: 0.0, done=lambda _: False))
    try:
        for schema in ('FILTERED', 'HOCKEY_INTENT_DPAD'):
            args = parse_cmdline([
                '--mode=model_vs_game', '--nn=ClassicAIV1', '--env=NHL94-Genesis-v0',
                '--state=PenguinsVsSenators.DefenseZone', f'--rf={name}', f'--action_type={schema}',
            ])
            env = build_single_nhl94_env(args, {'clip_reward': False}, use_frame_skip=False)
            try:
                env.reset(seed=7)
                model = ClassicAIV1Model(args)
                actual = get_game_state(env).team1.defense_control
                others = [slot for slot in range(5) if slot != actual]
                for frame in range(8):
                    action = model.predict_frame(get_game_state(env))[0]
                    details = model.defense_diagnostics
                    if frame == 0:
                        assert details['ideal_slot'] == others[1], details
                        assert details['desired_slot'] == others[0], details
                        assert details['mode'] == 'switch-request', details
                    if details.get('last_switch_result'):
                        result = details['last_switch_result']
                        assert result['outcome'] == 'confirmed', result
                        assert result['actual_slot'] == others[0], result
                        assert env.unwrapped.data.memory.extract(0xFFC320, '>i2') == others[0]
                        break
                    env.step(action)
                else:
                    raise AssertionError((schema, 'Useful switch was not observed', model.defense_diagnostics))
                print(f'PASS: {schema} obtains useful skater {others[0]} rather than waiting for '
                      f'unreachable ideal {others[1]}; ROM confirms after {frame} frames')
            finally:
                env.close()
    finally:
        TASKS.pop(name)


def live_switch_feedback():
    for game, spec in GAMES.items():
        env = make_retro(game=game, num_players=2)
        confirmed = unexpected = requests = 0
        last_result = None
        try:
            env.reset(seed=7)
            state = NHL94GameState(spec.skaters_per_team)
            state.BeginFrame(env.data.lookup_all(), [0] * 6)
            model = ClassicAIV1Model()
            for _ in range(1800):
                action = model.predict_frame(state)[0]
                details = model.defense_diagnostics
                requests += details.get('mode') == 'switch-request'
                result = details.get('last_switch_result')
                if result and result != last_result:
                    if result['outcome'] in ('confirmed', 'unexpected-selection'):
                        confirmed += result['outcome'] == 'confirmed'
                        unexpected += result['outcome'] == 'unexpected-selection'
                        assert result['from_slot'] != result['actual_slot']
                        assert 1 <= result['elapsed_frames'] <= 8
                        # Faceoff placement can change the nearest skater
                        # between B press/release. Follow actual ROM control.
                        assert details['acting_slot'] == result['actual_slot'] == state.team1.defense_control
                        assert model.defense.pending_switch is None
                    last_result = result
                state.EndFrame()
                *_, info = env.step(np.concatenate((action, np.zeros(12, dtype=np.int8))))
                state.BeginFrame(info, [0] * 6)
            if spec.skaters_per_team == 1:
                assert requests == 0
            else:
                assert confirmed > 0, (game, requests)
            print(f'PASS: {game}: {confirmed}/{requests} requests confirmed, {unexpected} unexpected selections '
                  'handled in a bounded two-controller trace')
        finally:
            env.close()


def _coverage_setup(env, game):
    get_task('DefenseZone').initialize(env, game)
    memory = env.data.memory
    actual = memory.extract(0xFFC320, '>i2')
    carrier = memory.extract(0xFFB7AA, '>i2')
    others = [slot for slot in range(5) if slot != actual]
    attackers = [slot for slot in range(6, 11) if slot != carrier]
    _place_object(memory, actual, (-40, -175))
    for slot, point in zip(others, ((0, -180), (110, 0), (-110, 0), (80, 60))):
        _place_object(memory, slot, point)
    _place_object(memory, carrier, (0, -150))
    memory.assign(0xFFB04A + carrier * 0x80 + 0x54, '>u2', 4)
    for slot, point in zip(attackers, ((40, -175), (-110, 100), (0, 100), (110, 100))):
        _place_object(memory, slot, point)
    _place_object(memory, 14, (0, -160))
    _rebuild_object_order(memory)


def teammate_coverage():
    name = 'ClassicCoverageProbe'
    register_task(name, replace(get_task('DefenseZone'), initialize=_coverage_setup,
                               reward=lambda _: 0.0, done=lambda _: False))
    try:
        for schema in ('FILTERED', 'HOCKEY_INTENT_DPAD'):
            args = parse_cmdline([
                '--mode=model_vs_game', '--nn=ClassicAIV1', '--env=NHL94-Genesis-v0',
                '--state=PenguinsVsSenators.DefenseZone', f'--rf={name}', f'--action_type={schema}',
            ])
            env = build_single_nhl94_env(args, {'clip_reward': False}, use_frame_skip=False)
            try:
                env.reset(seed=7)
                state = get_game_state(env)
                actual, carrier = state.team1.defense_control, state.engine.puck_owner
                assert carrier in range(6, 11), carrier
                model = ClassicAIV1Model(args)
                action = model.predict_frame(state)[0]
                details = model.defense_diagnostics
                assert details['decision'] == 'deny-reception', details
                assert details['threat_slot'] != carrier
                assert len(details['reserved_slots']) == 1, details
                blocker = details['reserved_slots'][0]
                assert model.defense._likely_switch(state) == blocker, details
                assert details['desired_slot'] == actual, details
                env.step(action)
                assert get_game_state(env).team1.defense_control == actual
                # Inject drift only in this isolated ROM scenario. The next
                # decoded frame must no longer count the teammate as cover.
                env.unwrapped.data.memory.assign(0xFFB04A + blocker * 0x80 + 0x28, '>i2', 6000)
                env.step(np.zeros_like(action))
                model.predict_frame(get_game_state(env))
                details = model.defense_diagnostics
                assert all(blocker not in lane['blockers'] for lane in details['lanes']), details
                assert not details['reserved_slots'], details
                print(f'PASS: {schema} covers another shooting lane without stealing blocker {blocker}; '
                      'live velocity feedback removes drifting cover')
            finally:
                env.close()
    finally:
        TASKS.pop(name)


def _check_setup(env, game):
    get_task('DefenseZone').initialize(env, game)
    memory = env.data.memory
    actual, carrier = memory.extract(0xFFC320, '>i2'), memory.extract(0xFFB7AA, '>i2')
    _place_object(memory, actual, (0, -174))
    for index, slot in enumerate(slot for slot in range(5) if slot != actual):
        _place_object(memory, slot, (-100 + index * 65, 50))
    _place_object(memory, carrier, (0, -150))
    for index, slot in enumerate(slot for slot in range(6, 11) if slot != carrier):
        _place_object(memory, slot, (-100 + index * 65, 120))
    # Isolated 4-versus-8 weight matchup; these are not benchmark roster edits.
    for slot, weight in ((actual, 32), (carrier, 64)):
        base = 0xFFB04A + slot * 0x80
        memory.assign(base + 0x67, '|u1', weight)
        memory.assign(base + 0x54, '>u2', 0)
    roster = memory.extract(0xFFB04A + actual * 0x80 + 0x66, '|u1')
    memory.assign(0xFFC700 + roster * 2, '>u2', 4096)
    _place_object(memory, 14, (0, -140))
    _rebuild_object_order(memory)


def deliberate_body_check():
    name = 'ClassicBodyCheckProbe'
    register_task(name, replace(get_task('DefenseZone'), initialize=_check_setup,
                               reward=lambda _: 0.0, done=lambda _: False))
    try:
        for schema in ('FILTERED', 'HOCKEY_INTENT_DPAD'):
            args = parse_cmdline([
                '--mode=model_vs_game', '--nn=ClassicAIV1', '--env=NHL94-Genesis-v0',
                '--state=PenguinsVsSenators.DefenseZone', f'--rf={name}', f'--action_type={schema}',
            ])
            env = build_single_nhl94_env(args, {'clip_reward': False}, use_frame_skip=False)
            try:
                results = {}
                for checking in (False, True):
                    env.reset(seed=7)
                    state = get_game_state(env)
                    actual, carrier = state.team1.defense_control, state.engine.puck_owner
                    assert actual in range(5) and carrier in range(6, 11)
                    model = ClassicAIV1Model(args)
                    action = model.predict_frame(state)[0]
                    details = model.defense_diagnostics
                    assert details['mode'] == 'check-request', details
                    before = state.team1.stats.bodychecks
                    trace = []
                    memory = env.unwrapped.data.memory
                    for frame in range(16):
                        env.step(action if checking and frame == 0 else np.zeros_like(action))
                        state = get_game_state(env)
                        base = 0xFFB04A + carrier * 0x80
                        trace.append({
                            'frame': frame + 1, 'owner': state.engine.puck_owner,
                            'checker_anim': memory.extract(0xFFB04A + actual * 0x80 + 0x58, '>u2'),
                            'carrier_anim': memory.extract(base + 0x58, '>u2'),
                            'impact_player': memory.extract(base + 0x2E, '>i2'),
                            'impact': memory.extract(base + 0x32, '>u2'),
                            'checks': state.team1.stats.bodychecks - before,
                        })
                    results[checking] = trace
                trace = results[True]
                assert any(row['checker_anim'] == 0xC5E for row in trace), (schema, trace)
                assert any(row['impact_player'] == actual and row['impact'] > 0 for row in trace), (schema, trace)
                assert any(row['checks'] == 1 and row['owner'] != carrier for row in trace), (schema, trace)
                assert all(row['checks'] == 0 for row in results[False]), (schema, results[False])
                recovered = any(row['owner'] == actual for row in trace)
                print(f'PASS: {schema} fresh C produces a recorded body check and carrier possession loss; '
                      f'controlled recovery={recovered}; paired neutral input records no check')
            finally:
                env.close()
    finally:
        TASKS.pop(name)


if __name__ == '__main__':
    telemetry()
    useful_switch()
    teammate_coverage()
    deliberate_body_check()
    live_switch_feedback()
    for action_type in ('FILTERED', 'HOCKEY_INTENT_DPAD'):
        assert capture(action_type, True) == capture(action_type, False), action_type
        print(f'PASS: {action_type} per-frame and four-frame callers have identical actions and diagnostics')
