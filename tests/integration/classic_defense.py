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
        confirmed = requests = 0
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
                    assert result['outcome'] != 'unexpected-selection', (game, result)
                    if result['outcome'] == 'confirmed':
                        confirmed += 1
                        assert result['from_slot'] != result['actual_slot']
                        assert 1 <= result['elapsed_frames'] <= 8
                    last_result = result
                state.EndFrame()
                *_, info = env.step(np.concatenate((action, np.zeros(12, dtype=np.int8))))
                state.BeginFrame(info, [0] * 6)
            if spec.skaters_per_team == 1:
                assert requests == 0
            else:
                assert confirmed > 0, (game, requests)
            print(f'PASS: {game}: {confirmed}/{requests} requests confirmed in a bounded two-controller trace')
        finally:
            env.close()


if __name__ == '__main__':
    telemetry()
    useful_switch()
    live_switch_feedback()
    for action_type in ('FILTERED', 'HOCKEY_INTENT_DPAD'):
        assert capture(action_type, True) == capture(action_type, False), action_type
        print(f'PASS: {action_type} per-frame and four-frame callers have identical actions and diagnostics')
