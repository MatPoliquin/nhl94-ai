"""Explicit SNES ROM checks for the shared Classic dispatch and state contract.

Run with the local NHL94-Snes-v0 integration installed. ROMs and save states
remain in stable-retro; no ROM bytes are written to test fixtures.
"""
from collections import Counter
import hashlib
from pathlib import Path

import numpy as np

from nhl94_ai.agents.base import AgentInput, configure_scripted_frames
from nhl94_ai.agents.registry import create_scripted
from nhl94_ai.config import load_hyperparams, resolve_hyperparams_for_model
from nhl94_ai.env.factory import build_single_nhl94_env, get_button_names
from nhl94_ai.evaluation.play import NHL94Player, parse_cmdline
from nhl94_ai.training.datasets import get_game_state


def options():
    return parse_cmdline(['--mode', 'model_vs_game', '--env', 'NHL94-Snes-v0',
                         '--nn', 'ClassicAIV1', '--rf', 'PostPlay', '--goalie-policy', 'off'])


def play_period():
    args = options()
    config = Path(__file__).resolve().parents[2] / 'configs/training/nhl94.json'
    params = resolve_hyperparams_for_model(load_hyperparams(str(config)), args.nn)
    params['frame_skip'] = 4
    assert get_button_names(args) == list(('B', 'Y', 'SELECT', 'START', 'UP', 'DOWN',
                                          'LEFT', 'RIGHT', 'A', 'X', 'L', 'R'))
    env = build_single_nhl94_env(args, params, num_players=1, use_sticky_action=False)
    try:
        observation, info = env.reset(seed=7)
        assert info['nhl94_platform'] == 'snes'
        assert info['native_puck_owner'] == -2
        state = get_game_state(env)
        assert state.team1.controlled_scnum() == 2
        assert not state.team1.players[2].is_one_timer
        agent = create_scripted('classic-v1', args, env)
        configure_scripted_frames(agent, env, record=True)
        digest = hashlib.sha256()
        decisions = Counter()
        frames = 0
        completed = False
        for _ in range(15000):
            state = get_game_state(env)
            action = agent.act(AgentInput(state, observation)).action
            assert env.action_space.contains(action)
            observation, _, terminated, truncated, info = env.step(action)
            assert np.isfinite(observation).all()
            for _, applied, _, _, diagnostics in info['scripted_frames']:
                digest.update(np.asarray(applied, dtype=np.int8).tobytes())
                frames += 1
                decisions[diagnostics['decision']] += 1
            if terminated or truncated:
                completed = terminated and not truncated
                break
        state = get_game_state(env)
        assert completed and state.time == 0, (state.time, frames)
        assert state.team1.stats.shots > 0 and state.team2.stats.shots > 0
        assert decisions['faceoff'] and decisions['carry'] and decisions['shoot']
        assert decisions['goalie-outlet'] and decisions['advance-pass']
        assert state.team1.pass_attempts > 0 and state.team1.stats.passing > 0
        result = dict(frames=frames, actions_sha256=digest.hexdigest(),
                      shots=[state.team1.stats.shots, state.team2.stats.shots],
                      goals=[state.team1.stats.score, state.team2.stats.score],
                      passes=[state.team1.stats.passing, state.team1.pass_attempts])
        observation, reset_info = env.reset(seed=7)
        agent.reset()
        assert reset_info['native_puck_owner'] == -2
        assert get_game_state(env).time == 300
        assert agent.controller.offense.pass_action.pending is None
        return result
    finally:
        env.close()


def main():
    first, repeated = play_period(), play_period()
    assert first == repeated, (first, repeated)
    print('PASS: shared SNES Classic completes reproducible periods, resets, and native shot/pass feedback', first)
    for mode, side, frames in (('model_vs_game', 'home', 4000), ('model_vs_game', 'away', 4500),
                               ('model_vs_model', 'home', 4000), ('model_vs_model', 'away', 1200),
                               ('player_vs_model', 'home', 600), ('player_vs_model', 'away', 600)):
        check_playback(mode, side, frames)


def check_playback(mode, side, frames):
    """Exercise the public playback dispatcher at one action per native frame."""
    args = options()
    args.mode, args.side, args.goalie_policy = mode, side, 'selective'
    args.num_players = 2 if mode in ('model_vs_model', 'player_vs_model') else 1
    config = Path(__file__).resolve().parents[2] / 'configs/training/nhl94.json'
    args.hyperparams_dict = resolve_hyperparams_for_model(load_hyperparams(str(config)), args.nn)
    player = NHL94Player(args, None, need_display=False)
    expected = ((1, 2) if args.num_players == 2 else (1, 0))
    if side == 'away':
        expected = tuple(3 - team if team else 0 for team in expected)
    step = player.display_env.step
    applied = Counter()
    count = 0

    class CheckedFrames(Exception):
        """Finish the bounded native-frame check without altering game state."""

    def checked_step(action):
        nonlocal count
        native = np.asarray(action[0])
        assert native.shape == (args.num_players * 12,)
        applied['first_active'] += int(native[:12].any())
        applied['second_active'] += int(native[12:].any())
        observation, reward, done, info = step(action)
        assert not np.any(done), 'bounded check unexpectedly reached period end'
        assert np.isfinite(observation).all()
        assert tuple(info[0][f'defense_team{i}'] for i in (1, 2)) == expected
        for joystick, team in enumerate(expected):
            if team:
                assert info[0][f'snes_team_controller_count_{team - 1}'] == 1
                slot = info[0][f'defense_control{joystick + 1}']
                assert slot == -1 or (team - 1) * 6 <= slot < team * 6
        count += 1
        if count == frames:
            raise CheckedFrames
        return observation, reward, done, info

    player.display_env.step = checked_step
    try:
        try:
            player.play(continuous=False)
        except CheckedFrames:
            pass
        assert count == frames
        assert applied['first_active'] > 0
        if mode == 'model_vs_model':
            assert applied['second_active'] > 0
        elif mode == 'player_vs_model':
            # The headless human joystick stays neutral independently of AI.
            assert applied['second_active'] == 0
        goalie = player.ai_sys.models[1].controller.goalie
        assert goalie.metrics['handoff_timeouts'] == 0, dict(goalie.metrics)
        if frames >= 4000:
            assert goalie.metrics['takeovers'] > 0
        if mode == 'model_vs_game' and side == 'home':
            assert goalie.metrics['save_animations'] > 0
            assert goalie.metrics['controlled_catches'] > 0
            assert goalie.metrics['outlets_launched'] > 0
        player.display_env.reset()
        assert tuple(player.display_env.reset_infos[0][f'defense_team{i}'] for i in (1, 2)) == expected
        print('PASS: SNES playback', mode, side, count, 'native frames', dict(goalie.metrics), flush=True)
    finally:
        player.close()


if __name__ == '__main__':
    main()
