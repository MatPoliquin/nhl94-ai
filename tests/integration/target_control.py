"""Explicit ROM checks for target execution, then paired defensive baselines.

The isolated skating probe marks other skaters inactive only in its test task.
The paired evaluations use the unmodified, full-team DefenseZone initializer.
Optionally pass --model to compare a trained target checkpoint on the same seeds.
"""
import argparse
from dataclasses import replace
from functools import partial
import math
from pathlib import Path
import tempfile

import numpy as np
import torch
from stable_baselines3 import PPO

from nhl94_ai.artifacts import load_policy, run_metadata, save_checkpoint
from nhl94_ai.agents.base import AgentInput, FrameRepeatAgent, LearnedAgent
from nhl94_ai.config import default_config_path, load_hyperparams, resolve_hyperparams_for_model
from nhl94_ai.env.factory import build_single_nhl94_env
from nhl94_ai.env.target_control import normalize_position
from nhl94_ai.evaluation.play import parse_cmdline
from nhl94_ai.models.factory import build_MlpPolicy
from nhl94_ai.tasks.defense_setup import _place_object
from nhl94_ai.tasks.registry import TASKS, get_task, register_task
from nhl94_ai.training.datasets import get_game_state


def _args(task='DefenseZone', action_type='TARGET_POSITION'):
    return parse_cmdline([
        '--mode=model_vs_game', '--env=NHL94-Genesis-v0', '--nn=MlpPolicy',
        '--rf=' + task, '--state=PenguinsVsSenators.DefenseZone', '--action_type=' + action_type,
    ])


def _tracking_setup(env, game, start=(0, 0)):
    get_task('DefenseZone').initialize(env, game)
    memory = env.data.memory
    selected = memory.extract(0xFFC320, '>i2')
    for slot in range(12):
        if slot != selected and slot not in (5, 11):
            memory.assign(0xFFB04A + slot * 0x80 + 0x34, '>i2', -1)
    _place_object(memory, selected, start)
    memory.assign(0xFFB04A + selected * 0x80 + 0x54, '>u2', 0)


def check_tracking(params):
    name = 'TargetTrackingProbe'
    register_task(name, replace(get_task('DefenseZone'), initialize=_tracking_setup,
                               reward=lambda _: 0.0, done=lambda _: False, target_bounds=None))
    env = None
    worst_error = 0.0
    boosts = 0
    try:
        env = build_single_nhl94_env(_args(name), params, use_frame_skip=False)
        cases = [((0, 0), target, 180) for target in ((0, 40), (40, 0), (0, -40), (-40, 0), (40, 40), (0, 90))]
        cases.extend([((-60, sign * 265), (60, sign * 265), 500) for sign in (-1, 1)])
        for start, target, frames in cases:
            env.init_function = partial(_tracking_setup, start=start)
            env.reset(seed=0)
            errors = []
            last_b = last_c = False
            for _ in range(frames):
                action = np.asarray(target, dtype=np.float32) / np.array([120, 270], dtype=np.float32)
                _, _, _, _, info = env.step(action)
                diagnostics = info['target_control']
                assert diagnostics['actual_slot'] == 1, diagnostics
                buttons = diagnostics['buttons']
                assert not (buttons[0] and last_b), 'B must have a release frame'
                assert not (buttons[8] and last_c), 'C must have a release frame'
                last_b, last_c = bool(buttons[0]), bool(buttons[8])
                boosts += int(last_c)
                errors.append(diagnostics['distance'])
            settling_error = max(errors[-30:])
            assert settling_error <= 6, (target, settling_error)
            worst_error = max(worst_error, settling_error)
        assert boosts > 0, 'The long-distance case must exercise real boost presses'
        print(f'PASS: eight ROM skating/net-routing cases settle within {worst_error:.2f} units for 30 frames; pulsed boosts.')
    finally:
        if env is not None:
            env.close()
        TASKS.pop(name)


def _scripted_target(state):
    # A deliberately simple lane-position baseline, not part of the controller.
    position = (state.puck.x * 0.65, state.puck.y * 0.65 + state.team1.net.y * 0.35)
    return normalize_position(position, get_task('DefenseZone').target_bounds).clip(-1, 1)


def check_replay_timing(params, model=None):
    class Targets:
        def predict(self, observation, **kwargs):
            return np.asarray([0.15, -0.65], dtype=np.float32), None

    traces = []
    for frame_skip in (True, False):
        env = build_single_nhl94_env(_args(), params, use_frame_skip=frame_skip)
        try:
            observation, _ = env.reset(seed=0)
            interval = params['frame_skip']
            agent = FrameRepeatAgent(LearnedAgent(model if model is not None else Targets(), 'TARGET_POSITION'),
                                     1 if frame_skip else interval)
            trace, reward_sum = [], 0.0
            for step in range(80 if frame_skip else 80 * interval):
                action = agent.act(AgentInput(None, observation)).action
                observation, reward, done, truncated, info = env.step(action)
                reward_sum += reward
                if frame_skip or (step + 1) % interval == 0 or done or truncated:
                    trace.append((observation.copy(), reward_sum, info['target_control']))
                    reward_sum = 0.0
                if done or truncated:
                    break
            traces.append(trace)
        finally:
            env.close()
    assert len(traces[0]) == len(traces[1])
    for training, replay in zip(*traces):
        np.testing.assert_array_equal(training[0], replay[0])
        assert training[1:] == replay[1:]
    label = 'fixed' if model is None else 'squashed PPO'
    print(f'PASS: {label}: training frame skip and frame-by-frame replay produce identical observations/rewards/actions.')


def check_squashed_policy():
    path = Path(__file__).resolve().parents[2] / 'configs/training/defense-target-hyperparams.json'
    params = load_hyperparams(str(path))
    args = _args()
    env = build_single_nhl94_env(args, params)
    try:
        policy, kwargs = build_MlpPolicy(256, params)
        model = PPO(policy, env, policy_kwargs=kwargs, n_steps=32, batch_size=16, n_epochs=2,
                    ent_coef=params['ent_coef'], learning_rate=params['learning_rate'], device='cpu', seed=0)
        assert model.policy.squash_output
        torch.testing.assert_close(model.policy.log_std.exp(), torch.full((2,), 0.5))
        observation, _ = env.reset(seed=0)
        observations = model.policy.obs_to_tensor(observation)[0].repeat(4096, 1)
        with torch.no_grad():
            actions, _, log_prob = model.policy(observations)
        boundary_fraction = (actions.abs() >= 0.98).float().mean().item()
        assert boundary_fraction < 0.01, ('Initial targets concentrate on the boundaries', boundary_fraction)
        assert torch.all(torch.isfinite(log_prob))
        model.learn(total_timesteps=128)
        assert np.all(np.abs(model.rollout_buffer.actions) <= 1)
        assert np.all(np.isfinite(model.rollout_buffer.log_probs))
        assert all(torch.all(torch.isfinite(parameter)) for parameter in model.policy.parameters())
        with tempfile.TemporaryDirectory() as directory:
            artifact = save_checkpoint(model, str(Path(directory) / 'target'), args, params)
            restored = load_policy(artifact, env=env, expected=run_metadata(args, params), device='cpu')
            np.testing.assert_array_equal(model.predict(observation, deterministic=True)[0],
                                          restored.predict(observation, deterministic=True)[0])
    finally:
        env.close()
    check_replay_timing(params, restored)
    print(f'PASS: squashed PPO: 128 ROM training decisions, save/reload; '
          f'initial target coordinates near boundaries={boundary_fraction:.2%}.')


def check_defensive_targets(params):
    env = build_single_nhl94_env(_args(), params)
    try:
        env.reset(seed=0)
        samples = [[x, y] for x in (-1, 0, 1) for y in (-1, 0, 1)]
        samples.extend(np.random.default_rng(0).normal(size=(256, 2)).clip(-1, 1))
        for index, action in enumerate(samples):
            _, _, done, truncated, info = env.step(np.asarray(action, dtype=np.float32))
            for field in ('target', 'destination'):
                x, y = info['target_control'][field]
                assert -120 <= x <= 120 and -270 <= y <= -88, (action, field, x, y)
            if done or truncated:
                env.reset(seed=index + 1)
        print('PASS: 265 real-ROM boundary/random target decisions stay inside the defensive zone.')
    finally:
        env.close()


def check_unselected_controller(params):
    """Inject a ROM selection gap at the same decoded-frame boundary as a step."""
    env = build_single_nhl94_env(_args(), params, use_frame_skip=False)
    try:
        _, info = env.reset(seed=0)
        memory = env.unwrapped.data.memory
        flags = 0xFFB04A + info['p1_control_slot'] * 0x80 + 0x62
        memory.assign(flags, '|u1', memory.extract(flags, '|u1') & ~0x08)
        memory.assign(0xFFC320, '>i2', -1)
        info = env.unwrapped.data.lookup_all()
        env.game_state.BeginFrame(info, [0] * 6)
        env.game_state.EndFrame()
        env.target_info = info
        env.target_controller.observe(env.game_state, info)
        gaps = 0
        for _ in range(2400):
            was_unselected = env.target_controller.actual_slot == -1
            observation, _, done, truncated, info = env.step(np.array([0.1, -0.65], dtype=np.float32))
            assert np.all(np.isfinite(observation)) and observation.shape == (322,)
            diagnostics = info['target_control']
            assert -270 <= diagnostics['target'][1] <= -88
            assert -270 <= diagnostics['destination'][1] <= -88
            if was_unselected:
                assert not any(diagnostics['buttons']), 'No buttons should execute without a selected player'
            if diagnostics['actual_slot'] == -1:
                gaps += 1
                assert diagnostics['distance'] is None
                assert not diagnostics['selection_available']
                assert env.game_state.team1.get_controlled_player() is None
            if done or truncated:
                break
        else:
            raise AssertionError('Selection-gap episode did not reach the normal task boundary')
        assert gaps > 0
        env.reset(seed=0)
        assert env.target_controller.actual_slot in range(5)
        print(f'PASS: injected ROM selection gap ({gaps} frames) completes normally without indexing a player.')
    finally:
        env.close()


def check_target_camera(params):
    """Check native screen projection against visible faceoff dots as the camera scrolls."""
    env = build_single_nhl94_env(_args(), params, use_frame_skip=False)
    cameras, checked = set(), 0
    try:
        assert env.unwrapped.data.crop_info() == (0, 0, 256, 224)
        for seed in range(12):
            env.reset(seed=seed)
            for _ in range(60):
                _, _, done, truncated, info = env.step([0, 0])
                frame = env.render()
                camera = info['target_camera_x'], info['target_camera_y']
                cameras.add(camera)
                state = get_game_state(env)
                objects = [state.puck] + [
                    player for team in (state.team1, state.team2) for player in (*team.players, team.goalie)
                ]
                for world_x in (-72, 72):
                    x, y = 128 + world_x - camera[0], 112 + 208 + camera[1]
                    if 8 <= x < 248 and 8 <= y < 180:
                        # Moving sprites can cover the painted dot without a projection error.
                        if any(abs(player.x - world_x) < 24 and abs(player.y + 208) < 40 for player in objects):
                            continue
                        pixels = frame[y - 2:y + 3, x - 2:x + 3].astype(int)
                        red = (pixels[:, :, 0] > 120) & (pixels[:, :, 0] > pixels[:, :, 1] + 40)
                        assert np.any(red), ('Faceoff dot does not match camera projection', camera, x, y)
                        checked += 1
                if done or truncated:
                    break
        assert len(cameras) >= 50 and checked >= 200, (len(cameras), checked)
        print(f'PASS: {checked} rendered faceoff dots match projection across {len(cameras)} camera positions.')
    finally:
        env.close()


def paired_defense(params, episodes, model_path=None):
    signatures = {}
    policies = ['passive', 'scripted-target']
    if model_path:
        policies.append('learned-target')
    for policy in policies:
        args = _args(action_type='FILTERED' if policy == 'passive' else 'TARGET_POSITION')
        env = build_single_nhl94_env(args, params, use_frame_skip=False)
        totals = dict(recovery=0, shot_failure=0, conceded=0, timeout=0, controlled=0, teammate=0,
                      goalie_catches=0, goalie_frames=0)
        returns, distances, shot_counts = [], [], []
        selection_gaps = 0
        try:
            model = load_policy(model_path, env=env, expected=run_metadata(args, params)) if policy == 'learned-target' else None
            for seed in range(episodes):
                observation, info = env.reset(seed=seed)
                state = get_game_state(env)
                positions = tuple((p.x, p.y) for team in (state.team1, state.team2) for p in (*team.players, team.goalie))
                if policy == 'passive':
                    signatures[seed] = positions
                else:
                    assert positions == signatures[seed], 'Paired policies must receive identical starts'
                total = 0.0
                shot_count = state.team2.stats.shots
                last_shots = shot_count
                previous_owner = info['puck_owner']
                opponent_score = info['p2_score']
                action = np.zeros(12, dtype=np.int8)
                for frame in range(2400):
                    acting_slot = env.unwrapped.data.memory.extract(0xFFC320, '>i2')
                    if frame % params.get('frame_skip', 4) == 0 and policy != 'passive':
                        action = model.predict(observation, deterministic=True)[0] if model else _scripted_target(state)
                    observation, reward, terminated, truncated, info = env.step(action)
                    state = get_game_state(env)
                    total += reward
                    owner = info['puck_owner']
                    shots = state.team2.stats.shots
                    shot = shots > last_shots
                    conceded = info['p2_score'] > opponent_score
                    expected_reward = -1.0 if shot or conceded else float(owner in range(5))
                    assert reward == expected_reward, (
                        'Reward must contain only possession, shots and goals',
                        policy, seed, frame, reward, expected_reward)
                    expected_done = shot or conceded or owner in range(5) or state.time < 200
                    assert terminated == expected_done, (
                        'Episode must stop on the first shot, goal or skater recovery, or at timeout',
                        policy, seed, frame, terminated, expected_done)
                    if owner == 5:
                        totals['goalie_catches'] += int(previous_owner != owner)
                        totals['goalie_frames'] += 1
                    previous_owner, last_shots = owner, shots
                    if policy != 'passive':
                        assert observation.shape == (322,)
                        diagnostics = info['target_control']
                        assert diagnostics['actual_slot'] == max(-1, env.unwrapped.data.memory.extract(0xFFC320, '>i2'))
                        assert state.team1.controlled_scnum() == diagnostics['actual_slot']
                        if diagnostics['distance'] is None:
                            selection_gaps += 1
                        else:
                            distances.append(diagnostics['distance'])
                    if terminated or truncated:
                        if conceded:
                            totals['conceded'] += 1
                        elif shot:
                            totals['shot_failure'] += 1
                        elif owner in range(5):
                            totals['recovery'] += 1
                            totals['controlled' if owner == acting_slot else 'teammate'] += 1
                        else:
                            totals['timeout'] += 1
                        break
                else:
                    raise AssertionError(f'Episode exceeded the defensive clock cutoff: seed={seed}')
                assert total in (-1.0, 0.0, 1.0), ('Unexpected episode return', policy, seed, total)
                shot_counts.append(state.team2.stats.shots - shot_count)
                returns.append(total)
            print(f'{policy}: {totals}; mean return={np.mean(returns):.3f}; '
                  f'mean shots={np.mean(shot_counts):.2f}'
                  + (f'; mean target error={np.mean(distances):.1f}' if distances else '')
                  + (f'; selection-gap frames={selection_gaps}' if policy != 'passive' else ''))
        finally:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', help='Optional target-policy checkpoint for evaluation only')
    parser.add_argument('--episodes', type=int, default=50)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error('--episodes must be positive')
    params = resolve_hyperparams_for_model(load_hyperparams(default_config_path('nhl94')), 'MlpPolicy')
    check_tracking(params)
    check_replay_timing(params)
    check_squashed_policy()
    check_defensive_targets(params)
    check_target_camera(params)
    check_unselected_controller(params)
    paired_defense(params, args.episodes, args.model)


if __name__ == '__main__':
    main()
