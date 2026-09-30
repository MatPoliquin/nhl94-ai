"""ROM-dependent V1 smoke checks and benchmark reproducibility; run explicitly."""
from pathlib import Path
import json

import numpy as np

from nhl94_ai.agents.base import AgentInput, configure_scripted_frames
from nhl94_ai.agents.registry import create_scripted
from nhl94_ai.config import load_hyperparams, resolve_hyperparams_for_model
from nhl94_ai.env.factory import build_single_nhl94_env
from nhl94_ai.evaluation.benchmark import match
from nhl94_ai.evaluation.play import parse_cmdline
from nhl94_ai.training.datasets import get_game_state


def smoke(game, schema):
    args = parse_cmdline(['--mode=model_vs_game', f'--env={game}', '--nn=ClassicAIV1',
                         '--rf=PostPlay', f'--action_type={schema}'])
    root = Path(__file__).resolve().parents[2]
    params = resolve_hyperparams_for_model(load_hyperparams(str(root / 'configs/training/nhl94.json')), args.nn)
    params['frame_skip'] = 4
    env = build_single_nhl94_env(args, params, use_sticky_action=False)
    try:
        observation, _ = env.reset(seed=7)
        agent = create_scripted('classic-v1', args, env)
        configure_scripted_frames(agent, env)
        goalie_moved = False
        for _ in range(300):
            state = get_game_state(env)
            for goalie in (state.team1.goalie, state.team2.goalie):
                assert goalie.motion_x is not None and goalie.motion_y is not None
                goalie_moved |= abs(goalie.motion_x) + abs(goalie.motion_y) > 0
            action = agent.act(AgentInput(state, observation)).action
            assert env.action_space.contains(action), (game, schema, action)
            observation, _, terminated, truncated, _ = env.step(action)
            assert np.isfinite(observation).all()
            if terminated or truncated:
                observation, _ = env.reset(seed=7)
                agent.reset()
        if game == 'NHL94-Genesis-v0':
            assert goalie_moved, (game, schema, 'Live goalie motion never changed')
        print(f'PASS: V1 {game} {schema}')
    finally:
        env.close()


def main():
    for game in ('NHL941on1-Genesis-v0', 'NHL942on2-Genesis-v0', 'NHL94-Genesis-v0'):
        smoke(game, 'FILTERED')
    smoke('NHL94-Genesis-v0', 'HOCKEY_INTENT_DPAD')
    first = match(('classic-v1', 'classic-v1-direct', 42, 60, 4))
    repeat = match(('classic-v1', 'classic-v1-direct', 42, 60, 4))
    other_seed = match(('classic-v1', 'classic-v1-direct', 43, 60, 4))
    assert first['completed'] and repeat['completed'] and other_seed['completed']
    for key in ('actions_sha256', 'goals', 'shots', 'frames', 'initial_state_sha256'):
        assert first[key] == repeat[key], key
    assert first['actions_sha256'] != other_seed['actions_sha256'], 'ROM seeds did not vary the trajectory'
    print('PASS: identical ROM seeds reproduce matches; different seeds change trajectories')
    direct = match(('classic-v1-direct', 'classic-v1', 2000, 300, 4))
    assert direct['completed'] and direct['one_timer_setups'][0] == 0
    print('PASS: disabling one-timers suppresses deliberate setups through a full period')
    one_timer = match(('classic-v1', 'classic-v1-direct', 3000, 300, 4))
    assert one_timer['completed'] and one_timer['one_timer_setups'][0] > 0
    print('PASS: V1 still attempts deliberate setups; execution is checked by the isolated one-timer replay')


if __name__ == '__main__':
    main()
