"""Bounded headless matches for any gameplay agent."""
from nhl94_ai.agents.registry import ALIASES
from dataclasses import asdict, dataclass
import argparse
import json
from itertools import count
import numpy as np
from nhl94_ai.agents.base import AgentInput, LearnedAgent, configure_scripted_frames
from nhl94_ai.agents.registry import create_scripted
from nhl94_ai.artifacts import load_policy, run_metadata
from nhl94_ai.config import EvaluationConfig, default_config_path, load_hyperparams, resolve_hyperparams_for_model
from nhl94_ai.env.factory import build_single_nhl94_env
from nhl94_ai.training.datasets import get_game_state


@dataclass
class EvaluationResult:
    rewards: list
    lengths: list
    terminations: list
    truncations: list
    budget_exhausted: list
    metadata: dict


def evaluate(agent, config, *, env):
    configure_scripted_frames(agent, env)
    rewards, lengths, terminations, truncations, exhausted = [], [], [], [], []
    defense_episodes = []
    for episode in range(config.episodes):
        observation, _ = env.reset(seed=config.seed + episode)
        agent.reset()
        total = 0.0
        last_defense = {}
        for step in (range(config.max_steps) if config.max_steps is not None else count()):
            action = agent.act(AgentInput(get_game_state(env), observation), config.deterministic).action
            observation, reward, terminated, truncated, info = env.step(action)
            if info.get('classic_defense'):
                last_defense = info['classic_defense']
            total += float(reward)
            if terminated or truncated:
                break
        rewards.append(total)
        lengths.append(step + 1)
        terminations.append(bool(terminated))
        truncations.append(bool(truncated))
        exhausted.append(not (terminated or truncated))
        defense_episodes.append(last_defense)
    metadata = asdict(config)
    if any(defense_episodes):
        metadata['last_defense_by_episode'] = defense_episodes
    return EvaluationResult(rewards, lengths, terminations, truncations, exhausted, metadata)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--agent', default='classic-v1', choices=list(ALIASES))
    parser.add_argument('--model', default='')
    parser.add_argument('--env', default='NHL94-Genesis-v0')
    parser.add_argument('--state', default=None)
    parser.add_argument('--nn', default='MlpPolicy')
    parser.add_argument('--rf', default='PostPlay')
    parser.add_argument('--action_type', default='FILTERED')
    parser.add_argument('--num_players', type=int, default=1)
    parser.add_argument('--seq_len', type=int, default=16)
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--max_steps', type=int, default=4500)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hyperparams', default=default_config_path('nhl94'))
    parser.add_argument('--output', default=None)
    return parser


def run(args):
    if args.action_type.upper() == 'TARGET_POSITION' and not args.model:
        raise ValueError('TARGET_POSITION evaluation requires --model with a target-policy checkpoint')
    config = EvaluationConfig(args.episodes, args.max_steps, args.seed)
    params = resolve_hyperparams_for_model(load_hyperparams(args.hyperparams), args.nn)
    env = build_single_nhl94_env(args, params)
    try:
        agent = (LearnedAgent(load_policy(args.model, env=env, expected=run_metadata(args, params)), args.action_type)
                 if args.model else create_scripted(args.agent, args, env))
        result = evaluate(agent, config, env=env)
        result.metadata.update(run_metadata(args, params))
        payload = json.dumps(asdict(result), indent=2)
        if args.output:
            from pathlib import Path
            path = Path(args.output).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(payload + '\n', encoding='utf-8')
        print(payload)
        return result
    finally:
        env.close()
