#!/usr/bin/env python3
"""Collect NHL94 ClassicAI demonstrations for imitation learning."""
from nhl94_ai.agents.registry import ALIASES, add_classic_arguments

from nhl94_ai.config import default_config_path
from nhl94_ai.model_inputs import add_model_input_arguments

import argparse
import json
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import numpy as np

from nhl94_ai.agents.registry import create_scripted
from nhl94_ai.env.intents import HOCKEY_INTENT_DPAD_ACTIONS
from nhl94_ai.training.datasets import build_single_nhl94_env, ensure_box_observation, get_game_state, normalize_reset_result, normalize_step_result, save_demo_shard
from nhl94_ai.config import load_hyperparams, resolve_clip_reward, resolve_hyperparams_for_model


def build_parser():
    parser = argparse.ArgumentParser(description="Collect NHL94 ClassicAI demonstration shards")
    parser.add_argument("--env", type=str, default="NHL94-Genesis-v0")
    parser.add_argument("--state", type=str, default=None)
    parser.add_argument("--rf", type=str, default="PostPlay")
    parser.add_argument("--nn", type=str, default="ResidualMlpPolicy")
    parser.add_argument("--alg", type=str, default="ppo2")
    parser.add_argument("--nnsize", type=int, default=256)
    parser.add_argument("--num_players", type=int, default=1)
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=4500)
    parser.add_argument("--output", type=str, default="~/OUTPUT/classic_ai_demos")
    parser.add_argument("--shard_name", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hyperparams", type=str, default=default_config_path("nhl94"))
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--action_type", type=str, default="HOCKEY_INTENT_DPAD", choices=["FILTERED", "DISCRETE", "MULTI_DISCRETE", "HOCKEY_INTENT_DPAD"])
    parser.add_argument("--opponent", type=str, default="game", choices=["game", "noop"])
    parser.add_argument("--no_frame_skip", default=False, action="store_true")
    parser.add_argument("--clip_reward", default=None, action="store_true")
    parser.add_argument("--no_clip_reward", dest="clip_reward", action="store_false")
    parser.add_argument("--agent", default="classic-v1", choices=list(ALIASES))
    return add_classic_arguments(add_model_input_arguments(parser))


def parse_cmdline(argv):
    return build_parser().parse_args(argv)


def _validate_args(args):
    if min(args.num_episodes, args.max_steps, args.num_workers) < 1:
        raise ValueError("num_episodes, max_steps, and num_workers must be positive")
    args.action_type = args.action_type.upper()
    if args.action_type not in ("FILTERED", "HOCKEY_INTENT_DPAD"):
        raise NotImplementedError("ClassicAI cloning currently supports FILTERED and HOCKEY_INTENT_DPAD.")
    if args.action_type == "HOCKEY_INTENT_DPAD" and args.opponent == "noop":
        raise ValueError("HOCKEY_INTENT_DPAD currently supports --opponent=game only.")
    if args.action_type == "HOCKEY_INTENT_DPAD" and args.num_players != 1:
        raise ValueError("HOCKEY_INTENT_DPAD currently expects --num_players=1.")
    if args.opponent == "noop" and args.num_players != 2:
        args.num_players = 2
    if args.opponent == "game" and args.num_players != 1:
        raise ValueError("--opponent=game expects --num_players=1 so the built-in game AI controls the opponent.")


def _make_output_path(args):
    output_dir = os.path.expanduser(args.output)
    os.makedirs(output_dir, exist_ok=True)
    if args.shard_name:
        shard_name = args.shard_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        shard_name = f"classic_ai_{args.env}_{args.rf}_{args.nn}_{timestamp}.npz"
    if not shard_name.endswith(".npz"):
        shard_name += ".npz"
    return os.path.join(output_dir, shard_name)


def _split_episode_ids(num_episodes, num_workers):
    worker_count = max(1, min(int(num_workers), int(num_episodes)))
    chunks = [[] for _ in range(worker_count)]
    for index, episode_id in enumerate(range(int(num_episodes))):
        chunks[index % worker_count].append(episode_id)
    return [chunk for chunk in chunks if chunk]


def _collect_episode_batch(args_payload, hyperparams, episode_ids, agent=None):
    args = argparse.Namespace(**args_payload)
    np.random.seed(args.seed + min(episode_ids))
    env = build_single_nhl94_env(
        args,
        hyperparams,
        num_players=args.num_players,
        use_sticky_action=False,
        use_frame_skip=not args.no_frame_skip,
    )
    expert = agent if agent is not None else create_scripted(getattr(args, 'agent', 'classic-v1'), args, env)
    from nhl94_ai.agents.base import AgentInput, configure_scripted_frames
    configure_scripted_frames(expert, env, record=True)
    clip_rewards = resolve_clip_reward(args, hyperparams)

    observations = []
    actions = []
    rewards = []
    dones = []
    sample_episode_ids = []
    steps = []
    decisions = []
    targets = []

    try:
        for episode_id in episode_ids:
            reset_result = env.reset(seed=args.seed + episode_id)
            obs, _ = normalize_reset_result(reset_result)
            expert.reset()
            sample_step = 0

            for step in range(args.max_steps):
                obs_array = ensure_box_observation(obs, args.nn)
                game_state = get_game_state(env)
                result = expert.act(AgentInput(game_state, obs), deterministic=True)
                action = np.asarray(result.action, dtype=np.int8)

                obs, reward, done, info = normalize_step_result(env.step(action))
                samples = info.get('scripted_frames', [(obs_array, action, reward, done, result.diagnostics)])
                for before, emitted, frame_reward, frame_done, diagnostics in samples:
                    observations.append(before.copy())
                    actions.append(emitted.copy())
                    sample_episode_ids.append(episode_id)
                    steps.append(sample_step)
                    sample_step += 1
                    decisions.append(str(diagnostics.get('decision', '')))
                    targets.append(np.asarray(diagnostics.get('target', (0, 0)), dtype=np.float32))
                    rewards.append(float(np.clip(frame_reward, -1, 1)) if clip_rewards
                                   else float(frame_reward))
                    dones.append(bool(frame_done))

                if done:
                    break
    finally:
        env.close()

    if not observations:
        raise RuntimeError("No demonstrations were collected.")

    return {
        "observations": np.stack(observations).astype(np.float32),
        "actions": np.stack(actions).astype(np.int8),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "dones": np.asarray(dones, dtype=np.bool_),
        "episode_ids": np.asarray(sample_episode_ids, dtype=np.int64),
        "steps": np.asarray(steps, dtype=np.int64),
        "decisions": np.asarray(decisions, dtype="U64"),
        "targets": np.stack(targets).astype(np.float32),
    }


def _concat_arrays(shards):
    keys = shards[0].keys()
    return {key: np.concatenate([shard[key] for shard in shards], axis=0) for key in keys}


def collect_demos(args, hyperparams, agent=None):
    _validate_args(args)
    if agent is not None and args.num_workers != 1:
        raise ValueError("Agent instances require num_workers=1; use a registered agent name for parallel collection")
    worker_count = max(1, int(getattr(args, "num_workers", 1)))

    episode_chunks = _split_episode_ids(args.num_episodes, worker_count)
    if len(episode_chunks) == 1:
        arrays = _collect_episode_batch(vars(args), hyperparams, episode_chunks[0], agent=agent)
    else:
        print(f"Collecting {args.num_episodes} episodes with {len(episode_chunks)} workers", flush=True)
        arrays_by_chunk = []
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=len(episode_chunks), mp_context=context) as executor:
            futures = [
                executor.submit(_collect_episode_batch, vars(args), hyperparams, chunk)
                for chunk in episode_chunks
            ]
            for future in as_completed(futures):
                arrays_by_chunk.append(future.result())
        arrays_by_chunk.sort(key=lambda shard: int(shard["episode_ids"].min()))
        arrays = _concat_arrays(arrays_by_chunk)

    arrays = {
        key: arrays[key]
        for key in (
            "observations",
            "actions",
            "rewards",
            "dones",
            "episode_ids",
            "steps",
            "decisions",
            "targets",
        )
    }
    action_counts = arrays["actions"].sum(axis=0).astype(int).tolist()
    metadata = {
        "env": args.env,
        "state": args.state,
        "rf": args.rf,
        "nn": args.nn,
        "action_type": args.action_type,
        "opponent": args.opponent,
        "num_players": args.num_players,
        "num_episodes_requested": args.num_episodes,
        "num_workers": len(episode_chunks),
        "num_samples": int(arrays["actions"].shape[0]),
        "observation_shape": list(arrays["observations"].shape[1:]),
        "action_shape": list(arrays["actions"].shape[1:]),
        "action_counts": action_counts,
        "button_counts": action_counts if args.action_type == "FILTERED" else None,
    }
    if args.action_type == "HOCKEY_INTENT_DPAD":
        metadata["intent_counts"] = np.bincount(arrays["actions"][:, 0], minlength=len(HOCKEY_INTENT_DPAD_ACTIONS)).astype(int).tolist()
    from nhl94_ai.artifacts import run_metadata
    from nhl94_ai.agents.base import ScriptedAgent
    if agent is None or isinstance(agent, ScriptedAgent):
        metadata['teacher_decision_interval'] = 1 if args.no_frame_skip else int(hyperparams.get('frame_skip', 4))
        metadata['sample_interval_frames'] = 1
        metadata.update(run_metadata(args, dict(hyperparams, frame_skip=1)))
    else:
        metadata.update(run_metadata(args, hyperparams))
    return arrays, metadata


def main(argv):
    return run(parse_cmdline(argv[1:]))


def run(args):
    hyperparams = resolve_hyperparams_for_model(load_hyperparams(
        args.hyperparams,
        required=True,
        base_dir=os.path.dirname(__file__),
    ), args.nn)
    arrays, metadata = collect_demos(args, hyperparams)
    output_path = _make_output_path(args)
    save_demo_shard(output_path, arrays, metadata)
    print(json.dumps({"saved": output_path, **metadata}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv)


def collect_demonstrations(agent, config):
    from nhl94_ai.config import RunConfig, namespace_from_mapping
    from nhl94_ai.artifacts import Dataset
    if isinstance(config, (dict, RunConfig)):
        values = config.options if isinstance(config, RunConfig) else config
        config = namespace_from_mapping(build_parser(), values)
    hyperparams = resolve_hyperparams_for_model(load_hyperparams(config.hyperparams), config.nn)
    arrays, metadata = collect_demos(config, hyperparams, agent=agent)
    path = _make_output_path(config)
    save_demo_shard(path, arrays, metadata)
    return Dataset(path, metadata)
