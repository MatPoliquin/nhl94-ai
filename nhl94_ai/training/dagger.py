#!/usr/bin/env python3
"""Run DAgger rounds with ClassicAI as the NHL94 oracle."""
from nhl94_ai.agents.registry import ALIASES, add_classic_arguments

from nhl94_ai.artifacts import load_policy

from nhl94_ai.config import default_config_path
from nhl94_ai.model_inputs import add_model_input_arguments

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import numpy as np
from stable_baselines3 import PPO

from nhl94_ai.agents.registry import create_scripted
from nhl94_ai.game.constants import GameConsts
from nhl94_ai.env.intents import HOCKEY_INTENT_DPAD_ACTIONS, HOCKEY_INTENT_DPAD_ACTION_SPACE
from nhl94_ai.training.datasets import build_single_nhl94_env, ensure_box_observation, get_game_state, normalize_reset_result, normalize_step_result, save_demo_shard
from nhl94_ai.config import load_hyperparams, resolve_hyperparams_for_model


def build_parser():
    parser = argparse.ArgumentParser(description="Run ClassicAI DAgger aggregation rounds")
    parser.add_argument("--model", required=True, help="Initial BC checkpoint")
    parser.add_argument("--base_datasets", nargs="+", required=True, help="Original expert demo shards/dirs/globs")
    parser.add_argument("--output_dir", type=str, default="~/OUTPUT/classic_ai_dagger")
    parser.add_argument("--env", type=str, default="NHL94-Genesis-v0")
    parser.add_argument("--state", type=str, default=None)
    parser.add_argument("--rf", type=str, default="PostPlay")
    parser.add_argument("--nn", type=str, default="ResidualMlpPolicy")
    parser.add_argument("--alg", type=str, default="ppo2")
    parser.add_argument("--nnsize", type=int, default=256)
    parser.add_argument("--num_players", type=int, default=1)
    parser.add_argument("--hyperparams", type=str, default=default_config_path("nhl94"))
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--action_type", type=str, default="HOCKEY_INTENT_DPAD", choices=["FILTERED", "DISCRETE", "MULTI_DISCRETE", "HOCKEY_INTENT_DPAD"])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--episodes_per_round", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=4500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bc_epochs", type=int, default=3)
    parser.add_argument("--bc_batch_size", type=int, default=1024)
    parser.add_argument("--bc_learning_rate", type=float, default=1e-4)
    parser.add_argument("--rare_weight", type=float, default=4.0)
    parser.add_argument("--positive_weight_max", type=float, default=12.0)
    parser.add_argument("--release_weight", type=float, default=8.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--skip_retrain", default=False, action="store_true")
    parser.add_argument("--no_frame_skip", default=False, action="store_true")
    parser.add_argument("--clip_reward", default=None, action="store_true")
    parser.add_argument("--no_clip_reward", dest="clip_reward", action="store_false")
    parser.add_argument("--agent", default="classic-v1", choices=list(ALIASES))
    return add_classic_arguments(add_model_input_arguments(parser))


def parse_cmdline(argv):
    return build_parser().parse_args(argv)


def _contradictory(action, a, b):
    return bool(action[a] and action[b])


def is_plausible_expert_action(action):
    if len(action) == len(HOCKEY_INTENT_DPAD_ACTION_SPACE):
        action = np.asarray(action, dtype=np.int64)
        if np.any(action < 0):
            return False
        if np.any(action >= np.asarray(HOCKEY_INTENT_DPAD_ACTION_SPACE, dtype=np.int64)):
            return False
        if _contradictory(action, 1, 2):
            return False
        if _contradictory(action, 3, 4):
            return False
        return True

    if len(action) != GameConsts.INPUT_MAX:
        return False
    if _contradictory(action, GameConsts.INPUT_UP, GameConsts.INPUT_DOWN):
        return False
    if _contradictory(action, GameConsts.INPUT_LEFT, GameConsts.INPUT_RIGHT):
        return False
    return True


def _make_round_paths(args, round_index):
    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    shard_path = os.path.join(output_dir, f"classic_ai_dagger_round_{round_index:02d}.npz")
    model_path = os.path.join(output_dir, f"classic_ai_dagger_round_{round_index:02d}.zip")
    return shard_path, model_path


def _split_episode_ids(num_episodes, num_workers):
    worker_count = max(1, min(int(num_workers), int(num_episodes)))
    chunks = [[] for _ in range(worker_count)]
    for index, episode_id in enumerate(range(int(num_episodes))):
        chunks[index % worker_count].append(episode_id)
    return [chunk for chunk in chunks if chunk]


def _concat_arrays(shards):
    keys = shards[0].keys()
    return {key: np.concatenate([shard[key] for shard in shards], axis=0) for key in keys}


def _collect_dagger_episode_batch(args_payload, hyperparams, model_path, round_index, episode_ids):
    args = argparse.Namespace(**args_payload)
    env = build_single_nhl94_env(
        args,
        hyperparams,
        num_players=args.num_players,
        use_sticky_action=False,
        use_frame_skip=not args.no_frame_skip,
    )
    try:
        model = load_policy(os.path.expanduser(model_path), env=env, device=args.device)
        expert = create_scripted(getattr(args, 'agent', 'classic-v1'), args, env)
        from nhl94_ai.agents.base import configure_scripted_frames
        configure_scripted_frames(expert, env, record=True, teacher_only=True)
        from nhl94_ai.config import resolve_clip_reward
        clip_rewards = resolve_clip_reward(args, hyperparams)
    except BaseException:
        env.close()
        raise

    observations = []
    expert_actions = []
    clone_actions = []
    rewards = []
    dones = []
    sample_episode_ids = []
    steps = []
    decisions = []
    dropped = 0

    try:
        for episode_id in episode_ids:
            reset_result = env.reset(seed=args.seed + round_index * 10000 + episode_id)
            obs, _ = normalize_reset_result(reset_result)
            expert.reset()
            sample_step = 0

            for step in range(args.max_steps):
                obs_array = ensure_box_observation(obs, args.nn)
                game_state = get_game_state(env)
                expert_action = np.asarray(expert.predict_game_state(game_state)[0], dtype=np.int8)
                clone_action = np.asarray(model.predict(obs, deterministic=True)[0], dtype=np.int8)
                diagnostics = dict(expert.last_output.diagnostics, teacher_action=expert_action)
                obs, reward, done, info = normalize_step_result(env.step(clone_action))
                samples = info.get('scripted_frames', [(obs_array, clone_action, reward, done, diagnostics)])
                for before, emitted, frame_reward, frame_done, details in samples:
                    label = details['teacher_action']
                    if is_plausible_expert_action(label):
                        observations.append(before.copy())
                        expert_actions.append(label.copy())
                        clone_actions.append(emitted.copy())
                        rewards.append(float(np.clip(frame_reward, -1, 1)) if clip_rewards
                                       else float(frame_reward))
                        dones.append(bool(frame_done))
                        sample_episode_ids.append(episode_id)
                        steps.append(sample_step)
                        decisions.append(str(details.get('decision', '')))
                    else:
                        dropped += 1
                    sample_step += 1

                if done:
                    break
    finally:
        env.close()

    if not observations:
        raise RuntimeError("No DAgger examples were collected.")

    return {
        "arrays": {
            "observations": np.stack(observations).astype(np.float32),
            "actions": np.stack(expert_actions).astype(np.int8),
            "clone_actions": np.stack(clone_actions).astype(np.int8),
            "rewards": np.asarray(rewards, dtype=np.float32),
            "dones": np.asarray(dones, dtype=np.bool_),
            "episode_ids": np.asarray(sample_episode_ids, dtype=np.int64),
            "steps": np.asarray(steps, dtype=np.int64),
            "decisions": np.asarray(decisions, dtype="U64"),
        },
        "dropped": dropped,
    }


def collect_dagger_round(args, hyperparams, model_path, round_index):
    worker_count = max(1, int(getattr(args, "num_workers", 1)))
    episode_chunks = _split_episode_ids(args.episodes_per_round, worker_count)

    if len(episode_chunks) == 1:
        result = _collect_dagger_episode_batch(vars(args), hyperparams, model_path, round_index, episode_chunks[0])
        arrays = result["arrays"]
        dropped = int(result["dropped"])
    else:
        print(
            f"Collecting DAgger round {round_index} with {len(episode_chunks)} workers",
            flush=True,
        )
        results = []
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=len(episode_chunks), mp_context=context) as executor:
            futures = [
                executor.submit(
                    _collect_dagger_episode_batch,
                    vars(args),
                    hyperparams,
                    model_path,
                    round_index,
                    chunk,
                )
                for chunk in episode_chunks
            ]
            for future in as_completed(futures):
                results.append(future.result())

        results.sort(key=lambda result: int(result["arrays"]["episode_ids"].min()))
        arrays = _concat_arrays([result["arrays"] for result in results])
        dropped = sum(int(result["dropped"]) for result in results)

    action_counts = arrays["actions"].sum(axis=0).astype(int).tolist()
    metadata = {
        "round": round_index,
        "source_model": model_path,
        "env": args.env,
        "state": args.state,
        "rf": args.rf,
        "nn": args.nn,
        "action_type": args.action_type,
        "num_workers": len(episode_chunks),
        "num_samples": int(arrays["actions"].shape[0]),
        "dropped_implausible": int(dropped),
        "observation_shape": list(arrays["observations"].shape[1:]),
        "action_shape": list(arrays["actions"].shape[1:]),
        "action_counts": action_counts,
        "button_counts": action_counts if args.action_type == "FILTERED" else None,
    }
    if args.action_type == "HOCKEY_INTENT_DPAD":
        metadata["intent_counts"] = np.bincount(arrays["actions"][:, 0], minlength=len(HOCKEY_INTENT_DPAD_ACTIONS)).astype(int).tolist()
    from nhl94_ai.artifacts import run_metadata
    metadata.update(run_metadata(args, dict(hyperparams, frame_skip=1)))
    metadata['sample_interval_frames'] = 1
    metadata['teacher_decision_interval'] = 1 if args.no_frame_skip else int(hyperparams.get('frame_skip', 4))
    return arrays, metadata


def run_bc_retrain(args, current_model, datasets, output_model):
    from nhl94_ai.training import bc
    from nhl94_ai.config import namespace_from_mapping
    names = ('env', 'state', 'rf', 'nn', 'alg', 'nnsize', 'num_players', 'hyperparams',
             'seq_len', 'action_type', 'rare_weight', 'positive_weight_max',
             'release_weight', 'seed', 'device', 'no_frame_skip', 'clip_reward')
    values = {name: getattr(args, name) for name in names}
    values.update(datasets=datasets, load_model=current_model, output_model=output_model,
                  epochs=args.bc_epochs, batch_size=args.bc_batch_size,
                  learning_rate=args.bc_learning_rate)
    bc_args = namespace_from_mapping(bc.build_parser(), values)
    params = resolve_hyperparams_for_model(load_hyperparams(args.hyperparams), args.nn)
    return bc.train_bc(bc_args, params)


def run_dagger(args, hyperparams):
    if min(args.rounds, args.episodes_per_round, args.max_steps, args.num_workers) < 1:
        raise ValueError("rounds, episodes_per_round, max_steps, and num_workers must be positive")
    args.action_type = args.action_type.upper()
    if args.action_type not in ("FILTERED", "HOCKEY_INTENT_DPAD"):
        raise NotImplementedError("DAgger currently supports FILTERED and HOCKEY_INTENT_DPAD.")

    current_model = os.path.expanduser(args.model)
    all_datasets = [os.path.expanduser(path) for path in args.base_datasets]
    summary = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "initial_model": current_model,
        "rounds": [],
    }

    for round_index in range(1, args.rounds + 1):
        shard_path, next_model = _make_round_paths(args, round_index)
        arrays, metadata = collect_dagger_round(args, hyperparams, current_model, round_index)
        save_demo_shard(shard_path, arrays, metadata)
        all_datasets.append(shard_path)

        if args.skip_retrain:
            trained_model = current_model
        else:
            trained_model = run_bc_retrain(args, current_model, all_datasets, next_model)
            current_model = trained_model

        row = {"shard": shard_path, "model": trained_model, **metadata}
        summary["rounds"].append(row)
        print(json.dumps(row, indent=2, sort_keys=True))

    summary["final_model"] = current_model
    summary_path = os.path.join(os.path.expanduser(args.output_dir), "dagger_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps({"summary": summary_path, "final_model": current_model}, indent=2, sort_keys=True))
    return current_model


def main(argv):
    return run(parse_cmdline(argv[1:]))


def run(args):
    hyperparams = resolve_hyperparams_for_model(load_hyperparams(
        args.hyperparams,
        required=True,
        base_dir=os.path.dirname(__file__),
    ), args.nn)
    run_dagger(args, hyperparams)


if __name__ == "__main__":
    main(sys.argv)
