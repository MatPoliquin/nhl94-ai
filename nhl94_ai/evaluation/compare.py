#!/usr/bin/env python3
"""Compare two models by training them and recording a side-by-side video."""

from nhl94_ai.artifacts import load_policy
from nhl94_ai.evaluation.frames import get_env_frame, prepare_action_info

import argparse
import copy
import io
import json
import math
import os
import time
from contextlib import redirect_stdout
from typing import List, Optional, Tuple

# Hide the pygame prompt before importing pygame
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import cv2  # type: ignore
import numpy as np  # type: ignore
from stable_baselines3 import PPO  # type: ignore
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # type: ignore

from nhl94_ai.training.logging import init_logger
from nhl94_ai.env.factory import init_env, get_button_names
from nhl94_ai.models.factory import get_num_parameters, get_model_probabilities
from torchsummary import summary
import nhl94_ai.env.components as games
import nhl94_ai.training.rl as train
from nhl94_ai.config import load_hyperparams, resolve_hyperparams_for_model


DEFAULT_OUTPUT_DIR = os.path.expanduser("~/OUTPUT/compare_models")
DEFAULT_FPS = 60
DEFAULT_DURATION_SECONDS = 120
from nhl94_ai.config import default_config_path
DEFAULT_HYPERPARAMS = default_config_path()
DEFAULT_WINDOW_WIDTH = 1920
DEFAULT_WINDOW_HEIGHT = 1080
DEFAULT_MARGIN = 20
DEFAULT_FOOTER_HEIGHT = 120
DEFAULT_FONT = "Arial"
BUTTON_LABEL_OVERRIDES = {
    "UP": "↑",
    "DOWN": "↓",
    "LEFT": "←",
    "RIGHT": "→",
    "START": "S",
    "MODE": "M",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train two models and record a side-by-side comparison video.")

    parser.add_argument("--env", type=str, default="NHL941on1-Genesis-v0",
                        help="Retro environment identifier (default: NHL941on1-Genesis-v0)")
    parser.add_argument("--state", type=str, default=None,
                        help="Specific environment state to load (default: None)")
    parser.add_argument("--action-type", type=str, default="FILTERED",
                        choices=["FILTERED", "DISCRETE", "MULTI_DISCRETE", "HOCKEY_INTENT_DPAD"],
                        help="Action space type to use (default: FILTERED)")
    parser.add_argument("--timesteps", type=int, default=1_000_000,
                        help="Number of environment timesteps to train each model (default: 1_000_000)")
    parser.add_argument("--train-num-env", type=int, default=8,
                        help="Number of parallel environments to use during training (default: 8)")
    parser.add_argument("--model1-policy", type=str, default="CnnPolicy",
                        help="Policy class name for model 1 (default: CnnPolicy)")
    parser.add_argument("--model2-policy", type=str, default="MlpPolicy",
                        help="Policy class name for model 2 (default: MlpPolicy)")
    parser.add_argument("--model1-seq-len", type=int, default=16,
                        help="Sequence length for model 1 when using temporal policies")
    parser.add_argument("--model2-seq-len", type=int, default=16,
                        help="Sequence length for model 2 when using temporal policies")
    parser.add_argument("--model1-hyperparams", type=str, default=DEFAULT_HYPERPARAMS,
                        help="Path to hyperparameter JSON for model 1")
    parser.add_argument("--model2-hyperparams", type=str, default=DEFAULT_HYPERPARAMS,
                        help="Path to hyperparameter JSON for model 2")
    parser.add_argument("--video-output", type=str, default="model_comparison.mp4",
                        help="Output path for the comparison video (default: model_comparison.mp4)")
    parser.add_argument("--video-duration", type=float, default=DEFAULT_DURATION_SECONDS,
                        help="Length of the recorded video in seconds (default: 120)")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS,
                        help="Frames per second for the recorded video (default: 60)")
    parser.add_argument("--window-width", type=int, default=DEFAULT_WINDOW_WIDTH,
                        help="Width of the comparison window (default: 1920)")
    parser.add_argument("--window-height", type=int, default=DEFAULT_WINDOW_HEIGHT,
                        help="Height of the comparison window (default: 1080)")
    parser.add_argument("--display-height", type=int, dest="window_height",
                        help="Deprecated alias for --window-height")
    parser.add_argument("--output-basedir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Base directory for training artifacts (default: ~/OUTPUT/compare_models)")
    parser.add_argument("--algorithm", type=str, default="ppo2",
                        help="Training algorithm identifier (PPO, PPO2, ES - case-insensitive; default: ppo2)")

    parser.add_argument("--model1-label", type=str, default=None,
                        help="Custom label to display under model 1 (default: policy name)")
    parser.add_argument("--model2-label", type=str, default=None,
                        help="Custom label to display under model 2 (default: policy name)")
    parser.add_argument("--summary-duration", type=float, default=20.0,
                        help="Duration in seconds of the pre-roll model summary segment (default: 20)")
    parser.add_argument("--hyperparams-duration", type=float, default=10.0,
                        help="Duration in seconds of the pre-roll hyperparameters segment (default: 10)")
    parser.add_argument("--reward-curve-duration", type=float, default=12.0,
                        help="Duration in seconds of the pre-roll reward curve segment (default: 12)")

    return parser.parse_args()


def ensure_hyperparams(path: str) -> str:
    abs_path = os.path.abspath(path)
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(f"Hyperparameters file not found: {abs_path}")
    return abs_path


def normalize_algorithm(name: str) -> str:
    if not name:
        return "ppo2"

    lowered = name.strip().lower()

    if lowered in {"ppo", "ppo2"}:
        return "ppo2"
    if lowered in {"es"}:
        return "es"

    raise ValueError(
        f"Unsupported algorithm '{name}'. Supported values are: PPO, PPO2, ES."
    )


def build_train_args(base, policy, hyperparams_path, run_id, seq_len):
    from nhl94_ai.config import namespace_from_mapping
    args = namespace_from_mapping(train.build_parser(), {
        'alg': base.algorithm, 'nn': policy, 'seq_len': seq_len, 'env': base.env,
        'num_timesteps': base.timesteps, 'num_env': base.train_num_env,
        'output_basedir': os.path.join(base.output_basedir, run_id),
        'hyperparams': hyperparams_path, 'action_type': base.action_type,
        'display_width': base.window_width, 'display_height': base.window_height,
        'state': base.state, 'num_players': 1, 'play': False, 'selfplay': False,
    })
    args.model1_desc = base.model1_label or policy
    args.model2_desc = base.model2_label or policy
    return args


def train_model(train_args: argparse.Namespace) -> Tuple[str, argparse.Namespace, str]:
    os.makedirs(os.path.expanduser(train_args.output_basedir), exist_ok=True)
    logger = init_logger(train_args)
    log_dir = getattr(logger, "dir", None)
    trainer = train.ModelTrainer(train_args, logger)
    try:
        model_save_root = trainer.train()
    finally:
        trainer.env.close()
    if hasattr(trainer.p1_model, "env") and trainer.p1_model.env is not None:
        trainer.p1_model.env.close()
    model_path = f"{model_save_root}.zip"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Expected trained model at {model_path}, but it was not created.")
    return model_path, train_args, log_dir or trainer.output_fullpath


def generate_model_summary(model: PPO, env, policy_name: str) -> str:
    obs_space = getattr(model, "observation_space", None) or env.observation_space

    try:
        if policy_name in ("MlpPolicy", "CustomMlpPolicy", "MlpDropoutPolicy", "CombinedPolicy", "AttentionMLPPolicy", "HockeyMultiHeadPolicy", "ResidualMlpPolicy", "HybridMambaPolicy", "GRUMlpPolicy"):
            if hasattr(obs_space, "shape") and obs_space.shape:
                input_shape = tuple(obs_space.shape)
            else:
                return f"Model summary unavailable for policy {policy_name}: unsupported observation space"
        elif policy_name in ("CnnPolicy", "ImpalaCnnPolicy", "CustomCnnPolicy", "CnnTransformerPolicy", "ViTPolicy", "DartPolicy"):
            if hasattr(obs_space, "shape") and len(obs_space.shape) >= 3:
                input_shape = (obs_space.shape[-3], obs_space.shape[-2], obs_space.shape[-1])
            else:
                return f"Model summary unavailable for policy {policy_name}: unsupported observation space"
        else:
            return f"Model summary unavailable for policy {policy_name}"

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            summary(model.policy, input_shape)
        return buffer.getvalue()
    except Exception as exc:  # pylint: disable=broad-except
        return f"Model summary unavailable: {exc}"


def load_hyperparams_text(hyperparams_path: str) -> str:
    try:
        with open(hyperparams_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return json.dumps(data, indent=2)
    except Exception as exc:  # pylint: disable=broad-except
        return f"Unable to load hyperparameters: {exc}"


def load_reward_series(log_dir: str, tag: str = "rollout/ep_rew_mean") -> List[Tuple[float, float]]:
    if not log_dir or not os.path.isdir(log_dir):
        return []

    try:
        accumulator = EventAccumulator(log_dir, size_guidance={"scalars": 0})
        accumulator.Reload()
        scalar_tags = accumulator.Tags().get("scalars", [])
        if tag not in scalar_tags:
            return []
        events = accumulator.Scalars(tag)
        return [(event.step, event.value) for event in events]
    except Exception:  # pylint: disable=broad-except
        return []


def extract_final_reward(reward_series: List[Tuple[float, float]]) -> float:
    for _, value in reversed(reward_series):
        if value is None:
            continue
        if isinstance(value, (int, float)) and not math.isnan(float(value)):
            return float(value)
    return 0.0


def init_play_env(train_args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, any]:
    play_args = copy.deepcopy(train_args)
    play_args.num_env = 1
    play_args.num_players = 1
    play_args.output_basedir = None
    play_args.alg_verbose = False
    play_args.info_verbose = False

    games.get_bindings(play_args)
    hyperparams = resolve_hyperparams_for_model(load_hyperparams(
        getattr(play_args, "hyperparams", None),
        required=True,
        base_dir=os.path.dirname(__file__),
    ), play_args.nn)
    env = init_env(
        None,
        1,
        play_args.state,
        play_args.num_players,
        play_args,
        hyperparams,
        use_sticky_action=False,
        use_display=False,
        use_frame_skip=False,
    )
    obs = env.reset()
    frame = get_env_frame(env)
    return obs, frame, env


def load_trained_model(model_path: str, env) -> PPO:
    model = load_policy(model_path, env=env)
    return model


def main() -> None:
    from nhl94_ai.ui.comparison import create_reward_curve_surface, play_and_record
    args = parse_args()

    try:
        args.algorithm = normalize_algorithm(args.algorithm)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    args.model1_hyperparams = ensure_hyperparams(args.model1_hyperparams)
    args.model2_hyperparams = ensure_hyperparams(args.model2_hyperparams)

    os.makedirs(args.output_basedir, exist_ok=True)

    print("--- Training Model 1 ---")
    train_args1 = build_train_args(
        args,
        args.model1_policy,
        args.model1_hyperparams,
        "model1",
        args.model1_seq_len,
    )
    model_path1, trained_args1, log_dir1 = train_model(train_args1)
    label1 = args.model1_label or args.model1_policy

    print("--- Training Model 2 ---")
    train_args2 = build_train_args(
        args,
        args.model2_policy,
        args.model2_hyperparams,
        "model2",
        args.model2_seq_len,
    )
    model_path2, trained_args2, log_dir2 = train_model(train_args2)
    label2 = args.model2_label or args.model2_policy

    print("--- Preparing environments ---")
    obs1, initial_frame1, env1 = init_play_env(trained_args1)
    obs2, initial_frame2, env2 = init_play_env(trained_args2)

    button_names = list(get_button_names(trained_args1))

    model1 = load_trained_model(model_path1, env1)
    model2 = load_trained_model(model_path2, env2)

    summary_text1 = generate_model_summary(model1, env1, trained_args1.nn)
    summary_text2 = generate_model_summary(model2, env2, trained_args2.nn)
    hyperparams_text1 = load_hyperparams_text(trained_args1.hyperparams)
    hyperparams_text2 = load_hyperparams_text(trained_args2.hyperparams)
    reward_series1 = load_reward_series(log_dir1)
    reward_series2 = load_reward_series(log_dir2)

    params1 = get_num_parameters(model1)
    params2 = get_num_parameters(model2)
    algo_display1 = getattr(trained_args1, "alg", args.algorithm)
    algo_display2 = getattr(trained_args2, "alg", args.algorithm)
    final_training_reward1 = extract_final_reward(reward_series1)
    final_training_reward2 = extract_final_reward(reward_series2)

    print(f"--- Recording video to {args.video_output} ---")
    try:
        play_and_record(
            model1,
            env1,
            obs1,
            label1,
            params1,
            algo_display1,
            final_training_reward1,
            model2,
            env2,
            obs2,
            label2,
            params2,
            algo_display2,
            final_training_reward2,
            button_names,
            reward_series1,
            reward_series2,
            args.reward_curve_duration,
            hyperparams_text1,
            hyperparams_text2,
            args.hyperparams_duration,
            summary_text1,
            summary_text2,
            args.summary_duration,
            initial_frame1,
            initial_frame2,
            args.video_output,
            args.video_duration,
            args.fps,
            args.window_width,
            args.window_height,
        )
    finally:
        env1.close()
        env2.close()

    print("Comparison video created successfully.")


if __name__ == "__main__":
    main()
