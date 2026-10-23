"""Legacy run options retained for configuration and existing-model evaluation."""
import argparse
import os
from typing import Optional, Sequence
from nhl94_ai.config import default_config_path, load_hyperparams, resolve_hyperparams_for_model
from nhl94_ai.model_inputs import add_model_input_arguments
from nhl94_ai.tasks.registry import resolve_task_model_input


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Legacy run configuration; training is currently unavailable.")

    parser.add_argument("--alg", choices=("ppo2",), default="ppo2")
    parser.add_argument("--nn", type=str, default="CnnPolicy")
    parser.add_argument("--nnsize", type=int, default=256)
    parser.add_argument("--env", type=str, default="NHL941on1-Genesis-v0")
    parser.add_argument("--state", type=str, default=None)
    parser.add_argument("--num_players", type=int, default=1)
    parser.add_argument("--num_env", type=int, default=24)
    parser.add_argument("--num_timesteps", type=int, default=6_000_000)
    parser.add_argument("--output_basedir", type=str, default="~/OUTPUT")
    parser.add_argument("--load_p1_model", type=str, default="")
    parser.add_argument("--load_opponent_model", type=str, default="")
    parser.add_argument("--display_width", type=int, default=1440)
    parser.add_argument("--display_height", type=int, default=810)
    parser.add_argument("--alg_verbose", default=True, action="store_true")
    parser.add_argument("--info_verbose", default=True, action="store_true")
    parser.add_argument("--play", default=False, action="store_true")
    parser.add_argument("--rf", type=str, default="")
    parser.add_argument("--deterministic", default=True, action="store_true")
    parser.add_argument("--hyperparams", type=str, default=default_config_path("default"))
    parser.add_argument("--selfplay", default=False, action="store_true")
    parser.add_argument("--selfplay_role", type=str, default="offense", choices=["offense", "defense"])
    parser.add_argument("--selfplay-snapshot-freq", dest="selfplay_snapshot_freq", type=int, default=100_000)
    parser.add_argument("--selfplay-max-snapshots", dest="selfplay_max_snapshots", type=int, default=8)
    parser.add_argument(
        "--seq_len",
        type=int,
        default=16,
        help="Frame history length for temporal policies such as HybridMambaPolicy or GRUMlpPolicy",
    )
    parser.add_argument(
        "--action_type",
        type=str,
        default="FILTERED",
        choices=["FILTERED", "DISCRETE", "MULTI_DISCRETE", "HOCKEY_INTENT_DPAD", "TARGET_POSITION"],
        help="Buttons, hockey intents, or target-only positioning",
    )
    parser.add_argument("--fullscreen", default=False, action="store_true")
    parser.add_argument("--headless", default=False, action="store_true")

    parser.add_argument("--live-eval-freq", dest="live_eval_freq", type=int, default=10_000)
    parser.add_argument("--live-eval-episodes", dest="live_eval_episodes", type=int, default=5)
    parser.add_argument("--live-window-width", dest="live_window_width", type=int, default=1280)
    parser.add_argument("--live-window-height", dest="live_window_height", type=int, default=720)
    parser.add_argument("--live-graph-width", dest="live_graph_width", type=int, default=420)
    parser.add_argument("--live-fps", dest="live_fps", type=int, default=60)
    parser.add_argument(
        "--live-sleep-per-step",
        dest="live_sleep_per_step",
        type=float,
        default=0.0,
        help="Extra seconds to wait after each display step (default 0 for ~60 FPS playback).",
    )

    parser.add_argument("--seed", type=int, default=0, help="Environment random seed")
    parser.add_argument("--policy_seed", type=int, default=None, help="Optional PPO initialization seed")
    return add_model_input_arguments(parser)


def parse_cmdline(argv: Sequence[str]) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args


def prepare_args(args: argparse.Namespace, *, hyperparams_base_dir: Optional[str] = None) -> argparse.Namespace:
    if getattr(args, "hyperparams_dict", None) is None:
        args.hyperparams_dict = resolve_hyperparams_for_model(load_hyperparams(
            args.hyperparams,
            required=True,
            base_dir=hyperparams_base_dir or os.path.join(os.path.dirname(__file__), "training"),
        ), args.nn)

    if getattr(args, "selfplay", False):
        if not getattr(args, "rf", ""):
            args.rf = "SelfPlayOffenseFinetune" if args.selfplay_role == "offense" else "SelfPlayDefenseFinetune"
        if not getattr(args, "load_opponent_model", ""):
            args.load_opponent_model = getattr(args, "load_p1_model", "")

    args.hyperparams_dict = resolve_task_model_input(args, args.hyperparams_dict)
    return args
