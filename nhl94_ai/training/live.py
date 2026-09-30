#!/usr/bin/env python3
"""Training sessions with evaluation callbacks and an optional live display."""

from __future__ import annotations
from nhl94_ai.agents.registry import CONTROLLERS

from nhl94_ai.artifacts import save_checkpoint

from nhl94_ai.config import default_config_path, EvaluationConfig

import argparse
import os
import random
import sys
import threading
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, List, Optional, Sequence, Tuple

# Hide the pygame greeting before importing pygame modules
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

from nhl94_ai.training.logging import com_print, create_output_dir, get_model_file_name, init_logger
from nhl94_ai.env.factory import get_button_names, init_env
from nhl94_ai.models.factory import get_model_probabilities, init_model
from nhl94_ai.config import load_hyperparams, resolve_hyperparams_for_model
import nhl94_ai.env.components as games
from nhl94_ai.game.state import NHL94GameState
from nhl94_ai.evaluation.metrics import LiveTeamTotals, LiveEvaluationResult, evaluate_policy_with_totals
from nhl94_ai.training.events import LiveTrainingState, LiveTrainingResult
from nhl94_ai.training.rl import ModelTrainer


SIM_STEPS_PER_SECOND = 60
NHL94_ENVS = {"NHL941on1-Genesis-v0", "NHL942on2-Genesis-v0", "NHL94-Genesis-v0"}


from nhl94_ai.artifacts import ensure_zip_path


class LiveTrainingCallback(BaseCallback):
    """Periodically evaluates the policy and updates the shared live state."""

    def __init__(
        self,
        shared_state: LiveTrainingState,
        eval_env,
        env_name: str,
        num_players: int,
        eval_freq: int,
        eval_episodes: int,
        status_reporter: Optional[Callable[[int, float, float, int, LiveTeamTotals, LiveTeamTotals], None]] = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.shared_state = shared_state
        self.eval_env = eval_env
        self.env_name = env_name
        self.num_players = num_players
        self.eval_freq = max(1, eval_freq)
        self.eval_episodes = max(1, eval_episodes)
        self.evaluation_config = EvaluationConfig(episodes=self.eval_episodes, max_steps=None)
        self.status_reporter = status_reporter
        self._last_eval_step = 0

    def _on_step(self) -> bool:
        with self.shared_state.lock:
            if self.shared_state.stop_requested:
                com_print("Live window requested stop – terminating training loop.")
                return False
        return True

    def _on_rollout_end(self) -> bool:
        if self.num_timesteps - self._last_eval_step < self.eval_freq:
            return True

        self._run_evaluation()
        return True

    def _on_training_end(self) -> None:
        if self.num_timesteps <= 0 or self._last_eval_step == self.num_timesteps:
            return

        self._run_evaluation()

    def _run_evaluation(self) -> None:
        mean_reward, _, team1_totals, team2_totals = evaluate_policy_with_totals(
            self.model,
            self.eval_env,
            n_eval_episodes=self.eval_episodes,
            config=self.evaluation_config,
            deterministic=True,
            env_name=self.env_name,
            num_players=self.num_players,
        )

        self._last_eval_step = self.num_timesteps
        best_reward = float("-inf")

        with self.shared_state.lock:
            self.shared_state.latest_eval_reward = mean_reward
            self.shared_state.latest_eval_timesteps = self.num_timesteps
            self.shared_state.reward_history.append((self.num_timesteps, mean_reward))

            if mean_reward > self.shared_state.best_mean_reward:
                save_path = ensure_zip_path(self.shared_state.best_model_path)
                save_checkpoint(self.model, save_path)
                self.shared_state.best_mean_reward = mean_reward

            if self.shared_state.latest_model_path is not None:
                save_checkpoint(self.model, ensure_zip_path(self.shared_state.latest_model_path))
                self.shared_state.latest_model_timesteps = self.num_timesteps
                self.shared_state.model_version += 1

            best_reward = self.shared_state.best_mean_reward

        if self.status_reporter is not None:
            self.status_reporter(
                self.num_timesteps,
                mean_reward,
                best_reward,
                self.eval_episodes,
                team1_totals,
                team2_totals,
            )

        if self.verbose:
            com_print(
                f"[Live eval] steps={self.num_timesteps:,} "
                f"reward={mean_reward:.3f} best={best_reward:.3f}"
            )


class SelfPlaySnapshotCallback(BaseCallback):
    """Periodically snapshots the learner and rotates the frozen self-play opponent."""

    def __init__(
        self,
        *,
        output_dir: str,
        best_model_path: str,
        eval_env,
        initial_opponent_path: str,
        snapshot_freq: int,
        max_snapshots: int,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.output_dir = output_dir
        self.best_model_path = ensure_zip_path(best_model_path)
        self.eval_env = eval_env
        self.snapshot_freq = max(1, snapshot_freq)
        self.max_snapshots = max(1, max_snapshots)
        self._last_snapshot_step = 0
        self._current_opponent_path = ensure_zip_path(initial_opponent_path) if initial_opponent_path else ""
        self._latest_snapshot_path = self._current_opponent_path
        self._historical_paths: List[str] = []
        if self._current_opponent_path:
            self._historical_paths.append(self._current_opponent_path)

    def _unwrap_vec_env(self, env):
        current = env
        while hasattr(current, "venv"):
            current = current.venv
        return current

    def _set_opponent_model(self, env, path: str) -> None:
        if not path:
            return

        target_env = self._unwrap_vec_env(env)
        if hasattr(target_env, "env_method"):
            target_env.env_method("set_opponent_model", path)

    def _save_snapshot(self) -> str:
        snapshot_path = os.path.join(self.output_dir, f"selfplay_snapshot_{self.num_timesteps}.zip")
        save_checkpoint(self.model, snapshot_path)
        return snapshot_path

    def _record_snapshot(self, snapshot_path: str) -> None:
        self._latest_snapshot_path = snapshot_path
        if snapshot_path in self._historical_paths:
            self._historical_paths.remove(snapshot_path)
        self._historical_paths.append(snapshot_path)

        while len(self._historical_paths) > self.max_snapshots:
            removed = self._historical_paths.pop(0)
            if removed == self._current_opponent_path:
                self._current_opponent_path = self._historical_paths[0] if self._historical_paths else ""

    def _sample_opponent_path(self) -> str:
        if not self._historical_paths:
            return self._current_opponent_path

        latest_candidate = self._latest_snapshot_path or self._historical_paths[-1]
        best_candidate = self.best_model_path if os.path.exists(self.best_model_path) else latest_candidate

        roll = random.random()
        if roll < 0.4:
            return latest_candidate
        if roll < 0.8:
            return random.choice(self._historical_paths)
        return best_candidate

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> bool:
        if self.num_timesteps - self._last_snapshot_step < self.snapshot_freq:
            return True

        snapshot_path = self._save_snapshot()
        self._record_snapshot(snapshot_path)
        self._last_snapshot_step = self.num_timesteps

        sampled_path = self._sample_opponent_path()
        if sampled_path:
            self._current_opponent_path = sampled_path
            self._set_opponent_model(self.training_env, sampled_path)
            self._set_opponent_model(self.eval_env, sampled_path)

        if self.verbose:
            com_print(
                f"[Self-play] steps={self.num_timesteps:,} snapshot={os.path.basename(snapshot_path)} "
                f"opponent={os.path.basename(sampled_path) if sampled_path else 'none'}"
            )

        return True


class LiveTrainer(ModelTrainer):
    """Coordinates training and live visualization."""

    def __init__(self, args, logger):
        super().__init__(args, logger)
        self.logger = logger
        self.model = self.p1_model
        self.callback_envs = []
        self.best_model_savepath = self.model_savepath + '_best_live'
        self.latest_model_savepath = self.model_savepath + '_latest_live'

    def build_callback(
        self,
        shared_state: LiveTrainingState,
        status_reporter: Optional[Callable[[int, float, float, int, LiveTeamTotals, LiveTeamTotals], None]] = None,
    ) -> BaseCallback:
        eval_env = init_env(
            None,
            1,
            self.args.state,
            self.args.num_players,
            self.args,
            self.args.hyperparams_dict,
            use_sticky_action=False,
        )
        self.callback_envs = [eval_env]

        live_callback = LiveTrainingCallback(
            shared_state=shared_state,
            eval_env=eval_env,
            env_name=self.args.env,
            num_players=self.args.num_players,
            eval_freq=self.args.live_eval_freq,
            eval_episodes=self.args.live_eval_episodes,
            status_reporter=status_reporter,
            verbose=1 if self.args.alg_verbose else 0,
        )

        if not getattr(self.args, "selfplay", False):
            return live_callback

        snapshot_callback = SelfPlaySnapshotCallback(
            output_dir=self.output_fullpath,
            best_model_path=self.best_model_savepath,
            eval_env=eval_env,
            initial_opponent_path=getattr(self.args, "load_opponent_model", ""),
            snapshot_freq=max(
                self.args.live_eval_freq,
                getattr(self.args, "selfplay_snapshot_freq", 100_000),
            ),
            max_snapshots=getattr(self.args, "selfplay_max_snapshots", 8),
            verbose=1 if self.args.alg_verbose else 0,
        )
        return CallbackList([live_callback, snapshot_callback])

    def train(self, callback: BaseCallback = None) -> None:
        if self.args.alg == "es":
            raise NotImplementedError("Live training does not currently support Evolution Strategies.")
        if self.args.nn in CONTROLLERS:
            raise NotImplementedError(f'{self.args.nn} is inference-only. Use nhl94 play to view it.')

        com_print("========= Live Training ==========")
        com_print(f"OUTPUT PATH:   {self.output_fullpath}")
        com_print(f"ENV:           {self.args.env}")
        com_print(f"STATE:         {self.args.state}")
        com_print(f"NN:            {self.args.nn}")
        com_print(f"ALGO:          {self.args.alg}")
        com_print(f"TIMESTEPS:     {self.args.num_timesteps:,}")

        super().train(callback=callback)

        com_print("========= Training Complete ==========")

        com_print(f"Model saved to: {self.model_savepath}.zip")

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
        for callback_env in self.callback_envs:
            callback_env.close()
        self.callback_envs = []


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a model while streaming live visuals.")

    parser.add_argument("--alg", type=str, default="ppo2")
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
    return parser


def parse_cmdline(argv: Sequence[str]) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args


def prepare_args(args: argparse.Namespace, *, hyperparams_base_dir: Optional[str] = None) -> argparse.Namespace:
    if getattr(args, "hyperparams_dict", None) is None:
        args.hyperparams_dict = resolve_hyperparams_for_model(load_hyperparams(
            args.hyperparams,
            required=True,
            base_dir=hyperparams_base_dir or os.path.dirname(__file__),
        ), args.nn)

    if getattr(args, "selfplay", False):
        if not getattr(args, "rf", ""):
            args.rf = "SelfPlayOffenseFinetune" if args.selfplay_role == "offense" else "SelfPlayDefenseFinetune"
        if not getattr(args, "load_opponent_model", ""):
            args.load_opponent_model = getattr(args, "load_p1_model", "")

    return args


def run_training_session(
    args: argparse.Namespace,
    logger=None,
    status_reporter: Optional[Callable[[int, float, float, int, LiveTeamTotals, LiveTeamTotals], None]] = None,
) -> LiveTrainingResult:
    prepare_args(args)

    if logger is None:
        logger = init_logger(args)

    com_print("=========== Live Params ===========")
    com_print(args)

    from nhl94_ai.config import TrainingConfig
    TrainingConfig.from_args(args)
    if args.alg == 'es':
        raise ValueError('Use python -m nhl94_ai.training.rl --alg es for Evolution Strategies')
    trainer = LiveTrainer(args, logger)
    shared_state = LiveTrainingState(
        best_model_path=trainer.best_model_savepath,
        reward_history=deque(maxlen=512),
        latest_model_path=trainer.latest_model_savepath if not getattr(args, 'headless', False) else None,
    )

    display = None
    try:
        if not getattr(args, 'headless', False):
            from nhl94_ai.ui.live import LiveTrainingDisplay
            display = LiveTrainingDisplay(args=args, shared_state=shared_state,
                                          fps=args.live_fps, graph_width=args.live_graph_width)
        callback = trainer.build_callback(shared_state, status_reporter=status_reporter)
        if display is not None:
            display.start()
        trainer.train(callback)
    except KeyboardInterrupt:
        com_print("Training interrupted by user; saving partial checkpoint.")
        save_checkpoint(trainer.model, trainer.model_savepath)
    finally:
        shared_state.training_active = False
        if display is not None:
            display.stop()
            if display.ident is not None:
                display.join(timeout=5.0)
        trainer.close()

    if args.play:
        com_print("Play-after-train is not yet available in live mode.")

    return LiveTrainingResult(
        final_model_path=ensure_zip_path(trainer.model_savepath),
        best_model_path=ensure_zip_path(trainer.best_model_savepath),
        output_dir=trainer.output_fullpath,
    )


def run_evaluation_session(
    args: argparse.Namespace,
    model_path: str,
    *,
    eval_episodes: int = 5,
    logger=None,
    status_reporter: Optional[Callable[[int, float, LiveTeamTotals, LiveTeamTotals], None]] = None,
) -> LiveEvaluationResult:
    prepare_args(args)

    if logger is None:
        logger = init_logger(args)

    eval_env = init_env(
        None,
        1,
        args.state,
        args.num_players,
        args,
        args.hyperparams_dict,
        use_sticky_action=False,
    )

    try:
        model = init_model(
            None,
            model_path,
            args.alg,
            args,
            eval_env,
            logger,
            args.hyperparams_dict,
        )
    except BaseException:
        eval_env.close()
        raise

    try:
        mean_reward, std_reward, team1_totals, team2_totals = evaluate_policy_with_totals(
            model,
            eval_env,
            n_eval_episodes=max(1, eval_episodes),
            deterministic=bool(getattr(args, "deterministic", True)),
            env_name=args.env,
            num_players=args.num_players,
            episode_reporter=status_reporter,
        )
    finally:
        eval_env.close()

    return LiveEvaluationResult(
        model_path=ensure_zip_path(model_path),
        mean_reward=float(mean_reward),
        std_reward=float(std_reward),
        episodes=max(1, eval_episodes),
        team1_totals=team1_totals,
        team2_totals=team2_totals,
    )


def main(argv: Sequence[str]) -> None:
    args = parse_cmdline(argv[1:])
    prepare_args(args, hyperparams_base_dir=os.path.dirname(__file__))
    run_training_session(args)


if __name__ == "__main__":
    main(sys.argv)
