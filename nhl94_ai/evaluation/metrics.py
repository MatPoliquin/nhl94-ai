from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple
import numpy as np
from stable_baselines3.common.vec_env import VecEnvWrapper
from nhl94_ai.game.ram import decode_shot_count
from nhl94_ai.game.state import NHL94GameState, Stats
from nhl94_ai.game.specs import GAMES
from nhl94_ai.config import EvaluationConfig
NHL94_ENVS = set(GAMES)

def resolve_nhl94_players_per_team(env_name, fallback_num_players):
    return GAMES[env_name].skaters_per_team


@dataclass
class LiveEvaluationResult:
    model_path: str
    mean_reward: float
    std_reward: float
    episodes: int
    team1_totals: "LiveTeamTotals"
    team2_totals: "LiveTeamTotals"


@dataclass
class LiveTeamTotals:
    goals: int = 0
    shots: int = 0
    passes: int = 0
    one_timers: int = 0
    cross_checks: int = 0

    @classmethod
    def from_game_stats(cls, stats) -> "LiveTeamTotals":
        return cls(
            goals=int(getattr(stats, "score", 0) or 0),
            shots=int(getattr(stats, "shots", 0) or 0),
            passes=int(getattr(stats, "passing", 0) or 0),
            one_timers=int(getattr(stats, "onetimer", 0) or 0) & 0xFFFF,
            cross_checks=int(getattr(stats, "bodychecks", 0) or 0),
        )

    @classmethod
    def from_info(cls, info, team) -> "LiveTeamTotals":
        return cls.from_game_stats(Stats(
            score=info[f'p{team}_score'], shots=decode_shot_count(info[f'p{team}_shots']),
            passing=info[f'p{team}_passing'], onetimer=info[f'p{team}_onetimer'],
            bodychecks=info[f'p{team}_bodychecks'],
        ))

    def add_delta(self, previous: Optional["LiveTeamTotals"], current: "LiveTeamTotals") -> None:
        self.goals += current.goals if previous is None or current.goals < previous.goals else current.goals - previous.goals
        self.shots += current.shots if previous is None or current.shots < previous.shots else current.shots - previous.shots
        self.passes += current.passes if previous is None or current.passes < previous.passes else current.passes - previous.passes
        self.one_timers += (
            current.one_timers
            if previous is None or current.one_timers < previous.one_timers
            else current.one_timers - previous.one_timers
        )
        self.cross_checks += (
            current.cross_checks
            if previous is None or current.cross_checks < previous.cross_checks
            else current.cross_checks - previous.cross_checks
        )


def _first_item(value, default=None):
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return default
        return value.reshape(-1)[0]

    if isinstance(value, (list, tuple)):
        if not value:
            return default
        return value[0]

    return value if value is not None else default


def reset_team_totals(env):
    """Read the reset boundary, including through observation-only VecEnv wrappers."""
    while isinstance(env, VecEnvWrapper):
        env = env.venv
    return tuple(LiveTeamTotals.from_info(env.reset_infos[0], team) for team in (1, 2))


def evaluate_policy_with_totals(
    model,
    eval_env,
    *,
    n_eval_episodes: int,
    config: Optional[EvaluationConfig] = None,
    deterministic: bool,
    env_name: str,
    num_players: int,
    episode_reporter: Optional[Callable[[int, float, LiveTeamTotals, LiveTeamTotals], None]] = None,
) -> Tuple[float, float, LiveTeamTotals, LiveTeamTotals]:
    config = config or EvaluationConfig(episodes=max(1, int(n_eval_episodes)), max_steps=None, deterministic=deterministic)
    total_episodes = config.episodes
    deterministic = config.deterministic
    episode_rewards: List[float] = []
    team1_totals = LiveTeamTotals()
    team2_totals = LiveTeamTotals()
    uses_nhl94_gamestate = env_name in NHL94_ENVS
    nhl94_players_per_team = resolve_nhl94_players_per_team(env_name, num_players)
    game_state = NHL94GameState(nhl94_players_per_team) if uses_nhl94_gamestate else None

    observation = eval_env.reset()
    baselines = reset_team_totals(eval_env) if game_state is not None else None
    episode_reward = 0.0
    episodes_completed = 0
    episode_steps = 0

    while episodes_completed < total_episodes:
        action, _ = model.predict(observation, deterministic=deterministic)
        observation, reward, done, info = eval_env.step(action)
        episode_reward += float(_first_item(reward, 0.0) or 0.0)
        episode_steps += 1

        if game_state is not None:
            frame_info = _first_item(info)
            if isinstance(frame_info, dict):
                game_state.BeginFrame(frame_info, [0] * 6)
                game_state.EndFrame()

        budget_exhausted = config.max_steps is not None and episode_steps >= config.max_steps
        if not bool(_first_item(done, False)) and not budget_exhausted:
            continue

        episode_rewards.append(episode_reward)
        episode_reward = 0.0
        episode_steps = 0

        if game_state is not None:
            team1_totals.add_delta(baselines[0], LiveTeamTotals.from_game_stats(game_state.team1.stats))
            team2_totals.add_delta(baselines[1], LiveTeamTotals.from_game_stats(game_state.team2.stats))
            game_state = NHL94GameState(nhl94_players_per_team)

        episodes_completed += 1
        if episode_reporter is not None:
            running_mean = float(np.mean(episode_rewards)) if episode_rewards else 0.0
            episode_reporter(
                episodes_completed,
                running_mean,
                LiveTeamTotals(
                    goals=team1_totals.goals,
                    shots=team1_totals.shots,
                    passes=team1_totals.passes,
                    one_timers=team1_totals.one_timers,
                    cross_checks=team1_totals.cross_checks,
                ),
                LiveTeamTotals(
                    goals=team2_totals.goals,
                    shots=team2_totals.shots,
                    passes=team2_totals.passes,
                    one_timers=team2_totals.one_timers,
                    cross_checks=team2_totals.cross_checks,
                ),
            )

        if episodes_completed < total_episodes:
            observation = eval_env.reset()
            if game_state is not None:
                baselines = reset_team_totals(eval_env)

    mean_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
    std_reward = float(np.std(episode_rewards)) if episode_rewards else 0.0
    return mean_reward, std_reward, team1_totals, team2_totals
