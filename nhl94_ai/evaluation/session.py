"""Evaluate an existing checkpoint using legacy run configuration."""
import argparse
from typing import Callable, Optional
from nhl94_ai.artifacts import ensure_zip_path
from nhl94_ai.env.factory import init_env
from nhl94_ai.evaluation.metrics import LiveEvaluationResult, LiveTeamTotals, evaluate_policy_with_totals
from nhl94_ai.logging import init_logger
from nhl94_ai.models.factory import init_model
from nhl94_ai.run_options import prepare_args


def run_evaluation_session(
    args: argparse.Namespace,
    model_path: str,
    *,
    eval_episodes: int = 5,
    logger=None,
    status_reporter: Optional[Callable[[int, Optional[float], LiveTeamTotals, LiveTeamTotals], None]] = None,
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
        mean_reward=mean_reward,
        std_reward=std_reward,
        episodes=max(1, eval_episodes),
        team1_totals=team1_totals,
        team2_totals=team2_totals,
        reward_status='unavailable' if mean_reward is None else 'available',
    )
