"""Task setup, episode progression, and legacy zero-reward contracts."""
from dataclasses import dataclass
from typing import Callable
from nhl94_ai.game.constants import GameConsts
from nhl94_ai.tasks.legacy import _reward_function_map
from nhl94_ai.tasks.scoregoal import update_scoregoal_v2, update_crosscrease_v2
from nhl94_ai.tasks.createopp import update_createopp


@dataclass(frozen=True)
class TaskDefinition:
    initialize: Callable
    reward: Callable
    done: Callable
    observation_size: Callable
    encode: Callable
    restrict_action: Callable
    target_control: str | None = None
    target_bounds: tuple[float, float, float, float] | None = None
    model_input_variant: str | None = None
    advance: Callable | None = None
    reward_available: bool = True


TASKS = {name: TaskDefinition(
            *functions, target_control='defense' if name == 'DefenseZone' else None,
            target_bounds=(-GameConsts.MAX_PLAYER_X, GameConsts.MAX_PLAYER_X,
                           -GameConsts.MAX_PLAYER_Y, GameConsts.DEFENSEZONE_POS_Y) if name == 'DefenseZone' else None,
            model_input_variant='pvg' if name == 'PvG' else None,
            advance={'ScoreGoalV2': update_scoregoal_v2, 'CrossCreaseV2': update_crosscrease_v2,
                     'CreateOpp': update_createopp}.get(name),
            reward_available=False,
         )
         for name, functions in _reward_function_map.items()}


def register_task(name, task):
    if name in TASKS:
        raise ValueError(f'Task already registered: {name}')
    if not isinstance(task, TaskDefinition):
        raise TypeError('task must be a TaskDefinition')
    TASKS[name] = task


def get_task(name):
    try:
        return TASKS[name]
    except KeyError as error:
        raise ValueError(f'Unsupported Reward Function: {name}') from error


def resolve_task_model_input(args, hyperparams, *, task=None):
    from nhl94_ai.env.encoding import _normalize_model_input_config
    from nhl94_ai.model_inputs import resolve_model_input
    if task is None and getattr(args, 'rf', None):
        task = get_task(args.rf)
    params = resolve_model_input(args, hyperparams, task=task)
    params['model_input']['groups'] = _normalize_model_input_config(params['model_input'])
    from nhl94_ai.game.specs import GAMES
    spec = GAMES.get(getattr(args, 'env', None))
    if spec is not None and spec.backend.normalization:
        params['model_input']['normalization'] = spec.backend.normalization
    return params
