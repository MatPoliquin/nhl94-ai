"""Named task definitions adapting the existing reward implementations."""
from dataclasses import dataclass
from typing import Callable
from nhl94_ai.game.constants import GameConsts
from nhl94_ai.tasks.legacy import _reward_function_map


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


TASKS = {name: TaskDefinition(
            *functions, target_control='defense' if name == 'DefenseZone' else None,
            target_bounds=(-GameConsts.MAX_PLAYER_X, GameConsts.MAX_PLAYER_X,
                           -GameConsts.MAX_PLAYER_Y, GameConsts.DEFENSEZONE_POS_Y) if name == 'DefenseZone' else None,
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
