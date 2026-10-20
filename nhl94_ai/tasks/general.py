"""
NHL94 general reward functions.
"""

from nhl94_ai.game.constants import GameConsts
from nhl94_ai.tasks.setup import init_general


def isdone_general(state):
    t1 = state.team1
    t2 = state.team2

    if state.time < 100:
        return True

    if t1.stats.score > t1.last_stats.score:
        return True
    if t2.stats.score > t2.last_stats.score:
        return True

    return False


def rf_general(state):
    """Reward unavailable pending the ML redesign; preserve the numeric API."""
    return 0.0


def init_general_v2(env, env_name):
    init_general(env, env_name)


def isdone_general_v2(state):
    return isdone_general(state)


def rf_general_v2(state):
    """Reward unavailable pending the ML redesign; preserve the numeric API."""
    return 0.0
