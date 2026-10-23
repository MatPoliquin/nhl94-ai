"""
NHL94 keep-puck reward functions.
"""

import math

from nhl94_ai.game.constants import GameConsts
from nhl94_ai.tasks.setup import SampleRandomPos, SetRandomSkaterPositions


def init_keeppuck(env, env_name=None):
    if env_name is None:
        env_name = 'NHL941on1-Genesis-v0'

    SetRandomSkaterPositions(env, env_name, SampleRandomPos)


def isdone_keeppuck(state):
    t1 = state.team1

    if t1.player_haspuck is False:
        return True

    if state.puck.y < 100:
        return True


def rf_keeppuck(state):
    """Reward unavailable pending the ML redesign; preserve the numeric API."""
    return 0.0
