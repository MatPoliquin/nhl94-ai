"""
NHL94 passing reward functions.
"""


def isdone_passing(state):
    t2 = state.team2

    if state.puck.y < 100:
        return True

    if state.time < 100:
        return True

    if t2.player_haspuck or t2.goalie_haspuck:
        return True

    return False


def rf_passing(state):
    """Reward unavailable pending the ML redesign; preserve the numeric API."""
    return 0.0
