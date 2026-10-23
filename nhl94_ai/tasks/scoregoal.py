"""
NHL94 score-goal style reward functions.
"""

import math

from nhl94_ai.game.constants import GameConsts


def isdone_scoregoal_cc(state):
    t1 = state.team1
    t2 = state.team2

    if t1.stats.score > t1.last_stats.score:
        return True

    if t2.player_haspuck or t2.goalie_haspuck:
        return True

    if state.puck.y < 100:
        return True

    if state.time < 100:
        return True

    return False


def rf_scoregoal_cc(state):
    """Reward unavailable pending the ML redesign; preserve the numeric API."""
    return 0.0


def isdone_scoregoal_ot(state):
    t1 = state.team1
    t2 = state.team2

    if t1.stats.score > t1.last_stats.score:
        return True

    if t2.player_haspuck or t2.goalie_haspuck:
        return True

    if state.puck.y < 0:
        return True

    if state.time < 100:
        return True

    return False


def rf_scoregoal_ot(state):
    """Reward unavailable pending the ML redesign; preserve the numeric API."""
    return 0.0


def isdone_scoregoal(state):
    t1 = state.team1

    if t1.stats.score > t1.last_stats.score:
        return True

    if state.puck.y < 100:
        return True

    if state.time < 100:
        return True


    return False


def rf_scoregoal(state):
    """Reward unavailable pending the ML redesign; preserve the numeric API."""
    return 0.0


def isdone_scoregoal_v2(state):
    t1 = state.team1

    if t1.stats.score > t1.last_stats.score:
        return True

    if state.puck.y < GameConsts.ATACKZONE_POS_Y:
        return True

    if state.time < 100:
        return True

    tracker = getattr(state, "_scoregoal_v2_tracker", None)
    if tracker is None:
        return False

    if tracker["stall_frames"] >= 45:
        return True

    if tracker["possession_frames"] >= 240 and not tracker["shot_taken_once"]:
        return True

    return False


def rf_scoregoal_v2(state):
    """Reward unavailable pending the ML redesign; preserve the numeric API."""
    return 0.0


def _x_side(value: float, threshold: float = 6.0) -> int:
    if value > threshold:
        return 1
    if value < -threshold:
        return -1
    return 0


def isdone_crosscrease_v2(state):
    if isdone_scoregoal(state):
        return True

    tracker = getattr(state, "_crosscrease_v2_tracker", None)
    if tracker is None:
        return False

    if tracker["stall_frames"] >= 35:
        return True

    if tracker["possession_frames"] >= 220 and not tracker["shot_taken_once"]:
        return True

    if tracker["cross_pass_window"] == 0 and tracker["cross_pass_completed"] and not tracker["shot_taken_once"]:
        return True

    return False


def rf_crosscrease_v2(state):
    """Reward unavailable pending the ML redesign; preserve the numeric API."""
    return 0.0


def isdone_scoregoal_v4(state):
    return isdone_scoregoal(state)


def rf_scoregoal_v4(state):
    """Reward unavailable pending the ML redesign; preserve the numeric API."""
    return 0.0


def update_scoregoal_v2(state):
    """Advance only the possession/shot/stall state used by episode completion."""
    team = state.team1
    tracker = getattr(state, '_scoregoal_v2_tracker', None)
    if tracker is None:
        tracker = dict(possession_frames=0, stall_frames=0, shot_taken_once=False,
                       last_puck_x=state.puck.x, last_puck_y=state.puck.y,
                       prev_shot_taken=bool(state.engine.shot_taken))
        state._scoregoal_v2_tracker = tracker
    if team.stats.score > team.last_stats.score:
        return
    if state.team2.player_haspuck or state.team2.goalie_haspuck or state.puck.y < GameConsts.ATACKZONE_POS_Y:
        tracker.update(possession_frames=0, stall_frames=0, shot_taken_once=False,
                       last_puck_x=state.puck.x, last_puck_y=state.puck.y,
                       prev_shot_taken=bool(state.engine.shot_taken))
        return
    player = team.get_controlled_player()
    if team.player_haspuck:
        tracker['possession_frames'] += 1
        stationary = math.hypot(state.puck.x-tracker['last_puck_x'], state.puck.y-tracker['last_puck_y']) < 2
        if (stationary and math.hypot(player.vx, player.vy) < 4 and not state.engine.shot_mode_active
                and not state.action[4] and not state.action[5]):
            tracker['stall_frames'] += 1
        else:
            tracker['stall_frames'] = max(0, tracker['stall_frames'] - 2)
    else:
        tracker.update(possession_frames=0, stall_frames=0)
    if team.stats.passing > team.last_stats.passing:
        tracker['stall_frames'] = max(0, tracker['stall_frames'] - 10)
    if state.engine.shot_taken and not tracker['prev_shot_taken']:
        tracker.update(shot_taken_once=True, stall_frames=0)
    tracker.update(prev_shot_taken=bool(state.engine.shot_taken),
                   last_puck_x=state.puck.x, last_puck_y=state.puck.y)


def update_crosscrease_v2(state):
    """Track the existing possession deadline and completed cross-pass window."""
    team = state.team1
    tracker = getattr(state, '_crosscrease_v2_tracker', None)
    if tracker is None:
        tracker = dict(possession_frames=0, stall_frames=0, setup_side=0,
                       cross_pass_window=0, cross_pass_completed=False, shot_taken_once=False,
                       last_puck_x=state.puck.x, last_puck_y=state.puck.y,
                       prev_shot_taken=bool(state.engine.shot_taken))
        state._crosscrease_v2_tracker = tracker
    if (team.stats.score > team.last_stats.score or state.team2.player_haspuck
            or state.team2.goalie_haspuck or state.puck.y < GameConsts.ATACKZONE_POS_Y):
        return
    player = team.get_controlled_player()
    if team.player_haspuck:
        tracker['possession_frames'] += 1
        stationary = math.hypot(state.puck.x-tracker['last_puck_x'], state.puck.y-tracker['last_puck_y']) < 2
        if stationary and math.hypot(player.vx, player.vy) < 4 and not state.action[4] and not state.action[5]:
            tracker['stall_frames'] += 1
        else:
            tracker['stall_frames'] = max(0, tracker['stall_frames'] - 2)
    else:
        tracker.update(possession_frames=0, stall_frames=0)
    side = _x_side(player.x)
    if player.y > GameConsts.CREASE_LOWER_BOUND - 10 and GameConsts.CREASE_MAX_X < abs(player.x) < 75 and side:
        tracker['setup_side'] = side
    if team.stats.passing > team.last_stats.passing:
        puck_side = _x_side(state.puck.x)
        receiver_area = player.y > GameConsts.CREASE_LOWER_BOUND and abs(player.x) < GameConsts.CREASE_MAX_X * 2
        if tracker['setup_side'] and puck_side and puck_side != tracker['setup_side'] and receiver_area:
            tracker.update(cross_pass_window=30, cross_pass_completed=True, stall_frames=0)
        else:
            tracker['cross_pass_window'] = 0
    if state.engine.shot_taken and not tracker['prev_shot_taken']:
        tracker.update(shot_taken_once=True, cross_pass_window=0, stall_frames=0)
    if tracker['cross_pass_window'] > 0:
        tracker['cross_pass_window'] -= 1
    tracker.update(prev_shot_taken=bool(state.engine.shot_taken),
                   last_puck_x=state.puck.x, last_puck_y=state.puck.y)
