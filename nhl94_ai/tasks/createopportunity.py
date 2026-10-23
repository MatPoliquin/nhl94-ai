"""
NHL94 create-opportunity reward functions.
"""

import math

from nhl94_ai.game.constants import GameConsts
from nhl94_ai.tasks.scoregoal import _x_side


def _clamp01(value):
    return max(0.0, min(1.0, value))


def _line_circle_intersection(start, end, center, radius):
    line_x = end[0] - start[0]
    line_y = end[1] - start[1]
    line_len_sq = line_x * line_x + line_y * line_y
    if line_len_sq <= 0:
        return GameConsts.Distance(start, center) <= radius

    projection = ((center[0] - start[0]) * line_x + (center[1] - start[1]) * line_y) / line_len_sq
    projection = _clamp01(projection)
    closest = (start[0] + projection * line_x, start[1] + projection * line_y)
    return GameConsts.Distance(closest, center) <= radius


def _lane_clear_to_net(receiver, opponent_team):
    receiver_pos = (receiver.x, receiver.y)
    target_y = GameConsts.P2_NET_Y
    target_xs = (
        GameConsts.P2_NET_LEFT_POLL + 3,
        0,
        GameConsts.P2_NET_RIGHT_POLL - 3,
    )

    best_lane = 0.0
    obstacles = list(opponent_team.players) + [opponent_team.goalie]
    for target_x in target_xs:
        target_pos = (target_x, target_y)
        lane_clear = True
        for obstacle in obstacles:
            radius = 10 if obstacle is opponent_team.goalie else 8
            if _line_circle_intersection(receiver_pos, target_pos, (obstacle.x, obstacle.y), radius):
                lane_clear = False
                break

        if lane_clear:
            shot_angle = _clamp01(1.0 - abs(receiver.x - target_x) / 95.0)
            depth = _clamp01((receiver.y - GameConsts.ATACKZONE_POS_Y) / 130.0)
            best_lane = max(best_lane, 0.45 + 0.35 * shot_angle + 0.20 * depth)

    return best_lane


def _create_opportunity_scores(state):
    t1 = state.team1
    t2 = state.team2

    if not t1.player_haspuck or t1.control <= 0:
        return 0.0, 0.0

    controlled_idx = t1.control - 1
    passer = t1.players[controlled_idx]
    if passer.y >= GameConsts.P2_NET_Y - 12:
        return 0.0, 0.0

    best_one_timer = 0.0
    best_cross_crease = 0.0

    for index, receiver in enumerate(t1.players):
        if index == controlled_idx:
            continue
        if receiver.y < GameConsts.ATACKZONE_POS_Y:
            continue

        pass_lane = 1.0 if receiver.passing_lane_clear else 0.0
        if pass_lane <= 0.0:
            continue

        lane_to_net = _lane_clear_to_net(receiver, t2)
        if lane_to_net <= 0.0:
            continue

        pass_distance = GameConsts.Distance((passer.x, passer.y), (receiver.x, receiver.y))
        distance_score = _clamp01(1.0 - abs(pass_distance - 75.0) / 75.0)
        lateral_score = _clamp01(abs(receiver.x - passer.x) / 80.0)
        depth_score = _clamp01((receiver.y - 125.0) / 100.0)
        slot_score = _clamp01(1.0 - abs(receiver.x) / 85.0)
        one_timer_ready = 1.0 if receiver.is_one_timer else 0.0

        one_timer_score = (
            0.25 * pass_lane
            + 0.30 * lane_to_net
            + 0.15 * distance_score
            + 0.15 * lateral_score
            + 0.10 * depth_score
            + 0.05 * one_timer_ready
        )
        best_one_timer = max(best_one_timer, one_timer_score)

        passer_side = _x_side(passer.x, threshold=GameConsts.CREASE_MAX_X)
        receiver_side = _x_side(receiver.x, threshold=GameConsts.CREASE_MAX_X * 0.5)
        opposite_side = passer_side != 0 and receiver_side != 0 and passer_side != receiver_side
        passer_near_crease = passer.y > (GameConsts.CREASE_LOWER_BOUND - 20)
        passer_side_setup = passer_near_crease and passer.y < (GameConsts.P2_NET_Y - 12) and abs(passer.x) > GameConsts.CREASE_MAX_X and abs(passer.x) < 75
        receiver_near_crease = receiver.y > (GameConsts.CREASE_LOWER_BOUND - 20)
        receiver_danger_area = receiver_near_crease and abs(receiver.x) < GameConsts.CREASE_MAX_X * 2.5
        goalie_pulled = abs(t2.goalie.x - passer.x) < abs(t2.goalie.x - receiver.x)

        cross_crease_score = 0.0
        if opposite_side and passer_side_setup and receiver_danger_area:
            cross_crease_score = (
                0.30 * pass_lane
                + 0.25 * lane_to_net
                + 0.15 * lateral_score
                + 0.15 * slot_score
                + 0.10 * distance_score
                + (0.05 if goalie_pulled or t2.goalie.is_pad_stack or t2.goalie.is_dive else 0.0)
            )
        best_cross_crease = max(best_cross_crease, cross_crease_score)

    return min(best_one_timer, 1.0), min(best_cross_crease, 1.0)


def _ensure_create_opportunity_tracker(state):
    tracker = getattr(state, "_create_opportunity_tracker", None)
    if tracker is None:
        tracker = {
            "possession_frames": 0,
            "stall_frames": 0,
            "behind_net_frames": 0,
            "last_puck_x": state.puck.x,
            "last_puck_y": state.puck.y,
            "best_setup_score": 0.0,
            "last_one_timer_score": 0.0,
            "last_cross_crease_score": 0.0,
            "opportunity_completed": False,
            "prev_shot_taken": bool(state.engine.shot_taken),
        }
        setattr(state, "_create_opportunity_tracker", tracker)
    return tracker


def isdone_createopportunity(state):
    t1 = state.team1
    t2 = state.team2
    tracker = getattr(state, "_create_opportunity_tracker", None)

    if tracker is not None and tracker["opportunity_completed"]:
        return True

    if t1.stats.score > t1.last_stats.score or t2.stats.score > t2.last_stats.score:
        return True

    if t1.stats.shots > t1.last_stats.shots:
        return True

    if t2.player_haspuck or t2.goalie_haspuck:
        return True

    if state.puck.y < GameConsts.ATACKZONE_POS_Y:
        return True

    if state.time < 100:
        return True

    if tracker is None:
        return False

    if tracker["stall_frames"] >= 45:
        return True

    if tracker["behind_net_frames"] >= 36:
        return True

    if tracker["possession_frames"] >= 240:
        return True

    return False


def rf_createopportunity(state):
    """Reward unavailable pending the ML redesign; preserve the numeric API."""
    return 0.0


def update_createopportunity(state):
    """Retain completion tracking for callers of this legacy unregistered task."""
    team, opponent = state.team1, state.team2
    tracker = _ensure_create_opportunity_tracker(state)
    if (team.stats.score > team.last_stats.score or opponent.stats.score > opponent.last_stats.score
            or team.stats.shots > team.last_stats.shots):
        return
    if (opponent.player_haspuck or opponent.goalie_haspuck
            or state.puck.y < GameConsts.ATACKZONE_POS_Y):
        return
    one_timer, cross_crease = _create_opportunity_scores(state)
    if team.player_haspuck:
        tracker['possession_frames'] += 1
        player = team.get_controlled_player()
        stationary = math.hypot(state.puck.x-tracker['last_puck_x'], state.puck.y-tracker['last_puck_y']) < 2
        if stationary and math.hypot(player.vx, player.vy) < 4 and not state.action[4] and not state.action[5]:
            tracker['stall_frames'] += 1
        else:
            tracker['stall_frames'] = max(0, tracker['stall_frames'] - 2)
        if player.y >= GameConsts.P2_NET_Y - 10:
            tracker['behind_net_frames'] += 1
        else:
            tracker['behind_net_frames'] = max(0, tracker['behind_net_frames'] - 3)
    else:
        tracker.update(possession_frames=0, stall_frames=0, behind_net_frames=0)
    if team.stats.passing > team.last_stats.passing:
        if max(tracker['last_one_timer_score'], tracker['last_cross_crease_score']) >= 0.65:
            tracker['opportunity_completed'] = True
        tracker['stall_frames'] = 0
    tracker.update(last_one_timer_score=one_timer, last_cross_crease_score=cross_crease,
                   last_puck_x=state.puck.x, last_puck_y=state.puck.y)
