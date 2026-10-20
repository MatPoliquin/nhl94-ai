"""
NHL94 create-opportunity reward functions.
"""

from nhl94_ai.game.constants import GameConsts
from nhl94_ai.env.intents import HOCKEY_INTENT_NOOP, HOCKEY_INTENT_NORMAL_SHOOT, HOCKEY_INTENT_ONE_TIMER, HOCKEY_INTENT_POKE_CHECK, HOCKEY_INTENT_SLAPSHOT
from nhl94_ai.tasks.scoregoal import _x_side


CREATE_OPP_CROSS_THRESHOLD = 0.75
CREATE_OPP_ONE_TIMER_THRESHOLD = 0.70
CREATE_OPP_MASKED_INTENTS = {
    HOCKEY_INTENT_NORMAL_SHOOT,
    HOCKEY_INTENT_SLAPSHOT,
    HOCKEY_INTENT_ONE_TIMER,
    HOCKEY_INTENT_POKE_CHECK,
}


def input_overide_createopp(ac):
    if not hasattr(ac, "__len__"):
        return

    if len(ac) == 6:
        if int(ac[0]) in CREATE_OPP_MASKED_INTENTS:
            ac[0] = HOCKEY_INTENT_NOOP
        return

    if len(ac) == GameConsts.INPUT_MAX:
        ac[GameConsts.INPUT_C] = 0
        return

    if len(ac) == 3 and ac[2] == 2:
        ac[2] = 0


def _clamp01(value):
    return max(0.0, min(1.0, value))


def _goalie_vulnerability_score(state, carrier):
    goalie = state.team2.goalie
    engine = state.engine

    lateral_pull = _clamp01(abs(goalie.x) / 34.0)
    goalie_state = max(float(goalie.is_pad_stack), float(goalie.is_dive), float(engine.goalie_box_small))
    inside_lane = 1.0 if carrier.y >= GameConsts.CREASE_LOWER_BOUND - 10 else 0.0
    return max(0.5 * lateral_pull + 0.5 * goalie_state, goalie_state * inside_lane)


def _defender_gap_score(receiver, opponent_team):
    if not opponent_team.players:
        return 1.0

    nearest_defender = min(
        GameConsts.Distance((receiver.x, receiver.y), (player.x, player.y))
        for player in opponent_team.players
    )
    return _clamp01((nearest_defender - 18.0) / 42.0)


def _receiver_slot_score(receiver):
    in_front = _clamp01((receiver.y - (GameConsts.CREASE_LOWER_BOUND - 10)) / 40.0)
    central = _clamp01(1.0 - abs(receiver.x) / (GameConsts.CREASE_MAX_X * 3.0))
    return 0.55 * in_front + 0.45 * central


def _receiver_net_drive_score(receiver):
    return _clamp01(receiver.vy / 12.0)


def _best_opportunity_scores(state):
    team = state.team1
    opponent = state.team2

    if not team.player_haspuck or team.control <= 0:
        return 0.0, 0.0, 0.0

    carrier = team.get_controlled_player()
    if carrier is None or carrier.y < GameConsts.ATACKZONE_POS_Y:
        return 0.0, 0.0, 0.0

    goalie_score = _goalie_vulnerability_score(state, carrier)
    carrier_side = _x_side(carrier.x, threshold=GameConsts.CREASE_MAX_X)

    best_one_timer = 0.0
    best_cross_crease = 0.0

    for receiver in team.players:
        if receiver is carrier:
            continue
        if receiver.y < GameConsts.ATACKZONE_POS_Y:
            continue

        slot_score = _receiver_slot_score(receiver)
        gap_score = _defender_gap_score(receiver, opponent)
        pass_lane = 1.0 if receiver.passing_lane_clear else 0.0

        if receiver.one_timer_lane_good:
            one_timer_score = 0.50 + 0.25 * slot_score + 0.25 * gap_score
            if pass_lane:
                one_timer_score += 0.10
            best_one_timer = max(best_one_timer, min(one_timer_score, 1.0))

        receiver_side = _x_side(receiver.x, threshold=GameConsts.CREASE_MAX_X * 0.5)
        opposite_side = carrier_side != 0 and receiver_side != 0 and carrier_side != receiver_side
        if opposite_side and pass_lane:
            drive_score = _receiver_net_drive_score(receiver)
            cross_crease_score = 0.25 + 0.20 * slot_score + 0.20 * gap_score + 0.20 * goalie_score + 0.15 * drive_score
            best_cross_crease = max(best_cross_crease, min(cross_crease_score, 1.0))

    return best_one_timer, best_cross_crease, min(goalie_score, 1.0)


def _ensure_createopp_state(state):
    reward_state = getattr(state, "_createopp_state", None)
    if reward_state is None:
        reward_state = {
            "completed": False,
            "last_cross_crease": 0.0,
            "last_one_timer": 0.0,
        }
        setattr(state, "_createopp_state", reward_state)
    return reward_state


def isdone_createopp(state):
    team = state.team1
    opponent = state.team2
    reward_state = getattr(state, "_createopp_state", None)

    if reward_state is not None and reward_state["completed"]:
        return True

    if team.stats.score > team.last_stats.score or opponent.stats.score > opponent.last_stats.score:
        return True

    if opponent.player_haspuck or opponent.goalie_haspuck:
        return True

    if state.puck.y < GameConsts.ATACKZONE_POS_Y:
        return True

    if state.time < 100:
        return True

    return False


def rf_createopp(state):
    """Reward unavailable pending the ML redesign; preserve the numeric API."""
    return 0.0


def update_createopp(state):
    """Keep the existing completed-opportunity condition independently of reward."""
    team, opponent = state.team1, state.team2
    tracker = _ensure_createopp_state(state)
    if (team.stats.score > team.last_stats.score or opponent.stats.score > opponent.last_stats.score
            or opponent.player_haspuck or opponent.goalie_haspuck or state.puck.y < GameConsts.ATACKZONE_POS_Y):
        return
    if state.action[5] or state.engine.shot_taken or state.engine.shot_mode_active:
        return
    one_timer, cross_crease, _ = _best_opportunity_scores(state)
    if team.stats.passing > team.last_stats.passing:
        if tracker['last_one_timer'] >= CREATE_OPP_ONE_TIMER_THRESHOLD or tracker['last_cross_crease'] >= CREATE_OPP_CROSS_THRESHOLD:
            tracker['completed'] = True
    tracker.update(last_one_timer=one_timer, last_cross_crease=cross_crease)
