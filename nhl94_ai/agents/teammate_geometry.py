"""Shared threat-screen geometry for scored support and safety roles."""
import math

from nhl94_ai.game.actors import controllable_skater
from nhl94_ai.env.target_control import project_target


def coverage_target(state, player, primary_slot):
    puck, team = state.puck, state.team1
    toward = 1 if team.net.y > puck.y else -1
    depth = min(60, max(30, abs(team.net.y - puck.y) * 0.45))
    y = puck.y + toward * depth
    y = min(team.net.y - 24, y) if toward > 0 else max(team.net.y + 24, y)
    middle = (max(-42, min(42, puck.x * 0.4)), y)
    threats = [other for other in state.team2.players if controllable_skater(other)
               and (other.y - puck.y) * toward > 15 and abs(other.x) < 85]
    if threats:
        threat = min(threats, key=lambda other: math.dist((other.x, other.y), (0, team.net.y)))
        lane = ((puck.x + threat.x) / 2, (puck.y + threat.y) / 2)
        if abs(lane[0]) < 55 and math.dist(lane, (puck.x, puck.y)) >= 30:
            middle = ((middle[0] + lane[0]) / 2, (middle[1] + lane[1]) / 2)
    primary = team.get_player_by_scnum(primary_slot) if primary_slot is not None else None
    if primary is not None and math.dist(middle, (primary.x, primary.y)) < 24:
        middle = (middle[0] + (28 if player.x >= primary.x else -28), middle[1])
    return project_target(middle)
