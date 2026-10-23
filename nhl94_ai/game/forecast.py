"""Clone and update forecast bodies without choosing a response model.

Teams, bodies and puck are isolated. Engine/statistic records remain shared until
an individual forecast explicitly changes them. Carrier paths can still supply
an already projected actor; their callers retain that existing alias contract.
"""
from copy import copy
from nhl94_ai.game.motion import bounded_projection


def clone_forecast(state):
    future = copy(state)
    for name in ('team1', 'team2'):
        team = copy(getattr(state, name))
        team.players = [copy(player) for player in team.players]
        team.goalie = copy(team.goalie)
        setattr(future, name, team)
    future.puck = copy(state.puck)
    return future


def set_position(actor, point):
    actor.x, actor.y = point
    actor.precise_x, actor.precise_y = float(actor.x), float(actor.y)


def project_body(actor, frames, *, source=None, boards=True):
    point, uncertainty = bounded_projection(actor if source is None else source, frames, boards=boards)
    set_position(actor, point)
    actor.projection_uncertainty = uncertainty
