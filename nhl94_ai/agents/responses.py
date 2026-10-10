"""Action-conditioned skating scenarios, not certificates of future CPU decisions."""
from copy import copy
import math

from nhl94_ai.agents.carry import CONTACT_RADIUS, _segment_clearance
from nhl94_ai.agents.defense import on_ice
from nhl94_ai.agents.motion import VELOCITY_SCALE, bounded_projection, velocity
from nhl94_ai.agents.passing import rom_direction
from nhl94_ai.agents.skating import grounded_step


_PADS = ((0, 1), (1, 1), (1, 0), (1, -1),
         (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 0), (0, 0))
_FIELDS = ('motion_x', 'motion_y', 'weight', 'agility', 'speed', 'energy',
           'steering_timer', 'steering', 'assignment')


def support_zone(puck, sign, *, wing=False):
    y = puck.y * sign
    projected = y + (round(velocity(puck)[1] / VELOCITY_SCALE) >> 6) * sign if wing else y
    return 0 if projected < -88 else 8 if y < 88 else 16 if projected < 264 else 24


def _high_byte_difference(first, second):
    return ((round(first / VELOCITY_SCALE) >> 8)
            - (round(second / VELOCITY_SCALE) >> 8) + 128) % 256 - 128


def _evade(player, puck, direction, sign, offsides):
    dx = player.x - puck.x + _high_byte_difference(velocity(player)[0], velocity(puck)[0])
    dy = player.y - puck.y + _high_byte_difference(velocity(player)[1], velocity(puck)[1])
    if abs(dx) > 40 or abs(dy) > 40:
        return direction
    if offsides and -50 <= player.y * sign - 88 <= 10:
        return 2 if player.x > puck.x else 6
    return rom_direction(player.x - puck.x, player.y - puck.y)


def _target(player, puck, sign, friendly):
    if friendly and player.assignment == 1:
        x = (80 if player.role == 2 else -80) * sign
        return (puck.x if (puck.x < 0) == (x < 0) else x + (int(puck.x) >> 1), 98 * sign)
    if friendly and player.assignment in (4, 6):
        x, y = player.cpu_target
        return ((x if player.role == 5 or player.assignment == 6 else -x) * sign, y * sign)
    if not friendly and player.assignment == 0x11:
        px = int(puck.x) + (round(velocity(puck)[0] / VELOCITY_SCALE) >> 9)
        py = int(puck.y) + (round(velocity(puck)[1] / VELOCITY_SCALE) >> 9)
        return px + (-px >> 1), py + ((-244 * sign - py) >> 1)
    return None


def response_step(player, puck, *, sign, friendly, offsides=False):
    """Keep the live steering countdown and native skateto/EvadePC quantization."""
    if not _supported(player, friendly):
        raise ValueError('Response stepping requires supported live stable-assignment feedback.')
    future = grounded_step(player, (0, 0))
    timer = player.steering_timer - 1
    direction = player.steering
    if timer < 0:
        timer += 12
        target = _target(player, puck, sign, friendly)
        dx = target[0] - future.x - (round(velocity(future)[0] / VELOCITY_SCALE) >> 8)
        dy = target[1] - future.y - (round(velocity(future)[1] / VELOCITY_SCALE) >> 8)
        direction = 9 if abs(dx) <= 12 and abs(dy) <= 12 else rom_direction(dx, dy)
        if friendly:
            direction = _evade(future, puck, direction, sign, offsides)
    initial = copy(player)
    if direction == 9 and not any(velocity(future)):
        wanted = rom_direction(puck.x - future.x, puck.y - future.y)
        heading = player.facing if player.facing is not None else player.orientation
        if wanted != heading:
            heading = (heading + (1 if (wanted - heading) % 8 <= 4 else -1)) % 8
            initial.facing = heading
            initial.facing_phase = heading + ((player.facing_phase or 0) % 1)
    if not friendly and abs(puck.x) <= 71 and puck.y * -sign >= 88:
        initial.speed = min(30, initial.speed + 6 + initial.movement_bonus)
    future = grounded_step(initial, _PADS[direction], stop=direction == 9)
    future.speed = player.speed
    future.steering_timer, future.steering = timer, direction
    return future


def _supported(player, friendly):
    if not player.cpu_response_available or any(getattr(player, name) is None for name in _FIELDS):
        return False
    if player.steering not in range(10) or player.cpu_target is None:
        return False
    return (player.assignment in (1, 4, 6) if friendly else
            player.assignment == 0x11 and player.cpu_target[0] <= 0)


def _puck_at(state, original, carrier):
    puck = copy(state.puck)
    puck.x, puck.y = (state.puck.x + carrier.x - original.x,
                      state.puck.y + carrier.y - original.y)
    puck.motion_x, puck.motion_y = velocity(carrier)
    return puck


def response_future(state, path):
    """Model stable support assignments; expose unsupported actors and zone transitions."""
    future = copy(state)
    future.puck = copy(state.puck)
    owner = state.engine.puck_owner
    original = state.team1.get_player_by_scnum(owner)
    if original is None or not path:
        raise ValueError('Response forecasting requires a live carrier and nonempty native path.')
    sign = 1 if state.team2.net.y > state.team1.net.y else -1
    tracks, modeled, unsupported = {}, [], []
    for name in ('team1', 'team2'):
        team = copy(getattr(state, name))
        team.players = [copy(player) for player in team.players]
        team.goalie = copy(team.goalie)
        setattr(future, name, team)
        for index, player in enumerate((*team.players, team.goalie)):
            slot = team.skater_scnum_base() + index
            if slot == owner:
                tracks[slot] = path
                team.players[index] = path[-1]
                continue
            if not on_ice(player):
                continue
            friendly = name == 'team1'
            supported = _supported(player, friendly)
            if supported and friendly and player.assignment in (4, 6):
                supported = all(support_zone(
                    _puck_at(state, original, carrier), sign, wing=player.assignment == 4
                ) == player.support_zone for carrier in path[1:])
            track = [player]
            for elapsed, carrier in enumerate(path[1:], 1):
                if supported:
                    puck = _puck_at(state, original, carrier)
                    player = response_step(player, puck, sign=sign if friendly else -sign,
                                           friendly=friendly, offsides=state.engine.offsides_enabled)
                else:
                    player = copy(track[0])
                    (player.x, player.y), player.projection_uncertainty = bounded_projection(
                        track[0], elapsed, boards=index < len(team.players))
                    player.precise_x, player.precise_y = float(player.x), float(player.y)
                track.append(player)
            tracks[slot] = track
            (modeled if supported else unsupported).append(slot)
            if index < len(team.players):
                team.players[index] = player
            else:
                team.goalie = player
    future.puck.x = state.puck.x + path[-1].x - original.x
    future.puck.y = state.puck.y + path[-1].y - original.y
    future.puck.motion_x, future.puck.motion_y = velocity(path[-1])
    clearance = math.inf
    for slot, track in tracks.items():
        if slot == owner:
            continue
        for first, second, other_first, other_second in zip(path, path[1:], track, track[1:]):
            clearance = min(clearance, _segment_clearance(
                (first.x - other_first.x, first.y - other_first.y),
                (second.x - other_second.x, second.y - other_second.y))
                - CONTACT_RADIUS - max(other_first.projection_uncertainty, other_second.projection_uncertainty))
    return future, {'response_model': 'native-stable-assignment-scenario',
                    'response_slots': modeled, 'response_unsupported_slots': unsupported,
                    'response_clearance': clearance}
