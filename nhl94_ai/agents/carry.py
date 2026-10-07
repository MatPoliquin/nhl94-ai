"""Bounded ordinary carries and defender reach, using native skating updates."""
from copy import copy
import math

import numpy as np

from nhl94_ai.agents.defense import on_ice
from nhl94_ai.agents.motion import (
    VELOCITY_SCALE, bounded_projection, facing, skating, velocity,
)
from nhl94_ai.agents.passing import _contact_offsets
from nhl94_ai.agents.skating import grounded_step
from nhl94_ai.env.target_control import project_target


CARRY_FRAMES = 18
CONTACT_RADIUS = 16
INTERCEPTION_MARGIN = 3
BURST_LOCK_FRAMES = 24
CARRY_SAMPLES = (4, 8, 12, CARRY_FRAMES)
RISK_PADS = ((0, 0), (0, 1), (1, 1), (1, 0), (1, -1),
             (0, -1), (-1, -1), (-1, 0), (-1, 1))
RISK_FIELDS = ('x', 'y', 'precise_x', 'precise_y', 'motion_x', 'motion_y', 'vx', 'vy',
               'facing', 'facing_phase', 'orientation', 'sprite_flipped_x',
               'weight', 'agility', 'speed', 'energy', 'movement_bonus', 'burst_full_energy')
ENVELOPE_FIELDS = (*RISK_FIELDS, 'facing_phase')
PROJECTION_FIELDS = ('x', 'y', 'motion_x', 'motion_y', 'vx', 'vy', 'projection_uncertainty', 'role')


def carry_pad(player, target):
    """The ordinary controller holds these buttons until its next decision."""
    return tuple(1 if delta > 3 else -1 if delta < -3 else 0
                 for delta in (target[0] - player.x, target[1] - player.y))


def carry_path(player, target, frames=CARRY_FRAMES, *, decision_interval=4):
    if frames < 0 or decision_interval < 1:
        raise ValueError('Carry horizons must be nonnegative and decision intervals positive')
    carrier = copy(player)
    path = [carrier]
    for elapsed in range(frames):
        if elapsed % decision_interval == 0:
            pad = (0, 0) if target is None else carry_pad(carrier, target)
        carrier = grounded_step(carrier, pad)
        if math.dist((carrier.x, carrier.y), project_target((carrier.x, carrier.y))) > 1e-6:
            return None
        path.append(carrier)
    return path


def _segment_clearance(start, end):
    dx, dy = end[0] - start[0], end[1] - start[1]
    fraction = max(0, min(1, -(start[0] * dx + start[1] * dy) / max(0.01, dx * dx + dy * dy)))
    return math.hypot(start[0] + dx * fraction, start[1] + dy * fraction)


def reach_envelopes(player, frames):
    """Enclose arbitrary steering/braking and bursts at any unlocked time."""
    acceleration, _, _ = skating(player, optimistic=True)
    control = max(acceleration + math.sqrt(2) * VELOCITY_SCALE, 150 * math.sqrt(2) * VELOCITY_SCALE)
    scenarios = [copy(player)]
    boosted = copy(player)
    energy = (4096 if player.burst_full_energy is not False or player.energy is None
              else max(0, min(4096, player.energy)))
    impulse = (energy // 128) * 200 * VELOCITY_SCALE
    boosted.motion_x, boosted.motion_y = (
        component + round(direction * 200) * (energy // 128) * VELOCITY_SCALE
        for component, direction in zip(velocity(player), facing(player)))
    scenarios.append(boosted)
    envelopes = []
    for elapsed in range(frames + 1):
        # Native friction is nonexpansive; input acts after position integration.
        radius = control * elapsed * (elapsed - 1) / 2 + (math.sqrt(2) if elapsed else 0)
        delayed = sum(elapsed - launch for launch in range(1, elapsed, BURST_LOCK_FRAMES)) * impulse
        repeated = sum(elapsed - launch for launch in range(
            BURST_LOCK_FRAMES, elapsed, BURST_LOCK_FRAMES)) * impulse
        envelopes.append((tuple((p.x, p.y) for p in scenarios), (radius + delayed, radius + repeated)))
        scenarios = [grounded_step(p, (0, 0)) for p in scenarios]
    return envelopes


def cached_reach_envelopes(player, frames, cache):
    if cache is None:
        return reach_envelopes(player, frames)
    key = 'reach-envelopes', tuple(getattr(player, field) for field in ENVELOPE_FIELDS), frames
    if key not in cache:
        cache[key] = reach_envelopes(player, frames)
    return cache[key]


def _contact_reach(player, body):
    offsets = _contact_offsets(player)
    reach = CONTACT_RADIUS if body else max(
        [8, *(14 + math.hypot(*offset) for offset in offsets[1:])])
    if not body and player.shot_projection_offsets_x is not None and player.shot_projection_offsets_y is not None:
        reach = max(reach, 14 + max(math.hypot(x, y) for x in player.shot_projection_offsets_x
                                   for y in player.shot_projection_offsets_y))
    return reach


def interception_time(state, player, point, deadline, *, body=False, envelopes=None, extra_radius=0):
    """A lower bound on contact time, never a failed-pursuit arrival estimate."""
    deadline = math.ceil(deadline)
    if not on_ice(player):
        return deadline + 1
    envelopes = reach_envelopes(player, deadline) if envelopes is None else envelopes
    reach = _contact_reach(player, body)
    previous = envelopes[0][0]
    for elapsed, (centers, radii) in enumerate(envelopes[:deadline + 1]):
        if any(state._line_intersects_circle(
                start, end, point, reach + player.projection_uncertainty + radius + extra_radius)
               for start, end, radius in zip(previous, centers, radii)):
            return elapsed
        previous = centers
    return deadline + 1


def interception_times(player, points, deadlines, bodies, envelopes, *, puck_extent=0):
    """Batch the scalar segment/circle test over the identical reach envelopes."""
    deadlines = np.ceil(deadlines).astype(np.int64)
    if not on_ice(player):
        return deadlines + 1
    centers = np.asarray([entry[0] for entry in envelopes], dtype=np.float64)
    radii = np.asarray([entry[1] for entry in envelopes], dtype=np.float64)
    starts = np.concatenate((centers[:1], centers[:-1]))
    deltas = centers - starts
    lengths = deltas[..., 0]**2 + deltas[..., 1]**2
    targets = np.asarray(points, dtype=np.float64)[:, None, None, :]
    relative = targets - starts
    projection = relative[..., 0] * deltas[..., 0] + relative[..., 1] * deltas[..., 1]
    fractions = np.zeros_like(projection)
    np.divide(projection, lengths, out=fractions, where=lengths != 0)
    np.clip(fractions, 0, 1, out=fractions)
    nearest = starts + fractions[..., None] * deltas
    distance = targets - nearest
    squared = distance[..., 0]**2 + distance[..., 1]**2
    body_reach, puck_reach = _contact_reach(player, True), _contact_reach(player, False)
    reaches = np.asarray([body_reach if body else puck_reach for body in bodies])[:, None, None]
    extra = np.asarray([0 if body else puck_extent for body in bodies])[:, None, None]
    boundaries = reaches + player.projection_uncertainty + radii + extra
    contact = np.any(squared <= boundaries**2, axis=2)
    times = np.arange(len(envelopes))
    contact &= times <= deadlines[:, None]
    return np.where(np.any(contact, axis=1), np.argmax(contact, axis=1), deadlines + 1)


def risk_tracks(player, frames, cache):
    """Sample native steering scenarios; their failure never certifies safety."""
    key = tuple(getattr(player, field) for field in RISK_FIELDS), frames
    if key not in cache:
        boosted = copy(player)
        energy = (4096 if player.burst_full_energy is not False else player.energy)
        boosted.motion_x, boosted.motion_y = (
            component + round(direction * 200) * (energy // 128) * VELOCITY_SCALE
            for component, direction in zip(velocity(player), facing(player)))
        paths = []
        for start in (player, boosted):
            for pad in RISK_PADS:
                future, track = copy(start), [copy(start)]
                for _ in range(frames):
                    future = grounded_step(future, pad)
                    track.append(future)
                paths.append(track)
        cache[key] = paths
    return cache[key]


def assess_path_risk(state, player, path, pressure_state, delay, interval, cache, puck_extent):
    """Report sampled contact timing separately from conservative certification."""
    frames = len(path) - 1
    first = None
    offset = state.puck.x - player.x, state.puck.y - player.y
    for other in pressure_state.team2.players:
        if not on_ice(other):
            continue
        if any(getattr(other, field) is None
               for field in ('motion_x', 'motion_y', 'weight', 'agility', 'speed', 'energy')):
            return {'carry_viable': False, 'carry_status': 'unmeasured-defender-risk', 'carry_risk': 1.0}
        for track in risk_tracks(other, math.ceil(delay + frames), cache):
            for elapsed, carrier in enumerate(path[1:], 1):
                if first is not None and elapsed >= first:
                    break
                defender = track[math.ceil(delay + elapsed)]
                previous = track[math.ceil(delay + elapsed - 1)]
                body = _segment_clearance(
                    (path[elapsed - 1].x - previous.x, path[elapsed - 1].y - previous.y),
                    (carrier.x - defender.x, carrier.y - defender.y))
                stick = max((math.hypot(*value) for value in _contact_offsets(other)[1:]), default=0)
                puck = math.dist((carrier.x + offset[0], carrier.y + offset[1]), (defender.x, defender.y))
                if (body <= CONTACT_RADIUS + other.projection_uncertainty or
                        puck <= max(8, 14 + stick if stick else 8) + other.projection_uncertainty
                        + (puck_extent or 0)):
                    first = elapsed
                    break
    horizon = frames + INTERCEPTION_MARGIN
    risk = min(1.0, max(interval / horizon, (horizon - first) / horizon if first is not None else 0))
    return {
        'carry_viable': first is None,
        'carry_status': 'imminent-modeled-contact' if first is not None and first <= interval
        else 'uncertified-modeled-contact' if first is not None else 'uncertified-clear-samples',
        'carry_risk': risk, 'carry_sampled_contact_frame': first,
        'carry_risk_model': 'native-directional-samples-not-probability',
    }


def forecast_carry(state, player, target, *, decision_interval=4, pressure_state=None, pressure_delay=0,
                   assess_uncertainty=False, risk_cache=None):
    path = carry_path(player, target, decision_interval=decision_interval)
    return forecast_path(
        state, player, path, pressure_state=pressure_state, pressure_delay=pressure_delay,
        assess_uncertainty=assess_uncertainty, decision_interval=decision_interval, risk_cache=risk_cache)


def forecast_path(state, player, path, *, pressure_state=None, pressure_delay=0, puck_extent=None,
                  assess_uncertainty=False, decision_interval=4, risk_cache=None):
    """Check every frame of a held-input carry or a coasting shot windup."""
    details = {
        'carry_bounded': path is not None, 'carry_safe': False,
        'carry_clearance': -math.inf, 'carry_pressure': -math.inf,
        'carry_progress': -math.inf, 'carry_shot_value': 0.0,
        'carry_pressure_model': 'native-reach-envelope',
        'carry_viable': False, 'carry_status': 'unbounded-route', 'carry_risk': 1.0,
    }
    if path is None:
        return path, details
    frames = len(path) - 1
    details['carry_forecast_frames'] = frames
    clearance = body_clearance(state, player, path, cache=risk_cache)
    offset = state.puck.x - player.x, state.puck.y - player.y
    margin = 60.0
    pressure_state = state if pressure_state is None else pressure_state
    opponents = [other for other in pressure_state.team2.players if on_ice(other)]
    envelopes = {
        id(other): cached_reach_envelopes(other, math.ceil(
            pressure_delay + frames + INTERCEPTION_MARGIN), risk_cache) for other in opponents}
    if any(getattr(other, field) is None for other in opponents
           for field in ('motion_x', 'motion_y', 'weight', 'agility', 'speed', 'energy')):
        details['carry_pressure_model'] = 'conservative-missing-defender-feedback'
    if frames and opponents:
        points = [point for carrier in path[1:] for point in (
            (carrier.x, carrier.y),
            (carrier.x + offset[0], carrier.y + offset[1]) if puck_extent is None else (carrier.x, carrier.y))]
        bodies = [body for _ in path[1:] for body in (True, False)]
        elapsed = np.repeat(np.arange(1, frames + 1), 2)
        deadlines = pressure_delay + elapsed + INTERCEPTION_MARGIN
        for other in opponents:
            contacts = interception_times(other, points, deadlines, bodies, envelopes[id(other)],
                                          puck_extent=0 if puck_extent is None else puck_extent)
            margin = min(margin, float(np.min(contacts - pressure_delay - elapsed)))
    sign = 1 if state.team2.net.y > state.team1.net.y else -1
    details.update(
        carry_safe=clearance > 0 and margin >= INTERCEPTION_MARGIN,
        carry_clearance=clearance, carry_pressure=margin,
        carry_progress=(path[-1].y - player.y) * sign)
    details.update(carry_viable=details['carry_safe'],
                   carry_status='certified-clear' if details['carry_safe'] else
                   'projected-body-contact' if clearance <= 0 else 'uncertified')
    if details['carry_safe']:
        details['carry_risk'] = 0.0
    elif clearance > 0 and assess_uncertainty:
        details.update(assess_path_risk(
            state, player, path, pressure_state, pressure_delay, decision_interval,
            {} if risk_cache is None else risk_cache, puck_extent))
    return path, details


def body_clearance(state, player, path, *, cache=None):
    """Swept body separation shared by the primary route and achievable exits."""
    clearance = math.inf
    obstacles = [other for team in (state.team1, state.team2)
                 for other in (*team.players, team.goalie) if other is not player and on_ice(other)]
    for other in obstacles:
        key = 'body-projection', tuple(getattr(other, field) for field in PROJECTION_FIELDS), len(path)
        projections = None if cache is None else cache.get(key)
        if projections is None:
            projections = [bounded_projection(other, elapsed, boards=other.role != 0)
                           for elapsed in range(1, len(path))]
            if cache is not None:
                cache[key] = projections
        previous = player.x - other.x, player.y - other.y
        clearance = min(clearance, math.hypot(*previous) - CONTACT_RADIUS - other.projection_uncertainty)
        for carrier, (position, uncertainty) in zip(path[1:], projections):
            relative = carrier.x - position[0], carrier.y - position[1]
            clearance = min(clearance, _segment_clearance(previous, relative) - CONTACT_RADIUS - uncertainty)
            previous = relative
    return clearance
