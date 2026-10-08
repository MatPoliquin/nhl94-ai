"""Live correction of ordinary receptions after native pass selection."""
import math

from nhl94_ai.agents.motion import VELOCITY_SCALE, arrival_time, puck_path, velocity


def passive_reception(state, receiver):
    """Measure body/stick clearance along the observed puck flight.

    assPassRec coasts with pfdoff until its timer expires. A marginal contact
    after that expiry depends on a fresh CPU decision and sprite change; the
    launch-time constant-velocity/stick estimate cannot certify it.
    """
    if any(value is None for value in (receiver.selection_flags, receiver.stick_x, receiver.stick_y,
                                       receiver.motion_x, receiver.motion_y, state.puck.motion_x,
                                       state.puck.motion_y, state.puck.height)):
        return None
    x = receiver.precise_x if receiver.precise_x is not None else float(receiver.x)
    y = receiver.precise_y if receiver.precise_y is not None else float(receiver.y)
    vx, vy = (round(component / VELOCITY_SCALE) for component in velocity(receiver))
    shift = 9 if receiver.selection_flags & 1 else 6
    closest, best_gap = None, math.inf
    for frame, point, height in puck_path(state.puck, 40, sample_interval=1):
        vx, vy = (component - ((component >> shift) or bool(component)) for component in (vx, vy))
        x, y = x + vx * VELOCITY_SCALE, y + vy * VELOCITY_SCALE
        if height > 8:
            continue
        gap = min(math.dist(point, (x, y)) - 8,
                  math.dist(point, (x + receiver.stick_x, y + receiver.stick_y)) - 14)
        if gap < best_gap:
            best_gap = gap
            closest = {'gap': gap, 'frames': frame, 'point': point}
    return closest


def reception_target(state, receiver):
    """Replan from the live flight, stopping at uncertain boards/net contact."""
    path = [(time, point) for time, point, height in puck_path(state.puck, 48, sample_interval=1)
            if height <= 8]
    if not path:
        return None
    return next((point for time, point in path if arrival_time(receiver, point) <= time), path[-1][1])
