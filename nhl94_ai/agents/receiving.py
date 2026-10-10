"""Live correction of ordinary receptions after native pass selection."""
import math

from nhl94_ai.agents.motion import VELOCITY_SCALE, arrival_time, puck_path, velocity

INCOMPATIBLE = ('shot_placement', 'possession_value', 'possession_ablation', 'receiver_selection',
                'offense_lookahead', 'uncertain_carry', 'chance_creation', 'cross_crease',
                'deke', 'classic_refinements')


def passive_reception(state, receiver, *, stick_only=False):
    """Measure contact clearance along the observed puck flight.

    assPassRec coasts with pfdoff until its timer expires. A marginal contact
    after that expiry depends on a fresh CPU decision and sprite change; the
    launch-time constant-velocity/stick estimate cannot certify it.

    The experimental stick-only forecast excludes body proximity: native
    checkpuckcoll calls puckstick only within radius 14 of GetHot and height 5.
    This remains a contact approximation, not a guarantee of native capture.
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
        if height > (5 if stick_only else 8):
            continue
        gap = math.dist(point, (x + receiver.stick_x, y + receiver.stick_y)) - 14
        if not stick_only:
            gap = min(math.dist(point, (x, y)) - 8, gap)
        if gap < best_gap:
            best_gap = gap
            closest = {'gap': gap, 'frames': frame, 'point': point}
    return closest


def reception_target(state, receiver, *, stick_only=False):
    """Replan from live flight and stick offset; arrival includes velocity/turning."""
    offset = (receiver.stick_x, receiver.stick_y) if stick_only else (0, 0)
    if any(value is None for value in offset):
        return None
    path = [(time, (point[0] - offset[0], point[1] - offset[1]))
            for time, point, height in puck_path(state.puck, 48, sample_interval=1)
            if height <= (5 if stick_only else 8)]
    if not path:
        return None
    return next((point for time, point in path if arrival_time(receiver, point) <= time), path[-1][1])
