"""Bounded, ROM-backed pass selection and interception estimates."""
from dataclasses import dataclass
from copy import copy
import math

import numpy as np

from nhl94_ai.agents.defense import eligible, on_ice
from nhl94_ai.agents.motion import VELOCITY_SCALE, arrival_time, bounded_projection, facing, rom_direction, skating, velocity
from nhl94_ai.agents.skating import grounded_step
from nhl94_ai.env.target_control import _clear_segment, project_target
from nhl94_ai.env.actions import HOCKEY_PASS_PRESS_FRAMES
from nhl94_ai.game.geometry import aim_pass


PASS_HORIZON = 48
RELEASE_FRAMES = 4
LIVE_RELEASE_FRAMES = 2
SAMPLES = 8
# Fixed common scenarios keep replanning and paired runs reproducible.
_SCENARIOS = np.random.default_rng(94).uniform(-1, 1, (SAMPLES, 2))


@dataclass(frozen=True)
class PassOption:
    index: int
    slot: int
    point: tuple
    flight_frames: float
    direction: int
    margin: float
    robustness: float
    forward_gain: float
    bypassed: int
    shot_value: float
    value: float
    continuation_value: float = 0.0
    continuation_target: tuple | None = None
    continuation_frames: int = 0
    continuation_position_value: float = 0.0
    continuation_finish_value: float = 0.0
    reception_value: float | None = None
    continuation_safe: bool | None = None
    continuation_risk: float = 0.0
    contact: tuple | None = None
    receiver_origin: tuple | None = None
    release_frames: int | None = None

    @property
    def opportunity(self):
        return max(self.shot_value if self.reception_value is None else self.reception_value,
                   self.continuation_value)


def pad_direction(passer, receiver):
    buttons = np.zeros(12, dtype=np.int8)
    aim_pass(buttons, passer, receiver)
    dx, dy = int(buttons[7] - buttons[6]), int(buttons[4] - buttons[5])
    return round(math.atan2(dx, dy) * 4 / math.pi) % 8 if dx or dy else 8


def selected_receiver(team, puck, passer_slot, direction):
    candidates = []
    for index, player in enumerate(team.players):
        slot = team.skater_scnum_base() + index
        if slot == passer_slot or player.role is not None and player.role <= 0 or player.unavailable & 4:
            continue
        dx, dy = player.x - puck.x, player.y - puck.y
        sector = rom_direction(dx, dy)
        if sector == 8:
            continue
        difference = (sector - direction + 4) % 8 - 4
        if abs(difference) <= 1:
            candidates.append((dx * dx + dy * dy + (difference * 256)**2, -slot, slot))
    return min(candidates)[2] if candidates else None


def pass_launch_view(state, passer):
    """The second held-B input selects and launches before later slots update.

    Earlier teammates have moved twice; later teammates have moved once. The
    puck is still at the end of the first frame, after pucknorm's quarter-step
    attraction to the carrier's stick. CPU steering and sprite offsets are held
    over this short forecast; this is not a prediction of new CPU decisions.
    """
    owner = state.engine.puck_owner
    team, puck = copy(state.team1), copy(state.puck)
    carrier = grounded_step(passer, (0, 0))
    team.players = []
    pads = ((0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 0), (0, 0))
    for index, player in enumerate(state.team1.players):
        slot = state.team1.skater_scnum_base() + index
        future = copy(player)
        if slot != owner and on_ice(player):
            known = (player.steering in range(10) if player.steering is not None else False)
            known &= all(getattr(player, field) is not None for field in ('weight', 'agility', 'speed', 'energy'))
            pad = pads[player.steering] if known else (0, 0)
            for _ in range(1 + (slot < owner)):
                future = grounded_step(future, pad, stop=known and player.steering == 9)
        team.players.append(future)
    shift = 9 if puck.friction == 511 / 512 else 6
    for axis, offset in (('x', passer.stick_x), ('y', passer.stick_y)):
        raw = round(velocity(puck)[axis == 'y'] / VELOCITY_SCALE)
        raw -= (raw >> shift) or bool(raw)
        precise = getattr(puck, 'precise_' + axis)
        integrated = math.floor((getattr(puck, axis) if precise is None else precise) + raw * VELOCITY_SCALE)
        hotspot = getattr(carrier, axis) + offset
        setattr(puck, axis, integrated + ((hotspot - integrated) >> 2))
    return team, puck


def option_receiver(receiver, option):
    """Effective pre-release origin for consumers of the shared pass forecast."""
    if option.receiver_origin is None:
        return receiver
    future = copy(receiver)
    future.x, future.y, future.motion_x, future.motion_y = option.receiver_origin
    return future


def pass_speed(passer):
    """Nominal rating-derived speed; live launch vectors include ROM quantization."""
    if passer.passing is None:
        return None
    speed = _nominal_speed(passer.passing, passer.role == 0)
    return speed * 65536 / (16 * 60) * VELOCITY_SCALE


def _nominal_speed(passing, goalie=False):
    speed = 160 + 2 * (8 if goalie else passing)
    return speed + speed // 16 if passing & 1 else speed


def _word(value):
    return (value + 32768) % 65536 - 32768


def _divide(value, divisor):
    quotient = abs(value) // abs(divisor)
    return -quotient if (value < 0) != (divisor < 0) else quotient


def _rom_sqrt(value):
    value &= 0xFFFFFFFF
    if value <= 1600:
        return math.isqrt(value)
    if value > 0xF00000:
        low, high = 0, 65535
        while True:
            middle = (low + high) // 2
            if middle == low:
                return low
            if middle * middle > value:
                high = middle
            else:
                low = middle
    guess = ((value >> 8) + 2) & 65535
    for _ in range(10):
        next_guess = ((value // guess + guess) & 65535) >> 1
        if next_guess == guess:
            break
        guess = next_guess
    return guess


def rom_pass_vector(dx, dy, vx, vy, passing, *, goalie=False):
    """passto's signed word arithmetic and eighth-second lead quantization."""
    speed = _nominal_speed(passing, goalie)
    qx, qy = _word((vx * 240) >> 16), _word((vy * 240) >> 16)
    ax, ay = dx >> 2, dy >> 2
    j = _word(2 * _word(qx * ax + qy * ay))
    k = qx * qx + qy * qy - (speed >> 2)**2
    root = _rom_sqrt(j * j - 4 * _word(k) * _word(ax * ax + ay * ay))
    divisor = _word(k) >> 2 or 1
    time = _divide(_word(root - j), divisor)
    if time < 0:
        time = _divide(_word(-root - j), divisor)
    time = time or 1
    if (time & 65535) > 24:
        time = 24
    result = []
    for distance, velocity in ((dx, qx), (dy, qy)):
        lead = velocity * time >> 1
        register = ((lead & 0xFFFF0000) | ((lead + distance) & 65535)) & 0xFFFFFFFF
        swapped = ((register << 16) | (register >> 16)) & 0xFFFFFFFF
        signed = (swapped + 0x80000000) % 0x100000000 - 0x80000000
        component = _divide(signed, time * 120)
        if not -32768 <= component <= 32767:
            return None
        result.append(component)
    return tuple(result)


def _live_pass_contact(puck, passer, receiver, delay):
    vx, vy = velocity(receiver)
    px, py = velocity(passer)
    start = math.floor(puck.x + px * delay), math.floor(puck.y + py * delay)
    target = (math.floor(receiver.x + vx * delay) + receiver.stick_x,
              math.floor(receiver.y + vy * delay) + receiver.stick_y)
    vector = rom_pass_vector(target[0] - start[0], target[1] - start[1],
                             round(vx / VELOCITY_SCALE), round(vy / VELOCITY_SCALE), passer.passing,
                             goalie=passer.role == 0)
    if vector is None:
        return None
    sx, sy = (component * VELOCITY_SCALE for component in vector)
    closest = math.inf, start, 0
    for frame in range(1, PASS_HORIZON + 1):
        travel = (1 - puck.friction**frame) / (1 - puck.friction) if puck.friction != 1 else frame
        point = start[0] + sx * travel, start[1] + sy * travel
        receiver_point = target[0] + vx * frame, target[1] + vy * frame
        distance = math.dist(point, receiver_point)
        if distance < closest[0]:
            closest = distance, point, frame
    distance, point, frame = closest
    if distance > 14 or math.dist(point, project_target(point)) > 1e-6:
        return None
    return start, point, frame, math.hypot(sx, sy) * puck.friction**frame


def pass_release_frames(receiver, delay=RELEASE_FRAMES):
    """Use the same release timing for launch prediction and collision deadlines."""
    return min(delay, LIVE_RELEASE_FRAMES) if receiver.stick_x is not None and receiver.stick_y is not None else delay


def pass_contact(puck, passer, receiver, delay=RELEASE_FRAMES):
    speed = pass_speed(passer)
    if speed is None or receiver.motion_x is None or receiver.motion_y is None:
        return None
    if receiver.stick_x is not None and receiver.stick_y is not None:
        return _live_pass_contact(puck, passer, receiver, pass_release_frames(receiver, delay))
    vx, vy = velocity(receiver)
    px, py = velocity(passer)
    fx, fy = facing(receiver)
    start = puck.x + px * delay, puck.y + py * delay
    # GetHot is sprite-dependent; use facing as an explicitly approximate hotspot.
    dx = receiver.x + vx * delay + fx * 10 - start[0]
    dy = receiver.y + vy * delay + fy * 10 - start[1]
    a, b, c = vx * vx + vy * vy - speed * speed, 2 * (dx * vx + dy * vy), dx * dx + dy * dy
    discriminant = b * b - 4 * a * c
    if discriminant < 0 or abs(a) < 0.01:
        return None
    roots = [t for t in ((-b - math.sqrt(discriminant)) / (2 * a),
                         (-b + math.sqrt(discriminant)) / (2 * a)) if t > 0]
    if not roots or min(roots) > PASS_HORIZON:
        return None
    time = min(roots)
    point = start[0] + dx + vx * time, start[1] + dy + vy * time
    if math.dist(point, project_target(point)) > 1e-6:
        return None
    return start, point, time, speed


def pass_point_at(puck, receiver, contact, time):
    start, point, flight, _ = contact
    live = receiver.stick_x is not None and receiver.stick_y is not None
    fraction = ((1 - puck.friction**time) / (1 - puck.friction**flight)
                if live and puck.friction != 1 else time / flight)
    return start[0] + (point[0] - start[0]) * fraction, start[1] + (point[1] - start[1]) * fraction


def one_timer_contact_frame(puck, receiver, contact, release):
    """First modeled body/stick contact, not the later closest-approach frame."""
    _, _, flight, _ = contact
    vx, vy = velocity(receiver)
    fx, fy = facing(receiver)
    offset = ((receiver.stick_x, receiver.stick_y) if receiver.stick_x is not None
              and receiver.stick_y is not None else (fx * 10, fy * 10))
    for time in (*range(1, math.ceil(flight)), flight):
        position = pass_point_at(puck, receiver, contact, time)
        body = receiver.x + vx * (release + time), receiver.y + vy * (release + time)
        stick = body[0] + offset[0], body[1] + offset[1]
        if math.dist(position, body) <= 8 or math.dist(position, stick) <= 14:
            return release + time
    return None


def pressure_margin(opponents, point, time):
    return min((arrival_time(p, (point[0] - offset[0], point[1] - offset[1]),
                             optimistic=True, boost=True) - time
                for p in opponents.players if p.role is None or p.role > 0
                for offset in _contact_offsets(p)), default=60)


def _contact_offsets(player):
    if player.stick_x is not None and player.stick_y is not None and (player.stick_x or player.stick_y):
        return (0, 0), (player.stick_x, player.stick_y)
    return ((0, 0),)


def _swept_contact(state, player, segment, times, radius, offset=(0, 0)):
    boards = player is not state.team1.goalie and player is not state.team2.goalie
    projections = [bounded_projection(player, time, boards=boards) for time in times]
    relative = [(point[0] - position[0] - offset[0], point[1] - position[1] - offset[1])
                for point, (position, _) in zip(segment, projections)]
    return state._line_intersects_circle(
        relative[0], relative[1], (0, 0), radius + max(uncertainty for _, uncertainty in projections))


def shot_value(state, shooter, point=None, delay=0):
    point = point or (shooter.x, shooter.y)
    sign = 1 if state.team2.net.y > state.team1.net.y else -1
    if abs(point[0]) > 71 or not 150 < point[1] * sign < 245:
        return 0.0
    goalie = state.team2.goalie
    gx, gy = velocity(goalie)
    targets = state._shot_target_points_for_net(state.team2.net)
    clear = 0
    for goal in targets:
        travel = math.dist(point, goal) / 4
        obstacles = [*state.team2.players, goalie]
        blocked = False
        for other in obstacles:
            if not on_ice(other):
                continue
            position, uncertainty = bounded_projection(other, delay + travel / 2, boards=other is not goalie)
            if state._line_intersects_circle(point, goal, position, (12 if other is goalie else 8) + uncertainty):
                blocked = True
                break
        clear += not blocked
    accuracy = shooter.shot_accuracy if shooter.shot_accuracy is not None else 15
    opening = min(20, abs(point[0] - goalie.x - gx * delay) / 3)
    return clear * 12 + max(0, point[1] * sign - 170) * 0.3 + accuracy * 0.3 + opening


def one_timer_position_ok(option, passer, current_shot):
    """Classic's comparative-position gate, shared with admission diagnostics."""
    return not (not (option.point[0] * passer.x <= 0 and abs(option.point[0] - passer.x) > 30)
                and option.shot_value <= current_shot + 12)


def evaluate_pass(state, passer, index, receiver, purpose='advance', *, decision_interval=4,
                  release_prediction=False, rank_risk=False, one_timer_window=True):
    """Evaluate native recipient/contact feasibility and estimated reception risk.

    rank_risk lets a caller price race/scenario uncertainty instead of applying
    the legacy thresholds. Physical obstructions and execution gates still veto.
    one_timer_window=False is an evaluation hook that skips only the shooting
    rectangle. Native contact, cue timing, offside and interception checks remain.
    """
    if decision_interval < 1:
        raise ValueError('Pass decision interval must be positive')
    slot = state.team1.skater_scnum_base() + index
    details = {'slot': slot, 'status': 'missing-feedback'}
    required = ('motion_x', 'motion_y', 'speed', 'agility', 'weight', 'energy', 'facing', 'stick')
    if (state.puck.height is None or not eligible(receiver)
            or any(getattr(p, field) is None for p in (passer, receiver)
                                     for field in required) or passer.passing is None):
        return None, details
    if receiver.projection_uncertainty:
        return None, {**details, 'status': 'uncertain-reception'}
    owner = state.engine.puck_owner
    direction = pad_direction(passer, receiver)
    release = pass_release_frames(receiver)
    live_launch = (release_prediction and passer.stick_x is not None and passer.stick_y is not None
                   and receiver.stick_x is not None and receiver.stick_y is not None)
    team, puck = pass_launch_view(state, passer) if live_launch else (state.team1, state.puck)
    recipient = selected_receiver(team, puck, owner, direction)
    details.update(recipient=recipient, recipient_model='native-slot-order-held-steering' if live_launch
                   else 'current-position',
                   recipient_unstable=recipient != selected_receiver(state.team1, state.puck, owner, direction))
    if recipient != slot:
        return None, {**details, 'status': 'different-rom-recipient'}
    contact = pass_contact(puck, passer, team.players[index], 0) if live_launch else pass_contact(
        state.puck, passer, receiver)
    if contact is None:
        return None, {**details, 'status': 'unreachable-reception'}
    start, point, flight, speed = contact
    if state.puck.height > 8 or not _clear_segment(start, point):
        return None, {**details, 'status': 'airborne-or-unsafe-route'}
    if live_launch:
        receiver = copy(team.players[index])
        receiver.x -= velocity(receiver)[0] * release
        receiver.y -= velocity(receiver)[1] * release
    total = flight + release
    sign = 1 if state.team2.net.y > state.team1.net.y else -1
    gain = (point[1] - passer.y) * sign
    details.update(point=point, flight_frames=flight, forward_gain=gain, direction=direction)
    live_geometry = receiver.stick_x is not None and receiver.stick_y is not None
    details.update(release_frames=release, geometry='rom-sprite-and-integer-launch' if live_geometry
                   else 'facing-hotspot-estimate')
    if purpose == 'one-timer':
        accuracy = receiver.shot_accuracy if receiver.shot_accuracy is not None else 15
        if one_timer_window and (abs(point[0]) > 70 or not 175 + max(0, 15 - accuracy) <= point[1] * sign < 245):
            return None, {**details, 'status': 'outside-one-timer-window'}
        cue = math.ceil(max(HOCKEY_PASS_PRESS_FRAMES, release) / decision_interval) * decision_interval
        details['cue_frame'] = cue
        first_contact = one_timer_contact_frame(state.puck, receiver, contact, release)
        details['first_contact_frame'] = first_contact
        if first_contact is None:
            return None, {**details, 'status': 'unreachable-reception'}
        if flight < 4 or first_contact <= cue:
            return None, {**details, 'status': 'too-short-for-one-timer-cue'}
    crosses_zone = state.puck.y * sign < 88 <= point[1] * sign
    if crosses_zone and state.engine.offsides_enabled is None:
        return None, {**details, 'status': 'unknown-offside-rule'}
    if (crosses_zone and state.engine.offsides_enabled
            and any(p is not passer and (
                p.y + velocity(p)[1] * (release + flight * (sign * 88 - start[1]) / (point[1] - start[1]))
            ) * sign >= 88 for p in state.team1.players)):
        return None, {**details, 'status': 'offside-receiver'}
    if purpose == 'advance' and gain < 20:
        return None, {**details, 'status': 'insufficient-progress'}
    if purpose != 'one-timer' and speed > (13000 + 350 * receiver.stick) * VELOCITY_SCALE:
        return None, {**details, 'status': 'too-fast-to-control'}
    opponents = [*state.team2.players, state.team2.goalie]
    if any(on_ice(p) and (p.motion_x is None or p.motion_y is None) for p in state.team2.players):
        return None, {**details, 'status': 'unknown-opponent-motion'}
    margin = 60
    safe = np.ones(SAMPLES, dtype=bool)
    steps = sorted(set([*range(4, math.ceil(flight), 4), flight]))
    previous_puck, previous_elapsed = start, release
    for time in steps:
        fraction = ((1 - state.puck.friction**time) / (1 - state.puck.friction**flight)
                    if live_geometry and state.puck.friction != 1 else time / flight)
        puck = start[0] + (point[0] - start[0]) * fraction, start[1] + (point[1] - start[1]) * fraction
        elapsed = time + release
        for friend in (*state.team1.players, state.team1.goalie):
            if friend is passer or friend is state.team1.players[index] or not on_ice(friend):
                continue
            if _swept_contact(state, friend, (previous_puck, puck), (previous_elapsed, elapsed), 8):
                return None, {**details, 'status': 'friendly-obstruction'}
            for offset in _contact_offsets(friend)[1:]:
                if _swept_contact(state, friend, (previous_puck, puck), (previous_elapsed, elapsed), 14, offset):
                    return None, {**details, 'status': 'friendly-stick-obstruction'}
        for other in opponents:
            if not on_ice(other):
                continue
            center, boundary_uncertainty = bounded_projection(
                other, elapsed, boards=other is not state.team2.goalie)
            body_radius = 14 if other is not state.team2.goalie else 16
            radius = body_radius + boundary_uncertainty
            if _swept_contact(state, other, (previous_puck, puck), (previous_elapsed, elapsed), body_radius):
                return None, {**details, 'status': 'moving-interception'}
            offsets = _contact_offsets(other)
            for offset in offsets[1:]:
                if _swept_contact(state, other, (previous_puck, puck), (previous_elapsed, elapsed), 14, offset):
                    return None, {**details, 'status': 'moving-stick-interception'}
            if other is not state.team2.goalie:
                arrival = min(arrival_time(other, (puck[0] - offset[0], puck[1] - offset[1]),
                                           optimistic=True, boost=True) for offset in offsets)
                margin = min(margin, arrival - elapsed)
                acceleration, _, _ = skating(other, optimistic=True)
                uncertainty = acceleration * min(elapsed, 12)**2 / 4
                for sample, offset in enumerate(_SCENARIOS):
                    safe[sample] &= all(math.dist((center[0] + contact[0] + offset[0] * uncertainty,
                                                   center[1] + contact[1] + offset[1] * uncertainty), puck) > radius
                                        for contact in offsets)
        previous_puck, previous_elapsed = puck, elapsed
    reception = pressure_margin(state.team2, point, total + (0 if purpose == 'one-timer' else 4))
    margin = min(margin, reception)
    robustness = float(safe.mean())
    details.update(margin=margin, robustness=robustness)
    if not rank_risk and (margin < 3 or robustness < 0.875):
        return None, {**details, 'status': 'contested-reception-or-lane'}
    bypassed = sum(passer.y * sign < p.y * sign < point[1] * sign
                   for p in state.team2.players if on_ice(p))
    shooting = shot_value(state, receiver, point, total)
    value = gain * 0.4 + bypassed * 16 + min(margin, 20) + shooting * 0.6
    option = PassOption(index, slot, point, flight, direction, margin, robustness, gain, bypassed, shooting, value,
                        contact=contact, receiver_origin=(receiver.x, receiver.y, *velocity(receiver)),
                        release_frames=release)
    return option, {**details, 'status': 'safe', 'bypassed': bypassed,
                    'shot_value': shooting, 'value': value}
