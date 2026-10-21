"""Bounded, ROM-backed pass selection and interception estimates."""
from dataclasses import dataclass
from copy import copy
import math

import numpy as np

from nhl94_ai.agents.classic_tuning import current as tuning

from nhl94_ai.game.actors import controllable_skater, observable_body
# rom_direction stays imported as a re-export for Genesis CPU-model callers.
from nhl94_ai.game.motion import (  # pylint: disable=unused-import
    VELOCITY_SCALE, arrival_time, bounded_projection, rom_direction, skating, velocity,
)
from nhl94_ai.env.target_control import _clear_segment
from nhl94_ai.env.actions import HOCKEY_PASS_PRESS_FRAMES
# Physical helpers remain available to callers of the original module.
from nhl94_ai.game.passing import (  # pylint: disable=unused-import
    _backend, PASS_HORIZON, RELEASE_FRAMES,
    LIVE_RELEASE_FRAMES, pad_direction,
    selected_receiver, pass_launch_view,
    option_receiver, pass_speed, _nominal_speed,
    _word, _divide, _rom_sqrt,
    rom_pass_vector, _live_pass_contact,
    pass_release_frames, pass_contact,
    pass_point_at, one_timer_contact_frame,
)


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
    gx, _gy = velocity(goalie)
    targets = state._shot_target_points_for_net(state.team2.net)
    clear = 0
    for goal in targets:
        travel = math.dist(point, goal) / 4
        obstacles = [*state.team2.players, goalie]
        blocked = False
        for other in obstacles:
            if not observable_body(other):
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
    backend = _backend(passer)
    native_ratings = backend.native_ratings
    if not native_ratings:
        required = ('motion_x', 'motion_y', 'facing')
    if (state.puck.height is None or not controllable_skater(receiver)
            or any(getattr(p, field) is None for p in (passer, receiver)
                                     for field in required) or native_ratings and passer.passing is None):
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
    details.update(recipient=recipient, recipient_model=backend.pass_recipient_models[0 if live_launch else 1],
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
        receiver.precise_x, receiver.precise_y = float(receiver.x), float(receiver.y)
    total = flight + release
    sign = 1 if state.team2.net.y > state.team1.net.y else -1
    gain = (point[1] - passer.y) * sign
    details.update(point=point, flight_frames=flight, forward_gain=gain, direction=direction)
    live_geometry = receiver.stick_x is not None and receiver.stick_y is not None
    details.update(release_frames=release, geometry=backend.pass_geometry_models[0 if live_geometry else 1])
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
    if purpose == 'advance' and gain < tuning().advance_pass_min_progress:
        return None, {**details, 'status': 'insufficient-progress'}
    if native_ratings and purpose != 'one-timer' and speed > (13000 + 350 * receiver.stick) * VELOCITY_SCALE:
        return None, {**details, 'status': 'too-fast-to-control'}
    opponents = [*state.team2.players, state.team2.goalie]
    if any(observable_body(p) and (p.motion_x is None or p.motion_y is None) for p in state.team2.players):
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
            if friend is passer or friend is state.team1.players[index] or not observable_body(friend):
                continue
            if _swept_contact(state, friend, (previous_puck, puck), (previous_elapsed, elapsed), 8):
                return None, {**details, 'status': 'friendly-obstruction'}
            for offset in _contact_offsets(friend)[1:]:
                if _swept_contact(state, friend, (previous_puck, puck), (previous_elapsed, elapsed), 14, offset):
                    return None, {**details, 'status': 'friendly-stick-obstruction'}
        for other in opponents:
            if not observable_body(other):
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
                   for p in state.team2.players if observable_body(p))
    shooting = shot_value(state, receiver, point, total)
    params = tuning()
    value = (gain * params.pass_progress_weight + bypassed * params.pass_bypassed_defender_weight
             + min(margin, 20) + shooting * params.pass_receiver_shot_weight)
    option = PassOption(index, slot, point, flight, direction, margin, robustness, gain, bypassed, shooting, value,
                        contact=contact, receiver_origin=(receiver.x, receiver.y, *velocity(receiver)),
                        release_frames=release)
    return option, {**details, 'status': 'safe', 'bypassed': bypassed,
                    'shot_value': shooting, 'value': value}
