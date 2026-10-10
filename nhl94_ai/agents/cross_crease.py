"""Opt-in held-C finishing: bounded opportunity estimates and native feedback."""
from collections import Counter
from copy import copy
from dataclasses import asdict, dataclass
from itertools import islice
import math

import numpy as np

from nhl94_ai.agents.defense import controlled_slot, eligible, on_ice
from nhl94_ai.agents.goalie import DIVE_ANIMATION, SAVE_ANIMATIONS
from nhl94_ai.agents.motion import arrival_time, coast_projection, skate_step, skating, stop_projection, velocity
from nhl94_ai.agents.offense import (
    GOALIE_SHOT_CLEARANCE, _braking, _carry_motion, goalie_avoidance,
)
from nhl94_ai.agents.passing import shot_value
from nhl94_ai.env.target_control import project_target, route_waypoint, steering_direction
from nhl94_ai.game.constants import GameConsts as Buttons


HOLD_FRAMES = 24
SWING_FRAMES = 14
EARLY_RELEASE_FRAMES = 2
REPLAN_FRAMES = 12
APPROACH_FRAMES = 160
SEQUENCE_FRAMES = APPROACH_FRAMES + 80
COOLDOWN_FRAMES = 96
VALUE_MARGIN = 8
# checkcx's combined player-body radius is 16; retain four units of uncertainty
# during the planned crossing and live windup.
WINDUP_CLEARANCE = 20


@dataclass(frozen=True)
class CrossCreasePlan:
    slot: int
    direction: int
    target: tuple
    approach_frames: int
    release_point: tuple
    aim_side: int
    value: float
    pressure: float
    clearance: float
    shot_frames: int
    route: tuple
    charge_point: tuple
    risk: float


def crossing_entry(state, player):
    sign = 1 if state.team2.net.y > state.team1.net.y else -1
    return player is not None and 88 <= player.y * sign <= 244 and 8 <= abs(player.x) <= 120


def shot_lane_clear(state, point, side, delay=0, *, include_goalie=True, goal_x=16):
    goal = side * goal_x, state.team2.net.y
    distance = math.dist(point, goal)
    obstacles = (*state.team2.players, state.team2.goalie) if include_goalie else state.team2.players
    for other in obstacles:
        if not on_ice(other):
            continue
        vx, vy = velocity(other)
        along = max(0, min(1, ((other.x - point[0]) * (goal[0] - point[0])
                              + (other.y - point[1]) * (goal[1] - point[1])) / max(1, distance**2)))
        time = min(12, delay + along * distance / 4)
        position = other.x + vx * time, other.y + vy * time
        if state._line_intersects_circle(point, goal, position, 12 if other is state.team2.goalie else 8):
            return False
    return True


def _segment_distance(start, end):
    dx, dy = end[0] - start[0], end[1] - start[1]
    fraction = max(0, min(1, -(start[0] * dx + start[1] * dy) / max(1e-9, dx * dx + dy * dy)))
    return math.hypot(start[0] + dx * fraction, start[1] + dy * fraction)


def _arrival_margin(state, point, elapsed):
    return min((arrival_time(other, point) - min(elapsed, 24)
                for other in state.team2.players if other.role is None or other.role > 0), default=60.0)


def _path_clear(state, player, samples, *, goalie_clearance=20, pressure=True, start=None, start_time=0):
    owner = state.team1.get_player_by_scnum(state.engine.puck_owner)
    if pressure and owner is None:
        return False, 'possession-lost', 60.0, None
    previous = (player.x, player.y) if start is None else start
    previous_time = start_time
    minimum, margin = None, 60.0
    for elapsed, point in samples:
        if math.dist(point, project_target(point)) > 1e-6 or route_waypoint(previous, point) != point:
            return False, 'unsafe-net-route', margin, minimum
        if pressure:
            offset = state.puck.x - owner.x, state.puck.y - owner.y
            puck_point = point[0] + offset[0], point[1] + offset[1]
            margin = min(margin, _arrival_margin(state, point, elapsed),
                         _arrival_margin(state, puck_point, elapsed))
        for team in (state.team1, state.team2):
            for other in (*team.players, team.goalie):
                if other is player or other is owner or not on_ice(other):
                    continue
                vx, vy = velocity(other)
                start_time, end_time = min(previous_time, 12), min(elapsed, 12)
                start = previous[0] - other.x - vx * start_time, previous[1] - other.y - vy * start_time
                end = point[0] - other.x - vx * end_time, point[1] - other.y - vy * end_time
                distance = _segment_distance(start, end)
                if other is state.team2.goalie:
                    minimum = distance if minimum is None else min(minimum, distance)
                radius = goalie_clearance if other is state.team2.goalie else 18
                # A clear route cannot pass through a stationary occupant.
                # Beyond live motion's horizon, possible pursuit is a score cost.
                known_obstacle = elapsed <= 12 or math.hypot(vx, vy) < 0.2
                if known_obstacle and distance <= radius:
                    return False, 'goalie-contact' if other is state.team2.goalie else 'occupied-crossing', margin, minimum
        previous, previous_time = point, elapsed
    return True, 'clear', margin, minimum


def crossing_clear(state, player, frames, *, pressure=True, goalie_clearance=GOALIE_SHOT_CLEARANCE, target=None):
    samples = []
    for elapsed in (*range(4, frames, 4), frames):
        if target is None:
            point = coast_projection(player, elapsed)
        else:
            motion = _carry_motion(player, target, elapsed)
            if motion is None:
                return False, 'unsafe-net-route', 60.0, None
            point = motion[0]
        samples.append((elapsed, point))
    return _path_clear(state, player, samples, goalie_clearance=goalie_clearance, pressure=pressure)


def _route_motion(player, route, *, horizon=APPROACH_FRAMES):
    """Project only the carrier, using the same steering as live execution."""
    carrier = copy(player)
    carrier.x, carrier.y = float(player.x), float(player.y)
    carrier.motion_x, carrier.motion_y = velocity(player)
    carrier.facing = player.facing if player.facing is not None else player.orientation
    acceleration, limit, _ = skating(player)
    turning, waypoint = 0, 0
    for frame in range(horizon + 1):
        if waypoint < len(route) - 1 and math.dist((carrier.x, carrier.y), route[waypoint]) <= 10:
            waypoint += 1
        yield frame, copy(carrier), waypoint
        pad, _ = steering_direction(carrier, route[waypoint])
        brake_target = carrier.x + pad[0] * 48, carrier.y + pad[1] * 48
        if _braking(carrier, brake_target):
            motion = stop_projection(carrier, 1)
            turning = 0
        else:
            motion, carrier.facing, turning = skate_step(
                ((carrier.x, carrier.y), velocity(carrier)), carrier.facing, turning, pad, acceleration, limit)
        (carrier.x, carrier.y), (carrier.motion_x, carrier.motion_y) = motion


def _risk_cost(margin):
    return min(18.0, max(0.0, 10 - margin) * 0.5)


def _shot_scene(state, carrier, direction, elapsed):
    """Potential windup-angle commitment, not an asserted future ROM save."""
    scene = copy(state)
    scene.team2 = copy(state.team2)
    scene.team2.players = [copy(other) for other in state.team2.players]
    for other in scene.team2.players:
        if not on_ice(other):
            continue
        vx, vy = velocity(other)
        moved = other.x + vx * min(elapsed, 12), other.y + vy * min(elapsed, 12)
        if math.dist(moved, project_target(moved)) < 1e-6:
            other.x, other.y = moved
        other.motion_x = other.motion_y = 0
    goalie = scene.team2.goalie = copy(state.team2.goalie)
    sign = 1 if state.team2.net.y > state.team1.net.y else -1
    bait = coast_projection(carrier, 8)
    vx, _ = velocity(carrier)
    u, v = bait[0] + direction * 6 + vx * 4, bait[1] - sign * 260
    length = max(30, math.hypot(u, v))
    goalie.x, goalie.y = 34 * u / length, sign * 260 + 26 * v / length
    goalie.motion_x = goalie.motion_y = 0
    return scene


def _goalie_sweep(state, carrier, projected, frames):
    goalie = state.team2.goalie
    previous = carrier.x - goalie.x, carrier.y - goalie.y
    minimum = math.hypot(*previous)
    for elapsed in (*range(4, frames, 4), frames):
        point = coast_projection(carrier, elapsed)
        fraction = min(1, elapsed / 24)
        center = (goalie.x + (projected.x - goalie.x) * fraction,
                  goalie.y + (projected.y - goalie.y) * fraction)
        relative = point[0] - center[0], point[1] - center[1]
        minimum = min(minimum, _segment_distance(previous, relative))
        previous = relative
    return minimum


def _windup_option(state, carrier, direction, route, elapsed, approach_margin=60, clearance=None):
    owner = state.team1.get_player_by_scnum(state.engine.puck_owner)
    if owner is None:
        return None, {'status': 'possession-lost'}
    shot_frames = HOLD_FRAMES + (8 if carrier.shot_power < 20 else SWING_FRAMES)
    body = coast_projection(carrier, shot_frames)
    sign = 1 if state.team2.net.y > state.team1.net.y else -1
    points = [(body[0] + direction * offset, body[1] - sign * 4) for offset in (6, 16)]
    current_puck_x = carrier.x + state.puck.x - owner.x
    if (current_puck_x * direction >= 6 or points[0][0] * direction < 0
            or (points[0][0] - current_puck_x) * direction < 24):
        return None, {'status': 'crossing-not-reachable-during-windup'}
    if max(abs(point[0]) for point in points) > 60 or not 160 <= points[0][1] * sign <= 245:
        return None, {'status': 'outside-release-window'}
    clear, reason, margin, coast_clearance = crossing_clear(
        state, carrier, shot_frames, goalie_clearance=WINDUP_CLEARANCE)
    if not clear:
        return None, {'status': reason, 'pressure': margin, 'clearance': coast_clearance}
    sides = [side for side in (direction, -direction) if any(
        shot_lane_clear(state, point, side, elapsed + shot_frames, include_goalie=False) for point in points)]
    if not sides:
        return None, {'status': 'blocked-release-lane'}
    scene = _shot_scene(state, carrier, direction, elapsed + shot_frames)
    anticipated_clearance = _goalie_sweep(state, carrier, scene.team2.goalie, shot_frames)
    if anticipated_clearance <= 18:
        return None, {'status': 'goalie-windup-contact', 'clearance': anticipated_clearance}
    side = sides[0]
    margin = min(approach_margin, margin)
    risk = _risk_cost(margin)
    value = min(shot_value(scene, carrier, point=point)
                for point in points) - elapsed * 0.08 - risk
    release = (points[0][0] + points[1][0]) / 2, points[0][1]
    minimum = coast_clearance if clearance is None else (
        clearance if coast_clearance is None else min(clearance, coast_clearance))
    option = CrossCreasePlan(
        state.engine.puck_owner, direction, route[-1], elapsed, release, side,
        value, margin, minimum if minimum is not None else 60.0, shot_frames,
        route, (carrier.x, carrier.y), risk)
    return option, {'status': 'feasible', 'value': value, 'pressure': margin, 'risk': risk,
                    'release_point': release, 'charge_point': (carrier.x, carrier.y)}


def evaluate_cross_crease(state, alternatives, *, prepare=True, max_depth=None):
    """Compare heuristic opportunity values, never calibrated goal probabilities."""
    player = state.team1.get_player_by_scnum(state.engine.puck_owner)
    diagnostics = {'alternatives': dict(alternatives), 'candidates': []}
    required = ('motion_x', 'motion_y', 'shot_power', 'handedness', 'live_anim', 'live_anim_frame')
    if player is None or player is state.team1.goalie:
        return None, {**diagnostics, 'status': 'missing-feedback'}
    missing = [field for field in required if getattr(player, field) is None]
    missing.extend('goalie.' + field for field in ('live_anim', 'motion_x', 'motion_y')
                   if getattr(state.team2.goalie, field) is None)
    if (not state.engine.puck_owner_known or controlled_slot(state.team1) != state.engine.puck_owner
            or missing):
        return None, {**diagnostics, 'status': 'missing-feedback', 'missing_feedback': missing}
    sign = 1 if state.team2.net.y > state.team1.net.y else -1
    if state.is_shootout_active or state.is_shooting or not eligible(player):
        return None, {**diagnostics, 'status': 'ineligible-state'}
    if not crossing_entry(state, player):
        return None, {**diagnostics, 'status': 'outside-crossing-entry'}
    direction = -1 if player.x > 0 else 1
    choices = []
    progress = player.y * sign
    depths = tuple(dict.fromkeys((min(220, max(192, progress)), 208, 220)))
    if max_depth is not None:
        depths = tuple(dict.fromkeys(min(max_depth, depth) for depth in depths))
    routes = [((direction * 50, sign * depth),) for depth in depths]
    if prepare:
        width = min(75, max(45, abs(player.x) - 15))
        routes.extend(((-direction * width, sign * depth), (direction * 50, sign * depth))
                      for depth in depths)
    immediate, _ = _windup_option(state, player, direction, routes[0], 0)
    if immediate is not None and immediate.value > max(alternatives.values(), default=0) + VALUE_MARGIN:
        return immediate, {**diagnostics, 'status': 'selected', 'plan': asdict(immediate),
                           'goalie_model': 'potential-windup-angle-commit'}
    for route in routes:
        details = {'route': route, 'target': route[-1]}
        diagnostics['candidates'].append(details)
        previous, previous_time = (player.x, player.y), 0
        margin, clearance = 60.0, None
        iterator = _route_motion(player, route) if prepare else ((0, player, 0),)
        for elapsed, carrier, waypoint in iterator:
            if elapsed % 4:
                continue
            samples = [(elapsed, (carrier.x, carrier.y))] if elapsed else []
            clear, reason, current_margin, current_clearance = _path_clear(
                state, player, samples, start=previous, start_time=previous_time)
            previous, previous_time = (carrier.x, carrier.y), elapsed
            margin = min(margin, current_margin)
            if current_clearance is not None:
                clearance = current_clearance if clearance is None else min(clearance, current_clearance)
            if not clear:
                details.update(status=reason, approach_frames=elapsed)
                break
            option, result = _windup_option(state, carrier, direction, route, elapsed, margin, clearance)
            details.update(result, approach_frames=elapsed)
            if option is not None:
                choices.append(option)
                if option.value > max(alternatives.values(), default=0) + VALUE_MARGIN:
                    break
            if waypoint == len(route) - 1 and carrier.x * direction > 8:
                break
    if not choices:
        return None, {**diagnostics, 'status': 'no-safe-crossing'}
    incumbent = max(alternatives.values(), default=0)
    worthwhile = [option for option in choices if option.value > incumbent + VALUE_MARGIN]
    if not worthwhile:
        return None, {**diagnostics, 'status': 'better-alternative',
                      'value': max(option.value for option in choices)}
    best_value = max(option.value for option in worthwhile)
    best = min((option for option in worthwhile if option.value >= best_value - 4),
               key=lambda option: option.approach_frames)
    return best, {**diagnostics, 'status': 'selected', 'plan': asdict(best),
                  'goalie_model': 'potential-windup-angle-commit'}


class CrossCreaseController:
    def __init__(self):
        self.plan = None
        self.phase = 'idle'
        self.started = self.pressed = self.release_at = self.retry_at = 0
        self.aim_side = 1
        self.committed = False
        self.event = None
        self.events = []
        self.metrics = Counter()
        self.diagnostics = {}
        self.waypoint = 0

    def start(self, plan, state, frame):
        if self.plan is not None:
            raise RuntimeError('Cannot replace an active cross-crease trajectory')
        self.plan, self.started = plan, frame
        self.phase, self.aim_side, self.committed = 'approach', plan.aim_side, False
        self.waypoint = 0
        self.metrics['plans'] += 1
        self.event = {
            'frame': frame, 'slot': plan.slot, 'plan': asdict(plan),
            'shots_before': state.team1.stats.shots, 'goals_before': state.team1.stats.score,
            'pressed': False, 'windup': False, 'committed': False, 'shot': False, 'contact': False,
            'goalie_anim_before': state.team2.goalie.live_anim,
            'goalie_position_before': (state.team2.goalie.x, state.team2.goalie.y),
            'alternatives': dict(self.diagnostics.get('alternatives', {})),
            'goalie_impact': state.team2.goalie.contact_impact or 0,
            'skater_impact': state.team1.get_player_by_scnum(plan.slot).contact_impact or 0,
            'start_position': (state.team1.get_player_by_scnum(plan.slot).x,
                               state.team1.get_player_by_scnum(plan.slot).y),
        }

    def _finish(self, frame, outcome):
        if self.event is not None:
            self.events.append({**self.event, 'end_frame': frame, 'outcome': outcome})
            self.metrics[outcome] += 1
        pressed = self.event is not None and self.event['pressed']
        self.plan = self.event = None
        self.phase = 'idle'
        self.retry_at = frame + (COOLDOWN_FRAMES if pressed else 12)
        self.diagnostics.update(outcome=outcome, phase='idle', mode='cross-crease-ended', metrics=dict(self.metrics))

    def observe(self, state, frame):
        if self.event is None:
            return
        event, slot = self.event, self.event['slot']
        goalie = state.team2.goalie
        player = state.team1.get_player_by_scnum(slot)
        goalie_impact = goalie.contact_impact or 0
        skater_impact = player.contact_impact or 0 if player is not None else 0
        goalie_contact = goalie.contact_player == slot and goalie_impact > event['goalie_impact']
        skater_contact = (player is not None and player.contact_player == state.team2.goalie_scnum()
                         and skater_impact > event['skater_impact'])
        event['goalie_impact'], event['skater_impact'] = goalie_impact, skater_impact
        if not event['contact'] and (goalie_contact or skater_contact):
            event['contact'] = True
            self.metrics['goalie_contacts'] += 1
        if (not event['shot'] and event['pressed'] and state.team1.stats.shots > event['shots_before']
                and state.engine.shot_player == slot):
            event['shot'] = True
            self.metrics['recorded_shots'] += 1
        if event['shot'] and state.team1.stats.score > event['goals_before'] and state.engine.shot_player == slot:
            self._finish(frame, 'goals')
        elif state.engine.puck_owner >= 0 and state.engine.puck_owner != slot:
            self._finish(frame, 'shot-ended' if event['shot'] else 'possession-lost')
        elif state.engine.clock_stopped:
            self._finish(frame, 'play-stopped')
        elif frame - self.started >= SEQUENCE_FRAMES:
            self._finish(frame, 'shot-ended' if event['shot'] else 'shot-unconfirmed')

    def step(self, state, frame, *, c_down=False):
        if self.plan is None:
            return None
        plan = self.plan
        player = state.team1.get_player_by_scnum(plan.slot)
        action = np.zeros(Buttons.INPUT_MAX, dtype=np.int8)
        if player is None or controlled_slot(state.team1) != plan.slot or player.is_falling:
            self._finish(frame, 'control-lost')
            return action
        missing = [field for field in ('motion_x', 'motion_y', 'live_anim', 'live_anim_frame')
                   if getattr(player, field) is None]
        missing.extend('goalie.' + field for field in ('motion_x', 'motion_y', 'live_anim')
                       if getattr(state.team2.goalie, field) is None)
        if missing and (self.phase == 'approach' or self.event['shot']):
            self.diagnostics['missing_feedback'] = missing
            self._finish(frame, 'feedback-lost')
            return action
        if self.phase == 'approach':
            if state.engine.puck_owner != plan.slot:
                self._finish(frame, 'possession-lost')
                return action
            elapsed = frame - self.started
            if self.waypoint < len(plan.route) - 1 and math.dist(
                    (player.x, player.y), plan.route[self.waypoint]) <= 10:
                self.waypoint += 1
            ready, preview = _windup_option(state, player, plan.direction, plan.route, 0)
            if ready is not None and ready.value <= max(self.event['alternatives'].values(), default=0) + VALUE_MARGIN:
                ready = None
            if ready is None:
                if elapsed >= APPROACH_FRAMES or self.waypoint == len(plan.route) - 1 and player.x * plan.direction > 12:
                    self._finish(frame, 'crossing-window-missed')
                    return action
                if elapsed and elapsed % REPLAN_FRAMES == 0:
                    sign = 1 if state.team2.net.y > state.team1.net.y else -1
                    replacement, details = evaluate_cross_crease(
                        state, self.event['alternatives'], max_depth=plan.target[1] * sign)
                    self.diagnostics['replan_status'] = details['status']
                    if replacement is not None and replacement.route != plan.route:
                        self.event.setdefault('replans', []).append({
                            'frame': frame, 'previous_route': plan.route, 'plan': asdict(replacement)})
                        self.metrics['route_replans'] += 1
                        self.plan = plan = replacement
                        self.waypoint = 0
                remaining = plan.route[self.waypoint:]
                samples = [(time, (future.x, future.y)) for time, future, _ in islice(
                    _route_motion(player, remaining), 13) if time and time % 4 == 0]
                clear, reason, _, _ = _path_clear(state, player, samples)
                target = remaining[0]
                if route_waypoint((player.x, player.y), target) != target or not clear:
                    self._finish(frame, 'approach-invalidated')
                    self.diagnostics['reason'] = reason
                    return action
                pad, _ = steering_direction(player, target)
                self._pad(action, pad)
                self.diagnostics.update(waypoint=target, arming_status=preview['status'])
            else:
                self.aim_side = ready.aim_side
                if not c_down:
                    self.plan = plan = ready
                    self.event['executed_plan'] = asdict(ready)
                    self.event['approach_elapsed'] = elapsed
                    self.phase, self.pressed = 'hold', frame
                    self.event['pressed'] = True
                    self.event['goalie_anim_before'] = state.team2.goalie.live_anim
                    self.event['goalie_position_before'] = (state.team2.goalie.x, state.team2.goalie.y)
                    self.event['charge_position'] = (player.x, player.y)
                    self.event['charge_frame'] = frame
                    self.metrics['attempts'] += 1
                    action[Buttons.INPUT_C] = 1
                action[Buttons.INPUT_RIGHT if self.aim_side > 0 else Buttons.INPUT_LEFT] = 1
        else:
            elapsed = frame - self.pressed
            goalie = state.team2.goalie
            windup = state.is_shooting and state.engine.puck_owner == plan.slot
            if windup and not self.event['windup']:
                self.event['windup'] = True
                self.metrics['accepted_windups'] += 1
            save_commit = goalie.live_anim in (*SAVE_ANIMATIONS, DIVE_ANIMATION)
            gx, _ = velocity(goalie)
            lateral_commit = gx * plan.direction < -0.12 and (
                state.puck.x - goalie.x) * plan.direction > 12
            before_y = self.event['goalie_position_before'][1]
            charge_y = self.event['charge_position'][1]
            advance = (goalie.y - before_y) * math.copysign(1, charge_y - before_y)
            angle_commit = advance >= 4
            committed = windup and (save_commit or lateral_commit or angle_commit)
            if committed and not self.committed:
                self.committed = self.event['committed'] = True
                self.event['commit_frame'] = frame
                if save_commit and goalie.live_anim != self.event['goalie_anim_before']:
                    self.metrics['pre_release_commits'] += 1
                    self.event['commit_kind'] = 'save-animation'
                elif lateral_commit:
                    self.metrics['lateral_commits'] += 1
                    self.event['commit_kind'] = 'wrong-way-motion'
                elif angle_commit:
                    self.metrics['angle_commits'] += 1
                    self.event['commit_kind'] = 'windup-advance'
            if not windup and not self.event['windup'] and elapsed >= 3:
                self._finish(frame, 'shot-not-started')
                return action
            if self.phase == 'hold':
                if missing:
                    clear, reason, margin, clearance, opening = False, 'feedback-lost', None, None, False
                    self.diagnostics['missing_feedback'] = missing
                else:
                    clear, reason, margin, clearance = crossing_clear(
                        state, player, 12, goalie_clearance=WINDUP_CLEARANCE if windup else GOALIE_SHOT_CLEARANCE)
                    # Releasing during the early windup can jump straight to the
                    # native release hotspot; do not assume a full 14-frame swing.
                    body = coast_projection(player, EARLY_RELEASE_FRAMES)
                    point = state.puck.x + body[0] - player.x, state.puck.y + body[1] - player.y
                    opening = (body[0] * plan.direction >= 4 and point[0] * plan.direction >= 4
                               and shot_lane_clear(state, point, self.aim_side, EARLY_RELEASE_FRAMES))
                windup_ended = not windup and elapsed >= 3
                forced_swing = player.live_anim_frame is not None and player.live_anim_frame >= 0x18
                opening_ready = self.committed and opening
                if windup_ended or elapsed >= HOLD_FRAMES or forced_swing or not clear or opening_ready:
                    self.phase, self.release_at = 'release', frame
                    self.metrics['c_releases'] += 1
                    self.diagnostics['release_reason'] = (
                        reason if not clear else 'committed-opening' if self.committed and opening
                        else 'bounded-windup')
                elif state.engine.puck_owner == plan.slot:
                    action[Buttons.INPUT_C] = 1
                self.diagnostics.update(pressure=margin, clearance=clearance, opening=opening)
            if self.phase == 'release' and self.event['shot']:
                self.phase = 'exit'
            if self.phase == 'exit' and not state.is_shooting:
                target = player.x + plan.direction * 40, player.y
                escape = goalie_avoidance(state, player, target)
                if escape is not None:
                    target = escape[0]
                dx, dy = target[0] - player.x, target[1] - player.y
                if abs(dx) > 3:
                    action[Buttons.INPUT_RIGHT if dx > 0 else Buttons.INPUT_LEFT] = 1
                if abs(dy) > 3:
                    action[Buttons.INPUT_UP if dy > 0 else Buttons.INPUT_DOWN] = 1
            else:
                action[Buttons.INPUT_RIGHT if self.aim_side > 0 else Buttons.INPUT_LEFT] = 1
        self.diagnostics.update(
            mode='cross-crease-' + self.phase, phase=self.phase, committed=self.committed,
            plan=asdict(plan), elapsed_frames=frame - self.started, metrics=dict(self.metrics))
        return action

    @staticmethod
    def _pad(action, pad):
        if pad[0]:
            action[Buttons.INPUT_RIGHT if pad[0] > 0 else Buttons.INPUT_LEFT] = 1
        if pad[1]:
            action[Buttons.INPUT_UP if pad[1] > 0 else Buttons.INPUT_DOWN] = 1
