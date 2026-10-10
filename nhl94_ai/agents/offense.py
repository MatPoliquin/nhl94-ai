"""Progression first; bounded cuts must create an identifiable opportunity."""
from copy import copy
from dataclasses import asdict, replace
import math

from nhl94_ai.agents.defense import controlled_slot, eligible, on_ice
from nhl94_ai.agents.lifecycle import PassLifecycle
from nhl94_ai.agents.carry import (
    CARRY_FRAMES, INTERCEPTION_MARGIN, _segment_clearance, assess_path_risk, body_clearance,
    carry_path, forecast_carry, forecast_path,
)
from nhl94_ai.agents.motion import (
    arrival_time, bounded_projection, coast_projection, facing, skate_step, skating, stop_projection, velocity,
)
from nhl94_ai.agents.passing import (
    evaluate_pass, one_timer_contact_frame, pass_contact, pass_point_at, pass_release_frames,
    one_timer_position_ok, option_receiver, pressure_margin, rom_direction, shot_value,
)
from nhl94_ai.agents.responses import response_future
from nhl94_ai.env.target_control import project_target, route_waypoint


FEINT_FRAMES = 18
CONTINUATION_TIME_COST = 0.3
POSITIONAL_CREDIT = 0.6
GOALIE_CLEARANCE = 32
GOALIE_SHOT_CLEARANCE = 24
GOALIE_HORIZON = 32
SHOT_RELEASE_DELAY = 2
CARRY_SAMPLES = (4, 8, 12, FEINT_FRAMES)


def normal_shot_release_frames(player, hold_frames):
    """Bound updateanim/ShotMode through contact, including the C-release jump."""
    durations = player.shot_durations or (4,) * 8
    index, timer = 0, -1
    for elapsed in range(1, hold_frames + 2 * sum(duration + 1 for duration in durations) + 3):
        if index >= 7:
            return elapsed + 1
        if timer < 0:
            timer = durations[index]
        else:
            timer -= 1
            if timer < 0:
                index += 1
                if index < 7:
                    timer = durations[index]
        if index >= 7:
            return elapsed + 1
        if index < 4 and (elapsed == hold_frames or player.shot_power is not None
                          and player.shot_power < 20 and index > 2):
            index = 7 - index
    raise RuntimeError('Shot animation did not reach native contact.')


def shot_release_in_front(state, player, hold_frames):
    sign = 1 if state.team2.net.y > state.team1.net.y else -1
    point = state.puck if state.engine.puck_owner_known else player
    _, y = coast_projection(player, normal_shot_release_frames(player, hold_frames))
    offsets = player.shot_offsets_y or (-20, 20)
    reach = max(offset * sign for offset in offsets)
    return max(point.y * sign, player.y * sign + reach, y * sign + reach) < state.team2.net.y * sign


def ordinary_shot_conditions(state, player, hold_frames, escape):
    """The live normal/early shot triggers, shared with receiving forecasts."""
    sign = 1 if state.team2.net.y > state.team1.net.y else -1
    accuracy = 15 if player.shot_accuracy is None else max(0, min(30, player.shot_accuracy))
    safe = shot_release_in_front(state, player, hold_frames) and goalie_contact_time(
        player, state.team2.goalie, clearance=GOALIE_SHOT_CLEARANCE) > GOALIE_HORIZON
    eligible_shot = abs(player.x) < 48 and safe
    return (player.y * sign > 218 + max(0, 15 - accuracy) * 0.4 and eligible_shot,
            player.y * sign > 175 and escape is not None and eligible_shot)


def carry_value(details):
    return max(details['carry_shot_value'], POSITIONAL_CREDIT * details.get(
        'carry_position_value', 0)) * (1 - details.get('carry_risk', 0)) - carry_risk_cost(details)


def carry_risk_cost(details):
    return details.get('carry_risk', 0) * CONTINUATION_TIME_COST * (
        details.get('carry_forecast_frames', CARRY_FRAMES) + INTERCEPTION_MARGIN)


def carry_viable(details):
    return details.get('carry_viable', details['carry_safe'])


def physical_clearance(details):
    return min(details['carry_clearance'], details.get('goalie_clearance', math.inf) - GOALIE_CLEARANCE)


def carry_projection(player, target, frames=FEINT_FRAMES):
    x, y = float(player.x), float(player.y)
    vx, vy = velocity(player)
    acceleration, limit, _ = skating(player)
    direction = player.facing if player.facing is not None else player.orientation
    turn_left = 0
    for _ in range(frames):
        dx, dy = target[0] - x, target[1] - y
        ux = math.copysign(1, dx) if abs(dx) > 3 else 0
        uy = math.copysign(1, dy) if abs(dy) > 3 else 0
        ((x, y), (vx, vy)), direction, turn_left = skate_step(
            ((x, y), (vx, vy)), direction, turn_left, (ux, uy), acceleration, limit)
        if math.dist((x, y), project_target((x, y))) > 1e-6:
            return None
    return (x, y), (vx, vy)


def _braking(player, target):
    vx, vy = velocity(player)
    dx, dy = target[0] - player.x, target[1] - player.y
    ux, uy = math.copysign(1, dx) if abs(dx) > 3 else 0, math.copysign(1, dy) if abs(dy) > 3 else 0
    wanted = round(math.atan2(ux, uy) * 4 / math.pi) % 8 if ux or uy else None
    direction = player.facing if player.facing is not None else player.orientation
    return (wanted == (direction + 4) % 8 and (rom_direction(vx, vy) - direction + 2) % 8 < 4
            and not (player.selection_flags or 0) & 0x10)


def _native_carrier(player):
    return all(getattr(player, field) is not None for field in (
        'motion_x', 'motion_y', 'weight', 'agility', 'speed', 'energy'))


def _carry_motion(player, target, frames=FEINT_FRAMES, *, decision_interval=None):
    if decision_interval is not None and _native_carrier(player):
        path = carry_path(player, target, frames, decision_interval=decision_interval)
        return ((path[-1].x, path[-1].y), velocity(path[-1])) if path is not None else None
    return (stop_projection(player, frames, bounded=True) if _braking(player, target)
            else carry_projection(player, target, frames))


def projected_state(state, target, frames=FEINT_FRAMES, *, decision_interval=None):
    original = state.team1.get_player_by_scnum(state.engine.puck_owner)
    if decision_interval is not None and _native_carrier(original):
        path = carry_path(original, target, frames, decision_interval=decision_interval)
        return carry_future(state, original, path[-1], frames=frames) if path is not None else None
    motion = _carry_motion(original, target, frames)
    if motion is None:
        return None
    future = copy(state)
    for name in ('team1', 'team2'):
        team = copy(getattr(state, name))
        team.players = [copy(p) for p in team.players]
        team.goalie = copy(team.goalie)
        carrier = team.get_player_by_scnum(state.engine.puck_owner) if name == 'team1' else None
        for player in (*team.players, team.goalie):
            if player is carrier or not on_ice(player):
                continue
            (player.x, player.y), player.projection_uncertainty = bounded_projection(
                player, frames, boards=player is not team.goalie)
        setattr(future, name, team)
    player = future.team1.get_player_by_scnum(state.engine.puck_owner)
    (player.x, player.y), (player.motion_x, player.motion_y) = motion
    future.puck = copy(state.puck)
    future.puck.x = state.puck.x + player.x - original.x
    future.puck.y = state.puck.y + player.y - original.y
    return future


def reception_frames(state, option):
    receiver = option_receiver(state.team1.players[option.index], option)
    passer = state.team1.get_player_by_scnum(state.engine.puck_owner)
    contact = option.contact or pass_contact(state.puck, passer, receiver)
    if contact is None:
        raise ValueError('A safe pass option must have a reachable reception')
    return one_timer_contact_frame(state.puck, receiver, contact, pass_release_frames(receiver))


def reception_state(state, option, elapsed=None):
    """Advance bodies to reception; the puck contact point is not a body center."""
    receiver = option_receiver(state.team1.players[option.index], option)
    elapsed = reception_frames(state, option) if elapsed is None else elapsed
    if elapsed is None:
        raise ValueError('Cannot project a reception without modeled body/stick contact')
    future = copy(state)
    for name in ('team1', 'team2'):
        team = copy(getattr(state, name))
        team.players = [copy(player) for player in team.players]
        team.goalie = copy(team.goalie)
        for player in (*team.players, team.goalie):
            if not on_ice(player):
                continue
            source = receiver if name == 'team1' and player is team.players[option.index] else player
            (player.x, player.y), player.projection_uncertainty = bounded_projection(
                source, elapsed, boards=player is not team.goalie)
            if source is receiver:
                player.motion_x, player.motion_y = velocity(receiver)
            player.precise_x, player.precise_y = float(player.x), float(player.y)
        setattr(future, name, team)
    future.engine = copy(state.engine)
    future.engine.puck_owner = option.slot
    future.team1.defense_control, future.team1.control = option.slot, option.index + 1
    future.puck = copy(state.puck)
    contact = option.contact or pass_contact(state.puck, state.team1.get_player_by_scnum(state.engine.puck_owner), receiver)
    future.puck.x, future.puck.y = pass_point_at(
        state.puck, receiver, contact, elapsed - pass_release_frames(receiver))
    future.puck.motion_x, future.puck.motion_y = velocity(receiver)
    return future


def carry_future(state, player, carrier, *, frames=CARRY_FRAMES):
    future = copy(state)
    for name in ('team1', 'team2'):
        team = copy(getattr(state, name))
        team.players = [copy(other) for other in team.players]
        team.goalie = copy(team.goalie)
        for index, other in enumerate(team.players):
            if getattr(state, name).players[index] is player:
                team.players[index] = carrier
            elif on_ice(other):
                (other.x, other.y), other.projection_uncertainty = bounded_projection(other, frames)
                other.precise_x, other.precise_y = float(other.x), float(other.y)
        if on_ice(team.goalie):
            (team.goalie.x, team.goalie.y), team.goalie.projection_uncertainty = bounded_projection(
                team.goalie, frames, boards=False)
            team.goalie.precise_x, team.goalie.precise_y = float(team.goalie.x), float(team.goalie.y)
        setattr(future, name, team)
    future.puck = copy(state.puck)
    future.puck.x += carrier.x - player.x
    future.puck.y += carrier.y - player.y
    return future


def carry_clearance(state, player, target, *, decision_interval=None):
    if decision_interval is not None and _native_carrier(player):
        path = carry_path(player, target, decision_interval=decision_interval)
        return body_clearance(state, player, path) if path is not None else -math.inf
    clearance = math.inf
    for frames in CARRY_SAMPLES:
        motion = _carry_motion(player, target, frames)
        if motion is None:
            return -math.inf
        point, _ = motion
        for team in (state.team1, state.team2):
            for other in (*team.players, team.goalie):
                if other is player or not on_ice(other):
                    continue
                position, uncertainty = bounded_projection(other, frames, boards=other is not team.goalie)
                clearance = min(clearance, math.dist(point, position) - 16 - uncertainty)
    return clearance


def carry_clear(state, player, target, *, decision_interval=None):
    return carry_clearance(state, player, target, decision_interval=decision_interval) > 0


def goalie_contact_time(player, goalie, clearance=GOALIE_CLEARANCE):
    if not on_ice(goalie):
        return math.inf
    vx, vy = velocity(player)
    gx, gy = velocity(goalie)
    dx, dy, rx, ry = player.x - goalie.x, player.y - goalie.y, vx - gx, vy - gy
    distance = dx * dx + dy * dy - clearance**2
    if distance <= 0:
        return 0.0
    speed = rx * rx + ry * ry
    dot = dx * rx + dy * ry
    discriminant = dot * dot - speed * distance
    if speed < 0.01 or dot >= 0 or discriminant < 0:
        return math.inf
    return (-dot - math.sqrt(discriminant)) / speed


def _goalie_clearance(player, goalie, target, *, decision_interval=None):
    if not on_ice(goalie):
        return math.inf
    gx, gy = velocity(goalie)
    previous = player.x - goalie.x, player.y - goalie.y
    clearance = math.hypot(*previous)
    if decision_interval is not None:
        path = carry_path(player, target, GOALIE_HORIZON, decision_interval=decision_interval)
        if path is None:
            return -math.inf
        for frames, carrier in enumerate(path[1:], 1):
            relative = carrier.x - goalie.x - gx * frames, carrier.y - goalie.y - gy * frames
            clearance = min(clearance, _segment_clearance(previous, relative))
            previous = relative
        return clearance
    for frames in range(4, GOALIE_HORIZON + 1, 4):
        motion = _carry_motion(player, target, frames)
        if motion is None:
            return -math.inf
        point, _ = motion
        relative = point[0] - goalie.x - gx * frames, point[1] - goalie.y - gy * frames
        dx, dy = relative[0] - previous[0], relative[1] - previous[1]
        fraction = max(0, min(1, -(previous[0] * dx + previous[1] * dy) / max(0.01, dx * dx + dy * dy)))
        clearance = min(clearance, math.hypot(previous[0] + dx * fraction, previous[1] + dy * fraction))
        previous = relative
    return clearance


def goalie_avoidance(state, player, target, *, decision_interval=None):
    goalie = state.team2.goalie
    if player is None or player is state.team1.goalie or not on_ice(goalie):
        return None
    if math.dist((player.x, player.y), (goalie.x, goalie.y)) > 120:
        return None
    native = decision_interval is not None and all(getattr(player, field) is not None
             for field in ('motion_x', 'motion_y', 'weight', 'agility', 'speed', 'energy'))
    interval = decision_interval if native else None
    clearance = _goalie_clearance(player, goalie, target, decision_interval=interval)
    contact = goalie_contact_time(player, goalie)
    if clearance >= GOALIE_CLEARANCE and (native or contact > GOALIE_HORIZON):
        return None
    if not native and contact <= GOALIE_HORIZON:
        vx, vy = velocity(player)
        gx, gy = velocity(goalie)
        dx, dy, rx, ry = player.x - goalie.x, player.y - goalie.y, vx - gx, vy - gy
        time = max(0, min(GOALIE_HORIZON, -(dx * rx + dy * ry) / max(0.01, rx * rx + ry * ry)))
        clearance = min(clearance, math.hypot(dx + rx * time, dy + ry * time))
    sign = 1 if state.team2.net.y > state.team1.net.y else -1
    candidates = []
    fx, fy = facing(player)
    points = [(player.x - fx * 48, player.y - fy * 48)]
    points.extend((player.x + dx, player.y + sign * dy)
                  for dx, dy in ((-48, 0), (48, 0), (-48, -24), (48, -24), (0, -48)))
    for point in points:
        point = project_target(point)
        if route_waypoint((player.x, player.y), point) != point:
            continue
        margin = _goalie_clearance(player, goalie, point, decision_interval=interval)
        if margin > clearance:
            candidates.append((margin, -math.dist(point, target), point))
    if not candidates:
        point, margin = project_target((player.x, player.y - sign * 48)), clearance
    else:
        margin, _, point = max(candidates)
    return point, {
        'goalie_clearance': margin, 'original_goalie_clearance': clearance,
        'goalie_avoidance_safe': margin >= GOALIE_CLEARANCE,
        'goalie_carry_model': 'native-held-input' if native else 'legacy-cut-or-missing-feedback',
        'reason': 'move clear of projected goalie contact' if margin >= GOALIE_CLEARANCE
        else 'maximize goalie clearance; contact may already be unavoidable',
    }


class OffenseController:
    @property
    def uses_lookahead(self):
        return True

    def __init__(self, *, one_timers=True, decision_interval=4, allow_uncertified=False,
                 chance_creation=False):
        self.one_timers = one_timers
        self.decision_interval = decision_interval
        self.allow_uncertified = allow_uncertified
        self.chance_creation = chance_creation
        self.chance_owner = None
        self.chance_until = self.chance_at = 0
        self.one_timer_at = 0
        self.pass_action = PassLifecycle()
        self.pass_at = 0
        self.feint_until = self.feint_at = 0
        self.feint_target = None
        self.feint_mode = 'feint'
        self.diagnostics = {}
        self.risk_cache = {}
        self.pass_timing = self.carry_motion = False
        self.rank_pass_risk = False

    @property
    def movement_interval(self):
        return self.decision_interval if self.carry_motion else None

    def clear_carry(self, state, player, target):
        return carry_clear(state, player, target, decision_interval=self.movement_interval)

    def cancel(self):
        self.pass_action.cancel()
        self.feint_target = None
        self.feint_until = 0
        self.diagnostics = {}
        self.risk_cache.clear()
        self.chance_owner = None
        self.chance_until = 0

    def observe(self, state, frame):
        if self.chance_owner is not None and state.engine.puck_owner != self.chance_owner:
            self.chance_owner = None
            self.chance_until = 0
        return self.pass_action.observe(state, frame)

    def start_pass(self, state, option, frame, purpose):
        self.pass_action.start(state, option, frame, purpose)
        self.pass_at = frame + 24
        self.feint_target = None
        self.feint_until = 0
        self.chance_owner = None
        self.chance_until = 0

    @staticmethod
    def carry_target(state, player):
        sign = 1 if state.team2.net.y > state.team1.net.y else -1
        x = (35 if player.x >= 0 else -35) if player.y * sign <= 140 else (
            -35 if state.team2.goalie.x >= 0 else 35)
        return x, sign * 235

    def _finish_value(self, state, player, *, carry_frames=0, pressure_state=None, pressure_delay=0):
        escape = goalie_avoidance(
            state, player, self.carry_target(state, player), decision_interval=self.decision_interval)
        normal, early = ordinary_shot_conditions(state, player, self.decision_interval, escape)
        # An early shot can lose live priority to a one-timer or its setup.
        if not normal and not (early and not self.one_timers):
            return 0.0, 0
        release = normal_shot_release_frames(player, self.decision_interval)
        path = carry_path(player, None, release)
        xs = player.shot_projection_offsets_x or (-20, 20)
        ys = player.shot_projection_offsets_y or (-20, 20)
        extent = max(math.hypot(x, y) for x in xs for y in ys)
        _, safety = forecast_path(
            state, player, path, pressure_state=pressure_state, pressure_delay=pressure_delay,
            puck_extent=max(extent, math.dist((state.puck.x, state.puck.y), (player.x, player.y))),
            assess_uncertainty=self.allow_uncertified,
            decision_interval=self.decision_interval, risk_cache=self.risk_cache)
        if not carry_viable(safety) or safety.get('carry_sampled_contact_frame') is not None:
            return 0.0, release
        return max(0.0, shot_value(state, player) - (
            carry_frames + release) * CONTINUATION_TIME_COST) * (1 - safety['carry_risk']), release

    def _threat_exit(self, state, player, path, contact, pressure_state, pressure_delay):
        replan = self.decision_interval
        if replan >= contact or replan >= len(path) - 1:
            return None
        future = carry_future(state, player, path[replan], frames=replan)
        carrier = path[replan]
        fx, fy = facing(carrier)
        sign = 1 if state.team2.net.y > state.team1.net.y else -1
        targets = [None, (carrier.x - fx * 48, carrier.y - fy * 48)]
        targets.extend((carrier.x + dx, carrier.y + sign * dy)
                       for dx in (-24, -12, 12, 24) for dy in (-24, 0, 24))

        def clear(candidate, extent=None):
            if candidate is None or body_clearance(state, player, candidate) <= 0:
                return False
            risk = assess_path_risk(
                state, player, candidate, pressure_state, pressure_delay,
                replan, self.risk_cache, extent)
            return (risk['carry_status'] != 'unmeasured-defender-risk'
                    and risk.get('carry_sampled_contact_frame') is None)

        escape = goalie_avoidance(future, carrier, self.carry_target(future, carrier),
                                  decision_interval=replan)
        normal, early = ordinary_shot_conditions(future, carrier, replan, escape)
        release = normal_shot_release_frames(carrier, replan)
        if (normal or early and not self.one_timers) and replan + release < contact:
            windup = carry_path(carrier, None, release)
            full = path[:replan] + windup if windup is not None else None
            extent = max(math.hypot(x, y) for x in (carrier.shot_projection_offsets_x or (-20, 20))
                         for y in (carrier.shot_projection_offsets_y or (-20, 20)))
            if clear(full, extent):
                return {'carry_exit_mode': 'shoot', 'carry_exit_frame': replan,
                        'carry_exit_target': None, 'carry_exit_release_frame': replan + release}
        for target in targets:
            if target is not None:
                target = route_waypoint((carrier.x, carrier.y), project_target(target))
                if goalie_avoidance(future, carrier, target, decision_interval=replan) is not None:
                    continue
            elif _goalie_clearance(carrier, future.team2.goalie, None,
                                   decision_interval=replan) < GOALIE_CLEARANCE:
                continue
            suffix = carry_path(carrier, target, len(path) - replan - 1, decision_interval=replan)
            full = path[:replan] + suffix if suffix is not None else None
            if clear(full):
                return {'carry_exit_mode': 'skate', 'carry_exit_frame': replan, 'carry_exit_target': target}
        return None

    def _carry_option(self, state, player, target, *, pressure_state=None, pressure_delay=0):
        target = route_waypoint((player.x, player.y), project_target(target))
        escape = goalie_avoidance(state, player, target, decision_interval=self.decision_interval)
        details = {}
        if escape is not None:
            target, details = escape
            target = route_waypoint((player.x, player.y), target)
        if any(getattr(player, field) is None for field in ('weight', 'agility', 'speed', 'energy')):
            motions = [_carry_motion(player, target, frames) for frames in CARRY_SAMPLES]
            bounded = all(motion is not None for motion in motions)
            clearance = carry_clearance(state, player, target)
            margin = min(pressure_margin(state.team2, motion[0], frames)
                         for motion, frames in zip(motions, CARRY_SAMPLES)) if bounded else -math.inf
            sign = 1 if state.team2.net.y > state.team1.net.y else -1
            safe = clearance > 0 and margin >= 3 and details.get('goalie_avoidance_safe', True)
            return target, {
                **details, 'carry_bounded': bounded, 'carry_safe': safe,
                'carry_clearance': clearance, 'carry_pressure': margin,
                'carry_progress': (motions[-1][0][1] - player.y) * sign if bounded else -math.inf,
                'carry_position_value': shot_value(state, player, motions[-1][0], FEINT_FRAMES) if safe else 0,
                'carry_shot_value': 0.0,
                'carry_viable': safe, 'carry_status': 'legacy-missing-carrier-feedback',
                'carry_risk': 0.0 if safe else 1.0,
                'carry_pressure_model': 'conservative-missing-carrier-feedback',
            }
        path, forecast = forecast_carry(
            state, player, target, decision_interval=self.decision_interval,
            pressure_state=pressure_state, pressure_delay=pressure_delay,
            assess_uncertainty=self.allow_uncertified, risk_cache=self.risk_cache)
        safe = forecast['carry_safe'] and details.get('goalie_avoidance_safe', True)
        viable = carry_viable(forecast) and details.get('goalie_avoidance_safe', True)
        contact = forecast.get('carry_sampled_contact_frame')
        value_frames = CARRY_FRAMES
        if (self.allow_uncertified and not viable and contact is not None
                and forecast['carry_clearance'] > 0 and details.get('goalie_avoidance_safe', True)):
            exit_plan = self._threat_exit(
                state, player, path, contact, state if pressure_state is None else pressure_state, pressure_delay)
            if exit_plan is not None:
                forecast.update(exit_plan, carry_status='uncertified-verified-exit')
                viable = True
                value_frames = self.decision_interval
            else:
                forecast['carry_status'] = 'no-achievable-exit-before-contact'
        position, finish, release = 0.0, 0.0, 0
        if viable:
            endpoint = path[value_frames]
            future = carry_future(state, player, endpoint, frames=value_frames)
            position = max(0.0, shot_value(future, endpoint) - value_frames * CONTINUATION_TIME_COST)
            finish, release = self._finish_value(
                future, endpoint, carry_frames=value_frames,
                pressure_state=state if pressure_state is None else pressure_state,
                pressure_delay=pressure_delay + value_frames)
        return target, {**details, **forecast, 'carry_safe': safe, 'carry_viable': viable,
                        'carry_status': forecast['carry_status'] if details.get(
                            'goalie_avoidance_safe', True) else 'projected-goalie-contact',
                        'carry_position_value': position, 'carry_shot_value': finish,
                        'carry_release_frames': release, 'carry_value_frames': value_frames,
                        'carry_risk_cost': carry_risk_cost(forecast),
                        'carry_progress': ((path[value_frames].y - player.y) * (
                            1 if state.team2.net.y > state.team1.net.y else -1))
                        if viable else forecast['carry_progress']}

    def short_carries(self, state, player, *, pressure_state=None, pressure_delay=0):
        sign = 1 if state.team2.net.y > state.team1.net.y else -1
        fx, fy = facing(player)
        targets = (self.carry_target(state, player),
                   (player.x, player.y + sign * 24),
                   (player.x - 12, player.y + sign * 24),
                   (player.x + 12, player.y + sign * 24),
                   (player.x - fx * 48, player.y - fy * 48))
        return [self._carry_option(state, player, target,
                                   pressure_state=pressure_state, pressure_delay=pressure_delay) for target in targets]

    def retained_opportunity(self, state, player):
        carries = self.short_carries(state, player)
        viable = [option for option in carries if carry_viable(option[1])]
        best = max(viable, key=lambda option: (
            carry_value(option[1]), option[1]['carry_progress']), default=None)
        current_finish, _ = self._finish_value(state, player)
        current = max(current_finish, POSITIONAL_CREDIT * shot_value(state, player))
        value = max(current, carry_value(best[1]) if best else 0.0)
        return value, best

    def _receiver_continuation(self, state, option):
        elapsed = reception_frames(state, option)
        if elapsed is None:
            return option, {'continuation_status': 'no-modeled-reception-contact'}
        future = reception_state(state, option, elapsed)
        receiver = future.team1.players[option.index]
        finish, _ = self._finish_value(future, receiver, pressure_state=state, pressure_delay=elapsed)
        option = replace(option, reception_value=max(
            finish, POSITIONAL_CREDIT * shot_value(future, receiver)))
        sign = 1 if state.team2.net.y > state.team1.net.y else -1
        _, limit, _ = skating(receiver)
        if receiver.projection_uncertainty or receiver.y * sign + max(
                limit, math.hypot(*velocity(receiver))) * CARRY_FRAMES <= 150:
            return option, {'continuation_status': 'outside-short-shooting-horizon'}
        carries = [carry for carry in self.short_carries(
            future, receiver, pressure_state=state, pressure_delay=elapsed) if carry_viable(carry[1])]
        best = max(carries, key=lambda carry: (
            carry_value(carry[1]), carry[1]['carry_progress']), default=None)
        if best is None:
            return option, {'continuation_status': 'no-viable-continuation'}
        value = carry_value(best[1])
        improved = replace(
            option, continuation_value=value, continuation_target=best[0], continuation_frames=CARRY_FRAMES,
            continuation_position_value=best[1]['carry_position_value'],
            continuation_finish_value=best[1]['carry_shot_value'],
            continuation_safe=best[1]['carry_safe'], continuation_risk=best[1]['carry_risk'],
            value=option.value + max(0.0, value - option.shot_value) * 0.6)
        return improved, {
            'continuation_status': 'safe' if best[1]['carry_safe'] else best[1]['carry_status'],
            'continuation_safe': best[1]['carry_safe'], 'continuation_risk': best[1]['carry_risk'],
            'continuation_value': value,
            'continuation_target': best[0], 'continuation_frames': CARRY_FRAMES,
            'continuation_position_value': best[1]['carry_position_value'],
            'continuation_finish_value': best[1]['carry_shot_value'],
            'continuation_release_frames': best[1]['carry_release_frames'],
            'continuation_value_frames': best[1]['carry_value_frames'],
            'continuation_risk_cost': best[1]['carry_risk_cost'],
            **{key: best[1][key] for key in (
                'carry_exit_mode', 'carry_exit_frame', 'carry_exit_target', 'carry_exit_release_frame',
            ) if key in best[1]},
            'reception_frames': elapsed,
        }

    def fallback_carry(self, state, player):
        preferred = self.carry_target(state, player)
        if (player.motion_x is None or player.motion_y is None
                or not state.engine.puck_owner_known):
            return route_waypoint((player.x, player.y), preferred), {
                'mode': 'carry', 'carry_safe': None,
                'reason': 'missing feedback; retain legacy carry/setup',
            }
        first = self._carry_option(state, player, preferred)
        if first[1]['carry_safe']:
            return first[0], {
                **first[1], 'mode': 'goalie-avoid' if 'goalie_clearance' in first[1] else 'carry',
                'reason': first[1].get('reason', 'verified carrying route with contact and interception clearance'),
            }
        sign = 1 if state.team2.net.y > state.team1.net.y else -1
        fx, fy = facing(player)
        brake = player.x - fx * 48, player.y - fy * 48
        targets = [brake, (player.x, player.y)]
        targets.extend((player.x + dx, player.y + sign * dy)
                       for dx, dy in ((-48, 0), (48, 0), (0, -48), (-48, -24), (48, -24),
                                      (-48, 24), (48, 24)))
        targets.extend((player.x + dx, player.y + sign * dy)
                       for dx in (-12, 12, -24, 24) for dy in (0, 24))
        options = [first, *(self._carry_option(state, player, target) for target in targets)]
        bounded = [option for option in options if option[1]['carry_bounded']]
        if not bounded:
            return route_waypoint((player.x, player.y), project_target(brake)), {
                'mode': 'carry-escape', 'carry_safe': False, 'carry_bounded': False,
                'reason': 'no bounded carry; brake before unavoidable wall or net contact',
            }
        def experimental_key(option):
            return (
                physical_clearance(option[1]) > 0, carry_viable(option[1]),
                carry_value(option[1]) if carry_viable(option[1]) else physical_clearance(option[1]),
                option[1]['carry_progress'] if carry_viable(option[1]) else min(option[1]['carry_pressure'], 12),
                option[1]['carry_safe'], min(physical_clearance(option[1]), 16),
                -math.dist(option[0], preferred))
        def certified_key(option):
            return (
                option[1]['carry_safe'], carry_value(option[1]),
                option[1]['carry_progress'] if option[1]['carry_safe'] else min(option[1]['carry_pressure'], 12),
                min(option[1]['carry_pressure'], 12) if option[1]['carry_safe'] else min(option[1]['carry_clearance'], 16),
                min(option[1]['carry_clearance'], 16), -math.dist(option[0], preferred))
        point, details = max(bounded, key=experimental_key if self.allow_uncertified else certified_key)
        mode = ('goalie-avoid' if 'goalie_clearance' in first[1] else
                'carry' if point == first[0] else 'carry-escape')
        return point, {
            **details, 'mode': mode,
            'reason': 'certified carrying escape' if details['carry_safe']
            else 'risk-assessed attacking route; defender safety uncertified' if carry_viable(details)
            else 'no viable carry; prioritize body and goalie separation',
        }

    def breakaway(self, state, player):
        sign = 1 if state.team2.net.y > state.team1.net.y else -1
        if any(other.y * sign >= player.y * sign - 8 and abs(other.x - player.x) < 90
               for other in state.team2.players if other.role is None or other.role > 0):
            return False
        target = OffenseController.carry_target(state, player)
        if not self.clear_carry(state, player, target):
            return False
        for frames in (8, 16, 24):
            motion = (_carry_motion(player, target, frames, decision_interval=self.movement_interval)
                      if self.carry_motion else carry_projection(player, target, frames))
            if motion is None:
                return False
            point, _ = motion
            if pressure_margin(state.team2, point, frames) < 6:
                return False
        return True

    def passes(self, state, purpose='advance', *, continuations=True):
        if purpose == 'one-timer' and not self.one_timers:
            return [], []
        player = state.team1.get_player_by_scnum(state.engine.puck_owner)
        if player is None:
            return [], []
        choices, diagnostics = [], []
        current_shot = shot_value(state, player) if purpose == 'one-timer' else 0
        for index, receiver in enumerate(state.team1.players):
            if receiver is player:
                continue
            option, details = evaluate_pass(
                state, player, index, receiver, purpose, decision_interval=self.decision_interval,
                release_prediction=self.pass_timing, rank_risk=self.rank_pass_risk)
            details['purpose'] = purpose
            if option is not None and purpose != 'one-timer' and continuations and self.uses_lookahead:
                option, continuation = self._receiver_continuation(state, option)
                details.update(continuation, value=option.value)
            if (purpose == 'one-timer' and option is not None
                    and not one_timer_position_ok(option, player, current_shot)):
                option = None
                details = {**details, 'status': 'weaker-one-timer-position'}
            diagnostics.append(details)
            if option is not None:
                choices.append(option)
        key = (lambda option: (-option.shot_value, -option.margin, option.slot)) if purpose == 'one-timer' else (
            lambda option: (-option.value, option.slot))
        return sorted(choices, key=key), diagnostics

    def _continue_cut(self, state, player, frame):
        if self.feint_target is None:
            return None
        target = self._cut_target(state, player, self.feint_target)
        if (frame < self.feint_until and pressure_margin(state.team2, (player.x, player.y), 4) >= 3
                and target == self.feint_target and self.clear_carry(state, player, self.feint_target)):
            return self._plan(state, self.feint_mode, self.feint_target,
                              'continue bounded one-timer setup' if self.feint_mode == 'one-timer-setup'
                              else 'continue bounded opportunity-creating cut')
        self.diagnostics['cut_cancelled'] = 'route or pressure changed; discard predicted opportunity'
        self.feint_target = None
        return None

    @staticmethod
    def _cut_target(state, player, target):
        escape = goalie_avoidance(state, player, target)
        return (escape[0] if escape[1]['goalie_avoidance_safe'] else None) if escape is not None else target

    def _cut_targets(self, state, player, sign):
        proposals = [project_target((player.x + direction * 26, player.y + sign * 8))
                     for direction in (-1, 1)]
        targets = [self._cut_target(state, player, point) for point in proposals]
        if targets != proposals:
            targets.extend(self._cut_target(state, player, project_target((player.x + direction * width, player.y)))
                           for direction in (-1, 1) for width in (26, 48))
        return tuple(dict.fromkeys(point for point in targets if point is not None))

    def _feint(self, state, player, frame, current_value, *, one_timer_only=False):
        if frame < self.feint_at:
            return None
        sign = 1 if state.team2.net.y > state.team1.net.y else -1
        if not 88 < player.y * sign < 218:
            return None
        forward_target = (player.x, sign * 235)
        straight_target = self._cut_target(state, player, forward_target)
        straight = (projected_state(state, straight_target, decision_interval=self.movement_interval)
                    if straight_target == forward_target and self.clear_carry(state, player, straight_target) else None)
        one_timer_open = bool(self.passes(state, 'one-timer')[0])
        if straight is not None:
            mover = straight.team1.get_player_by_scnum(state.engine.puck_owner)
            direct_passes, _ = self.passes(straight, 'position', continuations=False)
            one_timer_open |= bool(self.passes(straight, 'one-timer')[0])
            current_value = max(current_value, shot_value(straight, mover),
                                direct_passes[0].shot_value + min(direct_passes[0].margin, 12)
                                if direct_passes else 0)
        cuts = []
        for target in self._cut_targets(state, player, sign):
            if route_waypoint((player.x, player.y), target) != target or not self.clear_carry(state, player, target):
                continue
            future = projected_state(state, target, decision_interval=self.movement_interval)
            if future is None:
                continue
            mover = future.team1.get_player_by_scnum(state.engine.puck_owner)
            margin = pressure_margin(state.team2, (mover.x, mover.y), FEINT_FRAMES)
            if margin < 5:
                continue
            passes, _ = self.passes(future, 'position', continuations=False)
            one_timers, _ = self.passes(future, 'one-timer') if not one_timer_open else ([], [])
            shots = shot_value(future, mover)
            best = passes[0] if passes else None
            value = max(shots, best.shot_value + min(best.margin, 12) if best else 0)
            if one_timers:
                option = one_timers[0]
                one_timer_value = option.shot_value + min(option.margin, 12)
                if one_timer_value >= current_value:
                    cuts.append((one_timer_value, target, 'open a safe one-timer lane',
                                 'one-timer-setup'))
            if not one_timer_only and value > current_value + 10:
                cuts.append((value, target, 'open stronger receiving position' if best else 'open shooting lane',
                             'feint'))
        if not cuts:
            return None
        _, self.feint_target, reason, self.feint_mode = max(cuts, key=lambda row: row[0])
        self.feint_until, self.feint_at = frame + FEINT_FRAMES, frame + 72
        return self._plan(state, self.feint_mode, self.feint_target, reason)

    def _plan(self, state, mode, target, reason, option=None):
        actual = controlled_slot(state.team1)
        self.diagnostics.update(
            mode=mode, decision=mode, reason=reason, target=target, destination=target, waypoint=target,
            actual_slot=actual, desired_slot=option.slot if option else actual,
            receiver=asdict(option) if option else None, last_pass=self.pass_action.last_pass)
        return mode, target, option

    def _passing_window(self, state, frame=0):
        windows = []
        for purpose in ('position', 'one-timer'):
            if frame < (self.one_timer_at if purpose == 'one-timer' else self.pass_at):
                continue
            options, _ = self.passes(state, purpose, continuations=False)
            windows.extend((option.shot_value + min(option.margin, 12) * CONTINUATION_TIME_COST,
                            option.slot, purpose) for option in options if option.shot_value > 0)
        return max(windows, default=(0.0, None, None))

    def _create_chance(self, state, player, frame):
        if self.chance_owner is not None and (
                state.engine.puck_owner != self.chance_owner or frame >= self.chance_until):
            self.chance_owner = None
            self.chance_at = frame + 72
        if frame < self.chance_at or not 88 < player.y * (
                1 if state.team2.net.y > state.team1.net.y else -1) < 218:
            return None
        horizon = min(CARRY_FRAMES, self.chance_until - frame) if self.chance_owner is not None else CARRY_FRAMES
        baseline = max(shot_value(state, player), self._passing_window(state, frame)[0])
        sign = 1 if state.team2.net.y > state.team1.net.y else -1
        fx, fy = facing(player)
        proposals = (self.carry_target(state, player),
                     (player.x - 26, player.y + sign * 8),
                     (player.x + 26, player.y + sign * 8),
                     (player.x - fx * 48, player.y - fy * 48),
                     (player.x, player.y))
        rows, choices, seen = [], [], set()
        for index, proposal in enumerate(proposals):
            target, safety = self._carry_option(state, player, proposal)
            if target in seen:
                continue
            seen.add(target)
            row = {'target': target, 'carry_safe': safety['carry_safe'],
                   'status': 'uncertified-carry'}
            rows.append(row)
            # Response assumptions never relax the original control-reach certificate.
            if not safety['carry_safe'] or safety.get('carry_value_frames') != CARRY_FRAMES:
                continue
            path = carry_path(player, target, frames=horizon, decision_interval=self.decision_interval)
            future, response = response_future(state, path)
            row.update(response)
            if response['response_clearance'] <= 0:
                row['status'] = 'response-body-contact'
                continue
            if not response['response_slots']:
                row['status'] = 'no-supported-response'
                continue
            window, receiver, purpose = self._passing_window(future, frame + horizon)
            value = window - horizon * CONTINUATION_TIME_COST
            row.update(status='modeled-window', value=value, receiver=receiver, purpose=purpose,
                       forecast_frames=horizon)
            if index == 0:
                baseline = max(baseline, value, shot_value(
                    future, path[-1]) - horizon * CONTINUATION_TIME_COST)
            elif receiver is not None:
                choices.append((value, target, receiver, purpose, safety))
        self.diagnostics.update(chance_candidates=rows, chance_baseline=baseline)
        best = max(choices, key=lambda option: option[0], default=None)
        if best is None or best[0] <= baseline + 10:
            return None
        value, target, receiver, purpose, safety = best
        if self.chance_owner is None:
            self.chance_owner, self.chance_until = state.engine.puck_owner, frame + CARRY_FRAMES
        self.diagnostics.update(safety, chance_value=value, chance_receiver=receiver,
                                chance_purpose=purpose, chance_until=self.chance_until,
                                chance_window_status='predicted-not-observed')
        return self._plan(state, 'create-chance', target,
                          'safe short carry models a stronger passing window; recheck live before release')

    @staticmethod
    def _worthwhile(option, player, sign, purpose, current, retained_progress=0):
        if purpose == 'advance':
            return option.forward_gain >= 20 and (
                option.bypassed > 0 or option.forward_gain >= max(50, retained_progress + 20)
                or player.y * sign + max(0, retained_progress) < 88 <= option.point[1] * sign
                or option.opportunity > current + 12)
        return option.opportunity > current + 12

    def choose(self, state, frame):
        self.risk_cache.clear()
        player = state.team1.get_player_by_scnum(state.engine.puck_owner)
        self.diagnostics = {
            'candidates': [], 'scenario_samples': 8,
            'evaluation_frame': frame, 'evaluation_carrier': state.engine.puck_owner,
        }
        if (player is None or not eligible(player) or player.motion_x is None
                or player.motion_y is None or not state.engine.puck_owner_known):
            self.diagnostics['status'] = 'missing-feedback'
            return None
        sign = 1 if state.team2.net.y > state.team1.net.y else -1
        purpose = 'advance' if player.y * sign < 140 else 'position'
        choices, self.diagnostics['candidates'] = self.passes(state, purpose)
        retained_value, retained = self.retained_opportunity(state, player)
        current = shot_value(state, player)
        retained_progress = retained[1]['carry_progress'] if retained else 0
        self.diagnostics.update(retained_value=retained_value, retained_carry=retained,
                                retained_progress=retained_progress)
        choices = [option for option in choices if self._worthwhile(
            option, player, sign, purpose, retained_value, retained_progress)]
        worthwhile_slots = {option.slot for option in choices}
        for candidate in self.diagnostics['candidates']:
            if candidate['status'] == 'safe':
                candidate['worthwhile'] = candidate['slot'] in worthwhile_slots
        finish_depth = 198 + max(0, 15 - (player.shot_accuracy if player.shot_accuracy is not None else 15)) * 0.4
        if (player.y * sign > finish_depth and abs(player.x) < 48
                and shot_release_in_front(state, player, self.decision_interval)
                and pressure_margin(state.team2, (player.x, player.y), 8) >= 3):
            self.feint_target = None
            self.diagnostics['status'] = 'close-to-finish'
            return None
        if self.breakaway(state, player):
            self.feint_target = None
            return self._plan(state, 'carry-breakaway', self.carry_target(state, player),
                              'no reachable skater interception')
        if self.chance_creation and self.feint_target is None:
            chance = self._create_chance(state, player, frame)
            if chance is not None:
                return chance
        opportunity = max(current, choices[0].opportunity + min(choices[0].margin, 12) if choices else 0)
        if self.feint_mode == 'one-timer-setup':
            continuation = self._continue_cut(state, player, frame)
            if continuation:
                return continuation
        if choices and frame >= self.pass_at:
            best = choices[0]
            if purpose == 'position':
                setup = self._feint(state, player, frame, opportunity, one_timer_only=True)
                if setup:
                    return setup
            return self._plan(state, 'advance-pass' if purpose == 'advance' else 'position-pass',
                              best.point, 'advance beyond defenders with reception space', best)
        continuation = self._continue_cut(state, player, frame)
        if continuation:
            return continuation
        cut = self._feint(state, player, frame, opportunity)
        if cut:
            return cut
        if retained is not None and carry_value(retained[1]) > POSITIONAL_CREDIT * current + 10:
            self.diagnostics.update(retained[1])
            return self._plan(state, 'carry-opportunity', retained[0],
                              'risk-assessed short carry improves the opportunity; safety uncertified'
                              if not retained[1]['carry_safe'] else 'certified short carry improves the opportunity')
        self.diagnostics['status'] = 'no-better-safe-option'
        return None
