"""Progression first; bounded cuts must create an identifiable opportunity."""
from copy import copy
from dataclasses import asdict
import math

from nhl94_ai.agents.defense import controlled_slot, eligible, on_ice
from nhl94_ai.agents.motion import (
    arrival_time, bounded_projection, coast_projection, facing, skate_step, skating, stop_projection, velocity,
)
from nhl94_ai.agents.passing import evaluate_pass, pressure_margin, rom_direction, shot_value
from nhl94_ai.env.target_control import project_target, route_waypoint


FEINT_FRAMES = 18
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


def _carry_motion(player, target, frames=FEINT_FRAMES):
    return (stop_projection(player, frames, bounded=True) if _braking(player, target)
            else carry_projection(player, target, frames))


def projected_state(state, target, frames=FEINT_FRAMES):
    original = state.team1.get_player_by_scnum(state.engine.puck_owner)
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


def carry_clearance(state, player, target):
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


def carry_clear(state, player, target):
    return carry_clearance(state, player, target) > 0


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


def _goalie_clearance(player, goalie, target):
    if not on_ice(goalie):
        return math.inf
    gx, gy = velocity(goalie)
    previous = player.x - goalie.x, player.y - goalie.y
    clearance = math.hypot(*previous)
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


def goalie_avoidance(state, player, target):
    goalie = state.team2.goalie
    if player is None or player is state.team1.goalie or not on_ice(goalie):
        return None
    if math.dist((player.x, player.y), (goalie.x, goalie.y)) > 120:
        return None
    clearance = _goalie_clearance(player, goalie, target)
    contact = goalie_contact_time(player, goalie)
    if clearance >= GOALIE_CLEARANCE and contact > GOALIE_HORIZON:
        return None
    if contact <= GOALIE_HORIZON:
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
        margin = _goalie_clearance(player, goalie, point)
        if margin > clearance:
            candidates.append((margin, -math.dist(point, target), point))
    if not candidates:
        point, margin = project_target((player.x, player.y - sign * 48)), clearance
    else:
        margin, _, point = max(candidates)
    return point, {
        'goalie_clearance': margin, 'original_goalie_clearance': clearance,
        'goalie_avoidance_safe': margin >= GOALIE_CLEARANCE,
        'reason': 'move clear of projected goalie contact' if margin >= GOALIE_CLEARANCE
        else 'maximize goalie clearance; contact may already be unavoidable',
    }


class OffenseController:
    def __init__(self, *, one_timers=True, decision_interval=4):
        self.one_timers = one_timers
        self.decision_interval = decision_interval
        self.pending = None
        self.pass_at = 0
        self.feint_until = self.feint_at = 0
        self.feint_target = None
        self.feint_mode = 'feint'
        self.last_pass = None
        self.last_request = None
        self.diagnostics = {}

    def cancel(self):
        self.pending = None
        self.feint_target = None
        self.feint_until = 0
        self.diagnostics = {}

    def observe(self, state, frame):
        request = self.pending
        if request is None:
            return
        owner = state.engine.puck_owner
        if state.team2.owns_scnum(owner):
            result = 'intercepted' if request['launched'] else 'lost-before-release'
        elif owner == request['passer'] and request['flight_observed']:
            result = 'recovered-by-passer'
        elif frame >= request['deadline']:
            result = 'flight-timeout' if request['launched'] else 'not-released'
        elif owner >= 0 and owner != request['passer']:
            result = 'received' if owner == request['receiver'] else 'other-receiver'
        else:
            if owner < 0:
                request['flight_observed'] = True
            if (request['actual_receiver'] is None and state.team1.pass_attempts is not None
                    and request['passes_before'] is not None
                    and state.team1.pass_attempts > request['passes_before']):
                request['actual_receiver'] = state.engine.pass_target
                request['launched'] = True
            if owner < 0 and state.engine.last_puck_player == request['passer']:
                request['launched'] = True
            return
        self.last_pass = {**request, 'outcome': result, 'owner': owner, 'end_frame': frame}
        self.pending = None

    def start_pass(self, state, option, frame, purpose):
        self.pending = {
            'passer': state.engine.puck_owner, 'receiver': option.slot, 'purpose': purpose,
            'frame': frame, 'deadline': frame + math.ceil(option.flight_frames) + 20,
            'point': option.point, 'passes_before': state.team1.pass_attempts,
            'actual_receiver': None, 'launched': False, 'flight_observed': False,
        }
        self.last_request = dict(self.pending)
        self.pass_at = frame + 24
        self.feint_target = None
        self.feint_until = 0

    @staticmethod
    def carry_target(state, player):
        sign = 1 if state.team2.net.y > state.team1.net.y else -1
        x = (35 if player.x >= 0 else -35) if player.y * sign <= 140 else (
            -35 if state.team2.goalie.x >= 0 else 35)
        return x, sign * 235

    @staticmethod
    def _carry_option(state, player, target):
        target = route_waypoint((player.x, player.y), project_target(target))
        escape = goalie_avoidance(state, player, target)
        details = {}
        if escape is not None:
            target, details = escape
            target = route_waypoint((player.x, player.y), target)
        clearance = carry_clearance(state, player, target)
        motions = [_carry_motion(player, target, frames) for frames in CARRY_SAMPLES]
        bounded = all(motion is not None for motion in motions)
        margin = min(pressure_margin(state.team2, motion[0], frames)
                     for motion, frames in zip(motions, CARRY_SAMPLES)) if bounded else -math.inf
        safe = clearance > 0 and margin >= 3 and details.get('goalie_avoidance_safe', True)
        sign = 1 if state.team2.net.y > state.team1.net.y else -1
        progress = (motions[-1][0][1] - player.y) * sign if bounded else -math.inf
        opportunity = shot_value(state, player, motions[-1][0], FEINT_FRAMES) if safe else 0
        return target, {
            **details, 'carry_safe': safe, 'carry_bounded': bounded,
            'carry_clearance': clearance, 'carry_pressure': margin,
            'carry_progress': progress, 'carry_shot_value': opportunity,
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
        options = [first, *(self._carry_option(state, player, target) for target in targets)]
        bounded = [option for option in options if option[1]['carry_bounded']]
        if not bounded:
            return route_waypoint((player.x, player.y), project_target(brake)), {
                'mode': 'carry-escape', 'carry_safe': False, 'carry_bounded': False,
                'reason': 'no bounded carry; brake before unavoidable wall or net contact',
            }
        point, details = max(bounded, key=lambda option: (
            option[1]['carry_safe'], option[1]['carry_shot_value'],
            option[1]['carry_progress'] if option[1]['carry_safe'] else min(option[1]['carry_pressure'], 12),
            min(option[1]['carry_pressure'], 12) if option[1]['carry_safe'] else min(option[1]['carry_clearance'], 16),
            min(option[1]['carry_clearance'], 16), -math.dist(option[0], preferred)))
        mode = ('goalie-avoid' if 'goalie_clearance' in first[1] else
                'carry' if point == first[0] else 'carry-escape')
        return point, {
            **details, 'mode': mode,
            'reason': 'escape the unsafe default carry with interception and contact clearance'
            if details['carry_safe'] else 'no safe carry; maximize contact and interception clearance',
        }

    @staticmethod
    def breakaway(state, player):
        sign = 1 if state.team2.net.y > state.team1.net.y else -1
        if any(other.y * sign >= player.y * sign - 8 and abs(other.x - player.x) < 90
               for other in state.team2.players if other.role is None or other.role > 0):
            return False
        target = OffenseController.carry_target(state, player)
        if not carry_clear(state, player, target):
            return False
        for frames in (8, 16, 24):
            motion = carry_projection(player, target, frames)
            if motion is None:
                return False
            point, _ = motion
            if pressure_margin(state.team2, point, frames) < 6:
                return False
        return True

    def passes(self, state, purpose='advance'):
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
                state, player, index, receiver, purpose, decision_interval=self.decision_interval)
            if (purpose == 'one-timer' and option is not None
                    and not (option.point[0] * player.x <= 0 and abs(option.point[0] - player.x) > 30)
                    and option.shot_value <= current_shot + 12):
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
                and target == self.feint_target and carry_clear(state, player, self.feint_target)):
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
        straight = (projected_state(state, straight_target)
                    if straight_target == forward_target and carry_clear(state, player, straight_target) else None)
        one_timer_open = bool(self.passes(state, 'one-timer')[0])
        if straight is not None:
            mover = straight.team1.get_player_by_scnum(state.engine.puck_owner)
            direct_passes, _ = self.passes(straight, 'position')
            one_timer_open |= bool(self.passes(straight, 'one-timer')[0])
            current_value = max(current_value, shot_value(straight, mover),
                                direct_passes[0].shot_value + min(direct_passes[0].margin, 12)
                                if direct_passes else 0)
        cuts = []
        for target in self._cut_targets(state, player, sign):
            if route_waypoint((player.x, player.y), target) != target or not carry_clear(state, player, target):
                continue
            future = projected_state(state, target)
            if future is None:
                continue
            mover = future.team1.get_player_by_scnum(state.engine.puck_owner)
            margin = pressure_margin(state.team2, (mover.x, mover.y), FEINT_FRAMES)
            if margin < 5:
                continue
            passes, _ = self.passes(future, 'position')
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
            receiver=asdict(option) if option else None, last_pass=self.last_pass)
        return mode, target, option

    @staticmethod
    def _worthwhile(option, player, sign, purpose, current):
        if purpose == 'advance':
            return option.forward_gain >= 20 and (
                option.bypassed > 0 or option.forward_gain >= 50
                or player.y * sign < 88 <= option.point[1] * sign)
        return option.shot_value > current + 12

    def choose(self, state, frame):
        player = state.team1.get_player_by_scnum(state.engine.puck_owner)
        self.diagnostics = {'candidates': [], 'scenario_samples': 8}
        if (player is None or not eligible(player) or player.motion_x is None
                or player.motion_y is None or not state.engine.puck_owner_known):
            self.diagnostics['status'] = 'missing-feedback'
            return None
        sign = 1 if state.team2.net.y > state.team1.net.y else -1
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
        purpose = 'advance' if player.y * sign < 140 else 'position'
        choices, self.diagnostics['candidates'] = self.passes(state, purpose)
        current = shot_value(state, player)
        choices = [option for option in choices if self._worthwhile(option, player, sign, purpose, current)]
        worthwhile_slots = {option.slot for option in choices}
        for candidate in self.diagnostics['candidates']:
            if candidate['status'] == 'safe':
                candidate['worthwhile'] = candidate['slot'] in worthwhile_slots
        opportunity = max(current, choices[0].shot_value + min(choices[0].margin, 12) if choices else 0)
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
        self.diagnostics['status'] = 'no-better-safe-option'
        return None
