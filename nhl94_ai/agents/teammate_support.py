"""Bounded reachable support options using Classic's movement/pass geometry."""
from copy import copy
from nhl94_ai.game.forecast import clone_forecast, set_position
import math

from nhl94_ai.agents.defense import defensive_steering
from nhl94_ai.game.actors import controllable_skater
from nhl94_ai.game.motion import arrival_time, bounded_projection, velocity
from nhl94_ai.agents.passing import evaluate_pass, pressure_margin, shot_value
from nhl94_ai.game.skating import grounded_step
from nhl94_ai.env.target_control import project_target
from nhl94_ai.agents.teammate_offpuck import attack_options
from nhl94_ai.agents.teammate_geometry import coverage_target


FORECAST_FRAMES = 12
_PADS = ((0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1),
         (-1, 0), (-1, 1), (0, 0), (0, 0))


def support_forecast(state, slot, target, *, coast=False, frames=FORECAST_FRAMES, native_updates=None):
    """Project all actors to one release time, then let passing model the flight.

    The receiver follows execution's turning/braking pad. Other actors retain
    observed steering; new CPU decisions, contacts and animation are unknown.
    The puck follows the carrier displacement, preserving its current offset.
    No receiver is teleported to the target or assigned an invented zero speed.
    """
    # native_updates is an explicit sequence of D7 durations, not a count of
    # video frames. An empty sequence means no actor update. Callers without
    # a verified schedule retain the legacy, uncertified approximation.
    updates = [1] * frames if native_updates is None else list(native_updates)
    if any(isinstance(tick, bool) or not isinstance(tick, int) or tick < 1 for tick in updates):
        raise ValueError('Native update durations must be positive integers')
    future = clone_forecast(state)
    owner = state.engine.puck_owner

    def project(player, receiver=False):
        projected = copy(player)
        if any(getattr(player, key) is None for key in ('weight', 'agility', 'speed', 'energy')):
            point, _ = bounded_projection(player, sum(updates))
            set_position(projected, point)
            return projected
        for elapsed in updates:
            steering = player.steering if player.steering in range(10) else 8
            pad = ((0, 0) if coast else defensive_steering(projected, target)[0]) if receiver else _PADS[steering]
            projected = grounded_step(projected, pad, goalie=player.role == 0,
                                      stop=not receiver and steering == 9, elapsed_ticks=elapsed,
                                      friction_shift=(9 if native_updates is not None and (player.selection_flags or 0) & 1 else 6))
        return projected

    for name in ('team1', 'team2'):
        team = getattr(state, name)
        projected = getattr(future, name)
        projected.players = [project(player, name == 'team1' and team.skater_scnum_base() + i == slot)
                             for i, player in enumerate(team.players)]
        projected.goalie = project(team.goalie)
        setattr(future, name, projected)
    carrier, projected = state.team1.get_player_by_scnum(owner), future.team1.get_player_by_scnum(owner)
    if carrier is not None and projected is not None:
        for axis in ('x', 'y'):
            delta = getattr(projected, axis) - getattr(carrier, axis)
            setattr(future.puck, axis, getattr(state.puck, axis) + delta)
            precise = getattr(state.puck, 'precise_' + axis)
            setattr(future.puck, 'precise_' + axis, (getattr(state.puck, axis) if precise is None else precise) + delta)
        future.puck.motion_x, future.puck.motion_y = velocity(projected)
    return future


class SupportPlanner:
    def __init__(self):
        self.until = 0
        self.key = None
        self.target = (0, 0)
        self.role = 'coverage'
        self.diagnostics = {}
        self.shots = None
        self.recover_until = 0
        self.coast = False

    def choose(self, state, slot, frame, primary_slot=None, *, movement=False, reserved_targets=()):
        team, opponents, puck = state.team1, state.team2, state.puck
        player = team.get_player_by_scnum(slot)
        owner = state.engine.puck_owner
        carrier = team.get_player_by_scnum(owner)
        attack = 1 if opponents.net.y > team.net.y else -1
        if self.shots is not None and team.stats.shots > self.shots:
            self.recover_until = frame + 90
        self.shots = team.stats.shots
        phase = ('outlet' if team.goalie_haspuck else
                 'attack' if carrier is not None and carrier is not team.goalie and carrier.y * attack >= 88
                 else 'buildup' if carrier is not None and carrier is not team.goalie
                 else 'recovery' if frame < self.recover_until else 'defense')
        key = (slot, owner, attack, phase, primary_slot, movement, tuple(reserved_targets))
        if player is None or not controllable_skater(player):
            self.key, self.until, self.coast = None, 0, False
            self.target, self.role = (0, 0), 'coverage'
            self.diagnostics = {'phase': 'unavailable', 'candidates': []}
            return self.target, self.role
        if movement and player.unavailable & 2:
            # Animation and video-frame boundaries do not guarantee one grounded
            # update. Do not price a cut during a known animation transition.
            self.key, self.until, self.coast = None, 0, True
            self.target, self.role = (player.x, player.y), 'animation-hold'
            self.diagnostics = {'phase': 'animation-unavailable', 'candidates': [],
                                'coast': True, 'target_owner': 'support-planner'}
            return self.target, self.role
        imminent = owner < 0 and math.dist((player.x, player.y), (puck.x, puck.y)) < 24
        if key == self.key and frame < self.until and not imminent:
            return self.target, self.role
        self.key, self.until, self.coast = key, frame + (FORECAST_FRAMES if movement else 8), False
        if phase in ('defense', 'recovery'):
            self.target = self._coverage(state, player, primary_slot)
            self.role = 'coverage'
            self.diagnostics = {'phase': phase, 'target': self.target}
            return self.target, self.role
        self.role = 'outlet-option' if phase == 'outlet' else 'passing-option'
        if phase == 'outlet':
            carrier = team.goalie
        forward = carrier.y * attack
        flank = -1 if carrier.x >= 0 else 1
        points = [(flank * 45, forward + 40), (flank * 70, forward + 10),
                  (flank * 35, forward - 35), (-flank * 60, forward - 30),
                  (flank * 35, 210 if phase == 'attack' else forward + 65),
                  (player.x, player.y * attack)]
        labels = ['passing-option'] * len(points)
        if movement and phase != 'outlet':
            options = attack_options(carrier, attack, phase, frame)
            points = [point[:2] for point in options] + points
            labels = [point[2] for point in options] + labels
        labels.append('coasting-option')
        candidates = []
        for option_index, (x, y) in enumerate([*points, points[-1]]):
            coast = option_index == len(points)
            y = min(235, y)
            if state.engine.offsides_enabled is not False and puck.y * attack < 88:
                y = min(76, y)
            point = project_target((x, y * attack))
            if any(math.dist(point, reserved) < 24 for reserved in reserved_targets):
                continue
            arrival = arrival_time(player, point)
            spacing = math.dist((carrier.x, carrier.y), point)
            teammate_gap = min((math.dist(point, (other.x, other.y)) for other in team.players
                                if other is not player and other is not carrier and controllable_skater(other)), default=100)
            if arrival > 48 or spacing < 30 or teammate_gap < 18:
                continue
            pressure = pressure_margin(opponents, point, arrival)
            score = (0.3 * (point[1] * attack - forward) - arrival * 0.6
                     + min(18, pressure) * 1.5 - max(0, 30 - teammate_gap))
            candidates.append({'point': point, 'arrival': arrival, 'pressure': pressure, 'score': score,
                               'role': labels[option_index], 'coast': coast, 'stay': option_index >= len(points) - 1})
        # At most four forecasts. Always consider braking/staying and coasting
        # when their spacing is useful; movement is not itself the objective.
        finalists = sorted((item for item in candidates if not item['stay']),
                           key=lambda item: item['score'], reverse=True)[:2]
        finalists.extend(item for item in candidates if item['stay'])
        index = team.players.index(player) if player in team.players else None
        for candidate in finalists:
            future = support_forecast(state, slot, candidate['point'], coast=candidate['coast'])
            receiver = future.team1.get_player_by_scnum(slot)
            future_carrier = future.team1.get_player_by_scnum(owner)
            options = [evaluate_pass(future, future_carrier, index, receiver, purpose=purpose,
                                     release_prediction=True)
                       for purpose in ('support', 'one-timer')] if index is not None else [(None, {})] * 2
            feasible = [(option, details, purpose) for (option, details), purpose in
                        zip(options, ('support', 'one-timer')) if option is not None]
            option, details, purpose = max(feasible, key=lambda item: item[0].value) if feasible else (*options[0], 'support')
            candidate['pass_options'] = {purpose: details.get('status', 'unavailable')
                                         for (_, details), purpose in zip(options, ('support', 'one-timer'))}
            candidate['next_action'] = 'one-timer' if purpose == 'one-timer' else 'receive-and-carry-or-shoot'
            if (state.engine.offsides_enabled is not False and puck.y * attack < 88
                    and receiver.y * attack >= 88):
                candidate.update(score=-math.inf, pass_status='premature-zone-entry', feasible=False)
                continue
            candidate['pass_status'] = details.get('status', 'unavailable')
            candidate['feasible'] = option is not None
            candidate['receiver_position'] = (receiver.x, receiver.y)
            candidate['receiver_velocity'] = velocity(receiver)
            candidate['release_time'] = FORECAST_FRAMES
            candidate['next_shot'] = (shot_value(future, receiver, option.point, option.flight_frames)
                                      if option is not None else 0)
            candidate['safe_carry_margin'] = option.margin if option is not None else None
            # Price a realizable pass and continuation, including exposure if
            # support runs beyond the carrier. Infeasible passes earn no shot bonus.
            gap = min((math.dist((receiver.x, receiver.y), (other.x, other.y))
                       for other in future.team1.players if other is not receiver and other is not future_carrier
                       and controllable_skater(other)), default=100)
            exposure = pressure_margin(future.team2, (receiver.x, receiver.y), 0)
            candidate['score'] = (0.3 * (receiver.y - future_carrier.y) * attack
                                  - candidate['arrival'] * 0.6 + min(18, exposure) * 1.5
                                  - max(0, 30 - gap))
            candidate['score'] += (35 + min(12, option.margin) if option is not None else -15)
            candidate['score'] += candidate['next_shot'] * 0.4
            candidate['score'] -= max(0, (receiver.y - future_carrier.y) * attack - 70) * 0.3
        finalists = [item for item in finalists if math.isfinite(item['score'])]
        if finalists:
            chosen = max(finalists, key=lambda item: item['score'])
            self.target, self.coast = chosen['point'], chosen['coast']
            if movement:
                self.role = chosen['role']
        else:
            # A nearby legal outlet remains bounded when every forecast fails.
            y = min(76, player.y * attack) if state.engine.offsides_enabled is not False and puck.y * attack < 88 else player.y * attack
            self.target = project_target((player.x, y * attack))
        self.diagnostics = {'phase': phase, 'target': self.target, 'candidates': finalists,
                            'coast': self.coast, 'forecast': 'uncertified-native-update-schedule',
                            'forecast_certified': False,
                            'release_time': FORECAST_FRAMES, 'target_until': self.until,
                            'target_owner': 'support-planner', 'bursts': False}
        return self.target, self.role

    _coverage = staticmethod(coverage_target)
