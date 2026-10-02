"""Opt-in manual goalie positioning and feedback-confirmed joystick handoffs."""
from collections import Counter
from dataclasses import dataclass
import math
import warnings

import numpy as np

from nhl94_ai.agents.defense import controlled_slot, eligible, owns_puck
from nhl94_ai.agents.motion import VELOCITY_SCALE, arrival_time, blocks_shot_lane, skating, velocity
from nhl94_ai.agents.passing import evaluate_pass
from nhl94_ai.game.constants import GameConsts as Buttons
from nhl94_ai.game.geometry import aim_pass


TAKEOVER_FRAMES = 24
HANDOFF_TIMEOUT = 60
SAVE_ANIMATIONS = frozenset((0x146, 0x178, 0x1AA, 0x1EC, 0x21E, 0x250, 0x2A2,
                             0x148E, 0x14C0, 0x14F2, 0x1544))
DIVE_ANIMATION = 0x2F4


@dataclass(frozen=True)
class GoaliePlan:
    target: tuple
    reason: str
    crossing_frames: float | None = None
    crossing_x: float | None = None
    height: float | None = None
    deadline: float | None = None


def goalie_motion(player):
    """Near-net goalieacc accelerates in pad direction, without skater turn cost."""
    _, limit, _ = skating(player)
    factor = max(1, (64 - player.weight // 4 + 2 * player.agility + 16 + player.movement_bonus) // 2)
    acceleration = (200 * factor // 32) * VELOCITY_SCALE
    if player.has_puck:
        acceleration /= 2
        limit /= math.sqrt(2)
    return acceleration, limit


def goalie_arrival(player, target):
    acceleration, limit = goalie_motion(player)
    vx, vy = velocity(player)
    dx, dy = target[0] - player.x, target[1] - player.y
    distance = math.hypot(dx, dy)
    along = (vx * dx + vy * dy) / max(1, distance)
    distance = max(0, distance - 5)
    effective = max(acceleration / 2, acceleration - max(0, along) / 64)
    ramp = max(0, (limit - along) / effective)
    covered = along * ramp + effective * ramp * ramp / 2
    if covered >= distance:
        return max(0, (-along + math.sqrt(along * along + 2 * effective * distance)) / effective)
    return ramp + (distance - covered) / limit


def goalie_target(state):
    """Predict a released puck to the crease/goal planes, not through a rebound."""
    sign = 1 if state.team1.net.y > 0 else -1
    puck = state.puck
    vx, vy = velocity(puck)
    x, y, z, vz = float(puck.x), float(puck.y), max(0, puck.height), puck.motion_z or 0
    crease_crossing = None
    if state.engine.puck_owner < 0 and vy * sign > 0.05 and y * sign < 264:
        for frame in range(1, 97):
            if z <= 0:
                vx, vy = vx * puck.friction, vy * puck.friction
            previous = x, y, z
            x, y = x + vx, y + vy
            if z > 0 or vz:
                vz -= 102 * VELOCITY_SCALE
                z += vz
                if z < 0:
                    z, vz = 0, -vz / 2
            if abs(x) >= 128:
                break
            if crease_crossing is None and y * sign >= 246:
                crease_crossing = frame, x, max(0, z)
            if y * sign >= 264:
                fraction = (sign * 264 - previous[1]) / (y - previous[1])
                goal_x = previous[0] + (x - previous[0]) * fraction
                goal_z = max(0, previous[2] + (z - previous[2]) * fraction)
                if abs(goal_x) <= 25 and goal_z <= 24:
                    time, cross_x, height = crease_crossing or (frame, goal_x, goal_z)
                    return GoaliePlan((max(-32, min(32, cross_x)), sign * 246),
                                      'released puck threatens goal', time, cross_x, height, frame)
                break
    vx, vy = velocity(puck)
    reference = puck.x + vx * 8, max(-259, min(259, puck.y + vy * 8))
    reception = None
    receiver = (state.team2.get_player_by_scnum(state.engine.pass_target)
                if state.engine.pass_target is not None else None)
    if (state.engine.puck_owner < 0 and receiver is not None and receiver is not state.team2.goalie
            and state.engine.last_puck_player is not None
            and state.team2.owns_scnum(state.engine.last_puck_player)):
        rx, ry = velocity(receiver)
        dx, dy = receiver.x + (receiver.stick_x or 0) - puck.x, receiver.y + (receiver.stick_y or 0) - puck.y
        relative = vx - rx, vy - ry
        time = (dx * relative[0] + dy * relative[1]) / max(0.01, relative[0]**2 + relative[1]**2)
        if 0 < time < 40 and math.hypot(dx - relative[0] * time, dy - relative[1] * time) < 16:
            reference = receiver.x + rx * time, max(-259, min(259, receiver.y + ry * time))
            reception = time
    dx, dy = reference[0], reference[1] - sign * 260
    length = max(1, math.hypot(dx, dy))
    target = (max(-28, min(28, 26 * dx / length)),
              sign * max(236, min(254, 260 + sign * 18 * dy / length)))
    depth = puck.y * sign
    owner = state.engine.puck_owner
    deadline = None
    if state.team2.owns_scnum(owner) and 88 < depth < 264 and abs(puck.x) < 100:
        # Earliest useful shot is an estimate, not a guaranteed carrier action.
        deadline = max(4, (185 - depth) / max(0.7, vy * sign)) + math.dist(
            (puck.x, puck.y), (0, sign * 264)) / 6
    if reception is not None and reference[1] * sign > 88:
        deadline = reception + math.dist(reference, (0, sign * 264)) / 6
    return GoaliePlan(target, 'anticipate incoming receiver shooting angle' if reception is not None
                      else 'protect predicted shooting angle', deadline=deadline)


def goalie_steer(player, target):
    action = np.zeros(Buttons.INPUT_MAX, dtype=np.int8)
    for position, speed, destination, negative, positive in (
        (player.x, velocity(player)[0], target[0], Buttons.INPUT_LEFT, Buttons.INPUT_RIGHT),
        (player.y, velocity(player)[1], target[1], Buttons.INPUT_DOWN, Buttons.INPUT_UP),
    ):
        braking = speed * abs(speed) / (2 * 2000 * VELOCITY_SCALE)
        error = destination - position - braking
        if abs(error) > 2:
            action[positive if error > 0 else negative] = 1
    return action


class GoalieController:
    """Exclusive per-frame input owner; slot feedback, never gameplay RAM writes."""

    def __init__(self, policy):
        if policy not in ('selective', 'always'):
            raise ValueError('Enabled goalie policy must be selective or always')
        self.policy = policy
        self.frames = 0
        self.phase = 'skater'
        self.started = self.release_until = self.retry_at = self.save_at = 0
        self.possession_since = None
        self.outlet = None
        self.pending_save = None
        self.last_anim = None
        self.last_owner = None
        self.last_effective = False
        self.metrics = Counter()
        self.last_event = None
        self.diagnostics = {}

    @staticmethod
    def validate(state):
        engine, team = state.engine, state.team1
        if len(team.players) != 5:
            raise ValueError('Manual goalie AI supports full-team NHL94 only')
        if any(getattr(engine, field) is None for field in (
                'goalie_modes', 'goalie_hold_counts', 'controller_teams', 'input_block', 'clock_stopped', 'camera')):
            raise ValueError('Manual goalie AI requires live goalie/control telemetry')
        controllers = [i for i, side in enumerate(engine.controller_teams) if side == team.controller]
        if len(controllers) != 1:
            raise ValueError('Manual goalie AI requires exactly one joystick on the controlled team')
        controller = controllers[0]
        if engine.goalie_modes[controller] != 0 or engine.goalie_modes[team.controller - 1] != 0:
            raise ValueError('Manual goalie AI requires Manual goalie settings for joystick and team; '
                             'use SabresVsMightyDucks.ManualGoalie.Start')
        goalie = team.goalie
        required = ('role', 'selection_flags', 'live_state_flags', 'live_anim', 'assignment',
                    'speed', 'agility', 'weight', 'energy', 'motion_x', 'motion_y', 'passing', 'stick')
        if (any(getattr(goalie, field) is None for field in required)
                or any(getattr(state.puck, field) is None for field in ('height', 'motion_x', 'motion_y'))):
            raise ValueError('Missing live goalie motion, ratings or assignment feedback')
        return controller

    def _event(self, name, **details):
        self.metrics[name] += 1
        self.last_event = {'event': name, 'frame': self.frames, **details}

    def _timeout(self, request):
        self._event('handoff_timeouts', request=request)
        warnings.warn(f'Manual goalie {request} not confirmed within {HANDOFF_TIMEOUT} frames; '
                      'releasing input and backing off.', RuntimeWarning, stacklevel=3)
        self.phase, self.retry_at = 'backoff', self.frames + 180
        self.release_until = self.frames + 2

    def _takeover(self, state, plan, blocked):
        if blocked or self.frames < self.retry_at:
            return False, 'preserve pending skater action'
        if owns_puck(state.team1, state.engine.puck_owner):
            return False, 'retain attacking skater'
        goalie = state.team1.goalie
        if goalie.selection_flags & 0x20 or goalie.unavailable & 2:
            return False, 'retain CPU during goalie animation recovery'
        if (abs(goalie.x - state.engine.camera[0]) >= 108
                or abs(goalie.y - state.engine.camera[1]) >= 92):
            return False, 'goalie outside effective manual viewport'
        sign = 1 if state.team1.net.y > 0 else -1
        depth = state.puck.y * sign
        if not 88 < depth < 264:
            return False, 'retain skater outside defensive zone'
        if self.policy == 'always':
            return True, 'always-manual defensive ablation'
        if plan.deadline is None:
            return False, 'no projected goal threat'
        reach = goalie_arrival(goalie, plan.target)
        if plan.deadline < TAKEOVER_FRAMES + reach + 6:
            return False, 'too late for B hold plus goalie reach'
        player = state.team1.get_player_by_scnum(controlled_slot(state.team1))
        if player is not None and eligible(player):
            if arrival_time(player, (state.puck.x, state.puck.y), boost=True) <= TAKEOVER_FRAMES / 2:
                return False, 'skater can contest before takeover'
            if all(blocks_shot_lane(player, (state.puck.x, state.puck.y), (x, state.team1.net.y))
                   for x in (-12, 0, 12)):
                return False, 'retain useful controlled shooting-lane block'
        return True, 'reachable goal threat without useful controlled skater cover'

    def step(self, state, *, blocked=False):
        controller = self.validate(state)
        self.frames += 1
        team, goalie = state.team1, state.team1.goalie
        slot, actual = team.defense_goalie, controlled_slot(team)
        selected = actual == slot and bool(goalie.selection_flags & 8)
        effective = selected and not goalie.live_state_flags & 4
        if state.engine.puck_owner == slot and self.last_owner != slot and effective and self.last_effective:
            self._event('controlled_catches')
        plan = goalie_target(state)
        self.diagnostics = {
            'phase': 'goalie', 'mode': self.phase, 'decision': self.phase, 'target': None,
            'destination': plan.target, 'waypoint': plan.target, 'actual_slot': actual,
            'desired_slot': slot, 'reason': plan.reason, 'policy': self.policy,
            'crossing_frames': plan.crossing_frames, 'crossing_x': plan.crossing_x,
            'deadline': plan.deadline, 'reach_frames': goalie_arrival(goalie, plan.target),
            'hold_count': state.engine.goalie_hold_counts[controller],
            'animation': goalie.live_anim, 'assignment': goalie.assignment,
            'cpu_fallback': bool(goalie.live_state_flags & 4),
            'metrics': dict(self.metrics), 'last_event': self.last_event,
        }
        action = self._step(state, plan, slot, selected, blocked)
        self.diagnostics.update(mode=self.phase, decision=self.phase, metrics=dict(self.metrics),
                                last_event=self.last_event)
        if action is not None:
            self.diagnostics.update(target=plan.target, buttons=action.tolist())
        self.last_anim = goalie.live_anim
        self.last_owner, self.last_effective = state.engine.puck_owner, effective
        return action

    def _step(self, state, plan, slot, selected, blocked):
        goalie, owner = state.team1.goalie, state.engine.puck_owner
        neutral = np.zeros(Buttons.INPUT_MAX, dtype=np.int8)
        if self.pending_save is not None:
            request, started = self.pending_save
            expected = SAVE_ANIMATIONS if request == 'save' else (DIVE_ANIMATION,)
            if selected and goalie.live_anim in expected and goalie.live_anim != self.last_anim:
                self._event(request + '_animations')
                self.pending_save = None
            elif self.frames - started > 6:
                self._event(request + '_unconfirmed')
                self.pending_save = None
        if state.engine.clock_stopped or state.engine.input_block:
            if self.phase not in ('skater', 'stoppage'):
                self.release_until = self.frames + 2
            self.phase = 'stoppage'
            self.outlet = self.pending_save = None
            self.possession_since = None
            return neutral if selected or self.frames < self.release_until else None
        if self.phase == 'stoppage':
            self.phase = 'positioning' if selected else 'skater'
            self.retry_at = self.frames + 4
        if self.phase == 'backoff':
            if self.frames < self.release_until:
                return neutral
            self.phase = 'positioning' if selected else 'skater'
        if (self.phase in ('prepare-goalie', 'request-goalie')
                and not selected and owns_puck(state.team1, owner)):
            self._event('takeover_cancelled', reason='friendly recovery during B hold')
            self.phase, self.release_until, self.retry_at = 'neutral-handoff', self.frames + 2, self.frames + 36
            return neutral
        if self.phase == 'prepare-goalie':
            if self.frames < self.release_until:
                return neutral
            self.phase, self.started = 'request-goalie', self.frames
        if self.phase == 'request-goalie':
            if selected:
                self._event('takeovers', elapsed_frames=self.frames - self.started)
                self.phase, self.release_until = 'release-goalie', self.frames + 2
                return neutral
            if self.frames - self.started >= HANDOFF_TIMEOUT:
                self._timeout('takeover')
                return neutral
            neutral[Buttons.INPUT_B] = 1
            return neutral
        if self.phase == 'release-goalie' and self.frames < self.release_until:
            return neutral
        if self.phase == 'request-skater':
            actual = controlled_slot(state.team1)
            base = state.team1.skater_scnum_base()
            if base <= actual < base + len(state.team1.players):
                self._event('returns', actual_slot=actual)
                self.phase, self.release_until, self.retry_at = 'neutral-handoff', self.frames + 2, self.frames + 36
                return neutral
            if self.frames - self.started >= HANDOFF_TIMEOUT:
                self._timeout('return')
            elif self.frames - self.started < 2:
                neutral[Buttons.INPUT_B] = 1
            return neutral
        if self.phase == 'neutral-handoff':
            if self.frames < self.release_until:
                return neutral
            self.phase = 'skater'
        if self.outlet is not None:
            return self._outlet_feedback(state)
        if self.pending_save is not None and self.frames - self.pending_save[1] < 2:
            request = self.pending_save[0]
            if request == 'save':
                neutral[Buttons.INPUT_C] = 1
            else:
                neutral[Buttons.INPUT_A] = 1
                neutral[Buttons.INPUT_RIGHT if plan.target[0] > goalie.x else Buttons.INPUT_LEFT] = 1
            return neutral
        if goalie.role != 0 or goalie.unavailable & 4:
            self.diagnostics['reason'] = 'goalie pulled or unavailable'
            if selected:
                return self._return()
            self.phase = 'skater'
            return None
        if not selected:
            self.possession_since = None
            if self.phase not in ('skater', 'release-goalie'):
                self._event('automatic_handoffs', actual_slot=controlled_slot(state.team1))
                self.phase, self.release_until = 'neutral-handoff', self.frames + 2
                return neutral
            self.phase = 'skater'
            take, reason = self._takeover(state, plan, blocked)
            self.diagnostics['reason'] = reason
            if take:
                self._event('takeover_requests')
                self.phase, self.started = 'prepare-goalie', self.frames
                self.release_until = self.frames + 2
                # Release any skater B/C first; holding an old edge cannot select a goalie.
                return neutral
            return None
        fallback = bool(goalie.live_state_flags & 4)
        if fallback:
            self.metrics['cpu_fallback_frames'] += 1
        if goalie.selection_flags & 0x20 or goalie.unavailable & 2:
            self.phase = 'cpu-fallback' if fallback else 'save-recovery'
            return neutral
        sign = 1 if state.team1.net.y > 0 else -1
        if owner != slot and (owns_puck(state.team1, owner)
                              or state.puck.y * sign < 70 and plan.crossing_frames is None):
            return self._return()
        if fallback:
            self.phase = 'cpu-fallback'
            return neutral
        if owner == slot:
            return self._possess(state)
        self.possession_since = None
        self.phase = 'positioning'
        action = goalie_steer(goalie, plan.target)
        if plan.crossing_frames is not None and self.frames >= self.save_at:
            gap = abs(plan.crossing_x - goalie.x)
            if plan.crossing_frames <= 9 and gap > 18 and plan.height <= 5:
                action[:] = 0
                action[Buttons.INPUT_A] = 1
                action[Buttons.INPUT_RIGHT if plan.crossing_x > goalie.x else Buttons.INPUT_LEFT] = 1
                self._request_save('dive')
            elif plan.crossing_frames <= 12 and gap <= 18:
                action[:] = 0
                action[Buttons.INPUT_C] = 1
                self._request_save('save')
        return action

    def _return(self):
        self.phase, self.started = 'request-skater', self.frames
        self._event('return_requests')
        action = np.zeros(Buttons.INPUT_MAX, dtype=np.int8)
        action[Buttons.INPUT_B] = 1
        return action

    def _request_save(self, request):
        self.phase = request + '-request'
        self.pending_save = request, self.frames
        self.save_at = self.frames + 40
        self._event(request + '_requests')

    def _possess(self, state):
        self.phase = 'possession-outlet'
        if self.possession_since is None:
            self.possession_since = self.frames
        action = np.zeros(Buttons.INPUT_MAX, dtype=np.int8)
        if self.frames - self.possession_since < 8:
            return action
        candidates = [evaluate_pass(state, state.team1.goalie, index, player, 'outlet')
                      for index, player in enumerate(state.team1.players)]
        self.diagnostics['outlet_candidates'] = [details for _, details in candidates]
        options = [option for option, _ in candidates if option is not None]
        if options:
            option = max(options, key=lambda candidate: candidate.value)
            aim_pass(action, state.team1.goalie, state.team1.get_player_by_scnum(option.slot))
            action[Buttons.INPUT_B] = 1
            self.outlet = {'receiver': option.slot, 'started': self.frames,
                           'passes': state.team1.pass_attempts, 'launched': False,
                           'deadline': self.frames + math.ceil(option.flight_frames) + 20,
                           'buttons': action.copy()}
            self._event('outlet_requests', receiver=option.slot)
        elif self.frames - self.possession_since >= 90:
            # Manual neutral holding does not invoke the CPU cover timer. A clears.
            action[Buttons.INPUT_A] = 1
            sign = 1 if state.team2.net.y > state.team1.net.y else -1
            action[Buttons.INPUT_UP if sign > 0 else Buttons.INPUT_DOWN] = 1
            self.outlet = {'receiver': None, 'started': self.frames, 'passes': state.team1.pass_attempts,
                           'launched': False, 'deadline': self.frames + 60, 'buttons': action.copy()}
            self._event('clear_requests')
        return action

    def _outlet_feedback(self, state):
        request = self.outlet
        owner = state.engine.puck_owner
        neutral = np.zeros(Buttons.INPUT_MAX, dtype=np.int8)
        self.phase = 'outlet-flight'
        launched = (state.team1.pass_attempts is not None and request['passes'] is not None
                    and state.team1.pass_attempts > request['passes'])
        if not request['launched'] and launched:
            request['launched'] = True
            self._event('outlets_launched' if request['receiver'] is not None else 'clears_launched')
        elif not request['launched'] and owner != state.team1.defense_goalie:
            self._event('outlet_interrupted' if request['receiver'] is not None else 'clear_possession_ended')
            self.outlet = None
            self.phase, self.release_until = 'neutral-handoff', self.frames + 2
            return neutral
        if request['launched'] and owner >= 0 and owner != state.team1.defense_goalie:
            self._event('outlet_receptions' if owner == request['receiver'] else 'outlet_other_owner',
                        owner=owner)
            self.outlet = None
            self.phase, self.release_until = 'neutral-handoff', self.frames + 2
        elif self.frames >= request['deadline'] or (
                not request['launched'] and self.frames - request['started'] > 8):
            self._event('outlet_timeouts')
            self.outlet = None
            self.phase = 'positioning'
            self.possession_since = self.frames - 90
        elif (not request['launched'] and self.frames - request['started'] < 2
              and controlled_slot(state.team1) == state.team1.defense_goalie):
            return request['buttons'].copy()
        return neutral
