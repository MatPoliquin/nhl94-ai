"""Opt-in manual goalie positioning and feedback-confirmed joystick handoffs."""
from collections import Counter
from dataclasses import dataclass
import math
import warnings

import numpy as np

from nhl94_ai.agents.defense import DefenseController, controlled_slot, eligible, owns_puck
from nhl94_ai.agents.motion import VELOCITY_SCALE, arrival_time, blocks_shot_lane, skating, velocity
from nhl94_ai.agents.passing import evaluate_pass, one_timer_contact_frame, pass_contact, pass_release_frames
from nhl94_ai.agents.skating import grounded_step
from nhl94_ai.game.constants import GameConsts as Buttons
from nhl94_ai.game.geometry import aim_pass
from nhl94_ai.game.platforms import get_backend


TAKEOVER_FRAMES = get_backend().goalie_takeover_frames
HANDOFF_TIMEOUT = 60
SAVE_ANIMATIONS = get_backend().goalie_save_animations
DIVE_ANIMATION = next(iter(get_backend().goalie_dive_animations))


def outlet_options(state, *, decision_interval=4, limit=None, release_prediction=False):
    """Use the same eligible, receivable outlets for either goalie policy."""
    candidates = [evaluate_pass(state, state.team1.goalie, index, player, 'outlet',
                                decision_interval=decision_interval, release_prediction=release_prediction)
                  for index, player in enumerate(state.team1.players[:limit])]
    options = sorted((option for option, _ in candidates if option is not None),
                     key=lambda option: (-option.value, option.slot))
    return options, [details for _, details in candidates]


@dataclass(frozen=True)
class GoaliePlan:
    target: tuple
    reason: str
    crossing_frames: float | None = None
    crossing_x: float | None = None
    height: float | None = None
    deadline: float | None = None
    receiver: tuple | None = None
    reception_frames: float | None = None


@dataclass(frozen=True)
class IncomingPass:
    passer: int
    receiver: int
    passes: int
    expires: int
    period: int


def goalie_motion(player):
    """Near-net goalieacc accelerates in pad direction, without skater turn cost."""
    if getattr(player, 'platform', 'genesis') != 'genesis':
        return get_backend(player.platform).goalie_motion(player)
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


def _receiver_contact(state, windup, receiver_slot=None):
    target = state.engine.pass_target if receiver_slot is None else receiver_slot
    receiver = state.team2.get_player_by_scnum(target) if target is not None else None
    if receiver is None:
        return None
    # Genesis also locks skater control during a one-timer's shooting animation.
    winding_up = (receiver_slot is not None and state.platform.name == 'genesis'
                  and receiver.is_one_timer and receiver.live_anim in (0x7FC, 0x92E)
                  and (receiver.role is None or receiver.role > 0) and not receiver.unavailable & 4)
    if not eligible(receiver) and not winding_up:
        return None
    if windup is not None:
        passer = state.team2.get_player_by_scnum(windup[0])
        if passer is None or state.engine.puck_owner != windup[0]:
            return None
        contact = pass_contact(state.puck, passer, receiver)
        release = pass_release_frames(receiver)
    elif (state.engine.puck_owner < 0 and state.engine.last_puck_player is not None
          and state.team2.owns_scnum(state.engine.last_puck_player)):
        if (target == state.engine.last_puck_player or state.puck.height > 8):
            return None
        vx, vy = velocity(state.puck)
        rx, ry = velocity(receiver)
        offset = receiver.stick_x or 0, receiver.stick_y or 0
        dx, dy = receiver.x + offset[0] - state.puck.x, receiver.y + offset[1] - state.puck.y
        if dx * (vx - rx) + dy * (vy - ry) <= 0:
            return None
        closest = math.inf, None, 0
        for frame in range(1, 41):
            travel = (state.puck.friction * (1 - state.puck.friction**frame) / (1 - state.puck.friction)
                      if state.puck.friction != 1 else frame)
            point = state.puck.x + vx * travel, state.puck.y + vy * travel
            stick = receiver.x + rx * frame + offset[0], receiver.y + ry * frame + offset[1]
            distance = math.dist(point, stick)
            if distance < closest[0]:
                closest = distance, point, frame
        if closest[0] > 14:
            return None
        contact = ((state.puck.x, state.puck.y), closest[1], closest[2], math.hypot(vx, vy))
        release = 0
    else:
        return None
    if contact is None:
        return None
    time = one_timer_contact_frame(state.puck, receiver, contact, release)
    if time is None:
        return None
    rx, ry = velocity(receiver)
    return time, (receiver.x + rx * time, receiver.y + ry * time)


def goalie_target(state, windup=None, *, receiver_slot=None):
    """Predict contact at the goalie's depth, stopping before uncertain rebounds."""
    sign = 1 if state.team1.net.y > 0 else -1
    puck = state.puck
    vx, vy = velocity(puck)
    x, y, z, vz = float(puck.x), float(puck.y), max(0, puck.height), puck.motion_z or 0
    incoming = _receiver_contact(state, windup, receiver_slot)
    # Hold our live depth on a shot instead of skating back to a fixed plane.
    # If the puck has passed us, retreat within the crease for a last-chance save.
    depth = max(236, min(254, max(state.team1.goalie.y * sign, y * sign)))
    crease_crossing = (0.0, x, z) if y * sign >= depth else None
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
            if crease_crossing is None and y * sign >= depth:
                fraction = (sign * depth - previous[1]) / (y - previous[1])
                crease_crossing = (frame - 1 + fraction,
                                   previous[0] + (x - previous[0]) * fraction,
                                   max(0, previous[2] + (z - previous[2]) * fraction))
            if y * sign >= 264:
                fraction = (sign * 264 - previous[1]) / (y - previous[1])
                goal_x = previous[0] + (x - previous[0]) * fraction
                goal_z = max(0, previous[2] + (z - previous[2]) * fraction)
                if abs(goal_x) <= 25 and goal_z <= 24:
                    time, cross_x, height = crease_crossing
                    if incoming is not None and incoming[0] < time:
                        break
                    return GoaliePlan((max(-32, min(32, cross_x)), sign * depth),
                                      'released puck threatens goal', time, cross_x, height,
                                      frame - 1 + fraction)
                break
    vx, vy = velocity(puck)
    reference = puck.x + vx * 8, max(-259, min(259, puck.y + vy * 8))
    reception = None
    if incoming is not None:
        reception, point = incoming
        reference = point[0], max(-259, min(259, point[1]))
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
                      else 'protect predicted shooting angle', deadline=deadline,
                      receiver=reference if reception is not None else None, reception_frames=reception)


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


def moving_body_contact(state, action, horizon, *, save=False):
    """Estimate low-puck contact while moving or entering a momentum-quartering save.

    This is a short-horizon prediction, not a safety certificate. Twelve rink
    units is the smaller one-timer body radius; ordinary manual saves may have
    more reach. Do not use animation-specific reach for an upright goalie.
    The save branch projects friction-only animation recovery after C quarters
    velocity. Its normal/one-timer radius omits animation-specific extensions.
    """
    goalie, puck = state.team1.goalie, state.puck
    if not 0 < horizon <= 12 or any(getattr(goalie, name) is None for name in (
            'speed', 'agility', 'weight', 'energy')):
        return None
    pad = (int(action[Buttons.INPUT_RIGHT]) - int(action[Buttons.INPUT_LEFT]),
           int(action[Buttons.INPUT_UP]) - int(action[Buttons.INPUT_DOWN]))
    if not any(pad):
        return None
    x, y, z = float(puck.x), float(puck.y), max(0, puck.height)
    vx, vy = velocity(puck)
    vz = puck.motion_z or 0
    previous = (x - goalie.x, y - goalie.y)
    for frame in range(1, math.ceil(horizon) + 1):
        previous_height = z
        goalie = grounded_step(goalie, (0, 0) if save else pad, goalie=not save)
        if save and frame == 1:
            goalie.motion_x /= 4
            goalie.motion_y /= 4
        if z <= 0:
            vx, vy = vx * puck.friction, vy * puck.friction
        x, y = x + vx, y + vy
        if z > 0 or vz:
            vz -= 102 * VELOCITY_SCALE
            z += vz
            if z < 0:
                z, vz = 0, -vz / 2
        relative = (x - goalie.x, y - goalie.y)
        dx, dy = relative[0] - previous[0], relative[1] - previous[1]
        fraction = max(0, min(1, -(previous[0] * dx + previous[1] * dy)
                              / max(1e-9, dx * dx + dy * dy)))
        distance = math.hypot(previous[0] + fraction * dx, previous[1] + fraction * dy)
        contact_height = previous_height + fraction * (z - previous_height)
        radius = (12 if state.engine.native_one_timer_shot else 15) if save else 11
        if contact_height <= 15 and distance <= radius and frame - 1 + fraction <= horizon:
            return frame - 1 + fraction
        previous = relative
    return None


class GoalieController:
    """Exclusive per-frame input owner; slot feedback, never gameplay RAM writes."""

    def __init__(self, policy, *, pass_timing=False, save_timing='classic', platform=None):
        self.platform = platform or get_backend()
        if policy not in ('selective', 'always'):
            raise ValueError('Enabled goalie policy must be selective or always')
        self.policy = policy
        if save_timing not in ('classic', 'contact'):
            raise ValueError('Goalie save timing must be classic or contact')
        self.save_timing = save_timing
        self.pass_timing = pass_timing
        self.frames = 0
        self.phase = 'skater'
        self.started = self.release_until = self.retry_at = self.save_at = 0
        self.possession_since = None
        self.outlet = None
        self.pending_save = None
        self.last_anim = None
        self.last_owner = None
        self.last_effective = False
        self.last_passes = None
        self.pass_windup = None
        self.incoming_pass = None
        self.hold_sample = None
        self.hold_tick = None
        self.hold_stride = 1
        self.metrics = Counter()
        self.last_event = None
        self.diagnostics = {}

    @staticmethod
    def validate(state):
        if state.platform.name != 'genesis':
            return state.platform.validate_goalie(state)
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
        # A native assignment stack can be empty while the goalie is controlled.
        # Its decoded label is diagnostic only; motion and control remain required.
        required = ('role', 'selection_flags', 'live_state_flags', 'live_anim',
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

    def _takeover(self, state, plan, blocked, takeover_frames=None):
        if takeover_frames is None:
            takeover_frames = self.platform.goalie_takeover_frames
        if blocked or self.frames < self.retry_at:
            return False, 'preserve pending skater action'
        if owns_puck(state.team1, state.engine.puck_owner):
            return False, 'retain attacking skater'
        goalie = state.team1.goalie
        if goalie.selection_flags & 0x20 or goalie.unavailable & 2:
            return False, 'retain CPU during goalie animation recovery'
        viewport = self.platform.goalie_viewport
        if viewport is not None and (abs(goalie.x - state.engine.camera[0]) >= viewport[0]
                                     or abs(goalie.y - state.engine.camera[1]) >= viewport[1]):
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
        deadline = plan.crossing_frames if plan.crossing_frames is not None else plan.deadline
        if deadline < takeover_frames + reach + 6:
            return False, 'too late for goalie takeover plus reach'
        player = state.team1.get_player_by_scnum(controlled_slot(state.team1))
        if player is not None and eligible(player):
            if (plan.receiver is None and state.engine.puck_owner >= 0
                    and arrival_time(player, (state.puck.x, state.puck.y), boost=True)
                    <= self.platform.goalie_takeover_frames / 2):
                return False, 'skater can contest before takeover'
            if plan.receiver is not None and state.engine.puck_owner < 0:
                race = DefenseController().choose_target(state)
                if (race.mode == 'intercept-pass' and
                        arrival_time(player, race.target) + 5 <= race.puck_arrival):
                    return False, 'skater has a reachable pass interception'
            origin = plan.receiver or (state.puck.x, state.puck.y)
            delay = plan.reception_frames or 0
            if all(blocks_shot_lane(player, origin, (x, state.team1.net.y), delay)
                   for x in (-12, 0, 12)):
                return False, 'retain useful controlled shooting-lane block'
        return True, 'reachable goal threat without useful controlled skater cover'

    def _remaining_hold(self, count):
        if self.hold_sample is not None and count < self.hold_sample:
            if self.hold_tick is not None:
                self.hold_stride = max(self.hold_stride, (self.frames - self.hold_tick) /
                                       (self.hold_sample - count))
            self.hold_tick = self.frames
        self.hold_sample = count
        return max(0, count) * self.hold_stride + 2

    def _pass_feedback(self, state):
        passes = state.team2.pass_attempts
        if (passes is not None and self.last_passes is not None and passes > self.last_passes
                and state.team2.owns_scnum(state.engine.puck_owner)
                and state.engine.pass_target is not None):
            self.pass_windup = state.engine.puck_owner, state.engine.pass_target, self.frames
        self.last_passes = passes
        if self.pass_windup is not None and (
                state.engine.puck_owner != self.pass_windup[0]
                or state.engine.pass_target != self.pass_windup[1]
                or self.frames - self.pass_windup[2] > 6):
            self.pass_windup = None

    def _retained_receiver(self, state):
        engine, opponent = state.engine, state.team2
        opponent_touch = (engine.last_puck_player is not None
                          and opponent.owns_scnum(engine.last_puck_player))
        if (state.platform.name != 'genesis' or engine.clock_stopped
                or engine.puck_owner >= 0 or not opponent_touch
                or opponent.pass_attempts is None):
            self.incoming_pass = None
            return None
        pending = self.incoming_pass
        if pending is not None and (
                engine.last_puck_player != pending.passer or opponent.pass_attempts != pending.passes
                or state.period != pending.period or self.frames > pending.expires):
            self.incoming_pass = None
        target = engine.pass_target
        if target is not None and target >= 0:
            contact = _receiver_contact(state, None)
            if contact is None:
                self.incoming_pass = None
            else:
                self.incoming_pass = IncomingPass(engine.last_puck_player, target, opponent.pass_attempts,
                                                 self.frames + min(40, math.ceil(contact[0]) + 4), state.period)
            return None
        pending = self.incoming_pass
        if pending is not None:
            # The native target word can be cleared while the receiver winds up.
            # Recheck live interception geometry; never hold the old position.
            if _receiver_contact(state, None, pending.receiver) is not None:
                return pending.receiver
            self.incoming_pass = None
        return None

    def step(self, state, *, blocked=False):
        controller = self.validate(state)
        self.frames += 1
        team, goalie = state.team1, state.team1.goalie
        slot, actual = team.defense_goalie, controlled_slot(team)
        selected = actual == slot and bool(goalie.selection_flags & 8)
        effective = selected and not goalie.live_state_flags & 4
        if state.engine.puck_owner == slot and self.last_owner != slot and effective and self.last_effective:
            self._event('controlled_catches')
        self._pass_feedback(state)
        receiver_slot = self._retained_receiver(state)
        plan = (goalie_target(state, self.pass_windup) if receiver_slot is None else
                goalie_target(state, self.pass_windup, receiver_slot=receiver_slot))
        self.diagnostics = {
            'phase': 'goalie', 'mode': self.phase, 'decision': self.phase, 'target': None,
            'destination': plan.target, 'waypoint': plan.target, 'actual_slot': actual,
            'desired_slot': slot, 'reason': plan.reason, 'policy': self.policy,
            'crossing_frames': plan.crossing_frames, 'crossing_x': plan.crossing_x,
            'deadline': plan.deadline, 'reach_frames': goalie_arrival(goalie, plan.target),
            'receiver': plan.receiver, 'reception_frames': plan.reception_frames,
            'retained_receiver_slot': receiver_slot,
            'hold_count': (state.engine.goalie_hold_counts[controller]
                           if state.engine.goalie_hold_counts is not None else None),
            'animation': goalie.live_anim, 'assignment': goalie.assignment,
            'cpu_fallback': bool(goalie.live_state_flags & 4),
            'metrics': dict(self.metrics), 'last_event': self.last_event,
        }
        action = self._step(state, plan, slot, selected, blocked)
        self.diagnostics.update(mode=self.phase, decision=self.phase, metrics=dict(self.metrics),
                                last_event=self.last_event)
        if action is not None:
            self.diagnostics.update(target=plan.target, buttons=action.tolist())
            if not action[4:8].any():
                idle = ('save-commitment' if action[Buttons.INPUT_C] else
                        'dive-commitment' if action[Buttons.INPUT_A] else
                        'waiting-for-control' if action[Buttons.INPUT_B] else self.phase)
                idle = self.diagnostics.get('idle_reason', idle)
                self.diagnostics['idle_reason'] = idle
                self.metrics['idle:' + idle] += 1
        self.last_anim = goalie.live_anim
        self.last_owner, self.last_effective = state.engine.puck_owner, effective
        return action

    def _step(self, state, plan, slot, selected, blocked):
        goalie, owner = state.team1.goalie, state.engine.puck_owner
        neutral = np.zeros(Buttons.INPUT_MAX, dtype=np.int8)
        if self.pending_save is not None:
            request, started = self.pending_save
            expected = (self.platform.goalie_save_animations if request == 'save'
                        else self.platform.goalie_dive_animations)
            if (selected and not goalie.live_state_flags & 4
                    and goalie.live_anim in expected and goalie.live_anim != self.last_anim):
                self._event(request + '_animations')
                self.pending_save = None
            elif (not selected or owns_puck(state.team1, owner) or goalie.live_state_flags & 4
                  or goalie.selection_flags & 0x20):
                self._event(request + '_cancelled')
                self.pending_save = None
                self.save_at = self.frames + 2
            elif goalie.unavailable & 6 or goalie.role != 0:
                self._event(request + '_cancelled')
                self.pending_save = None
                self.save_at = self.frames + 2
            elif self.frames - started > 6:
                self._event(request + '_unconfirmed')
                self.pending_save = None
                self.save_at = self.frames + 2
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
        if self.phase in ('prepare-goalie', 'request-goalie') and not selected:
            remaining = (self._remaining_hold(state.engine.goalie_hold_counts[
                self.validate(state)]) if self.phase == 'request-goalie'
                and state.engine.goalie_hold_counts is not None else self.platform.goalie_takeover_frames)
            self.diagnostics['remaining_hold_frames'] = remaining
            take, reason = self._takeover(state, plan, blocked, remaining)
            if not take:
                self._event('takeover_cancelled', reason=reason)
                self.diagnostics['reason'] = reason
                self.phase, self.release_until, self.retry_at = 'neutral-handoff', self.frames + 2, self.frames + 36
                return neutral
        if self.phase == 'prepare-goalie':
            if self.frames < self.release_until:
                return neutral
            self.phase, self.started = 'request-goalie', self.frames
            self.hold_sample = self.hold_tick = None
            self.hold_stride = 1
        if self.phase == 'request-goalie':
            if selected:
                self._event('takeovers', elapsed_frames=self.frames - self.started)
                self.phase, self.release_until = 'release-goalie', self.frames + 2
                return neutral
            if self.frames - self.started >= HANDOFF_TIMEOUT:
                self._timeout('takeover')
                return neutral
            pulse = getattr(self.platform, 'goalie_takeover_pulse', None)
            if pulse is None or (self.frames - self.started) % (pulse * 2) < pulse:
                neutral[self.platform.goalie_takeover_button] = 1
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
                              or state.puck.y * sign < 70 and plan.deadline is None):
            return self._return()
        if fallback:
            self.phase = 'cpu-fallback'
            return neutral
        if owner == slot:
            return self._possess(state)
        self.possession_since = None
        self.phase = 'positioning'
        return self._position(state, plan)

    def _position(self, state, plan):
        """Choose an available goalie's movement/save; handoffs remain in _step."""
        goalie = state.team1.goalie
        action = goalie_steer(goalie, plan.target)
        if plan.crossing_frames is not None and self.frames >= self.save_at:
            gap = abs(plan.crossing_x - goalie.x)
            reach = goalie_arrival(goalie, plan.target)
            can_align = (gap > 8 and plan.height <= 15 and plan.crossing_frames >= 2
                         and reach + 1 <= plan.crossing_frames
                         and (action[Buttons.INPUT_LEFT] or action[Buttons.INPUT_RIGHT]))
            if can_align:
                # A normal C save quarters momentum; finish reachable alignment first.
                self.diagnostics.update(save_decision='align-before-save', alignment_gap=gap,
                                        alignment_frames=reach)
                return action
            if self.save_timing == 'contact' and gap > 8 and plan.height <= 15:
                contact = moving_body_contact(state, action, plan.crossing_frames)
                if contact is not None and moving_body_contact(state, action, plan.crossing_frames, save=True) is None:
                    self.diagnostics.update(save_decision='body-contact-before-save',
                                            body_contact_frames=contact)
                    return action
            if plan.crossing_frames <= 9 and gap > 18 and plan.height <= 5:
                action[:] = 0
                action[Buttons.INPUT_A] = 1
                action[Buttons.INPUT_RIGHT if plan.crossing_x > goalie.x else Buttons.INPUT_LEFT] = 1
                self._request_save('dive')
            elif plan.crossing_frames <= 12 and gap <= 18:
                action[:] = 0
                action[Buttons.INPUT_C] = 1
                self._request_save('save')
        if not action.any():
            self.diagnostics['idle_reason'] = 'holding-position'
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
        options, self.diagnostics['outlet_candidates'] = outlet_options(
            state, decision_interval=1, release_prediction=self.pass_timing)
        if options:
            option = options[0]
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
        clear_animation = getattr(self.platform, 'goalie_clear_animation', None)
        if request['receiver'] is None and clear_animation is not None:
            launched = (owner < 0 and state.team1.goalie.live_anim == clear_animation
                        and math.hypot(*velocity(state.puck)) > 0)
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
        elif (not request['launched']
              and (request['receiver'] is not None or self.frames - request['started'] < 2)
              and controlled_slot(state.team1) == state.team1.defense_goalie):
            # The native goalie may sample the aim on the third frame. Keep it
            # until the pass counter confirms launch, within the timeout above.
            return request['buttons'].copy()
        return neutral
