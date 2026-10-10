"""Opt-in skating deception: bait, observe tracking, cut, then take a quick shot."""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
import math

import numpy as np

from nhl94_ai.agents.cross_crease import _path_clear, crossing_clear, shot_lane_clear
from nhl94_ai.agents.defense import controlled_slot, eligible, on_ice
from nhl94_ai.agents.motion import VELOCITY_SCALE, coast_projection, velocity
from nhl94_ai.agents.offense import goalie_avoidance, normal_shot_release_frames, shot_release_in_front
from nhl94_ai.agents.passing import pressure_margin, rom_direction, shot_value
from nhl94_ai.agents.skating import grounded_step, skating_route
from nhl94_ai.env.target_control import SETTINGS, project_target, route_waypoint, steering_direction
from nhl94_ai.game.constants import GameConsts as Buttons


BAIT_FRAMES = 64
CUT_FRAMES = 96
SKATING_FRAMES = BAIT_FRAMES + CUT_FRAMES
SEQUENCE_FRAMES = SKATING_FRAMES + 64
COOLDOWN_FRAMES = 72
VALUE_MARGIN = 8
SHOT_HOLD_FRAMES = 4
SKATING_CLEARANCE = 32
FINISH_DEPTH = 202
LOOKAHEAD_FRAMES = 24


@dataclass(frozen=True)
class DekePlan:
    slot: int
    direction: int
    bait: tuple
    target: tuple
    frames: int
    value: float
    pressure: float
    aim_side: int


def deke_entry(state, player):
    sign = 1 if state.team2.net.y > state.team1.net.y else -1
    return player is not None and 150 <= player.y * sign <= 216 and abs(player.x) <= 60


def _missing_feedback(state, player):
    missing = [field for field in ('motion_x', 'motion_y', 'shot_power', 'handedness',
                                   'live_anim', 'live_anim_frame', 'precise_x', 'precise_y',
                                   'facing_phase', 'sprite_flipped_x', 'speed', 'agility', 'weight', 'energy',
                                   'shot_offsets_x', 'shot_offsets_y', 'shot_durations',
                                   'shot_projection_offsets_x', 'shot_projection_offsets_y', 'shot_projection_durations')
               if getattr(player, field) is None]
    missing.extend('goalie.' + field for field in ('motion_x', 'motion_y', 'live_anim', 'speed',
                                                  'agility', 'weight', 'decision_timer',
                                                  'decision_interval', 'steering', 'precise_x', 'precise_y',
                                                  'energy', 'cpu_tracking_active')
                   if getattr(state.team2.goalie, field) is None)
    if state.time is None:
        missing.append('time')
    return missing


def quick_finish(state, player, side, hold_frames=SHOT_HOLD_FRAMES):
    """Check the live puck lane and the entire short ShotMode coast."""
    frames = normal_shot_release_frames(player, hold_frames)
    clear, reason, margin, _ = crossing_clear(state, player, frames, goalie_clearance=24)
    body = coast_projection(player, frames)
    point = (state.puck.x + body[0] - player.x, state.puck.y + body[1] - player.y)
    bounds = player.shot_offsets_x
    points = [(body[0] + offset, point[1]) for offset in bounds]
    sign = 1 if state.team2.net.y > state.team1.net.y else -1
    safe = (clear and player.y * sign >= FINISH_DEPTH and margin >= 2
            and shot_release_in_front(state, player, hold_frames)
            and all(shot_lane_clear(state, release, side, frames, goal_x=13) for release in (*points, point)))
    return safe, {'release_frames': frames, 'hold_frames': hold_frames, 'release_point': point, 'pressure': margin,
                  'release_x_bounds': tuple(release[0] for release in points),
                  'status': 'open' if safe else reason if not clear else 'no-safe-opening'}


def _ordinary_goalie_scene(state):
    goalie = state.team2.goalie
    return (-52 <= goalie.x <= 52 and 210 < abs(goalie.y) <= 260 and goalie.y > -260
            and goalie.y * state.puck.y > 0 and 0 <= goalie.steering <= 8)


def _goalie_step(state):
    """Ordinary CPU tracking with live decision cadence; never teleport the goalie."""
    goalie = state.team2.goalie
    coast = grounded_step(goalie, (0, 0))
    if not goalie.cpu_tracking_active:
        return coast
    timer = goalie.decision_timer - 1
    direction = goalie.steering
    if timer < 0:
        timer = goalie.decision_interval
        carrier = state.team1.get_player_by_scnum(state.engine.puck_owner)
        reference_x = carrier.x + (state.puck.x - carrier.x) // 2
        multiplier = 160 + 16 * (int(state.time) % 8) - (64 if state.puck.y > 219 or state.puck.y <= -219 else 0)
        vx, vy = (round(component / VELOCITY_SCALE) for component in velocity(state.puck))
        sign = 1 if state.team2.net.y > state.team1.net.y else -1
        u = reference_x + (multiplier * vx // 65536)
        v = max(-259, min(259, state.puck.y + multiplier * vy // 65536)) - sign * 260
        if u * u + v * v > 900:
            length = math.isqrt(u * u + v * v) + 1
            u, v = math.trunc(26 * u / length), math.trunc(18 * v / length)
        dx = u - coast.x - round(coast.motion_x / VELOCITY_SCALE) // 256
        dy = sign * 260 + v - coast.y - round(coast.motion_y / VELOCITY_SCALE) // 256
        direction = 8 if abs(dx) <= 4 and abs(dy) <= 4 else rom_direction(dx, dy)
    angle = direction * math.pi / 4
    pad = ((round(math.sin(angle)), round(math.cos(angle))) if direction < 8 else (0, 0))
    future = grounded_step(goalie, pad, goalie=True)
    future.decision_timer, future.steering = timer, direction
    return future


def _advance_preview(scene, slot, action):
    carrier = scene.team1.get_player_by_scnum(slot)
    offset = scene.puck.x - carrier.x, scene.puck.y - carrier.y
    pad = (int(action[Buttons.INPUT_RIGHT]) - int(action[Buttons.INPUT_LEFT]),
           int(action[Buttons.INPUT_UP]) - int(action[Buttons.INPUT_DOWN]))
    future = grounded_step(carrier, pad)
    if future.facing != carrier.facing:
        future.shot_offsets_x = future.shot_projection_offsets_x
        future.shot_offsets_y = future.shot_projection_offsets_y
        future.shot_durations = future.shot_projection_durations
    index = slot - scene.team1.skater_scnum_base()
    scene.team1.players[index] = future
    scene.puck.x, scene.puck.y = future.x + offset[0], future.y + offset[1]
    scene.puck.motion_x, scene.puck.motion_y = future.motion_x, future.motion_y
    for team in (scene.team1, scene.team2):
        for index, other in enumerate(team.players):
            if other is future or not on_ice(other):
                continue
            team.players[index] = grounded_step(other, (0, 0))
    scene.team2.goalie = _goalie_step(scene)


def movement_safety(state, player, target):
    samples = [(elapsed, (future.x, future.y)) for elapsed, future in
               skating_route(player, target, LOOKAHEAD_FRAMES) if elapsed and elapsed % 4 == 0]
    clear, reason, margin, clearance = _path_clear(
        state, player, samples, goalie_clearance=SKATING_CLEARANCE)
    if clear and clearance is not None and clearance <= SKATING_CLEARANCE:
        clear, reason = False, 'projected-goalie-contact'
    if clear and pressure_margin(state.team2, (state.puck.x, state.puck.y), 4) < 0:
        clear, reason = False, 'imminent-puck-pressure'
    return clear, reason, margin


def preview_deke(state, plan, alternatives):
    """Run the live controller's phases and predicates on a projected scene."""
    scene = deepcopy(state)
    controller = DekeController()
    controller.diagnostics = {'alternatives': dict(alternatives)}
    controller.start(plan, scene, 0)
    margin = 60.0
    for elapsed in range(SKATING_FRAMES + 1):
        if not _ordinary_goalie_scene(scene):
            return None, {'status': 'unsupported-goalie-state', 'frames': elapsed}
        action = controller.step(scene, elapsed)
        if controller.plan is None:
            return None, {'status': controller.diagnostics.get('reason', controller.diagnostics['outcome']),
                          'outcome': controller.diagnostics['outcome'], 'frames': elapsed}
        margin = min(margin, controller.diagnostics.get('pressure', 60))
        if controller.event['pressed']:
            player = scene.team1.get_player_by_scnum(plan.slot)
            value = shot_value(scene, player, controller.diagnostics['release_point'])
            risk = min(24, max(0, 10 - margin) * 0.75)
            value -= elapsed * 0.08 + risk
            finish = DekePlan(plan.slot, plan.direction, plan.bait, plan.target,
                              elapsed, value, margin, plan.aim_side)
            return finish, {'status': 'feasible', 'frames': elapsed, 'pressure': margin, 'risk': risk,
                            'value': value, 'hold_frames': controller.event['hold_frames'],
                            'release_point': controller.diagnostics['release_point'],
                            'release_x_bounds': controller.diagnostics['release_x_bounds'],
                            'timeline': controller.event['timeline']}
        _advance_preview(scene, plan.slot, action)
    return None, {'status': 'cut-unreachable'}


def _candidate_routes(state, player):
    sign = 1 if state.team2.net.y > state.team1.net.y else -1
    depth = player.y * sign
    finish_depth = FINISH_DEPTH + SETTINGS.arrival_radius
    for direction in (-1, 1):
        already_wide = player.x * direction >= 18
        bait = (player.x if already_wide else max(-48, min(48, player.x + direction * 28)),
                sign * min(206, max(178, depth + (12 if already_wide else 8))))
        if not already_wide and (bait[0] - player.x) * direction < 12:
            continue
        for width in (20, 32):
            yield DekePlan(state.engine.puck_owner, direction, bait,
                           (-direction * width, sign * finish_depth), 0, 0, 60, -direction)


def evaluate_deke(state, alternatives):
    diagnostics = {'alternatives': dict(alternatives), 'candidates': [],
                   'goalie_model': 'ordinary-cpu-tracking-with-live-cadence',
                   'motion_model': 'native-grounded-forward-skating'}
    player = state.team1.get_player_by_scnum(state.engine.puck_owner)
    if player is None or player is state.team1.goalie:
        return None, {**diagnostics, 'status': 'missing-feedback'}
    missing = _missing_feedback(state, player)
    if (missing or not state.engine.puck_owner_known or controlled_slot(state.team1) != state.engine.puck_owner):
        return None, {**diagnostics, 'status': 'missing-feedback', 'missing_feedback': missing}
    if state.is_shootout_active or state.is_shooting or not eligible(player) or not on_ice(state.team2.goalie):
        return None, {**diagnostics, 'status': 'ineligible-state'}
    if not _ordinary_goalie_scene(state):
        return None, {**diagnostics, 'status': 'unsupported-goalie-state'}
    if not deke_entry(state, player):
        return None, {**diagnostics, 'status': 'outside-deke-entry'}
    sign = 1 if state.team2.net.y > state.team1.net.y else -1
    if player.y * sign >= FINISH_DEPTH and any(quick_finish(state, player, side)[0] for side in (-1, 1)):
        return None, {**diagnostics, 'status': 'shot-already-open'}
    choices = []
    for candidate in _candidate_routes(state, player):
        plan, row = preview_deke(state, candidate, alternatives)
        diagnostics['candidates'].append({'direction': candidate.direction, 'bait': candidate.bait,
                                          'target': candidate.target, **row})
        if plan is not None:
            choices.append(plan)
    if not choices:
        return None, {**diagnostics, 'status': 'no-safe-deke'}
    plan = max(choices, key=lambda option: option.value)
    if plan.value <= max(alternatives.values(), default=0) + VALUE_MARGIN:
        return None, {**diagnostics, 'status': 'better-alternative'}
    return plan, {**diagnostics, 'status': 'selected', 'plan': asdict(plan)}


class DekeController:
    def __init__(self):
        self.plan = self.event = None
        self.phase = 'idle'
        self.started = self.phase_at = self.pressed = self.retry_at = 0
        self.events = []
        self.metrics = Counter()
        self.diagnostics = {}

    def start(self, plan, state, frame):
        if self.plan is not None:
            raise RuntimeError('Cannot replace an active deke')
        self.plan, self.started, self.phase_at = plan, frame, frame
        self.phase = 'bait'
        self.metrics['plans'] += 1
        player, goalie = state.team1.get_player_by_scnum(plan.slot), state.team2.goalie
        self.event = {
            'frame': frame, 'slot': plan.slot, 'plan': asdict(plan),
            'start_position': (player.x, player.y), 'goalie_position': (goalie.x, goalie.y),
            'shots_before': state.team1.stats.shots, 'goals_before': state.team1.stats.score,
            'pressed': False, 'windup': False, 'released': False, 'shot': False, 'contact': False,
            'retries': 0,
            'goalie_impact': goalie.contact_impact or 0, 'skater_impact': player.contact_impact or 0,
            'alternatives': dict(self.diagnostics.get('alternatives', {})),
            'timeline': [{'frame': frame, 'phase': 'bait'}],
        }

    def _phase(self, phase, frame, reason):
        self.phase, self.phase_at = phase, frame
        self.diagnostics['phase'] = phase
        self.event['timeline'].append({'frame': frame, 'phase': phase, 'reason': reason})

    def _finish(self, frame, outcome):
        self.events.append({**self.event, 'end_frame': frame, 'outcome': outcome})
        self.metrics[outcome] += 1
        self.plan = self.event = None
        self.phase = 'idle'
        self.retry_at = frame + COOLDOWN_FRAMES
        self.diagnostics.update(phase='idle', outcome=outcome, metrics=dict(self.metrics))

    def observe(self, state, frame):
        if self.plan is None:
            return
        plan, event = self.plan, self.event
        player, goalie = state.team1.get_player_by_scnum(plan.slot), state.team2.goalie
        if player is None or (self.phase in ('bait', 'cut') and (
                controlled_slot(state.team1) != plan.slot or not eligible(player))):
            self._finish(frame, 'control-lost')
            return
        impacts = goalie.contact_impact or 0, player.contact_impact or 0
        contact = (goalie.contact_player == plan.slot and impacts[0] > event['goalie_impact']
                   or player.contact_player == state.team2.goalie_scnum() and impacts[1] > event['skater_impact'])
        event['goalie_impact'], event['skater_impact'] = impacts
        if contact and not event['contact']:
            event['contact'] = True
            self.metrics['goalie_contacts'] += 1
        if (event['pressed'] and state.is_shooting and state.engine.puck_owner == plan.slot
                and not event['windup']):
            event['windup'] = True
            self.metrics['accepted_windups'] += 1
        if (event['windup'] and not event['released'] and state.engine.puck_owner < 0
                and state.engine.shot_player == plan.slot and state.engine.last_puck_player == plan.slot):
            event['released'] = True
            self.metrics['observed_releases'] += 1
            self._phase('exit', frame, 'native-release')
        if (event['pressed'] and not event['shot'] and state.team1.stats.shots > event['shots_before']
                and state.engine.shot_player == plan.slot):
            event['shot'] = True
            self.metrics['recorded_shots'] += 1
        if event['shot'] and state.team1.stats.score > event['goals_before'] and state.engine.shot_player == plan.slot:
            self._finish(frame, 'goals')
        elif state.engine.puck_owner >= 0 and state.engine.puck_owner != plan.slot:
            self._finish(frame, 'shot-ended' if event['released'] or event['shot'] else 'possession-lost')
        elif state.engine.clock_stopped:
            self._finish(frame, 'play-stopped')
        elif frame - self.started >= SEQUENCE_FRAMES:
            self._finish(frame, 'shot-ended' if event['released'] else 'sequence-timeout')

    def _shoot(self, action, state, player, frame, c_down, reason):
        safe, details = quick_finish(state, player, self.plan.aim_side)
        if not safe:
            safe, details = quick_finish(state, player, self.plan.aim_side, 1)
        self.diagnostics.update(details)
        if not safe:
            return False
        value = shot_value(state, player, details['release_point'])
        if (reason == 'live-opening-after-cut'
                and value <= max(self.event['alternatives'].values(), default=0) + VALUE_MARGIN):
            self.diagnostics['status'] = 'opening-not-worth-delay'
            return False
        if not c_down:
            action[Buttons.INPUT_C] = 1
            self.event['pressed'] = True
            self.event['hold_frames'] = details.get('hold_frames', SHOT_HOLD_FRAMES)
            self.pressed = frame
            self.metrics['attempts'] += 1
            self._phase('hold', frame, reason)
            self.diagnostics['waypoint'] = (self.plan.aim_side * 13, state.team2.net.y)
        action[Buttons.INPUT_RIGHT if self.plan.aim_side > 0 else Buttons.INPUT_LEFT] = 1
        return True

    def step(self, state, frame, *, c_down=False):
        if self.plan is None:
            return None
        action = np.zeros(Buttons.INPUT_MAX, dtype=np.int8)
        plan, player = self.plan, state.team1.get_player_by_scnum(self.plan.slot)
        missing = ['player'] if player is None else _missing_feedback(state, player)
        if missing:
            self.diagnostics['missing_feedback'] = missing
            self._finish(frame, 'feedback-lost')
            return action
        if self.phase in ('hold', 'release', 'exit'):
            if controlled_slot(state.team1) != plan.slot:
                return action
            if self.phase in ('hold', 'release') or state.is_shooting:
                action[Buttons.INPUT_RIGHT if plan.aim_side > 0 else Buttons.INPUT_LEFT] = 1
                elapsed = frame - self.pressed
                hold_frames = self.event.get('hold_frames', SHOT_HOLD_FRAMES)
                if self.phase == 'hold':
                    action[Buttons.INPUT_C] = int(elapsed < hold_frames)
                    if elapsed >= hold_frames:
                        self._phase('release', frame, 'bounded-wrist-shot')
                if not self.event['windup'] and elapsed >= hold_frames + 3:
                    self.metrics['unaccepted_presses'] += 1
                    if self.event['retries'] < 2 and state.engine.puck_owner == plan.slot and eligible(player):
                        self.event['retries'] += 1
                        self._phase('cut', frame, 'unaccepted-shot-retry')
                    else:
                        self._finish(frame, 'shot-not-started')
                elif self.phase == 'release' and frame - self.pressed >= 32:
                    self._finish(frame, 'shot-unconfirmed')
            else:
                target = (player.x - plan.direction * 40, player.y)
                escape = goalie_avoidance(state, player, target)
                if escape is not None:
                    target = escape[0]
                clear, _, _, _ = crossing_clear(state, player, 8, pressure=False, target=target)
                if clear:
                    self._pad(action, steering_direction(player, target)[0])
                self.diagnostics['waypoint'] = target
            self.diagnostics.update(phase=self.phase)
            return action
        if state.engine.puck_owner != plan.slot:
            self._finish(frame, 'possession-lost')
            return action
        goalie = state.team2.goalie
        moved = (player.x - self.event['start_position'][0]) * plan.direction
        follow = (goalie.x - self.event['goalie_position'][0]) * plan.direction
        sign = 1 if state.team2.net.y > state.team1.net.y else -1
        approached = ((player.y - self.event['start_position'][1]) * sign >= 8
                      and player.x * plan.direction >= 18)
        tracking = follow >= 6 or (goalie.x * plan.direction >= 4
                                  and velocity(goalie)[0] * plan.direction > 0.12)
        if self.phase == 'bait' and (moved >= 16 or approached) and tracking:
            self.event['bait_position'] = (player.x, player.y)
            self.metrics['observed_tracking'] += 1
            self.metrics['cuts'] += 1
            self._phase('cut', frame, 'goalie-followed-bait')
        target = plan.bait if self.phase == 'bait' else plan.target
        clear, reason, margin = movement_safety(state, player, target)
        cut_distance = (self.event.get('bait_position', (player.x, player.y))[0] - player.x) * plan.direction
        if self.phase == 'cut' and cut_distance >= 10 and self._shoot(
                action, state, player, frame, c_down, 'live-opening-after-cut'):
            return action
        if not clear:
            if self.phase == 'cut' and self._shoot(action, state, player, frame, c_down, 'finish-before-pressure'):
                return action
            self.diagnostics['reason'] = reason
            self._finish(frame, 'route-invalidated')
            return action
        if frame - self.phase_at >= (BAIT_FRAMES if self.phase == 'bait' else CUT_FRAMES):
            self._finish(frame, 'bait-timeout' if self.phase == 'bait' else 'cut-window-missed')
            return action
        if route_waypoint((player.x, player.y), target) != target or math.dist(target, project_target(target)) > 1e-6:
            self._finish(frame, 'route-invalidated')
            return action
        self._pad(action, steering_direction(player, target)[0])
        self.diagnostics.update(phase=self.phase, waypoint=target, pressure=margin,
                                follow=follow, cut_distance=cut_distance)
        return action

    @staticmethod
    def _pad(action, pad):
        if pad[0]:
            action[Buttons.INPUT_RIGHT if pad[0] > 0 else Buttons.INPUT_LEFT] = 1
        if pad[1]:
            action[Buttons.INPUT_UP if pad[1] > 0 else Buttons.INPUT_DOWN] = 1
