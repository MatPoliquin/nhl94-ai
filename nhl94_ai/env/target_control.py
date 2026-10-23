"""Target-only actions executed with normal buttons, at emulator-frame cadence."""
from dataclasses import asdict, dataclass
import heapq
import math

import numpy as np

from nhl94_ai.game.constants import GameConsts as Buttons


ACTION_TYPE = 'TARGET_POSITION'
CONTROLLER_FIELDS = (
    'target_valid', 'target_x', 'target_y',
    'desired_slot_0', 'desired_slot_1', 'desired_slot_2', 'desired_slot_3', 'desired_slot_4',
    'switch_cooldown', 'boost_cooldown', 'poke_cooldown', 'switch_attempts',
)


@dataclass(frozen=True)
class TargetSettings:
    version: int = 1
    scale_x: float = 120.0
    scale_y: float = 270.0
    arrival_radius: float = 4.0
    steering_deadzone: float = 2.0
    braking_frames: float = 10.0
    switch_margin_frames: float = 12.0
    switch_cooldown: int = 24
    max_switch_attempts: int = 2
    boost_cooldown: int = 30
    boost_distance: float = 55.0
    poke_cooldown: int = 16
    poke_distance: float = 18.0


SETTINGS = TargetSettings()
_NETS = ((-28.0, 28.0, 254.0, 284.0), (-28.0, 28.0, -284.0, -254.0))
_ROUTE_POINTS = ((-32.0, 250.0), (32.0, 250.0), (-32.0, -250.0), (32.0, -250.0))


def validate_target_bounds(bounds):
    if bounds is None:
        return None
    values = np.asarray(bounds, dtype=np.float64)
    if values.shape != (4,) or not np.all(np.isfinite(values)):
        raise ValueError('Target bounds must be finite (x_min, x_max, y_min, y_max).')
    x_min, x_max, y_min, y_max = (float(value) for value in values)
    if not (-SETTINGS.scale_x <= x_min < x_max <= SETTINGS.scale_x
            and -SETTINGS.scale_y <= y_min < y_max <= SETTINGS.scale_y):
        raise ValueError('Target bounds must describe a nonempty rectangle inside the rink.')
    return x_min, x_max, y_min, y_max


def target_schema(profile, bounds=None):
    if profile != 'defense':
        raise ValueError(f'Unsupported target-control profile: {profile}')
    bounds = validate_target_bounds(bounds)
    schema = {
        'profile': profile,
        'coordinates': 'absolute-world-xy-v1',
        'state_feedback': 'authoritative-home-control-and-puck-owner-v1',
        'action_shape': [2],
        'action_dtype': 'float32',
        'observation_fields': list(CONTROLLER_FIELDS),
        'settings': asdict(SETTINGS),
    }
    if bounds is not None:
        schema['coordinates'] = 'task-bounded-world-xy-v1'
        schema['bounds'] = list(bounds)
    return schema


def validate_target(action):
    target = np.asarray(action, dtype=np.float32)
    if target.shape != (2,) or not np.all(np.isfinite(target)):
        raise ValueError('TARGET_POSITION requires two finite normalized coordinates.')
    if np.any(np.abs(target) > 1.0):
        raise ValueError('TARGET_POSITION coordinates must be in [-1, 1]; use the policy prediction, not raw network outputs.')
    return target


def target_position(action, bounds=None):
    """Decode normalized actions directly into the task's world-coordinate area."""
    target = validate_target(action)
    bounds = validate_target_bounds(bounds)
    if bounds is None:
        return float(target[0] * SETTINGS.scale_x), float(target[1] * SETTINGS.scale_y)
    x_min, x_max, y_min, y_max = bounds
    return (x_min + (float(target[0]) + 1) * (x_max - x_min) / 2,
            y_min + (float(target[1]) + 1) * (y_max - y_min) / 2)


def normalize_position(position, bounds=None):
    """Encode a world point; callers explicitly clip points outside the task area."""
    point = np.asarray(position, dtype=np.float64)
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        raise ValueError('A target position must contain two finite world coordinates.')
    bounds = validate_target_bounds(bounds)
    if bounds is None:
        return (point / [SETTINGS.scale_x, SETTINGS.scale_y]).astype(np.float32)
    x_min, x_max, y_min, y_max = bounds
    return np.asarray([(point[0] - x_min) * 2 / (x_max - x_min) - 1,
                       (point[1] - y_min) * 2 / (y_max - y_min) - 1], dtype=np.float32)


def project_target(position):
    """Conservative playable bounds, with rounded corners and net clearance."""
    x = min(max(float(position[0]), -SETTINGS.scale_x), SETTINGS.scale_x)
    y = min(max(float(position[1]), -SETTINGS.scale_y), SETTINGS.scale_y)
    corner_x, corner_y = max(0.0, abs(x) - 72), max(0.0, abs(y) - 222)
    radius = math.hypot(corner_x, corner_y)
    if radius > 48:
        x = math.copysign(72 + corner_x * 48 / radius, x)
        y = math.copysign(222 + corner_y * 48 / radius, y)
    if abs(x) < 28 and abs(y) > 254:
        if 28 - abs(x) < abs(y) - 254:
            x = math.copysign(28, x)
        else:
            y = math.copysign(254, y)
    return x, y


def _clear_segment(start, end):
    for left, right, bottom, top in _NETS:
        enter, leave = 0.0, 1.0
        for value, delta, low, high in (
            (start[0], end[0] - start[0], left, right),
            (start[1], end[1] - start[1], bottom, top),
        ):
            if abs(delta) < 1e-9:
                if not low < value < high:
                    break
            else:
                first, last = sorted(((low - value) / delta, (high - value) / delta))
                enter, leave = max(enter, first), min(leave, last)
            if enter >= leave:
                break
        else:
            if enter < leave:
                return False
    return True


def route_waypoint(position, target):
    escaped = project_target(position)
    if math.dist(position, escaped) > 0.5:
        return escaped
    if _clear_segment(position, target):
        return target
    points = tuple(dict.fromkeys((position, target, *_ROUTE_POINTS)))
    queue = [(0.0, 0, ())]
    visited = set()
    while queue:
        distance, index, path = heapq.heappop(queue)
        if index in visited:
            continue
        visited.add(index)
        if index == 1:
            return points[path[0]]
        for neighbor, point in enumerate(points):
            if neighbor not in visited and _clear_segment(points[index], point):
                heapq.heappush(queue, (
                    distance + math.dist(points[index], point), neighbor, (*path, neighbor),
                ))
    raise RuntimeError(f'No target route around nets: {position} -> {target}')


def steering_direction(player, waypoint):
    """Velocity-aware D-pad corrections shared by target and scripted routes."""
    dx, dy = waypoint[0] - player.x, waypoint[1] - player.y
    vx, vy = player.motion_x, player.motion_y
    if math.hypot(dx, dy) <= SETTINGS.arrival_radius and math.hypot(vx, vy) < 0.2:
        return (0, 0), 'holding'
    corrections = dx - SETTINGS.braking_frames * vx, dy - SETTINGS.braking_frames * vy
    pad = tuple(math.copysign(1, value) if abs(value) > SETTINGS.steering_deadzone else 0
                for value in corrections)
    mode = 'braking' if corrections[0] * vx + corrections[1] * vy < 0 else 'skating'
    return pad, mode


class TargetPositionController:
    """No tactical destinations: selection and local mechanics only."""

    def __init__(self, profile='defense', bounds=None):
        if profile != 'defense':
            raise ValueError(f'Unsupported target-control profile: {profile}')
        self.profile = profile
        self.bounds = validate_target_bounds(bounds)
        self.reset()

    def reset(self):
        self.target = None
        self.destination = None
        self.waypoint = None
        self.desired_slot = None
        self.switch_cooldown = 0
        self.boost_cooldown = 0
        self.poke_cooldown = 0
        self.switch_attempts = 0
        self.frames = 0
        self.actual_slot = None
        self.acting_slot = None
        self.switches = 0
        self.mode = 'reset'
        self.buttons = np.zeros(Buttons.INPUT_MAX, dtype=np.int8)

    def observe(self, state, info):
        required = ['p1_control_slot', 'target_controller_team', 'puck_owner',
                    'target_puck_vx', 'target_puck_vy']
        for slot in range(5):
            prefix = 'p1_' if slot == 0 else f'p1_{slot + 1}_'
            required.extend([f'target_{name}_{slot}' for name in ('role', 'flags', 'unavailable', 'facing')])
            required.extend([prefix + 'live_vel_x', prefix + 'live_vel_y'])
        missing = [name for name in required if name not in info]
        if missing:
            raise ValueError(f'TARGET_POSITION requires authoritative RAM fields: {", ".join(missing)}')
        raw_slot = int(info['p1_control_slot'])
        if info['target_controller_team'] != 1 or raw_slot >= 6:
            raise ValueError('TARGET_POSITION requires home controller 1 with a negative (unselected) slot or 0-5.')
        # ROM selection resets can write either FFFF or only the high byte (FFxx).
        slot = max(-1, raw_slot)
        if self.actual_slot is not None and slot != self.actual_slot:
            self.switches += 1
        self.actual_slot = slot
        if slot == self.desired_slot:
            self.switch_attempts = 0
        return self.diagnostics(state)

    def observation(self):
        target = (0.0, 0.0) if self.target is None else self.target
        return np.asarray([
            float(self.target is not None), *target,
            *(float(slot == self.desired_slot) for slot in range(5)),
            self.switch_cooldown / SETTINGS.switch_cooldown,
            self.boost_cooldown / SETTINGS.boost_cooldown,
            self.poke_cooldown / SETTINGS.poke_cooldown,
            self.switch_attempts / SETTINGS.max_switch_attempts,
        ], dtype=np.float32)

    @staticmethod
    def _eligible(state, info, slot):
        if slot not in range(5):
            return False
        player = state.team1.players[slot]
        return (info[f'target_role_{slot}'] > 0 and not player.is_falling
                and not info[f'target_flags_{slot}'] & 0x20
                and not info[f'target_unavailable_{slot}'] & 0x04)

    @staticmethod
    def _arrival_time(player, target):
        dx, dy = target[0] - player.x, target[1] - player.y
        distance = math.hypot(dx, dy)
        along = (dx * player.motion_x + dy * player.motion_y) / max(distance, 1.0)
        return distance / 1.5 - along * 6.0

    def _select_player(self, state, info):
        eligible = [slot for slot in range(5) if self._eligible(state, info, slot)]
        if not eligible:
            return None, None
        costs = {slot: self._arrival_time(state.team1.players[slot], self.destination)
                 for slot in eligible}
        best = min(costs, key=costs.get)
        if self.actual_slot in costs and costs[self.actual_slot] <= costs[best] + SETTINGS.switch_margin_frames:
            best = self.actual_slot
        if self.desired_slot in costs and costs[self.desired_slot] <= costs[best] + 4:
            best = self.desired_slot
        if best != self.desired_slot:
            self.switch_attempts = 0
        self.desired_slot = best
        # B selects near the projected puck, not near our destination. This is
        # an estimate only; observe() confirms the ROM's actual choice.
        puck = (state.puck.x + info['target_puck_vx'] * 17 / 65536 * 4,
                state.puck.y + info['target_puck_vy'] * 17 / 65536 * 4)
        candidates = [slot for slot in eligible if slot != self.actual_slot]
        likely = min(candidates, key=lambda slot: math.dist(
            (state.team1.players[slot].x, state.team1.players[slot].y), puck,
        )) if candidates else None
        return best, likely

    def _steer(self, player):
        self.waypoint = route_waypoint((player.x, player.y), self.destination)
        pad, self.mode = steering_direction(player, self.waypoint)
        for correction, negative, positive in (
            (pad[0], Buttons.INPUT_LEFT, Buttons.INPUT_RIGHT),
            (pad[1], Buttons.INPUT_DOWN, Buttons.INPUT_UP),
        ):
            if correction:
                self.buttons[positive if correction > 0 else negative] = 1

    def step(self, action, state, info):
        if self.actual_slot is None:
            raise RuntimeError('Target controller must observe a reset frame before stepping')
        self.target = validate_target(action).copy()
        self.destination = project_target(target_position(self.target, self.bounds))
        self.waypoint = self.destination
        self.frames += 1
        self.acting_slot = self.actual_slot
        self.buttons = np.zeros(Buttons.INPUT_MAX, dtype=np.int8)
        for name in ('switch_cooldown', 'boost_cooldown', 'poke_cooldown'):
            setattr(self, name, max(0, getattr(self, name) - 1))
        owner = state.engine.puck_owner
        if state.team1.owns_scnum(owner):
            self.mode = 'possession'
            return self.buttons.copy()
        if self.actual_slot == -1:
            self.mode = 'waiting-for-selection'
            return self.buttons.copy()
        best, likely = self._select_player(state, info)
        can_press_b = not state.action[4]
        can_switch = not self.switch_cooldown and can_press_b and self.switch_attempts < SETTINGS.max_switch_attempts
        if best is not None and best != self.actual_slot and best == likely and can_switch:
            self.buttons[Buttons.INPUT_B] = 1
            self.switch_cooldown = SETTINGS.switch_cooldown
            self.poke_cooldown = SETTINGS.poke_cooldown
            self.switch_attempts += 1
            self.mode = 'switch-request'
            return self.buttons.copy()
        if not self._eligible(state, info, self.actual_slot):
            self.mode = 'waiting-for-skater'
            return self.buttons.copy()
        player = state.team1.players[self.actual_slot]
        self._steer(player)
        angle = info[f'target_facing_{self.actual_slot}'] * math.pi / 4
        facing = math.sin(angle), math.cos(angle)
        puck_delta = state.puck.x - player.x, state.puck.y - player.y
        puck_distance = math.hypot(*puck_delta)
        poke_aligned = sum(a * b for a, b in zip(puck_delta, facing)) >= puck_distance * 0.7
        if (puck_distance <= SETTINGS.poke_distance and poke_aligned
                and not self.poke_cooldown and not self.switch_cooldown and can_press_b):
            self.buttons[Buttons.INPUT_B] = 1
            self.poke_cooldown = SETTINGS.poke_cooldown
            self.mode = 'poke-request'
        else:
            delta = self.waypoint[0] - player.x, self.waypoint[1] - player.y
            distance = math.hypot(*delta)
            aligned = sum(a * b for a, b in zip(delta, facing)) >= distance * 0.95
            can_boost = not self.boost_cooldown and not state.action[5]
            if (self.mode == 'skating' and distance >= SETTINGS.boost_distance and aligned
                    and math.hypot(player.motion_x, player.motion_y) < 1.8
                    and can_boost):
                self.buttons[Buttons.INPUT_C] = 1
                self.boost_cooldown = SETTINGS.boost_cooldown
                self.mode = 'boost-request'
        return self.buttons.copy()

    def diagnostics(self, state):
        player = state.team1.get_player_by_scnum(self.actual_slot) if self.actual_slot is not None else None
        return {
            'frame': self.frames,
            'target': None if self.target is None else list(target_position(self.target, self.bounds)),
            'bounds': self.bounds,
            'destination': self.destination, 'waypoint': self.waypoint,
            'desired_slot': self.desired_slot, 'actual_slot': self.actual_slot,
            'acting_slot': self.acting_slot, 'puck_owner': state.engine.puck_owner,
            'selection_available': self.actual_slot is not None and self.actual_slot >= 0,
            'mode': self.mode, 'buttons': self.buttons.tolist(),
            'distance': None if self.destination is None or player is None else math.dist(
                (player.x, player.y), self.destination),
            'switches': self.switches, 'switch_attempts': self.switch_attempts,
        }
