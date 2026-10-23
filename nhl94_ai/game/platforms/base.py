"""Console contracts expressed in rink units and native emulator frames."""
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class RinkGeometry:
    net_y: float = 264
    post_x: float = 19
    net_depth: float = 10
    player_x: float = 120
    player_y: float = 270
    puck_x: float = 130
    puck_y: float = 270
    max_velocity: float = 50


# These are hockey control channels, preserving the historical Genesis API.
# B = pass/switch/poke; C = shoot/check/burst; A = clear/dive. Console names
# enter only when these channels are encoded for stable-retro.
CONTROL_CHANNELS = ('pass', 'clear', 'mode', 'start', 'up', 'down', 'left',
                    'right', 'shoot', 'aux1', 'aux2', 'aux3')


@dataclass(frozen=True)
class HockeyCommand:
    values: tuple

    @classmethod
    def from_buttons(cls, buttons):
        values = np.asarray(buttons)
        if values.shape != (12,) or not np.isin(values, (0, 1)).all():
            raise ValueError('Hockey commands require twelve binary control channels')
        return cls(tuple(int(value) for value in values))

    def encode(self, buttons, mapping):
        result = np.zeros(len(buttons), dtype=np.int8)
        for channel, value in zip(CONTROL_CHANNELS, self.values):
            if value:
                result[buttons.index(mapping[channel])] = 1
        return result


# Each public operation represents a distinct part of the console contract.
class PlatformBackend:  # pylint: disable=too-many-public-methods
    name = 'genesis'
    rink = RinkGeometry()
    native_ratings = True
    velocity_scale = 17 / 65536
    skater_friction = 63 / 64
    normal_release_frames = None
    normalization = None

    def decode_info(self, info):
        return info

    def shot_release_frames(self, player, hold_frames):
        """Frames from the shot press to puck release for a planned button hold."""
        raise NotImplementedError

    def shot_speed(self, player, hold_frames, *, one_timer=False):
        """Puck speed in rink units per frame for a shot held hold_frames."""
        raise NotImplementedError

    def update_state(self, state, info):
        return None

    def episode_done(self, state, done):
        return done(state)

    def decode_player(self, player, info, prefix):
        player.platform = self.name
        player.shot_accuracy = info.get(prefix + 'shot_accuracy')
        for axis in ('x', 'y'):
            value = info.get(prefix + 'live_vel_' + axis)
            setattr(player, 'motion_' + axis, None if value is None else value * self.velocity_scale)
        player.anim = info.get(prefix + 'anim', 0) or 0
        player.anim_frame = info.get(prefix + 'anim_frame', 0) or 0
        player.state_flags = info.get(prefix + 'state_flags', 0) or 0
        player.has_anim = float(player.anim != 0)

    def encode_action(self, action):
        # Preserve all historical buttons, including unused Genesis channels.
        return HockeyCommand.from_buttons(action).encode(self.buttons, self.control_mapping)

    # Stage-2 mechanics contract. Shared tactical code reaches these through the
    # player/puck platform instead of assuming Genesis ROM rules. Methods whose
    # only platform input is a declared attribute (friction, skating estimates)
    # live here; structural ROM rules are implemented per console. A backend
    # without recovered ROM behavior must say "estimated" in the docstring and
    # in any diagnostic model name it exposes.
    pass_recipient_models = ('native-slot-order-held-steering', 'current-position')
    pass_geometry_models = ('rom-sprite-and-integer-launch', 'facing-hotspot-estimate')

    def direction(self, dx, dy):
        """Eight-way quantization of a rink displacement."""
        raise NotImplementedError

    def aim_pass(self, action, passer, receiver):
        """Write the pad direction that aims a pass from passer to receiver."""
        raise NotImplementedError

    def skating(self, player, optimistic=False):
        """(acceleration, speed limit, energy) in rink units per frame."""
        raise NotImplementedError

    def boost_impulse(self, player):
        """One-frame burst impulse in rink units."""
        raise NotImplementedError

    def coast_projection(self, player, frames):
        """Inertia-only projection decaying at the console's skater friction."""
        from nhl94_ai.game.motion import velocity
        x, y = float(player.x), float(player.y)
        vx, vy = velocity(player)
        for _ in range(frames):
            vx, vy = vx * self.skater_friction, vy * self.skater_friction
            x, y = x + vx, y + vy
        return x, y

    def stop_projection(self, player, frames, *, bounded=False):
        """Hard-brake projection: ((x, y), (vx, vy)) or None out of bounds."""
        raise NotImplementedError

    def skate_step(self, motion, heading, turning, pad, acceleration, limit):
        """One projected frame of eight-way acceleration with turning."""
        raise NotImplementedError

    def arrival_time(self, player, target, *, optimistic=False, boost=False):
        """Reach a stick-sized area, including turning and lateral momentum."""
        from nhl94_ai.game.motion import boost_safe, facing, velocity
        from nhl94_ai.env.target_control import project_target, route_waypoint
        target = project_target(target)
        waypoint = (target if max(abs(player.y), abs(target[1])) < 248
                    else route_waypoint((player.x, player.y), target))
        dx, dy = waypoint[0] - player.x, waypoint[1] - player.y
        distance = math.hypot(dx, dy)
        route_extra = math.dist(waypoint, target)
        if distance + route_extra < 7:
            return 0.0
        vx, vy = velocity(player)
        acceleration, limit, _ = self.skating(player, optimistic)
        along = (vx * dx + vy * dy) / max(distance, 1)
        lateral = abs(vx * dy - vy * dx) / max(distance, 1)
        facing_x, facing_y = facing(player)
        turn = 4 * (1 - (facing_x * dx + facing_y * dy) / max(distance, 1))
        if optimistic and player.projection_uncertainty:
            along, lateral, turn = math.hypot(vx, vy), 0, 0
        if boost and boost_safe(player, waypoint):
            along += self.boost_impulse(player)
        # Friction reduces acceleration as speed builds. Do not clamp an existing
        # burst/collision velocity to the ordinary acceleration acceptance limit.
        effective = max(acceleration * 0.5, acceleration - max(0, along) * (1 - self.skater_friction))
        distance = max(0, distance + route_extra - 7 - player.projection_uncertainty)
        if along > limit:
            travel = distance / max(limit, along * 0.85)
        else:
            accelerate = max(0, (limit - along) / effective)
            covered = along * accelerate + effective * accelerate * accelerate / 2
            if covered >= distance:
                travel = (-along + math.sqrt(along * along + 2 * effective * distance)) / effective
            else:
                travel = accelerate + (distance - covered) / limit
        return max(0, travel) + turn + min(20, lateral / (2 * acceleration))

    def puck_path(self, puck, horizon=64, *, sample_interval=4):
        """Sampled puck trajectory as (frame, (x, y), height) tuples."""
        raise NotImplementedError

    def selected_receiver(self, team, puck, passer_slot, direction):
        """Native recipient of a pass launched in the quantized direction."""
        raise NotImplementedError

    def _select_nearest_in_sector(self, team, puck, passer_slot, direction):
        candidates = []
        for index, player in enumerate(team.players):
            slot = team.skater_scnum_base() + index
            if slot == passer_slot or player.role is not None and player.role <= 0 or player.unavailable & 4:
                continue
            dx, dy = player.x - puck.x, player.y - puck.y
            sector = self.direction(dx, dy)
            if sector == 8:
                continue
            difference = (sector - direction + 4) % 8 - 4
            if abs(difference) <= 1:
                candidates.append((dx * dx + dy * dy + (difference * 256)**2, -slot, slot))
        return min(candidates)[2] if candidates else None

    def pass_launch_view(self, state, passer):
        """(team, puck) at native launch, or None without launch telemetry."""
        return None

    def pass_release_frames(self, receiver, delay):
        """Frames from the pass input to puck release for this receiver."""
        raise NotImplementedError

    def live_pass_contact(self, puck, passer, receiver, delay):
        """Sprite/integer launch contact, or None without sprite telemetry."""
        return None

    def pass_contact(self, puck, passer, receiver, delay):
        """(start, point, flight_frames, arrival_speed) or None."""
        raise NotImplementedError

    def estimated_pass_contact(self, puck, passer, receiver, delay):
        """Constant-speed interception solve against an approximate hotspot.

        GetHot is sprite-dependent; use facing as an explicitly approximate
        hotspot. This is an estimate on every console, never a ROM rule.
        """
        from nhl94_ai.game.motion import facing, velocity
        from nhl94_ai.game.passing import PASS_HORIZON, pass_speed
        from nhl94_ai.env.target_control import project_target
        speed = pass_speed(passer)
        if speed is None or receiver.motion_x is None or receiver.motion_y is None:
            return None
        vx, vy = velocity(receiver)
        px, py = velocity(passer)
        fx, fy = facing(receiver)
        start = puck.x + px * delay, puck.y + py * delay
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

    def reception_drift_shift(self, receiver):
        """Fixed-point coast shift for a player awaiting a reception."""
        raise NotImplementedError
