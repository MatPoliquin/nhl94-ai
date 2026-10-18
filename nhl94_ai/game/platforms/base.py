"""Console contracts expressed in rink units and native emulator frames."""
from dataclasses import dataclass

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


class PlatformBackend:
    name = 'genesis'
    rink = RinkGeometry()
    native_ratings = True
    velocity_scale = 17 / 65536
    skater_friction = 63 / 64
    normal_release_frames = None
    normalization = None

    def decode_info(self, info):
        return info

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

    def validate_options(self, args):
        return None
