"""SNES array RAM layout and estimated mechanics for the common Classic agent.

The integration's owner is a word-array byte offset (0, 2, ..., 22), whereas
shot_player already uses logical player IDs. Positions and facing use the
same center-origin, positive-up rink convention as Genesis. Native animation
pointers and selection flags must never be decoded as Genesis animation IDs.
"""
from copy import copy
from importlib.resources import files
import math

from nhl94_ai.game.platforms.base import PlatformBackend, RinkGeometry


PLAYER_PREFIXES = tuple(
    f'p{side}_' + (f'{index + 1}_' if index else '') if index < 5 else f'g{side}_'
    for side in (1, 2) for index in range(6)
)
WRAM = 0x7E0000
VELOCITY_X = WRAM + 0x0EC3
VELOCITY_Y = WRAM + 0x0EE3
PUCK_VELOCITY_X = WRAM + 0x0EDF
PUCK_VELOCITY_Y = WRAM + 0x0EFF
SELECTED = 0x0008
UNAVAILABLE = 0x1000
# Native animation pointers, verified with A/Y goalie inputs. These are not
# Genesis animation IDs. A chooses one of the directional kick/save sequences.
SAVE_ANIMATIONS = frozenset((0xE1E2, 0xE214, 0xE2AA, 0xE2FC, 0xE246, 0xE278,
                             0xE862, 0xE7D0, 0xE73E))
DIVE_ANIMATIONS = frozenset((0xE4B6,))


class SnesBackend(PlatformBackend):
    name = 'snes'
    buttons = ('B', 'Y', 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A', 'X', 'L', 'R')
    control_mapping = dict(zip(('pass', 'clear', 'mode', 'start', 'up', 'down', 'left',
                               'right', 'shoot', 'aux1', 'aux2', 'aux3'), buttons))
    rink = RinkGeometry(net_depth=8, puck_y=300)
    native_ratings = False
    normalization = 'nhl94-snes-game-constants-v1'
    velocity_scale = 1 / 4096
    skater_friction = 125 / 128
    normal_release_frames = 8
    boost_speed = 1.5
    # Bounded empirical defaults, not fabricated live player attributes.
    estimated_pass_speed = 3.4
    estimated_acceleration = 0.075
    estimated_speed = 2.3
    goalie_takeover_frames = 2
    goalie_takeover_button = 9  # X, an edge rather than Genesis's B hold.
    goalie_takeover_pulse = 4
    goalie_viewport = None
    goalie_save_animations = SAVE_ANIMATIONS
    goalie_dive_animations = DIVE_ANIMATIONS
    goalie_clear_animation = 0xE34E

    @property
    def scenario(self):
        return str(files('nhl94_ai').joinpath('data/snes-scenario.json'))

    def validate_options(self, args):
        from nhl94_ai.agents.registry import CONTROLLERS, ALIASES
        policy = getattr(args, 'nn', '')
        classic = policy in CONTROLLERS or policy == 'MlpPolicy' and getattr(args, 'agent', None) in ALIASES
        human = getattr(args, 'mode', None) == 'player_vs_game'
        if (not classic and not (human and policy == 'MlpPolicy')) or getattr(args, 'rf', 'PostPlay') not in ('', 'PostPlay'):
            raise ValueError('SNES currently supports Classic PostPlay; RAM-writing training tasks and learned policies require a SNES contract')
        if any(getattr(args, name, None) for name in ('model', 'model_1', 'model_2', 'load_p1_model', 'load_p2_model', 'retention_model')):
            raise ValueError('Genesis learned-policy artifacts cannot be loaded as SNES policies')
        model_input = getattr(args, 'model_input', 'default')
        if isinstance(model_input, str) and model_input.startswith('pvg'):
            raise ValueError('The compact PvG observation layout requires Genesis ROM attributes')
        if (getattr(args, 'action_type', 'FILTERED').upper() != 'FILTERED'
                or getattr(args, 'selfplay', False)):
            raise ValueError('SNES Classic requires FILTERED controls without self-play')
        mode = getattr(args, 'mode', 'model_vs_game')
        if mode not in ('model_vs_game', 'player_vs_game', 'player_vs_model', 'model_vs_model'):
            raise ValueError('Unsupported SNES playback mode')
        players = getattr(args, 'num_players', 1)
        if players != (2 if mode in ('player_vs_model', 'model_vs_model') else 1):
            raise ValueError('SNES playback requires one controller versus CPU or two for multiplayer')
        if (getattr(args, 'goalie_positioning', None) not in (None, 'classic')
                or getattr(args, 'goalie_model', None) or getattr(args, 'goalie_planner', None)
                or getattr(args, 'goalie_decision_interval', None) not in (None, 1)
                or getattr(args, 'goalie_save_timing', 'classic') != 'classic'):
            raise ValueError('SNES manual goalies use shared classic positioning with native SNES saves')
        if getattr(args, 'mode', None) is not None and getattr(args, 'seed', None) is not None:
            raise ValueError('SNES ROM RNG seeding is not mapped; omit --seed to use the saved RNG')

    def register(self, env, skaters):
        if skaters != 5:
            raise ValueError('SNES requires the full-team integration')
        for slot, prefix in enumerate(PLAYER_PREFIXES):
            for axis, base in (('x', VELOCITY_X), ('y', VELOCITY_Y)):
                env.data.set_variable(prefix + 'live_vel_' + axis,
                                      {'address': base + slot * 2, 'type': '<i2'})
        for axis, address in (('x', PUCK_VELOCITY_X), ('y', PUCK_VELOCITY_Y)):
            env.data.set_variable('puck_live_vel_' + axis, {'address': address, 'type': '<i2'})
        for name, base, count in (
                ('snes_controller_assignment_', 0x1C84, 5),
                ('snes_controller_team_', 0x19F6, 5),
                ('snes_controller_player_', 0x19EC, 5),
                ('snes_player_controller_', 0x19D4, 12),
                ('snes_team_controller_count_', 0x19D0, 2),
                ('snes_first_controller_', 0x0AF5, 2),
                ('snes_last_controller_', 0x0AF9, 2),
                ('snes_next_controller_', 0x0AFD, 2),
                ('snes_goalie_manual_', 0x1C90, 2)):
            for index in range(count):
                env.data.set_variable(name + str(index),
                                      {'address': WRAM + base + 2 * index, 'type': '<i2'})
        for name, address in (('snes_input_block', 0x15D6), ('snes_game_flags', 0x15D8)):
            env.data.set_variable(name, {'address': WRAM + address, 'type': '<i2'})

    def configure_controllers(self, env, side, players):
        """Apply the native player-select tables once, before the reset frame.

        Persistent menu assignment is 1=home/2=away/0=off; the live table is
        0=home/2=away/-2=off. Player and joystick references are byte offsets.
        Team controller counts and the three cursors preserve selection through
        automatic handoffs and subsequent faceoffs. Gameplay never rewrites
        these assignments.
        """
        info = env.data.lookup_all()
        teams = (1, 2) if side == 'home' else (2, 1)
        teams = teams[:players]
        # Preserve the supplied home-only state byte for byte when possible.
        current = tuple(info.get(f'snes_controller_assignment_{i}') for i in range(5))
        desired = teams + (0,) * (5 - players)
        if current == desired:
            return
        for slot, prefix in enumerate(PLAYER_PREFIXES):
            env.data.set_value(prefix + 'state_flags', (info[prefix + 'state_flags'] & ~SELECTED) | 2)
            env.data.set_value(f'snes_player_controller_{slot}', -1)
        for joystick, team in enumerate(desired):
            slot = (team - 1) * 6 + 2 if team else -1
            env.data.set_value(f'snes_controller_assignment_{joystick}', team)
            env.data.set_value(f'snes_controller_team_{joystick}', (team - 1) * 2)
            env.data.set_value(f'snes_controller_player_{joystick}', slot * 2 if team else -1)
            if team:
                env.data.set_value(f'snes_player_controller_{slot}', joystick * 2)
                prefix = PLAYER_PREFIXES[slot]
                env.data.set_value(prefix + 'state_flags', info[prefix + 'state_flags'] | SELECTED)
        for team in (1, 2):
            joystick = teams.index(team) * 2 if team in teams else -1
            env.data.set_value(f'snes_team_controller_count_{team - 1}', teams.count(team))
            for field in ('first', 'last', 'next'):
                env.data.set_value(f'snes_{field}_controller_{team - 1}', joystick)

    @staticmethod
    def player_id(value):
        if value is None or value < 0:
            return -1
        if value > 22 or value % 2:
            raise ValueError(f'Invalid SNES puck-owner byte offset: {value}')
        return value // 2

    def decode_info(self, info):
        if info.get('nhl94_platform') == self.name:
            return info
        result = dict(info)
        result['nhl94_platform'] = self.name
        result['native_puck_owner'] = info.get('puck_owner')
        result['puck_owner'] = self.player_id(info.get('puck_owner'))
        for joystick in (0, 1):
            native = info.get(f'snes_controller_team_{joystick}')
            if native is not None:
                result[f'defense_team{joystick + 1}'] = native // 2 + 1 if native >= 0 else 0
                result[f'defense_control{joystick + 1}'] = self.player_id(
                    info.get(f'snes_controller_player_{joystick}'))
        for side in (1, 2):
            slots = range((side - 1) * 6, side * 6)
            selected = [slot for slot in slots if info.get(PLAYER_PREFIXES[slot] + 'state_flags', 0) & SELECTED]
            if len(selected) > 1:
                raise ValueError(f'SNES team {side} has multiple selected players; same-team co-op is not supported')
            result[f'p{side}_control_slot'] = selected[0] if selected else -1
            owner = result['puck_owner']
            prefix = PLAYER_PREFIXES[owner] if owner in slots else None
            for star in ('fullstar', 'emptystar'):
                source = prefix if star == 'fullstar' else PLAYER_PREFIXES[selected[0]] if selected else None
                for axis in ('x', 'y'):
                    key = ('' if side == 1 and star == 'fullstar' else f'p{side}_') + star + '_' + axis
                    result[key] = info[source + axis] if source is not None else 1000000
        return result

    def frame_info(self, env, info):
        return self.decode_info(info)

    def episode_done(self, state, done):
        # The SNES integration exposes a separate countdown word, without the
        # Genesis timer's historical ten-second PostPlay cutoff.
        return state.time <= 0

    def load_team_telemetry(self, team, info):
        team.pass_attempts = info.get(f'p{team.controller}_pass_attempts')

    def update_engine_state(self, state, info):
        engine = state.engine
        engine.puck_owner = info['puck_owner']
        engine.puck_owner_known = info.get('native_puck_owner') is not None
        engine.shot_player = info.get('shot_player', -1)
        engine.pass_dir = info.get('pass_dir', 0)
        engine.pass_speed = info.get('pass_speed', 0)
        engine.controlled_is_shooter = float(engine.shot_player == state.team1.controlled_scnum()
                                             and engine.shot_player >= 0)
        engine.faceoff = info.get('native_puck_owner') == -2
        engine.controller_teams = tuple(info.get(f'defense_team{i}', 0) for i in (1, 2))
        modes = tuple(info.get(f'snes_goalie_manual_{i}') for i in (0, 1))
        engine.goalie_modes = None if None in modes else tuple(0 if mode == 1 else 1 for mode in modes)
        engine.input_block = info.get('snes_input_block')
        engine.clock_stopped = engine.faceoff or bool(info.get('snes_game_flags', 0) & 0x1000)

    def decode_player(self, player, info, prefix):
        super().decode_player(player, info, prefix)
        # 0x0008 is joystick selection, not Genesis's one-timer flag.
        player.is_one_timer = player.is_breakaway = player.is_falling = 0.0
        player.is_pad_stack = player.is_dive = 0.0
        player.native_state_flags = player.state_flags
        player.state_flags = 0
        player.unavailable = 4 if player.native_state_flags & UNAVAILABLE else 0
        player.selection_flags = (8 if player.native_state_flags & SELECTED else 0) | (32 if player.unavailable else 0)
        player.live_anim, player.live_anim_frame = player.anim, player.anim_frame
        player.live_state_flags = 0
        player.facing = player.orientation = info.get(prefix + 'ori', 0) % 8
        player.friction = self.skater_friction

    def update_state(self, state, info):
        state.puck.platform = self.name
        state.puck.height = max(0, info.get('puck_z', 0))
        state.puck.motion_z = None
        for axis in ('x', 'y'):
            value = info.get('puck_live_vel_' + axis)
            setattr(state.puck, 'motion_' + axis, None if value is None else value * self.velocity_scale)
        state.puck.friction = 255 / 256
        for team in (state.team1, state.team2):
            team.defense_control = team.controlled_scnum()
            team.defense_goalie = team.goalie_scnum()
            for slot, player in enumerate((*team.players, team.goalie), team.skater_scnum_base()):
                prefix = PLAYER_PREFIXES[slot]
                self.decode_player(player, info, prefix)
                player.ori_x = math.sin(player.orientation * math.pi / 4)
                player.ori_y = math.cos(player.orientation * math.pi / 4)
            team.goalie.role = 0
            if team.goalie.live_anim in SAVE_ANIMATIONS | DIVE_ANIMATIONS:
                team.goalie.selection_flags |= 0x20

    @staticmethod
    def validate_goalie(state):
        engine = state.engine
        if (engine.goalie_modes is None or engine.input_block is None
                or any(value is None for value in (state.team1.goalie.motion_x,
                       state.team1.goalie.motion_y, state.puck.motion_x, state.puck.motion_y))):
            raise ValueError('SNES manual goalie AI requires native control and motion telemetry')
        controllers = [i for i, team in enumerate(engine.controller_teams) if team == state.team1.controller]
        if len(controllers) != 1:
            raise ValueError('SNES manual goalie AI requires one joystick on the controlled team')
        if engine.goalie_modes[state.team1.controller - 1] != 0:
            raise ValueError('SNES manual goalie AI requires Manual goalie settings for the controlled team')
        return controllers[0]

    def goalie_motion(self, player):
        return self.estimated_acceleration * (0.5 if player.has_puck else 1), self.estimated_speed

    def skating(self, player, optimistic=False):
        return (self.estimated_acceleration * (1.3 if optimistic else 1),
                self.estimated_speed * (1.3 if optimistic else 1), 4096)

    def grounded_step(self, player, pad, *, goalie=False, stop=False):
        from nhl94_ai.agents.motion import velocity
        future = copy(player)
        vx, vy = velocity(player)
        vx, vy = vx * self.skater_friction, vy * self.skater_friction
        x = player.x if player.precise_x is None else player.precise_x
        y = player.y if player.precise_y is None else player.precise_y
        future.x = future.precise_x = x + vx
        future.y = future.precise_y = y + vy
        acceleration, limit, _ = self.skating(player)
        if stop:
            vx, vy = (math.copysign(max(0, abs(v) - acceleration), v) for v in (vx, vy))
        elif any(pad):
            length = math.hypot(*pad)
            wanted = round(math.atan2(pad[0], pad[1]) * 4 / math.pi) % 8
            heading = player.facing if player.facing is not None else player.orientation
            delta = (wanted - heading + 4) % 8 - 4
            future.facing = (heading + (1 if delta > 0 else -1 if delta < 0 else 0)) % 8
            proposed = vx + pad[0] / length * acceleration, vy + pad[1] / length * acceleration
            if math.hypot(*proposed) <= max(limit, math.hypot(vx, vy)):
                vx, vy = proposed
        future.motion_x, future.motion_y = vx, vy
        return future
