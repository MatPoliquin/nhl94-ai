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
# ROM-verified motion state (NHL94 SNES $9F8511 main loop / $9F8686 player loop):
# positions are 16.16 fixed point, split across int and frac word arrays.
POSITION_X_FRAC = WRAM + 0x0D77
POSITION_Y_FRAC = WRAM + 0x0DD9
PUCK_X_FRAC = WRAM + 0x0D93  # slot 14
PUCK_Y_FRAC = WRAM + 0x0DF5
PUCK_Z_FRAC = WRAM + 0x0DFD
PUCK_VELOCITY_Z = WRAM + 0x0E01
# Physics cadence: $83 counts rendered frames, $0D01 the last processed frame;
# one physics step covers n = ($83 - $0D01) + 1 pending frames when debt > 0.
FRAME_CLOCK = WRAM + 0x0083
PHYSICS_CLOCK = WRAM + 0x0D01
FRAME_STEP = WRAM + 0x0ADB  # n of the most recent step; also the friction index
# Live fields read by the skating acceleration/turn routines at $9EA719/$9EA9B6.
CARRIER_STATE = WRAM + 0x0F43
INPUT_LOCK_X = WRAM + 0x1103
INPUT_LOCK_Y = WRAM + 0x1123
FACING_PHASE = WRAM + 0x1183
MOTION_WEIGHT = WRAM + 0x12C3
MOTION_AGILITY = WRAM + 0x12E3
SPEED_RATING = WRAM + 0x1303
TURNING_STATE = WRAM + 0x1523
MOTION_FLAGS = WRAM + 0x1283
MOTION_PUCK_OWNER = WRAM + 0x0AD9
MOTION_MODE = WRAM + 0x0D07
# Friction byte table $9F8B64 (ROM 0xF8B64) indexed by step size n: per-step
# retention T[n]/256 approximates 0.98828**n (T[2]=250/256 == 125/128).
FRICTION_BY_STEP = (107, 253, 250, 247, 244, 241, 238, 235, 232, 230)
# Turn table $9EAAA3 indexed by (wanted-facing) % 8. The low/high signed
# product words advance the phase/facing pair at $1183/$1163.
TURN_RATE_BY_DELTA = (0, 17, 17, 17, 0, -17, -17, -17)
# Speed table $80D9C6 pairs; the second word is the accepted squared sum of
# signed velocity high bytes. The first word's role is still unrecovered.
SPEED_LIMIT_SQUARED_HIGH = (415, 458, 503, 549, 598, 649, 702, 757,
                            814, 874, 935, 998, 1064, 1131, 1201, 1273)
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
            for axis, base in (('x', POSITION_X_FRAC), ('y', POSITION_Y_FRAC)):
                env.data.set_variable(prefix + axis + '_frac',
                                      {'address': base + slot * 2, 'type': '<u2'})
            for name, base in (
                    ('carrier_state', CARRIER_STATE), ('input_lock_x', INPUT_LOCK_X),
                    ('input_lock_y', INPUT_LOCK_Y), ('native_facing_phase', FACING_PHASE),
                    ('motion_flags', MOTION_FLAGS), ('motion_weight', MOTION_WEIGHT),
                    ('motion_agility', MOTION_AGILITY), ('speed_rating', SPEED_RATING),
                    ('turning_state', TURNING_STATE)):
                env.data.set_variable(prefix + name,
                                      {'address': base + slot * 2, 'type': '<i2'})
        for axis, address in (('x', PUCK_VELOCITY_X), ('y', PUCK_VELOCITY_Y)):
            env.data.set_variable('puck_live_vel_' + axis, {'address': address, 'type': '<i2'})
        for axis, address in (('x', PUCK_X_FRAC), ('y', PUCK_Y_FRAC)):
            env.data.set_variable('puck_' + axis + '_frac', {'address': address, 'type': '<u2'})
        env.data.set_variable('puck_z_frac', {'address': PUCK_Z_FRAC, 'type': '<u2'})
        env.data.set_variable('puck_live_vel_z', {'address': PUCK_VELOCITY_Z, 'type': '<i2'})
        for name, address in (('snes_frame_clock', 0x0083), ('snes_physics_clock', 0x0D01),
                              ('snes_frame_step', 0x0ADB)):
            env.data.set_variable(name, {'address': WRAM + address, 'type': '<u2'})
        for name, address in (('snes_motion_puck_owner', MOTION_PUCK_OWNER),
                              ('snes_motion_mode', MOTION_MODE)):
            env.data.set_variable(name, {'address': address, 'type': '<i2'})
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
                # Faceoff/line-change transitions briefly flag two players; the native
                # player->joystick map identifies the one the controller drives.
                narrowed = [slot for slot in selected
                            if info.get(f'snes_player_controller_{slot}') not in (None, -1)]
                if len(narrowed) == 1:
                    selected = narrowed
                else:
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
        frame_clock = info.get('snes_frame_clock')
        physics_clock = info.get('snes_physics_clock')
        engine.physics_frame_debt = (None if frame_clock is None or physics_clock is None
                                     else frame_clock - physics_clock)
        engine.physics_step_frames = info.get('snes_frame_step')

    def decode_player(self, player, info, prefix):
        super().decode_player(player, info, prefix)
        # ROM: position is one 16.16 fixed-point value split across int/frac arrays.
        for axis in ('x', 'y'):
            integer = info.get(prefix + axis)
            frac = info.get(prefix + axis + '_frac')
            setattr(player, 'position_frac_' + axis, None if frac is None else int(frac))
            precise = None if integer is None or frac is None else integer + frac / 65536
            setattr(player, 'precise_' + axis, precise)
            raw = info.get(prefix + 'live_vel_' + axis)
            setattr(player, 'raw_vel_' + axis, None if raw is None else int(raw))
        for name in ('carrier_state', 'input_lock_x', 'input_lock_y', 'native_facing_phase',
                     'motion_flags', 'motion_weight', 'motion_agility', 'speed_rating',
                     'turning_state'):
            value = info.get(prefix + name)
            setattr(player, name, None if value is None else int(value))
        player.facing_phase = (None if player.native_facing_phase is None
                               else player.native_facing_phase / 65536)
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
        for axis in ('x', 'y'):
            integer = info.get('puck_' + axis)
            frac = info.get('puck_' + axis + '_frac')
            setattr(state.puck, 'position_frac_' + axis, None if frac is None else int(frac))
            precise = None if integer is None or frac is None else integer + frac / 65536
            setattr(state.puck, 'precise_' + axis, precise)
            value = info.get('puck_live_vel_' + axis)
            setattr(state.puck, 'raw_vel_' + axis, None if value is None else int(value))
            setattr(state.puck, 'motion_' + axis, None if value is None else value * self.velocity_scale)
        z_integer = info.get('puck_z')
        z_frac = info.get('puck_z_frac')
        z_precise = (None if z_integer is None else z_integer + (z_frac or 0) / 65536)
        state.puck.height = None if z_precise is None else max(0, z_precise)
        raw_z = info.get('puck_live_vel_z')
        state.puck.motion_z = None if raw_z is None else raw_z * self.velocity_scale
        state.puck.friction = 255 / 256
        state.puck.physics_frame_debt = state.engine.physics_frame_debt
        state.puck.physics_step_frames = state.engine.physics_step_frames
        for team in (state.team1, state.team2):
            team.defense_control = team.controlled_scnum()
            team.defense_goalie = team.goalie_scnum()
            for slot, player in enumerate((*team.players, team.goalie), team.skater_scnum_base()):
                prefix = PLAYER_PREFIXES[slot]
                self.decode_player(player, info, prefix)
                player.physics_frame_debt = state.engine.physics_frame_debt
                player.physics_step_frames = state.engine.physics_step_frames
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
        """Capture-observed AI goalie positioning (goalie-shot-* captures): shuffles at
        up to ~0.68 px/frame with ~0.05 px/frame^2 acceleration. Save and dive
        animations are separate states not covered by this positioning model."""
        return 0.05, 0.68

    def shot_release_frames(self, player, hold_frames):
        """ROM $9E:E146 windup: the $ED4A swing releases at anim counter 24 after ~27
        held frames; an early button-up mirrors the counter (24 - c) and finishes the
        swing sooner. Captures: hold 5 -> release 7, hold 12 -> 19, full hold -> 27."""
        if hold_frames <= 6:
            return hold_frames + 2
        return min(27, hold_frames + 2 + (hold_frames - 6) * 5 // 6)

    def shot_speed(self, player, hold_frames, *, one_timer=False):
        """ROM $9E:E220 launch: charge $0D05 = 15 + windup steps, launch scale
        (9584 * charge * 28) >> 16 at the neutral $1363 rating (4096), puck speed
        2*68*final in 8.8 raw. Captures: hold 5 -> 77, hold 12 -> 110, full -> 143."""
        charge = 15 if one_timer else min(35, 14 + hold_frames)
        final = (9584 * charge * 28) >> 16
        return max(1.0, 136 * final * self.velocity_scale)

    def skating(self, player, optimistic=False):
        """ROM acceleration factor and speed cap when live attributes exist."""
        weight = getattr(player, 'motion_weight', None)
        agility = getattr(player, 'motion_agility', None)
        speed = getattr(player, 'speed_rating', None)
        if weight is not None and agility is not None:
            # $9EA743-$A75D: 32 - weight/8 + agility + agility/2.
            factor = max(1, 32 - (weight >> 3) + agility + (agility >> 1))
            acceleration = 6 * factor * self.velocity_scale
        else:
            acceleration = self.estimated_acceleration
        if speed is None:
            limit = self.estimated_speed
        else:
            capped = min(max(speed, 0), len(SPEED_LIMIT_SQUARED_HIGH) - 1)
            limit = math.sqrt(SPEED_LIMIT_SQUARED_HIGH[capped]) * 256 * self.velocity_scale
        return (acceleration * (1.3 if optimistic else 1),
                limit * (1.3 if optimistic else 1), 4096)

    # Motion cadence, fixed-point state, friction and integration are ROM-derived.
    # Pass direction quantization, recipient selection, release timing, launch
    # vectors and puck flight are ROM-derived (see $9BB445/$9E:E9BF/$9E:EB2E).
    pass_recipient_models = ('rom-octant-distance-search', 'rom-octant-distance-search')
    pass_geometry_models = ('rom-receiver-lead-launch', 'rom-receiver-lead-launch')

    def direction(self, dx, dy):
        """ROM $9BB445: 2:1 sector boundaries; exactly 2:1 is diagonal. Octants are
        numbered 0=+y (up) clockwise: 1=up-right, 2=+x, ..., 7=up-left; 8 is zero."""
        if dx == 0 and dy == 0:
            return 8
        code = (1 if dx < 0 else 0) | (2 if dy < 0 else 0)
        if 2 * abs(dy) >= abs(dx):
            code |= 4
        if 2 * abs(dx) >= abs(dy):
            code |= 8
        return (1, 7, 3, 5, 0, 0, 4, 4, 2, 6, 2, 6, 1, 7, 3, 5)[code]

    def aim_pass(self, action, passer, receiver):
        """ROM-derived: pad bits must land in the receiver's $9BB445 octant."""
        from nhl94_ai.game.constants import GameConsts as Buttons
        dx, dy = receiver.x - passer.x, receiver.y - passer.y
        action[4:8] = 0
        if abs(dx) > max(3, 2 * abs(dy)):
            action[Buttons.INPUT_RIGHT if dx > 0 else Buttons.INPUT_LEFT] = 1
        if abs(dy) > max(3, 2 * abs(dx)):
            action[Buttons.INPUT_UP if dy > 0 else Buttons.INPUT_DOWN] = 1

    def boost_impulse(self, player):
        """Estimated constant burst; native boost stamina is unrecovered."""
        return self.boost_speed

    def coast_projection(self, player, frames):
        """ROM cadence/friction when native fixed-point state is decoded."""
        if getattr(player, 'physics_frame_debt', None) is None:
            return super().coast_projection(player, frames)
        future = copy(player)
        for _ in range(frames):
            future = self.grounded_step(future, (0, 0))
        return future.precise_x, future.precise_y

    def stop_projection(self, player, frames, *, bounded=False):
        """Estimated braking over ROM cadence/friction; native stop timing is unrecovered."""
        from nhl94_ai.env.target_control import project_target
        if getattr(player, 'physics_frame_debt', None) is not None:
            future = copy(player)
            for _ in range(frames):
                future = self.grounded_step(future, (0, 0), stop=True)
                point = future.precise_x, future.precise_y
                if bounded and math.dist(point, project_target(point)) > 1e-6:
                    return None
            return (future.precise_x, future.precise_y), (future.motion_x, future.motion_y)
        from nhl94_ai.agents.motion import velocity
        x, y = float(player.x), float(player.y)
        vx, vy = velocity(player)
        acceleration, _, _ = self.skating(player)
        for _ in range(frames):
            vx, vy = vx * self.skater_friction, vy * self.skater_friction
            vx = math.copysign(max(0, abs(vx) - acceleration), vx)
            vy = math.copysign(max(0, abs(vy) - acceleration), vy)
            x, y = x + vx, y + vy
            if bounded and math.dist((x, y), project_target((x, y))) > 1e-6:
                return None
        return (x, y), (vx, vy)

    def skate_step(self, motion, heading, turning, pad, acceleration, limit):
        """Estimated: one sector of turning per frame; native turn timing is unrecovered."""
        (x, y), (vx, vy) = motion
        ux, uy = pad
        wanted = round(math.atan2(ux, uy) * 4 / math.pi) % 8 if ux or uy else heading
        if wanted != heading:
            difference = (wanted - heading + 4) % 8 - 4
            heading = (heading + (1 if difference > 0 else -1)) % 8
        length = max(1, math.hypot(ux, uy))
        nx = vx * self.skater_friction + acceleration * ux / length
        ny = vy * self.skater_friction + acceleration * uy / length
        if math.hypot(nx, ny) <= max(limit, math.hypot(vx, vy)):
            vx, vy = nx, ny
        else:
            vx, vy = vx * self.skater_friction, vy * self.skater_friction
        return ((x + vx, y + vy), (vx, vy)), heading, 0

    def puck_path(self, puck, horizon=64, *, sample_interval=4):
        """ROM flight: trunc(vel*255/256) drag each substep, then pos += vel*16 in 16.16;
        z integrates zvel/16 after -96 gravity and bounces with zvel = -zvel/2.
        Modeled at one substep per rendered frame (the measured long-run cadence);
        load-dependent skip frames are not modeled."""
        from nhl94_ai.agents.motion import velocity
        from nhl94_ai.env.target_control import project_target
        x, y = float(puck.x), float(puck.y)
        vx, vy = velocity(puck)
        z = max(0.0, puck.height or 0.0)
        vz = (puck.motion_z or 0.0) / self.velocity_scale  # raw; z += vz/16 per substep
        points = []
        for frame in range(1, horizon + 1):
            vx *= puck.friction
            vy *= puck.friction
            x, y = x + vx, y + vy
            if z > 0 or vz:
                vz -= 96
                z += vz / 16
                if z < 0:
                    z, vz = 0.0, -vz / 2
                    if abs(vz) < 96:
                        vz = 0.0
            if math.dist((x, y), project_target((x, y))) > 1e-6:
                break
            if frame % sample_interval == 0:
                points.append((frame, (x, y), z))
        return points

    def selected_receiver(self, team, puck, passer_slot, direction):
        """ROM $9E:E9BF search: teammates within +-1 octant of the pressed direction;
        lowest quarter-resolution squared distance wins, an off-by-one octant adds 0x1000,
        and the lowest slot wins ties. Eligibility mirrors the ROM: carrier_state > 0
        (goalies and special states are 0) and motion_flags & 0x1000 clear."""
        candidates = []
        for index, player in enumerate(team.players):
            slot = team.skater_scnum_base() + index
            if slot == passer_slot or player.unavailable & 4:
                continue
            carrier = getattr(player, 'carrier_state', None)
            if carrier is not None:
                if carrier <= 0:
                    continue
            elif player.role is not None and player.role <= 0:
                continue
            flags = getattr(player, 'motion_flags', None)
            if flags is not None and flags & 0x1000:
                continue
            dx, dy = int(player.x) - int(puck.x), int(player.y) - int(puck.y)
            sector = self.direction(dx, dy)
            if sector == 8:
                continue
            difference = (sector - direction) & 7
            if difference not in (0, 1, 7):
                continue
            score = (abs(dx) >> 2) ** 2 + (abs(dy) >> 2) ** 2 + (0x1000 if difference else 0)
            candidates.append((score, slot))
        return min(candidates)[1] if candidates else None

    def pass_release_frames(self, receiver, delay):
        """ROM-derived: passes launch at the first physics tick after the input is
        processed (1-2 rendered frames; there is no Genesis-style windup)."""
        return min(delay, 2)

    def pass_contact(self, puck, passer, receiver, delay):
        """ROM $9E:EB2E: aim at receiver + anim hotspot + velocity lead, speed
        clamp((2*cheapdist)/3, 64, 80), launch velocity (speed*d/dist) << 8 per axis,
        then drag/gravity flight until the puck comes within 12px of the receiver.

        The receiver anim hotspot is modeled from the ROM-observed default receiving
        anim (13 rink px to the stick side, 10 forward); the +-(16 - rating) launch
        speed jitter is RNG noise and not modeled."""
        from nhl94_ai.agents.motion import velocity
        from nhl94_ai.agents.passing import PASS_HORIZON
        from nhl94_ai.env.target_control import project_target
        if receiver.motion_x is None or receiver.motion_y is None:
            return None
        rvx, rvy = velocity(receiver)
        pvx, pvy = velocity(passer)
        release = self.pass_release_frames(receiver, delay)
        # The ROM aims with integer positions: the puck int at launch and the
        # receiver's int position plus its anim hotspot and velocity lead.
        start = math.floor(puck.x + pvx * release), math.floor(puck.y + pvy * release)
        base_x = math.floor(receiver.x + rvx * release)
        base_y = math.floor(receiver.y + rvy * release)
        facing_x, facing_y = getattr(receiver, 'ori_x', None), getattr(receiver, 'ori_y', None)
        if facing_x is None or facing_y is None:
            facing_x, facing_y = 0.0, 1.0
        aim_x = base_x + round(13 * facing_y + 10 * facing_x)
        aim_y = base_y + round(-13 * facing_x + 10 * facing_y)
        lead_frames = 24
        for _ in range(2):
            dx = aim_x + math.floor(rvx * lead_frames) - start[0]
            dy = aim_y + math.floor(rvy * lead_frames) - start[1]
            dist = max(1, self._cheap_distance(dx, dy))
            raw_speed = 2 * dist // 3
            speed = min(80, max(64, raw_speed))
            wanted = dist // 4 if raw_speed < 64 else (dist // 5 if raw_speed >= 80 else 24)
            if wanted == lead_frames:
                break
            lead_frames = wanted
        qx = math.trunc(speed * dx / dist)
        qy = math.trunc(speed * dy / dist)
        raw_x, raw_y = qx << 8, qy << 8  # launch velocities are 8.8 fixed point
        x, y = float(start[0]), float(start[1])
        z, vz = 0.0, float(dist << 3)
        for frame in range(1, PASS_HORIZON + 1):
            raw_x = math.trunc(raw_x * 255 / 256)
            raw_y = math.trunc(raw_y * 255 / 256)
            x, y = x + raw_x / 4096, y + raw_y / 4096
            vz -= 96
            z += vz / 16
            if z < 0:
                z, vz = 0.0, -vz / 2
            point = x, y
            if math.dist(point, project_target(point)) > 1e-6:
                return None
            receiver_x, receiver_y = base_x + rvx * frame, base_y + rvy * frame
            if math.hypot(x - receiver_x, y - receiver_y) <= 12:
                return start, point, frame, math.hypot(raw_x, raw_y) / 4096
        return None

    @staticmethod
    def _cheap_distance(dx, dy):
        """ROM $9CA554: max + min/4, or max + 3*min/8 when min > max/2."""
        hi, lo = max(abs(dx), abs(dy)), min(abs(dx), abs(dy))
        if hi // 2 < lo:
            return hi + ((lo + lo // 2) >> 2)
        return hi + (lo >> 2)

    def reception_drift_shift(self, receiver):
        """Estimated: native reception drift is unrecovered; use the unprimed coast shift."""
        return 6

    @staticmethod
    def _friction_value(value, frames):
        coefficient = FRICTION_BY_STEP[min(frames, len(FRICTION_BY_STEP) - 1)]
        magnitude = abs(value) * coefficient // 256
        return math.copysign(magnitude, value)

    def _native_fixed_position(self, player, axis):
        integer = int(getattr(player, axis))
        fraction = getattr(player, 'position_frac_' + axis, None)
        if fraction is not None:
            return integer * 65536 + fraction
        precise = getattr(player, 'precise_' + axis, None)
        return round((integer if precise is None else precise) * 65536)

    def _set_native_fixed_position(self, player, axis, value):
        integer = value >> 16
        fraction = value & 0xFFFF
        setattr(player, axis, integer)
        setattr(player, 'position_frac_' + axis, fraction)
        setattr(player, 'precise_' + axis, integer + fraction / 65536)

    @staticmethod
    def _signed_word(value):
        return (value + 0x8000) % 0x10000 - 0x8000

    def _native_turn(self, player, future, wanted, frames, raw_vx, raw_vy):
        """Apply the ROM's speed-dependent fractional turn table."""
        facing = player.facing if player.facing is not None else player.orientation
        phase = getattr(player, 'native_facing_phase', None)
        if phase is None:
            delta = (wanted - facing + 4) % 8 - 4
            future.facing = future.orientation = (facing + (1 if delta > 0 else -1 if delta < 0 else 0)) % 8
            return
        rate = TURN_RATE_BY_DELTA[(wanted - facing) & 7]
        if not rate:
            future.facing = future.orientation = facing
            future.native_facing_phase = phase
            future.facing_phase = phase / 65536
            return
        high_x = raw_vx >> 8
        high_y = raw_vy >> 8
        speed2 = high_x * high_x + high_y * high_y
        turn_factor = max(0x180, 0x300 - speed2)
        product = rate * frames * turn_factor
        if getattr(player, 'turning_state', None):
            product = -product
        low_sum = (phase & 0xFFFF) + (product & 0xFFFF)
        carry = low_sum >> 16
        phase = self._signed_word(low_sum)
        facing = (facing + (product >> 16) + carry) & 7
        future.facing = future.orientation = facing
        future.native_facing_phase = phase
        future.facing_phase = phase / 65536

    def grounded_step(self, player, pad, *, goalie=False, stop=False):
        from nhl94_ai.agents.motion import velocity
        future = copy(player)
        debt = getattr(player, 'physics_frame_debt', None)
        if debt is not None:
            if debt < 0:
                raise ValueError('SNES physics frame debt cannot be negative')
            if debt == 0:
                future.physics_frame_debt = 1
                future.physics_step_frames = 0
                return future
            frames = debt + 1
            raw_x = getattr(player, 'raw_vel_x', None)
            raw_y = getattr(player, 'raw_vel_y', None)
            raw_vx = round(velocity(player)[0] / self.velocity_scale) if raw_x is None else raw_x
            raw_vy = round(velocity(player)[1] / self.velocity_scale) if raw_y is None else raw_y
            raw_vx = int(self._friction_value(raw_vx, frames))
            raw_vy = int(self._friction_value(raw_vy, frames))
            for axis, raw in (('x', raw_vx), ('y', raw_vy)):
                position = self._native_fixed_position(player, axis)
                self._set_native_fixed_position(future, axis, position + raw * 16 * frames)
            acceleration, limit, _ = self.skating(player)
            acceleration_raw = acceleration * frames / self.velocity_scale
            if stop:
                raw_vx = int(math.copysign(max(0, abs(raw_vx) - acceleration_raw), raw_vx))
                raw_vy = int(math.copysign(max(0, abs(raw_vy) - acceleration_raw), raw_vy))
            elif any(pad):
                length = math.hypot(*pad)
                wanted = round(math.atan2(pad[0], pad[1]) * 4 / math.pi) % 8
                self._native_turn(player, future, wanted, frames, raw_vx, raw_vy)
                if future.facing == wanted:
                    proposed = (raw_vx + pad[0] / length * acceleration_raw,
                                raw_vy + pad[1] / length * acceleration_raw)
                    speed_rating = getattr(player, 'speed_rating', None)
                    if speed_rating is None:
                        limit_raw = limit / self.velocity_scale
                        accepts = math.hypot(*proposed) <= max(limit_raw, math.hypot(raw_vx, raw_vy))
                    else:
                        rating = min(max(speed_rating, 0), len(SPEED_LIMIT_SQUARED_HIGH) - 1)
                        high_x = int(proposed[0]) >> 8
                        high_y = int(proposed[1]) >> 8
                        accepts = high_x * high_x + high_y * high_y <= SPEED_LIMIT_SQUARED_HIGH[rating]
                    if accepts:
                        raw_vx, raw_vy = proposed
            future.raw_vel_x = int(raw_vx)
            future.raw_vel_y = int(raw_vy)
            future.motion_x = raw_vx * self.velocity_scale
            future.motion_y = raw_vy * self.velocity_scale
            future.physics_frame_debt = 0
            future.physics_step_frames = frames
            return future

        # Hand-built SNES fixtures without native cadence retain the pre-decode estimate.
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
