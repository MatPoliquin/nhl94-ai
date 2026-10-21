"""Genesis integration and ROM telemetry behind the shared console interface."""
import math

from nhl94_ai.game.platforms.base import PlatformBackend


# Legacy Genesis shot primitives remain in offense/finishing until their extraction.
class GenesisBackend(PlatformBackend):  # pylint: disable=abstract-method
    goalie_takeover_frames = 24
    goalie_takeover_button = 0
    goalie_viewport = (108, 92)
    goalie_save_animations = frozenset((0x146, 0x178, 0x1AA, 0x1EC, 0x21E, 0x250, 0x2A2,
                                       0x148E, 0x14C0, 0x14F2, 0x1544))
    goalie_dive_animations = frozenset((0x2F4,))
    buttons = ('B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'C', 'Y', 'X', 'Z')
    control_mapping = dict(zip(('pass', 'clear', 'mode', 'start', 'up', 'down', 'left',
                               'right', 'shoot', 'aux1', 'aux2', 'aux3'), buttons))

    def decode_player(self, player, info, prefix):
        super().decode_player(player, info, prefix)
        player.is_one_timer = float(bool(player.state_flags & 8))
        player.is_breakaway = float(bool(player.state_flags & 2))
        player.is_falling = float(bool(player.state_flags & 32))
        player.is_pad_stack = float(player.anim in (0x250, 0x2A2))
        player.is_dive = float(player.anim == 0x2F4)

    def register(self, env, skaters):
        from nhl94_ai.game.ram import (
            register_defense_state, register_goalie_control, register_goalie_motion,
            register_pass_state, register_skater_ratings,
        )
        register_skater_ratings(env, skaters)
        register_pass_state(env)
        register_goalie_motion(env, skaters)
        register_goalie_control(env, skaters)
        register_defense_state(env, skaters)

    def frame_info(self, env, info):
        from nhl94_ai.game.ram import pass_geometry_info
        return pass_geometry_info(env, info)

    def load_team_telemetry(self, team, info):
        team.pass_attempts = info.get(f'p{team.controller}_pass_attempts')
        team.one_timer_attempts = info.get(f'p{team.controller}_one_timer_attempts')
        team.defense_control = None
        team.defense_goalie = None
        for player in team.players:
            player.assignment = player.decision_timer = player.steering_timer = player.steering = None
            player.cpu_target = player.support_zone = None
            player.cpu_response_available = False
        for name in ('role', 'facing', 'speed', 'agility', 'weight', 'stick', 'passing', 'puck_control',
                     'selection_flags', 'live_anim', 'live_anim_frame', 'animation_timer',
                     'live_state_flags', 'cover_timer', 'assignment', 'energy',
                     'contact_player', 'contact_impact', 'precise_x', 'precise_y',
                     'decision_timer', 'decision_interval', 'steering', 'cpu_tracking_active'):
            setattr(team.goalie, name, None)
        team.goalie.unavailable = 0
        if 'defense_control1' not in info:
            return
        team.defense_goalie = team.skater_scnum_base() + team.num_players
        for controller in range(1, 5):
            if info.get(f'defense_team{controller}') == team.controller:
                team.defense_control = max(-1, info[f'defense_control{controller}'])
        # Same-team controllers each receive their own authoritative view.
        if team.defense_control is not None and f'p{team.controller}_control_slot' in info:
            team.defense_control = max(-1, info[f'p{team.controller}_control_slot'])
        flags = info.get('defense_energy_override', 0)
        bonus = 2 if flags & 0x40 or info.get('defense_skill_boost', 0) & 2 else 0
        for index, player in enumerate(team.players):
            prefix = f'defense_{team.skater_scnum_base() + index}_'
            player.burst_full_energy = (bool(flags & 0x10)
                                        if 'defense_energy_override' in info else None)
            player.stick_x = info.get(f'offense_{team.skater_scnum_base() + index}_stick_x')
            player.stick_y = info.get(f'offense_{team.skater_scnum_base() + index}_stick_y')
            player.one_timer_offsets = info.get(f'offense_{team.skater_scnum_base() + index}_one_timer_offsets')
            player.shot_durations = info.get(f'offense_{team.skater_scnum_base() + index}_shot_durations')
            player.shot_offsets_y = info.get(f'offense_{team.skater_scnum_base() + index}_shot_offsets_y')
            player.shot_offsets_x = info.get(f'offense_{team.skater_scnum_base() + index}_shot_offsets_x')
            player.shot_projection_durations = info.get(
                f'offense_{team.skater_scnum_base() + index}_shot_projection_durations')
            for axis in ('x', 'y'):
                setattr(player, 'shot_projection_offsets_' + axis,
                        info.get(f'offense_{team.skater_scnum_base() + index}_shot_projection_offsets_' + axis))
                coordinate = info.get(prefix + 'precise_' + axis)
                setattr(player, 'precise_' + axis, None if coordinate is None else coordinate / 65536)
            phase = info.get(prefix + 'facing_phase')
            player.facing_phase = None if phase is None else phase / 65536
            player.sprite_flipped_x = info.get(f'offense_{team.skater_scnum_base() + index}_sprite_flipped_x')
            for name in ('role', 'facing', 'speed', 'agility', 'weight', 'stick', 'checking', 'endurance', 'passing',
                         'shot_power', 'handedness', 'live_anim', 'live_anim_frame', 'animation_timer',
                         'contact_player', 'contact_impact'):
                setattr(player, name, info.get(prefix + name))
            player.unavailable = info.get(prefix + 'unavailable', 0)
            player.selection_flags = info.get(prefix + 'flags')
            assignment_index = info.get(prefix + 'assignment_index')
            player.assignment = (info.get(prefix + f'assignment_{assignment_index}')
                                 if assignment_index in range(8) else None)
            player.decision_timer = info.get(prefix + 'decision_timer')
            player.steering_timer = info.get(prefix + 'steering_timer')
            player.steering = info.get(prefix + 'steering')
            target = info.get(prefix + 'target_x'), info.get(prefix + 'target_y')
            player.cpu_target = None if None in target else target
            player.support_zone = info.get(prefix + 'support_zone')
            player.cpu_response_available = (player.selection_flags is not None
                                             and not player.selection_flags & 0x3A
                                             and not player.unavailable & 0x27)
            player.movement_bonus = bonus
            roster = info.get(prefix + 'roster')
            player.energy = 4096 if flags & 0x10 else info.get(f'defense_energy_{team.controller}_{roster}')
            for axis in ('x', 'y'):
                velocity = info.get(prefix + 'v' + axis)
                if velocity is not None:
                    setattr(player, 'motion_' + axis, velocity * 17 / 65536)
        goalie = team.goalie
        prefix = f'g{team.controller}_control_'
        for axis in ('x', 'y'):
            coordinate = info.get(prefix + 'precise_' + axis)
            setattr(goalie, 'precise_' + axis, None if coordinate is None else coordinate / 65536)
        goalie.decision_timer = info.get(prefix + 'decision_timer')
        goalie.steering = info.get(prefix + 'steering')
        delay = info.get(prefix + 'decision_delay')
        goalie.decision_interval = None if delay is None else max(0, delay - bool(flags & 0x40)) // 4
        for name in ('role', 'facing', 'speed', 'agility', 'weight', 'stick', 'passing', 'puck_control'):
            setattr(goalie, name, info.get(prefix + name))
        goalie.selection_flags = info.get(prefix + 'flags')
        goalie.unavailable = info.get(prefix + 'unavailable', 0)
        goalie.cpu_tracking_active = (None if goalie.selection_flags is None else
                                     not bool(goalie.selection_flags & 0x20 or goalie.unavailable & 2))
        goalie.movement_bonus = bonus
        goalie.energy = 4096 if flags & 0x10 else info.get(
            f'defense_energy_{team.controller}_{info.get(prefix + "roster")}')
        goalie.live_anim, goalie.live_anim_frame = info.get(prefix + 'anim'), info.get(prefix + 'anim_frame')
        goalie.animation_timer = info.get(prefix + 'anim_timer')
        goalie.contact_player = info.get(prefix + 'contact_player')
        goalie.contact_impact = info.get(prefix + 'contact_impact')
        goalie.live_state_flags = info.get(prefix + 'state_flags')
        goalie.cover_timer = info.get(prefix + 'cover_timer')
        index = info.get(prefix + 'assignment_index')
        goalie.assignment = info.get(prefix + f'assignment_{index}') if index in range(8) else None

    def update_engine_state(self, state, info):
        state.engine.goalie_modes = tuple(info[f'goalie_mode_{i}'] for i in (1, 2)) if all(
            f'goalie_mode_{i}' in info for i in (1, 2)) else None
        state.engine.goalie_hold_counts = tuple(info[f'goalie_hold_b_{i}'] for i in (1, 2)) if all(
            f'goalie_hold_b_{i}' in info for i in (1, 2)) else None
        state.engine.controller_teams = tuple(info[f'defense_team{i}'] for i in range(1, info.get('teammate_controller_count', 2) + 1)) if all(
            f'defense_team{i}' in info for i in (1, 2)) else None
        state.engine.input_block = info.get('goalie_input_block')
        state.engine.clock_stopped = bool(info['goalie_clock_flags'] & 1) if 'goalie_clock_flags' in info else None
        state.engine.camera = (-64 - info['defense_scroll_x'], -info['defense_scroll_y']) if all(
            f'defense_scroll_{axis}' in info for axis in ('x', 'y')) else None
        puck_owner = info.get("puck_owner")
        shot_player = info.get("shot_player")

        state.engine.puck_owner = -1 if puck_owner is None else puck_owner
        state.engine.puck_owner_known = puck_owner is not None
        rules = info.get('offense_rule_flags')
        state.engine.offsides_enabled = None if rules is None else bool(rules & 0x20)
        state.engine.pass_target = info.get('pass_target')
        state.engine.last_puck_player = info.get('last_puck_player')
        state.engine.shot_player = -1 if shot_player is None else shot_player
        state.engine.pass_dir = info.get("pass_dir", 0) or 0
        state.engine.pass_speed = info.get("pass_speed", 0) or 0
        state.engine.goalie_chk_body = info.get("goalie_chk_body", 0) or 0
        state.engine.sflags = info.get("sflags", 0) or 0
        state.engine.sflags2 = info.get("sflags2", 0) or 0
        state.is_shooting = bool(state.engine.sflags & 0x0800)
        state.has_released_shot = state.has_released_shot or bool(state.engine.sflags2 & 0x1000)
        state.engine.ba_ps_flags = info.get("ba_ps_flags", 0) or 0
        state.is_shootout_active = bool(state.engine.ba_ps_flags & 0x0400)
        state.engine.word_ffc2f6 = info.get("word_ffc2f6", 0) or 0
        state.engine.word_ffc2f8 = info.get("word_ffc2f8", 0) or 0
        state.engine.word_ffc2fa = info.get("word_ffc2fa", 0) or 0

        state.engine.shot_mode_active = 1.0 if state.engine.sflags & (1 << 3) else 0.0
        state.engine.shot_taken = 1.0 if state.engine.sflags2 & (1 << 4) else 0.0
        state.engine.in_close_top_shelf = 1.0 if state.engine.word_ffc2f6 & (1 << 4) else 0.0
        state.engine.one_timer_collision_mode = 1.0 if state.engine.word_ffc2f8 & (1 << 1) else 0.0
        state.engine.breakaway_context = 1.0 if state.engine.word_ffc2fa & (1 << 4) else 0.0
        state.engine.controlled_is_shooter = float(
            state.engine.shot_player >= 0 and state.engine.shot_player == state.team1.controlled_scnum()
        )
        state.engine.goalie_box_small = 1.0 if state.engine.goalie_chk_body <= 0xC else 0.0

    def direction(self, dx, dy):
        from nhl94_ai.agents.motion import rom_direction
        return rom_direction(dx, dy)

    def aim_pass(self, action, passer, receiver):
        """Choose the nearest of eight pad directions, not two independent axes.

        The diagonal boundary is tan(22.5 degrees). Skating toward a point can
        use both axes for any displacement; aiming that way misdirects shallow
        passes.
        """
        from nhl94_ai.game.constants import GameConsts as Buttons
        dx, dy = receiver.x - passer.x, receiver.y - passer.y
        action[4:8] = 0
        if abs(dx) > max(3, abs(dy) * 0.41421356237):
            action[Buttons.INPUT_RIGHT if dx > 0 else Buttons.INPUT_LEFT] = 1
        if abs(dy) > max(3, abs(dx) * 0.41421356237):
            action[Buttons.INPUT_UP if dy > 0 else Buttons.INPUT_DOWN] = 1

    def skating(self, player, optimistic=False):
        """ROM playeracc/MaxSpeed units, with explicit conservative unknown ratings."""
        speed = player.speed if player.speed is not None else (30 if optimistic else 10)
        agility = player.agility if player.agility is not None else (30 if optimistic else 10)
        weight = player.weight if player.weight is not None else (32 if optimistic else 88)
        energy = player.energy if player.energy is not None else (4096 if optimistic else 2048)
        energy = max(0, min(4096, energy))
        speed = max(0, min(30, speed + player.movement_bonus))
        index = min(15, (speed // 2) * energy // 4096)
        limit = ((20 + index) * 275) ** 2
        if speed % 2 and index < 15:
            limit = (limit + ((21 + index) * 275) ** 2) / 2
        factor = max(1, (64 - weight // 4 + agility + player.movement_bonus) // 2)
        acceleration = (200 * factor // 32) * self.velocity_scale
        return acceleration, math.sqrt(limit) * self.velocity_scale, energy

    def boost_impulse(self, player):
        _, _, energy = self.skating(player)
        return max(0, energy - 204) // 128 * 200 * self.velocity_scale

    def stop_projection(self, player, frames, *, bounded=False):
        """Skater stopna subtracts 150 raw units per axis, without reversing velocity."""
        from nhl94_ai.agents.motion import velocity
        from nhl94_ai.env.target_control import project_target
        x, y = float(player.x), float(player.y)
        vx, vy = velocity(player)
        for _ in range(frames):
            vx = math.copysign(max(0, abs(vx) - 150 * self.velocity_scale), vx) * 63 / 64
            vy = math.copysign(max(0, abs(vy) - 150 * self.velocity_scale), vy) * 63 / 64
            x, y = x + vx, y + vy
            if bounded and math.dist((x, y), project_target((x, y))) > 1e-6:
                return None
        return (x, y), (vx, vy)

    def skate_step(self, motion, heading, turning, pad, acceleration, limit):
        """One frame of the shared conservative eight-way skating projection."""
        (x, y), (vx, vy) = motion
        ux, uy = pad
        wanted = round(math.atan2(ux, uy) * 4 / math.pi) % 8 if ux or uy else heading
        if wanted != heading:
            turning = turning or 6
            turning -= 1
            if turning == 0:
                difference = (wanted - heading + 4) % 8 - 4
                heading = (heading + (1 if difference > 0 else -1)) % 8
            ux = uy = 0
        length = max(1, math.hypot(ux, uy))
        nx = vx * self.skater_friction + acceleration * ux / length
        ny = vy * self.skater_friction + acceleration * uy / length
        if math.hypot(nx, ny) <= max(limit, math.hypot(vx, vy)):
            vx, vy = nx, ny
        else:
            vx, vy = vx * self.skater_friction, vy * self.skater_friction
        return ((x + vx, y + vy), (vx, vy)), heading, turning

    def puck_path(self, puck, horizon=64, *, sample_interval=4):
        """Stop at uncertain wall/net contacts; never invent an exact rebound."""
        from nhl94_ai.agents.motion import velocity
        from nhl94_ai.env.target_control import project_target
        x, y = float(puck.x), float(puck.y)
        vx, vy = velocity(puck)
        z = max(0, puck.height or 0)
        vz = puck.motion_z or 0
        points = []
        for frame in range(1, horizon + 1):
            if z <= 0:
                vx *= puck.friction
                vy *= puck.friction
            x, y = x + vx, y + vy
            if z > 0 or vz:
                vz -= 102 * self.velocity_scale
                z += vz
                if z < 0:
                    z, vz = 0.0, -vz / 2
                    if abs(vz) < 0.05:
                        vz = 0
            if math.dist((x, y), project_target((x, y))) > 1e-6:
                break
            if frame % sample_interval == 0:
                points.append((frame, (x, y), z))
        return points

    def selected_receiver(self, team, puck, passer_slot, direction):
        """passto's sector-and-distance recipient selection."""
        return self._select_nearest_in_sector(team, puck, passer_slot, direction)

    def pass_launch_view(self, state, passer):
        """The second held-B input selects and launches before later slots update.

        Earlier teammates have moved twice; later teammates have moved once. The
        puck is still at the end of the first frame, after pucknorm's quarter-step
        attraction to the carrier's stick. CPU steering and sprite offsets are held
        over this short forecast; this is not a prediction of new CPU decisions.
        """
        from copy import copy
        from nhl94_ai.agents.defense import on_ice
        from nhl94_ai.agents.motion import velocity
        from nhl94_ai.agents.skating import grounded_step
        owner = state.engine.puck_owner
        team, puck = copy(state.team1), copy(state.puck)
        carrier = grounded_step(passer, (0, 0))
        team.players = []
        pads = ((0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 0), (0, 0))
        for index, player in enumerate(state.team1.players):
            slot = state.team1.skater_scnum_base() + index
            future = copy(player)
            if slot != owner and on_ice(player):
                known = (player.steering in range(10) if player.steering is not None else False)
                known &= all(getattr(player, field) is not None for field in ('weight', 'agility', 'speed', 'energy'))
                pad = pads[player.steering] if known else (0, 0)
                for _ in range(1 + (slot < owner)):
                    future = grounded_step(future, pad, stop=known and player.steering == 9)
            team.players.append(future)
        shift = 9 if puck.friction == 511 / 512 else 6
        for axis, offset in (('x', passer.stick_x), ('y', passer.stick_y)):
            raw = round(velocity(puck)[axis == 'y'] / self.velocity_scale)
            raw -= (raw >> shift) or bool(raw)
            precise = getattr(puck, 'precise_' + axis)
            integrated = math.floor((getattr(puck, axis) if precise is None else precise) + raw * self.velocity_scale)
            hotspot = getattr(carrier, axis) + offset
            setattr(puck, axis, integrated + ((hotspot - integrated) >> 2))
        return team, puck

    def pass_release_frames(self, receiver, delay):
        """Use the same release timing for launch prediction and collision deadlines."""
        from nhl94_ai.agents.passing import LIVE_RELEASE_FRAMES
        live = receiver.stick_x is not None and receiver.stick_y is not None
        return min(delay, LIVE_RELEASE_FRAMES) if live else delay

    def live_pass_contact(self, puck, passer, receiver, delay):
        from nhl94_ai.agents.passing import _live_pass_contact
        return _live_pass_contact(puck, passer, receiver, delay)

    def pass_contact(self, puck, passer, receiver, delay):
        from nhl94_ai.agents.passing import pass_speed
        if pass_speed(passer) is None or receiver.motion_x is None or receiver.motion_y is None:
            return None
        if receiver.stick_x is not None and receiver.stick_y is not None:
            return self.live_pass_contact(puck, passer, receiver,
                                          self.pass_release_frames(receiver, delay))
        return self.estimated_pass_contact(puck, passer, receiver, delay)

    def reception_drift_shift(self, receiver):
        """assPassRec coasts with pfdoff, faster while its timer bit is set."""
        return 9 if receiver.selection_flags & 1 else 6
