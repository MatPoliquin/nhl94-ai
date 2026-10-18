"""Genesis integration and ROM telemetry behind the shared console interface."""
from nhl94_ai.game.platforms.base import PlatformBackend


class GenesisBackend(PlatformBackend):
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
        for controller in (1, 2):
            if info.get(f'defense_team{controller}') == team.controller:
                team.defense_control = max(-1, info[f'defense_control{controller}'])
        flags = info.get('defense_energy_override', 0)
        bonus = 2 if flags & 0x40 or info.get('defense_skill_boost', 0) & 2 else 0
        for index, player in enumerate(team.players):
            prefix = f'defense_{team.skater_scnum_base() + index}_'
            player.burst_full_energy = (bool(flags & 0x10)
                                        if 'defense_energy_override' in info else None)
            player.stick_x = info.get(f'offense_{team.skater_scnum_base() + index}_stick_x')
            player.stick_y = info.get(f'offense_{team.skater_scnum_base() + index}_stick_y')
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
        state.engine.controller_teams = tuple(info[f'defense_team{i}'] for i in (1, 2)) if all(
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
