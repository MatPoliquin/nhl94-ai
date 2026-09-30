"""RAM decoding and additional read-only observations owned by nhl94-ai."""


def decode_shot_count(value: int) -> int:
    """Legacy pN_shots reads include a neighboring word above the shot counter."""
    return value & 0xFFFF


def register_pass_state(env):
    """ROM symbols passplayer and lastplayer, verified against nhl94.xml/ROM."""
    for name, address in (('pass_target', 0xFFBEE0), ('last_puck_player', 0xFFBEDA)):
        env.data.set_variable(name, {'address': address, 'type': '>i2'})


def register_goalie_motion(env):
    """Full signed velocities; legacy gN_vel_x/y observations remain unchanged."""
    for side, slot in ((1, 5), (2, 11)):
        for axis, offset in (('x', 0x28), ('y', 0x2A)):
            env.data.set_variable(f'g{side}_live_vel_{axis}', {
                'address': 0xFFB04A + slot * 0x80 + offset, 'type': '>i2',
            })


def register_skater_ratings(env, skaters):
    """Live effective accuracy (0–30), including the ROM's roster adjustments.

    These named fields supplement GameState; they do not change neural inputs.
    Slot 5/11 is a goalie whose attribute layout differs from a skater's.
    """
    for side in (1, 2):
        for index in range(skaters):
            slot = (side - 1) * 6 + index
            prefix = f'p{side}_' + (f'{index + 1}_' if index else '')
            env.data.set_variable(prefix + 'shot_accuracy', {
                'address': 0xFFB04A + slot * 0x80 + 0x6D, 'type': '|u1',
            })


def register_target_state(env):
    """Full-team target control only; never change legacy observation aliases."""
    fields = {
        'p1_control_slot': (0xFFC320, '>i2'),
        'target_controller_team': (0xFFC328, '>u2'),
        'target_scroll_x': (0xFFBD1E, '>i2'),
        'target_scroll_y': (0xFFBD20, '>i2'),
        'target_puck_vx': (0xFFB74A + 0x28, '>i2'),
        'target_puck_vy': (0xFFB74A + 0x2A, '>i2'),
    }
    for slot in range(5):
        base = 0xFFB04A + slot * 0x80
        prefix = 'p1_' if slot == 0 else f'p1_{slot + 1}_'
        for axis, offset in (('x', 0x28), ('y', 0x2A)):
            fields[prefix + 'live_vel_' + axis] = base + offset, '>i2'
        for name, offset, kind in (
            ('role', 0x34, '>i2'), ('facing', 0x54, '>u2'),
            ('flags', 0x62, '|u1'), ('unavailable', 0x63, '|u1'),
        ):
            fields[f'target_{name}_{slot}'] = base + offset, kind
    for name, (address, kind) in fields.items():
        env.data.set_variable(name, {'address': address, 'type': kind})


def register_defense_state(env, skaters):
    """Read-only tactical feedback; separate from the saved neural input schema."""
    fields = {
        'defense_control1': (0xFFC320, '>i2'), 'defense_control2': (0xFFC322, '>i2'),
        'defense_team1': (0xFFC328, '>u2'), 'defense_team2': (0xFFC32A, '>u2'),
        'defense_scroll_x': (0xFFBD1E, '>i2'), 'defense_scroll_y': (0xFFBD20, '>i2'),
        'defense_energy_override': (0xFFC2FC, '|u1'),
        'defense_skill_boost': (0xFFC2FE, '|u1'),
    }
    for axis, offset in (('x', 0x28), ('y', 0x2A), ('z', 0x2C)):
        fields['defense_puck_v' + axis] = 0xFFB74A + offset, '>i2'
    fields['defense_puck_z'] = 0xFFB74A + 0x18, '>i4'
    fields['defense_puck_flags'] = 0xFFB74A + 0x62, '|u1'
    for side in (1, 2):
        for index in range(skaters):
            slot = (side - 1) * 6 + index
            base = 0xFFB04A + slot * 0x80
            prefix = f'defense_{slot}_'
            for name, offset, kind in (
                ('vx', 0x28, '>i2'), ('vy', 0x2A, '>i2'), ('role', 0x34, '>i2'),
                ('facing', 0x54, '>u2'), ('flags', 0x62, '|u1'),
                ('unavailable', 0x63, '|u1'), ('roster', 0x66, '|u1'),
                ('weight', 0x67, '|u1'), ('agility', 0x68, '|u1'),
                ('speed', 0x69, '|u1'), ('stick', 0x71, '|u1'),
                ('endurance', 0x72, '|u1'), ('checking', 0x75, '|u1'),
            ):
                fields[prefix + name] = base + offset, kind
        # getpde indexes the team energy array by roster identity, not ice slot.
        for roster in range(26):
            fields[f'defense_energy_{side}_{roster}'] = (
                0xFFC700 + (side - 1) * 0x364 + roster * 2, '>u2')
    for name, (address, kind) in fields.items():
        env.data.set_variable(name, {'address': address, 'type': kind})
