"""RAM decoding, tactical observations, and scoped controller assignment."""
import warnings
from functools import lru_cache
from pathlib import Path


def select_cpu_side(data, info, side, *, controller_prefix='defense', player_prefix='defense', slots=5):
    """Transfer a home-only joystick without leaving an idle human opponent."""
    if side not in (1, 2):
        raise ValueError('CPU side must be home (1) or away (2).')
    team = f'{controller_prefix}_team1'
    control = f'{controller_prefix}_control1'
    if (info[team], info[f'{controller_prefix}_team2']) != (1, 0):
        raise ValueError('CPU playback requires a home-only joystick save.')
    if side == 1:
        return
    previous = info[control]
    if not 0 <= previous < slots:
        raise ValueError('CPU playback requires an initially selected home skater.')
    target = previous + 6
    home_flags = f'{player_prefix}_{previous}_flags'
    away_flags = f'{player_prefix}_{target}_flags'
    released = (info[home_flags] & ~8) | 2
    assigned = info[away_flags] | 8
    data.set_value(team, 2)
    data.set_value(control, target)
    data.set_value(home_flags, released)
    data.set_value(away_flags, assigned)


def controller_team_info(info):
    """Map joystick slots to physical teams rather than stale controller stars."""
    corrected = dict(info)
    for side in (1, 2):
        slot = -1
        for controller in (1, 2):
            if info[f'defense_team{controller}'] == side:
                slot = info[f'defense_control{controller}']
        corrected[f'p{side}_control_slot'] = slot
    return corrected


def restore_away_control(data, info, *, controller_prefix='defense', player_prefix='defense', slots=5):
    """Keep scoped away playback on its team after a cross-team ROM reassignment."""
    if (info[f'{controller_prefix}_team1'], info[f'{controller_prefix}_team2']) != (2, 0):
        raise ValueError('Away playback no longer has a home CPU opponent.')
    control = f'{controller_prefix}_control1'
    slot = info[control]
    if slot < 0 or 6 <= slot <= 11:
        return info
    if not 0 <= slot < slots:
        raise ValueError(f'Invalid away controller slot: {slot}')
    target = slot + 6
    home_flags, away_flags = f'{player_prefix}_{slot}_flags', f'{player_prefix}_{target}_flags'
    if not info[home_flags] & 8:
        raise ValueError(f'Away controller selected home slot {slot} without joystick assignment.')
    released, assigned = (info[home_flags] & ~8) | 2, info[away_flags] | 8
    data.set_value(control, target)
    data.set_value(home_flags, released)
    data.set_value(away_flags, assigned)
    data.update_ram()
    warnings.warn(f'ROM reassigned the away joystick to home slot {slot}; restored away slot {target}.',
                  RuntimeWarning, stacklevel=2)
    return {**info, **data.lookup_all(), 'away_control_restored_from': slot}


def decode_shot_count(value: int) -> int:
    """Legacy pN_shots reads include a neighboring word above the shot counter."""
    return value & 0xFFFF


def register_pass_state(env):
    """ROM symbols passplayer and lastplayer, verified against nhl94.xml/ROM."""
    for name, address in (('pass_target', 0xFFBEE0), ('last_puck_player', 0xFFBEDA)):
        env.data.set_variable(name, {'address': address, 'type': '>i2'})


@lru_cache(maxsize=3)
def _stick_hotspots(game, rom_path=None):
    from stable_retro.data import get_romfile_path
    rom = Path(rom_path or get_romfile_path(game)).read_bytes()
    signature = bytes.fromhex('48e700c0206f000c424042414a6800066f00')
    entry = rom.find(signature)
    if entry < 0 or rom.find(signature, entry + 1) >= 0 or rom[entry + 20:entry + 22] != bytes.fromhex('227c'):
        raise ValueError(f'Unsupported GetHot routine in {game}; cannot decode pass geometry.')
    table = int.from_bytes(rom[entry + 22:entry + 26], 'big')
    if not 0 < table < len(rom) - 2:
        raise ValueError(f'Invalid stick-hotspot table in {game}.')
    return rom[table:]


def pass_geometry_info(env, info):
    """Decode read-only sprite stick offsets without altering neural observations."""
    game = getattr(env.unwrapped, 'gamename', None)
    if game is None:
        return info
    from nhl94_ai.game.specs import get_game
    skaters = get_game(game).skaters_per_team
    table = _stick_hotspots(game, getattr(env.unwrapped, 'pass_geometry_rom', None))
    corrected = dict(info)
    memory = env.data.memory
    for side in (1, 2):
        for index in range(skaters):
            slot = (side - 1) * 6 + index
            base = 0xFFB04A + slot * 0x80
            frame = memory.extract(base + 6, '>i2')
            flags = memory.extract(base + 4, '|u1')
            x = y = 0
            if frame > 0:
                if frame * 2 + 1 >= len(table):
                    raise ValueError(f'Invalid stick sprite frame: {frame}')
                x = int.from_bytes(table[frame * 2:frame * 2 + 1], 'big', signed=True)
                y = int.from_bytes(table[frame * 2 + 1:frame * 2 + 2], 'big', signed=True)
                x = -x if flags & 8 else x
                y = y if flags & 16 else -y
                if info.get('sflags', 0) & 0x8000:
                    x, y = y, -x
            corrected[f'offense_{slot}_stick_x'] = x
            corrected[f'offense_{slot}_stick_y'] = y
    return corrected


def register_goalie_motion(env, skaters=5):
    """Full signed velocities; legacy gN_vel_x/y observations remain unchanged."""
    for side, slot in ((1, skaters), (2, 6 + skaters)):
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
        'offense_rule_flags': (0xFFC2EA, '|u1'),
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
                ('passing', 0x6E, '|u1'),
                ('endurance', 0x72, '|u1'), ('checking', 0x75, '|u1'),
            ):
                fields[prefix + name] = base + offset, kind
        # getpde indexes the team energy array by roster identity, not ice slot.
        for roster in range(26):
            fields[f'defense_energy_{side}_{roster}'] = (
                0xFFC700 + (side - 1) * 0x364 + roster * 2, '>u2')
    for name, (address, kind) in fields.items():
        env.data.set_variable(name, {'address': address, 'type': kind})
