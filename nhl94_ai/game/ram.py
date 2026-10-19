"""RAM decoding, tactical observations, and scoped controller assignment."""
import warnings
from functools import lru_cache
from pathlib import Path


ROM_RNG_ADDRESS = 0xFFD066
SKATER_INPUT_ATTRIBUTES = {
    'weight': (0x67, 120), 'agility': (0x68, 30), 'speed': (0x69, 30),
    'offensive_delay': (0x6A, 15), 'defensive_delay': (0x6B, 15),
    'shot_power': (0x6C, 30), 'shot_accuracy': (0x6D, 30), 'passing': (0x6E, 30),
    'shot_bias': (0x70, 30), 'stick': (0x71, 30), 'endurance': (0x72, 30),
    'aggression': (0x73, 15), 'checking': (0x75, 30), 'handedness': (0x76, 1),
}
GOALIE_INPUT_ATTRIBUTES = {
    'weight': (0x67, 120), 'agility': (0x68, 30), 'speed': (0x69, 30),
    'defensive_delay': (0x6B, 15), 'puck_control': (0x6C, 30),
    'glove_right': (0x6E, 30), 'stick_left': (0x70, 30),
    'stick_right': (0x72, 30), 'glove_left': (0x73, 15), 'handedness': (0x76, 1),
}


def compact_input_info(env, info, skaters):
    """Read only selected skaters and goalies, without changing integration aliases."""
    memory = env.data.memory
    corrected = dict(info)
    slots = {skaters, 6 + skaters}
    for controller in range(1, 5):
        team = memory.extract(0xFFC328 + (controller - 1) * 2, '>u2')
        slot = memory.extract(0xFFC320 + (controller - 1) * 2, '>i2')
        corrected[f'defense_team{controller}'] = team
        corrected[f'defense_control{controller}'] = slot
        if team in (1, 2) and (team - 1) * 6 <= slot < (team - 1) * 6 + skaters:
            slots.add(slot)
    records = {}
    for slot in slots:
        base = 0xFFB04A + slot * 0x80
        if memory.extract(base + 0x66, '|u1') >= 26:
            records[slot] = None
            continue
        attributes = GOALIE_INPUT_ATTRIBUTES if slot in (skaters, 6 + skaters) else SKATER_INPUT_ATTRIBUTES
        record = {name: memory.extract(base + offset, '|u1') for name, (offset, _) in attributes.items()}
        record.update(
            vx=memory.extract(base + 0x28, '>i2'), vy=memory.extract(base + 0x2A, '>i2'),
            facing=memory.extract(base + 0x54, '>u2'),
            has_puck=int(memory.extract(0xFFB7AA, '>i2') == slot),
        )
        records[slot] = record
    corrected['model_input_players'] = records
    return corrected


def validate_rom_seed(seed):
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError('ROM seed must be a uint32 integer (0..4294967295).')


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
        for controller in range(1, 5):
            if info.get(f'defense_team{controller}') == side:
                slot = info[f'defense_control{controller}']
        corrected[f'p{side}_control_slot'] = slot
    return corrected


def register_teammate_state(env):
    """Keep the native clock separate from the legacy composite time alias."""
    env.data.set_variable('teammate_clock', {'address': 0xFFC468, 'type': '>u2'})


def configure_teammates(data, side=1, *, team_size=2, human=False):
    """Apply the native Two-Teammates assignment to a full-team exhibition save.

    Only reset writes live here. Skating, switching and possession subsequently
    belong to the ROM. No integration files or saved states are modified.
    """
    if side not in (1, 2):
        raise ValueError('Teammate side must be home (1) or away (2)')
    if isinstance(team_size, bool) or team_size not in (2, 3):
        raise ValueError('Teammate team_size must be 2 or 3')
    if (team_size == 3 or human) and side != 1:
        raise ValueError('Multitap teammate modes require home bots')
    memory = data.memory
    read = memory.extract
    roles = [read(0xFFB04A + slot * 0x80 + 0x34, '>i2') for slot in range(12)]
    if (read(0xFFD048, '>u2') != 0 or roles[5] != 0 or roles[11] != 0
            or any(role <= 0 for slot, role in enumerate(roles) if slot % 6 != 5)):
        raise ValueError('team_vs_game requires a full-team exhibition save with five active skaters per team')
    first = read(0xFFC320, '>i2')
    if not 0 <= first < 5:
        raise ValueError('team_vs_game requires an initially selected home skater')
    second = read(0xFFC322, '>i2')
    if not 0 <= second < 5 or second == first:
        second = next(slot for slot in range(5) if slot != first)
    first += (side - 1) * 6
    second += (side - 1) * 6
    writes = {
        0xFFD04A: 3,  # OptNOP: native Two - Teammates.
        0xFFD052: 1, 0xFFD05A: 1, 0xFFD05C: 1,  # 1 = automatic; 0 = manual.
        0xFFC328: side, 0xFFC32A: side, 0xFFC32C: 0, 0xFFC32E: 0,
        0xFFC320: first, 0xFFC322: second, 0xFFC324: -1, 0xFFC326: -1,
    }
    selected = [first, second]
    if team_size == 3:
        selected.append(next(slot for slot in range(5) if slot not in selected))
    if human:
        selected.append(6)
    if team_size == 3 or human:
        writes[0xFFD046] = 1
        for port in range(4):
            writes[0xFFC320 + port * 2] = selected[port] if port < len(selected) else -1
            writes[0xFFC328 + port * 2] = side if port < team_size else 3 - side if human and port == team_size else 0
    for address, value in writes.items():
        memory.assign(address, '>i2', value)
    for slot in range(12):
        address = 0xFFB04A + slot * 0x80 + 0x62
        flags = read(address, '|u1')
        memory.assign(address, '|u1', flags | 8 if slot in selected else (flags & ~8) | 2)
    data.update_ram()
    return (selected[0], selected[1]) if team_size == 2 else (selected[0], selected[1], selected[2])


def teammate_controller_info(info, controller=1, side=1, *, team_size=2, human=False):
    """A controller-specific view with physical team identities preserved."""
    if not 1 <= controller <= team_size:
        raise ValueError('Teammate controller is outside the bot team')
    if side not in (1, 2):
        raise ValueError('Teammate side must be home (1) or away (2)')
    # gmode bit 4 is the ROM's intermission highlight game. It temporarily
    # replaces teams, counters, clock and assignments, restoring them on exit.
    if info.get('goalie_clock_flags', 0) & 0x10:
        return {**info, 'p1_control_slot': -1, 'p2_control_slot': -1,
                'time': info['teammate_clock'], 'teammate_highlight': True}
    if team_size == 3 or human:
        from nhl94_ai.game.multitap import controller_snapshot
        if 'teammate_transient' not in info:
            info = controller_snapshot(info, team_size, human, side)
    if tuple(info.get(f'defense_team{i}') for i in range(1, team_size + 1)) != (side,) * team_size:
        raise ValueError('team_vs_game requires both controllers on the AI team and its opponent under CPU control')
    slots = tuple(info[f'defense_control{i}'] for i in range(1, team_size + 1))
    base = (side - 1) * 6
    invalid_slot = any(slot >= 0 and not base <= slot < base + 6 for slot in slots)
    if invalid_slot or len([slot for slot in slots if slot >= 0]) != len({slot for slot in slots if slot >= 0}):
        raise ValueError(f'Invalid teammate controller selections: {slots}')
    if any(info.get(f'goalie_mode_{i}') != 1 for i in (1, 2)):
        raise ValueError('team_vs_game requires automatic goalies')
    return {**info, f'p{side}_control_slot': slots[controller - 1], f'p{3-side}_control_slot': info.get(f'defense_control{team_size + 1}', -1) if human else -1,
            'time': info['teammate_clock'], 'teammate_highlight': False,
            'teammate_controller_count': team_size + human}


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
    """Read-only pass identity and native one-timer release counters."""
    for name, address in (('pass_target', 0xFFBEE0), ('last_puck_player', 0xFFBEDA)):
        env.data.set_variable(name, {'address': address, 'type': '>i2'})
    for side, address in enumerate((0xFFCA2A, 0xFFCD8E), start=1):
        env.data.set_variable(f'p{side}_one_timer_attempts', {'address': address, 'type': '>u2'})


def register_shootout_scores(env):
    """Native shootout counters update one frame before the ordinary scoreboard."""
    for side, address in enumerate((0xFFD574, 0xFFD576), start=1):
        env.data.set_variable(f'p{side}_score', {'address': address, 'type': '>u2'})


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


@lru_cache(maxsize=3)
def _animation_rom(game, rom_path=None):
    from stable_retro.data import get_romfile_path
    return Path(rom_path or get_romfile_path(game)).read_bytes()


@lru_cache(maxsize=128)
def _animation_frames(game, animation, rom_path=None):
    if animation == 0:
        return ((),) * 8
    rom = _animation_rom(game, rom_path)
    base = 0x5B1C + animation
    if base + 18 > len(rom):
        raise ValueError(f'Invalid animation header in {game}.')
    directions = []
    for direction in range(8):
        stream = base + int.from_bytes(rom[base + direction * 2:base + direction * 2 + 2],
                                       'big', signed=True)
        frames = []
        for index in range(0, 256, 4):
            if not 0 <= stream + index <= len(rom) - 4:
                raise ValueError(f'Invalid animation stream in {game}.')
            sprite = int.from_bytes(rom[stream + index:stream + index + 2], 'big')
            duration = int.from_bytes(rom[stream + index + 2:stream + index + 4], 'big', signed=True)
            frames.append((sprite, duration))
            if duration < 0:
                break
        else:
            raise ValueError(f'Unterminated animation in {game}.')
        directions.append(tuple(frames))
    return tuple(directions)


@lru_cache(maxsize=3)
def _shot_animations(game, rom_path=None):
    hot = _stick_hotspots(game, rom_path)
    directions = []
    for direction in range(8):
        ticks, offsets = [], []
        for animation in (0x7FC, 0x92E, 0x50C):
            durations = []
            for sprite, duration in _animation_frames(game, animation, rom_path)[direction]:
                if sprite * 2 + 2 > len(hot):
                    raise ValueError(f'Invalid shot frame in {game}.')
                durations.append(abs(duration))
                offsets.append(tuple(int.from_bytes(hot[sprite * 2 + axis:sprite * 2 + axis + 1],
                                                    'big', signed=True) for axis in (0, 1)))
            if animation != 0x50C:
                if len(durations) < 8:
                    raise ValueError(f'Truncated shot animation in {game}.')
                ticks.append(durations[:8])
        directions.append((tuple(max(values) for values in zip(*ticks)), tuple(offsets)))
    return tuple(directions)


@lru_cache(maxsize=3)
def _one_timer_hotspots(game, rom_path=None):
    """Contact poses at SPAnum 24 for either native one-timer animation."""
    hot = _stick_hotspots(game, rom_path)
    result = []
    for direction in range(8):
        offsets = []
        for animation in (0x7FC, 0x92E):
            frames = _animation_frames(game, animation, rom_path)[direction]
            if len(frames) < 7:
                raise ValueError(f'Truncated one-timer animation in {game}.')
            sprite = frames[6][0]
            offsets.append(tuple(int.from_bytes(hot[sprite * 2 + axis:sprite * 2 + axis + 1],
                                                'big', signed=True) for axis in (0, 1)))
        result.append(tuple(offsets))
    return tuple(result)


def pass_geometry_info(env, info):
    """Decode read-only sprite stick offsets without altering neural observations."""
    game = getattr(env.unwrapped, 'gamename', None)
    if game is None:
        return info
    from nhl94_ai.game.specs import get_game
    skaters = get_game(game).skaters_per_team
    table = _stick_hotspots(game, getattr(env.unwrapped, 'pass_geometry_rom', None))
    shots = _shot_animations(game, getattr(env.unwrapped, 'pass_geometry_rom', None))
    teammates = info.get('defense_team1') in (1, 2) and info.get('defense_team1') == info.get('defense_team2')
    one_timers = _one_timer_hotspots(game, getattr(env.unwrapped, 'pass_geometry_rom', None)) if teammates else None
    projection_x = max(abs(sx) for _, offsets in shots for sx, _ in offsets)
    projection_y = max(abs(sy) for _, offsets in shots for _, sy in offsets)
    projection_durations = tuple(max(values) for values in zip(*(durations for durations, _ in shots)))
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
            corrected[f'offense_{slot}_sprite_flipped_x'] = bool(flags & 8)
            direction = memory.extract(base + 0x54, '>u2')
            direction = (direction - (2 if info.get('sflags', 0) & 0x8000 else 0)) % 8
            direction = -direction % 8 if flags & 8 else direction
            if one_timers is not None:
                contact_offsets = []
                for sx, sy in one_timers[direction]:
                    sx, sy = (-sx if flags & 8 else sx), (sy if flags & 16 else -sy)
                    contact_offsets.append((sy, -sx) if info.get('sflags', 0) & 0x8000 else (sx, sy))
                corrected[f'offense_{slot}_one_timer_offsets'] = tuple(contact_offsets)
            durations, offsets = shots[direction]
            previous = _animation_frames(game, memory.extract(base + 0x58, '>u2'),
                                         getattr(env.unwrapped, 'pass_geometry_rom', None))[direction]
            previous_offsets = []
            for sprite, _ in previous:
                if sprite * 2 + 2 > len(table):
                    raise ValueError(f'Invalid pre-shot sprite in {game}.')
                previous_offsets.append(tuple(int.from_bytes(table[sprite * 2 + axis:sprite * 2 + axis + 1],
                                                            'big', signed=True) for axis in (0, 1)))
            ys = [(-sx if not flags & 8 else sx) if info.get('sflags', 0) & 0x8000
                  else (sy if flags & 16 else -sy) for sx, sy in (*offsets, *previous_offsets)]
            xs = [(sy if flags & 16 else -sy) if info.get('sflags', 0) & 0x8000
                  else (-sx if flags & 8 else sx) for sx, sy in (*offsets, *previous_offsets)]
            corrected[f'offense_{slot}_shot_durations'] = durations
            corrected[f'offense_{slot}_shot_offsets_y'] = min(ys), max(ys)
            corrected[f'offense_{slot}_shot_offsets_x'] = min(xs), max(xs)
            px, py = (projection_y, projection_x) if info.get('sflags', 0) & 0x8000 else (
                projection_x, projection_y)
            px, py = max(px, *(abs(value) for value in xs)), max(py, *(abs(value) for value in ys))
            corrected[f'offense_{slot}_shot_projection_offsets_x'] = -px, px
            corrected[f'offense_{slot}_shot_projection_offsets_y'] = -py, py
            corrected[f'offense_{slot}_shot_projection_durations'] = projection_durations
    return corrected


def register_goalie_motion(env, skaters=5):
    """Full signed velocities; legacy gN_vel_x/y observations remain unchanged."""
    for side, slot in ((1, skaters), (2, 6 + skaters)):
        for axis, offset in (('x', 0x28), ('y', 0x2A)):
            env.data.set_variable(f'g{side}_live_vel_{axis}', {
                'address': 0xFFB04A + slot * 0x80 + offset, 'type': '>i2',
            })


def register_goalie_control(env, skaters=5):
    """Manual availability and goalie state are optional, read-only feedback."""
    fields = {
        'goalie_mode_1': (0xFFD05A, '>u2'), 'goalie_mode_2': (0xFFD05C, '>u2'),
        'goalie_hold_b_1': (0xFFBF06, '|u1'), 'goalie_hold_b_2': (0xFFBF07, '|u1'),
        'goalie_input_block': (0xFFD412, '>u2'), 'goalie_clock_flags': (0xFFC2EA, '|u1'),
    }
    for side, slot in ((1, skaters), (2, 6 + skaters)):
        base = 0xFFB04A + slot * 0x80
        for name, offset, kind in (
            ('role', 0x34, '>i2'), ('facing', 0x54, '>u2'), ('flags', 0x62, '|u1'),
            ('precise_x', 0, '>i4'), ('precise_y', 0x14, '>i4'),
            ('decision_timer', 0x40, '|i1'), ('steering', 0x43, '|i1'), ('decision_delay', 0x6B, '|u1'),
            ('unavailable', 0x63, '|u1'), ('state_flags', 0x64, '|u1'),
            ('speed', 0x69, '|u1'), ('agility', 0x68, '|u1'), ('weight', 0x67, '|u1'),
            ('puck_control', 0x6C, '|u1'),
            ('roster', 0x66, '|u1'), ('stick', 0x71, '|u1'), ('passing', 0x6E, '|u1'),
            ('anim', 0x58, '>u2'), ('anim_frame', 0x5A, '>u2'), ('anim_timer', 0x5C, '|i1'),
            ('contact_player', 0x2E, '>i2'), ('contact_impact', 0x32, '>u2'),
            ('sprite', 6, '>i2'), ('assignment_index', 0x36, '>u2'), ('cover_timer', 0x48, '>i2'),
        ):
            fields[f'g{side}_control_{name}'] = base + offset, kind
        for index in range(8):
            fields[f'g{side}_control_assignment_{index}'] = base + 0x38 + index, '|u1'
    for name, (address, kind) in fields.items():
        env.data.set_variable(name, {'address': address, 'type': kind})


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
        'defense_team3': (0xFFC32C, '>u2'), 'defense_team4': (0xFFC32E, '>u2'),
        'defense_control3': (0xFFC324, '>i2'), 'defense_control4': (0xFFC326, '>i2'),
        'defense_scroll_x': (0xFFBD1E, '>i2'), 'defense_scroll_y': (0xFFBD20, '>i2'),
        'defense_energy_override': (0xFFC2FC, '|u1'),
        'defense_skill_boost': (0xFFC2FE, '|u1'),
        'offense_rule_flags': (0xFFC2EA, '|u1'),
    }
    for axis, offset in (('x', 0x28), ('y', 0x2A), ('z', 0x2C)):
        fields['defense_puck_v' + axis] = 0xFFB74A + offset, '>i2'
    fields['defense_puck_z'] = 0xFFB74A + 0x18, '>i4'
    fields['defense_puck_precise_x'] = 0xFFB74A, '>i4'
    fields['defense_puck_precise_y'] = 0xFFB74A + 0x14, '>i4'
    fields['defense_puck_flags'] = 0xFFB74A + 0x62, '|u1'
    for side in (1, 2):
        for index in range(skaters):
            slot = (side - 1) * 6 + index
            base = 0xFFB04A + slot * 0x80
            prefix = f'defense_{slot}_'
            for name, offset, kind in (
                ('vx', 0x28, '>i2'), ('vy', 0x2A, '>i2'), ('role', 0x34, '>i2'),
                ('facing', 0x54, '>u2'), ('flags', 0x62, '|u1'),
                ('precise_x', 0, '>i4'), ('precise_y', 0x14, '>i4'), ('facing_phase', 0x54, '>u4'),
                ('unavailable', 0x63, '|u1'), ('roster', 0x66, '|u1'),
                ('weight', 0x67, '|u1'), ('agility', 0x68, '|u1'),
                ('speed', 0x69, '|u1'), ('stick', 0x71, '|u1'),
                ('passing', 0x6E, '|u1'),
                ('shot_power', 0x6C, '|u1'), ('handedness', 0x76, '|u1'),
                ('live_anim', 0x58, '>u2'), ('live_anim_frame', 0x5A, '>u2'),
                ('animation_timer', 0x5C, '|i1'),
                ('contact_player', 0x2E, '>i2'), ('contact_impact', 0x32, '>u2'),
                ('endurance', 0x72, '|u1'), ('checking', 0x75, '|u1'),
                ('assignment_index', 0x36, '>u2'), ('decision_timer', 0x40, '|i1'),
                ('steering_timer', 0x42, '|i1'), ('steering', 0x43, '|u1'),
                ('target_x', 0x44, '>i2'), ('target_y', 0x46, '>i2'),
                ('support_zone', 0x48, '>i2'),
            ):
                fields[prefix + name] = base + offset, kind
            for assignment in range(8):
                fields[f'{prefix}assignment_{assignment}'] = base + 0x38 + assignment, '|u1'
        # getpde indexes the team energy array by roster identity, not ice slot.
        for roster in range(26):
            fields[f'defense_energy_{side}_{roster}'] = (
                0xFFC700 + (side - 1) * 0x364 + roster * 2, '>u2')
    for name, (address, kind) in fields.items():
        env.data.set_variable(name, {'address': address, 'type': kind})
