"""Genesis match RAM, controller selection, roster decoding, and team views.

Shared by gameplay environments and evaluation. These helpers retain the
historical bench/cpu telemetry names used by save states and report tooling.
"""
from copy import copy

from nhl94_ai.game.ram import ROM_RNG_ADDRESS, select_cpu_side


# NHL94 Genesis RAM symbols: random, SCnum[], teamselect[], clock, shots.
RAM = {
    'bench_rng': (ROM_RNG_ADDRESS, '>u4'),
    'bench_control1': (0xFFC320, '>i2'), 'bench_control2': (0xFFC322, '>i2'),
    'bench_team1': (0xFFC328, '>u2'), 'bench_team2': (0xFFC32A, '>u2'),
    'bench_clock': (0xFFC468, '>u2'),
    'bench_shots1': (0xFFC6CE, '>u2'), 'bench_shots2': (0xFFCA32, '>u2'),
    'bench_one_timers1': (0xFFCA2A, '>u2'), 'bench_one_timers2': (0xFFCD8E, '>u2'),
    'bench_one_timer_goals1': (0xFFCA2C, '>u2'), 'bench_one_timer_goals2': (0xFFCD90, '>u2'),
}


def _position(info, side, slot):
    index = slot - (side - 1) * 6
    if not 0 <= index <= 5:
        return 9999, 9999
    prefix = f'g{side}_' if index == 5 else f'p{side}_' + (f'{index + 1}_' if index else '')
    return info[prefix + 'x'], info[prefix + 'y']


def update_state(state, info, env=None):
    """Resolve both teams' possession/control from slots, not misaddressed stars."""
    if env is not None:
        from nhl94_ai.game.ram import pass_geometry_info
        info = pass_geometry_info(env, info)
    info = dict(info)
    owner = info['puck_owner']
    for side, team in enumerate((state.team1, state.team2), 1):
        full = 'fullstar_' if side == 1 else 'p2_fullstar_'
        info[full + 'x'], info[full + 'y'] = _position(info, side, owner)
        empty = f'p{side}_emptystar_'
        slot = info[f'bench_control{side}']
        info[empty + 'x'], info[empty + 'y'] = _position(info, side, slot)
        if team.owns_scnum(slot):
            team.control = 0 if slot == team.goalie_scnum() else slot - team.skater_scnum_base() + 1
        info[f'p{side}_shots'] = info[f'bench_shots{side}']
    state.BeginFrame(info, [0] * 6)
    # Avoid ambiguous stars when two players happen to occupy the same point.
    for side, team in enumerate((state.team1, state.team2), 1):
        slot = info[f'bench_control{side}']
        if team.owns_scnum(slot):
            team.control = 0 if slot == team.goalie_scnum() else slot - team.skater_scnum_base() + 1
        team.goalie_haspuck = owner == team.goalie_scnum()
        team.player_haspuck = team.owns_scnum(owner) and not team.goalie_haspuck


def away_view(state):
    """Swap team references while retaining physical coordinates and goal ends."""
    view = copy(state)
    view.team1, view.team2 = state.team2, state.team1
    return view


CPU_RAM = {
    **RAM,
    'home_team': (0xFFC330, '>u2'), 'away_team': (0xFFC332, '>u2'),
    'home_roster': (0xFFC6EC, '>u4'), 'away_roster': (0xFFCA50, '>u4'),
    **{f'cpu_{slot}_{field}': (0xFFB04A + slot * 0x80 + offset, '|u1')
       for slot in range(12)
       for field, offset in (('flags', 0x62), ('roster', 0x66), ('accuracy', 0x6D))},
}


def select_side(data, info, side):
    """Transfer the single joystick; the other team must run the game's CPU.

    Mirrors restorep1's pfjoycon/pfna flag changes. Leaving a second human
    controller assigned with zero input would not test the built-in opponent.
    """
    if (info['bench_team1'], info['bench_team2'], info['period']) != (1, 0, 0):
        raise ValueError('CPU benchmark requires a first-period, home-only joystick save.')
    select_cpu_side(data, info, side, controller_prefix='bench', player_prefix='cpu', slots=6)


def lineup(info, rom):
    """Resolve names from live roster indices, not save-state filenames."""
    players = []
    for slot in range(12):
        address = info['home_roster' if slot < 6 else 'away_roster']
        address += int.from_bytes(rom[address:address + 2], 'big')
        for _ in range(info[f'cpu_{slot}_roster']):
            address += int.from_bytes(rom[address:address + 2], 'big') + 8
        size = int.from_bytes(rom[address:address + 2], 'big')
        if not 2 < size < 64 or address + size >= len(rom):
            raise ValueError('Invalid ROM roster pointer/name.')
        name = rom[address + 2:address + size].rstrip(b'\0').decode('ascii')
        players.append({'slot': slot, 'name': name,
                        'shot_accuracy': info[f'cpu_{slot}_accuracy'] if slot % 6 != 5 else None})
    return players


def cpu_view(state, info, side):
    if side == 1:
        # Same inference used in normal play, including any stale star readings.
        state.BeginFrame(info, [0] * 6)
        return state
    corrected = dict(info, bench_control1=-1, bench_control2=info['bench_control1'])
    update_state(state, corrected)
    return away_view(state)
