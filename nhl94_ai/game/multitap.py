"""Opt-in EA 4-Way Play setup and conservative views of native RAM swaps."""
import stable_retro as retro

WAYPLAY_3_BUTTON = 1025
STATE_IO_OFFSET = 16 + 65536 + 8192 + 1 + 4


def require_multitap():
    if (getattr(retro.RetroEmulator, 'max_players', 2) < 4
            or not hasattr(retro.RetroEmulator, 'set_controller_port_device')):
        raise RuntimeError('Three/four controllers require the stable-retro controller-device patch. '
                           'Build the sibling stable-retro checkout with /home/mat/vretro/bin/python; '
                           'see docs/ARCHITECTURE.md (EA 4-Way Play).')


def prepare_multitap_save(env):
    """Initialize the adapter polling register once at reset, preserving ROM bytes.

    GENPLUS-GX 1.7.5 stores IO after work RAM, Z80 RAM, zstate and zbank.
    The original two-pad save has port B configured as a TH-only output. EA's
    adapter selector needs all seven output bits, as initialized by NHL94 boot.
    Never call this on a gameplay snapshot: protocol state belongs to the core.
    """
    require_multitap()
    snapshot = bytearray(env.em.get_state())
    if snapshot[:16] != b'GENPLUS-GX 1.7.5' or len(snapshot) != 1036288:
        raise ValueError('Unsupported Genesis save format for EA 4-Way Play initialization')
    if snapshot[STATE_IO_OFFSET + 4] != 0x40 or snapshot[STATE_IO_OFFSET + 5] not in (0x40, 0x7f):
        raise ValueError('Starting save has incompatible controller IO registers')
    snapshot[STATE_IO_OFFSET + 5] = 0x7f
    if not env.em.set_state(bytes(snapshot)):
        raise RuntimeError('Could not initialize EA 4-Way Play polling state')
    env.data.update_ram()


def controller_snapshot(info, team_size, human=False, side=1):
    """Expose no selection while the ROM temporarily aliases joystick tables.

    updateplayers/setchgplayer substitute C3/C4 into C1/C2 and restore the
    originals from the 68000 stack. A video frame may end inside that routine.
    Never invent the suspended selections or enforce them by writing live RAM.
    """
    expected = [side] * team_size + ([3 - side] if human else [])
    expected += [0] * (4 - len(expected))
    teams = [info[f'defense_team{i}'] for i in range(1, 5)]
    slots = [info[f'defense_control{i}'] for i in range(1, 5)]
    active = [slot for team, slot in zip(expected, slots) if team and slot >= 0]
    skaters = set(range(12)) - {5, 11}
    selected = {slot for slot in range(12) if info.get(f'defense_{slot}_flags', 0) & 8}
    transient = teams != expected or len(active) != len(set(active)) or (selected & skaters) != (set(active) & skaters)
    if any(slot >= 12 for slot in slots):
        raise ValueError(f'Invalid native controller selections: {slots}')
    if any(team not in (0, 1, 2) for team in teams):
        raise ValueError(f'Invalid native controller teams: {teams}')
    if not transient:
        for team, slot in zip(expected, slots):
            if team and slot >= 0 and not (team - 1) * 6 <= slot < team * 6:
                raise ValueError(f'Cross-team controller selection: {slots}')
    corrected = dict(info, teammate_transient=transient, native_controller_teams=teams, native_controller_slots=slots)
    if transient:
        for i, team in enumerate(expected, 1):
            corrected[f'defense_team{i}'] = team
            corrected[f'defense_control{i}'] = -1
    return corrected
