"""Jointly sampled full-team defensive starts, using the existing live-play save.

RAM layout and assignment IDs: docs/nhl94 Deep dive.md, sections 2.2, 2.3, 6.1.
Writes affect only this emulator instance, not the installed Retro integration.
"""
import math

from nhl94_ai.game.constants import GameConsts
from nhl94_ai.tasks.setup import _randint_inclusive


_OBJECTS = 0xFFB04A
_STRIDE = 0x80
_MIN_SPACING = 20


def _sample_formation(rng):
    def sample(low, high):
        return _randint_inclusive(rng, low, high)

    for _ in range(128):
        side = 1 if sample(0, 1) else -1
        carrier = (side * sample(0, 95), sample(-180, -125))
        dx, dy = -carrier[0], GameConsts.P1_NET_Y - carrier[1]
        distance = math.hypot(dx, dy)
        forward, lateral = sample(28, 48), sample(-22, 22)
        defender = (
            round(carrier[0] + (dx * forward - dy * lateral) / distance),
            round(carrier[1] + (dy * forward + dx * lateral) / distance),
        )
        friendly = [
            defender,
            (-side * sample(12, 32), sample(-222, -202)),
            (side * sample(45, 95), sample(-120, -92)),
            (sample(-22, 22), sample(-150, -105)),
            (-side * sample(60, 100), sample(-182, -140)),
        ]
        opponent = [
            carrier,
            (-side * sample(45, 85), sample(-205, -165)),
            (sample(-28, 28), sample(-158, -118)),
            (side * sample(50, 100), sample(-110, -92)),
            (-side * sample(50, 100), sample(-110, -92)),
        ]
        goalie_x = round(carrier[0] * 14 / (carrier[1] - GameConsts.P1_NET_Y))
        goalies = [(max(-14, min(14, goalie_x + sample(-3, 3))), -250),
                   (sample(-5, 5), 250)]
        positions = friendly + opponent + goalies
        if all(math.dist(first, second) >= _MIN_SPACING
               for index, first in enumerate(positions) for second in positions[index + 1:]):
            return friendly, opponent, goalies
    raise RuntimeError("Could not sample a non-overlapping defensive formation after 128 attempts.")


def _place_object(memory, slot, position):
    """Reset current/previous 16.16 coordinates, motion and contact state."""
    base = _OBJECTS + slot * _STRIDE
    for value, current, previous, velocity in (
        (position[0], 0x00, 0x1C, 0x28),
        (position[1], 0x14, 0x20, 0x2A),
        (0, 0x18, 0x24, 0x2C),
    ):
        memory.assign(base + current, '>i4', int(value * 65536))
        memory.assign(base + previous, '>i4', int(value * 65536))
        memory.assign(base + velocity, '>i2', 0)
    memory.assign(base + 0x2E, '>i2', -1)
    memory.assign(base + 0x30, '>u4', 0)


def _reset_assignment(memory, slot, assignment, temporary=()):
    base = _OBJECTS + slot * _STRIDE
    assignments = [assignment] + [0] * 7
    if temporary:
        assignments[-len(temporary):] = temporary
    memory.assign(base + 0x36, '>u2', 8 - len(temporary) if temporary else 0)
    for index, value in enumerate(assignments):
        memory.assign(base + 0x38 + index, '|u1', value)
    for offset in (0x40, 0x44):
        memory.assign(base + offset, '>u4', 0)
    memory.assign(base + 0x48, '>u2', 0)
    # 0x4A/0x4C are collision radii, not assignment scratch.
    memory.assign(base + 0x4E, '>u4', 0)


def _rebuild_object_order(memory):
    """Rebuild SprSort's collision/sprite tables after all object moves."""
    axis = 0 if memory.extract(0xFFC2EC, '|u1') & 0x80 else 0x14
    positions = [memory.extract(_OBJECTS + slot * _STRIDE + axis, '>i2') for slot in range(16)]
    for slot, y in enumerate(positions):
        memory.assign(0xFFB84A + slot * 2, '>i2', y)
    for index, slot in enumerate(sorted(range(16), key=positions.__getitem__)):
        memory.assign(0xFFB88A + index, '|u1', slot * 2)
        memory.assign(0xFFB86A + slot * 2, '>u2', index)


def _reset_player(memory, slot, position, facing, assignment, controlled, temporary=()):
    base = _OBJECTS + slot * _STRIDE
    _place_object(memory, slot, position)
    _reset_assignment(memory, slot, assignment, temporary)
    memory.assign(base + 0x54, '>u4', facing << 16)
    memory.assign(base + 0x58, '>u2', 2 if slot in (5, 11) else 0)
    memory.assign(base + 0x5A, '>u2', 0)
    memory.assign(base + 0x5C, '>u2', 8)
    memory.assign(base + 0x5E, '>u2', 0)
    # Preserve team/attack direction and unrelated flags; restart unlocked AI.
    flags = memory.extract(base + 0x62, '|u1') & ~0x39
    memory.assign(base + 0x62, '|u1', flags | 0x02 | (0x08 if controlled else 0))
    memory.assign(base + 0x63, '|u1', memory.extract(base + 0x63, '|u1') & ~0x06)
    memory.assign(base + 0x64, '|u1', memory.extract(base + 0x64, '|u1') & ~0x3B)
    memory.assign(base + 0x65, '|u1', 0)


def _facing(position, target):
    return round(math.atan2(target[0] - position[0], target[1] - position[1]) * 4 / math.pi) % 8


def init_full_team_defense(env):
    memory = env.data.memory
    roles = [memory.extract(_OBJECTS + slot * _STRIDE + 0x34, '>i2') for slot in range(12)]
    controlled = memory.extract(0xFFC320, '>i2')
    opponent_controlled = memory.extract(0xFFC322, '>i2')
    if not all((
        memory.extract(0xFFC466, '>u2') == 0,
        memory.extract(0xFFC468, '>u2') > 200,
        memory.extract(0xFFC328, '>u2') == 1,
        controlled in range(5),
        memory.extract(0xFFB7AA, '>i2') in range(6, 11),
        all(roles[slot] in (1, 2, 3, 4, 5) for slot in (*range(5), *range(6, 11))),
        roles[5] == 0,
        roles[11] == 0,
    )):
        raise ValueError(
            "DefenseZone requires a live first-period 5-on-5 save with home control, "
            "opponent skater possession and more than 200 seconds remaining "
            "(for example PenguinsVsSenators.DefenseZone)."
        )

    rng = env.np_random
    friendly, opponent, goalies = _sample_formation(rng)
    primary = controlled if _randint_inclusive(rng, 0, 3) else (
        controlled + _randint_inclusive(rng, 1, 4)
    ) % 5
    carrier = _randint_inclusive(rng, 6, 10)
    # Covering defenseman first; opposing defensemen occupy the two points.
    friendly_slots = [primary] + sorted(
        (slot for slot in range(5) if slot != primary),
        key=lambda slot: (roles[slot] not in (1, 2), slot),
    )
    opponent_slots = [carrier] + sorted(
        (slot for slot in range(6, 11) if slot != carrier),
        key=lambda slot: (roles[slot] in (1, 2), slot),
    )
    for slots, positions, attacking in ((friendly_slots, friendly, False), (opponent_slots, opponent, True)):
        for slot, position in zip(slots, positions):
            role = roles[slot]
            assignment = (1 if attacking else 2) if role in (1, 2) else (
                (6 if attacking else 5) if role == 4 else (4 if attacking else 3)
            )
            # The carrier must resume assnearest after a pass/loss so the ROM
            # can transfer that team's puck-handling assignment to the next owner.
            temporary = (0x10, 0x11) if slot == carrier else ((0x11,) if slot == primary else ())
            target = (0, GameConsts.P1_NET_Y) if slot == carrier else opponent[0]
            _reset_player(memory, slot, position, _facing(position, target), assignment,
                          slot in (controlled, opponent_controlled), temporary)
    for slot, position in zip((5, 11), goalies):
        _reset_player(memory, slot, position, _facing(position, opponent[0]), 0x0E,
                      slot == opponent_controlled)

    # Initialize asspuckc's normal scratch state without an immediate shot/pass
    # on the reset frame. Subsequent decisions use the ROM's usual awareness delay.
    base = _OBJECTS + carrier * _STRIDE
    memory.assign(base + 0x62, '|u1', memory.extract(base + 0x62, '|u1') & ~0x02)
    memory.assign(base + 0x40, '|u1', max(4, memory.extract(base + 0x6A, '|u1')))
    memory.assign(base + 0x42, '>u2', 8)
    memory.assign(base + 0x44, '>u2', _randint_inclusive(rng, 0, 3))

    _place_object(memory, 14, opponent[0])
    _reset_assignment(memory, 14, 0x18)
    memory.assign(0xFFB78C, '>u2', 120)  # pucknorm's loose-puck timeout.
    for address in (0xFFBEE6, 0xFFBEEA):
        memory.assign(address, '>i2', 0)
        memory.assign(address + 2, '>i2', -1)  # Stationary puck cannot cross a goal line.
    memory.assign(0xFFB7AA, '>i2', carrier)
    memory.assign(0xFFBEDA, '>i2', carrier)  # Last puck player.
    memory.assign(0xFFBEE0, '>i2', -1)  # No pending pass receiver.
    memory.assign(0xFFBED8, '>i2', -1)  # No pending shot.
    # ROM bit operations on these words address their high byte.
    memory.assign(0xFFC2EC, '>u2', memory.extract(0xFFC2EC, '>u2') & ~0x0C00)
    memory.assign(0xFFC2EE, '>u2', memory.extract(0xFFC2EE, '>u2') & ~0x1000)
    _rebuild_object_order(memory)
    memory.assign(0xFFD066, '>u4', _randint_inclusive(rng, 1, 2**32 - 1))
