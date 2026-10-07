"""Scoped ordinary-ROM player/CPU-goalie fixtures, not a training curriculum."""
import math
from dataclasses import dataclass

from nhl94_ai.agents.motion import VELOCITY_SCALE
from nhl94_ai.game.ram import GOALIE_INPUT_ATTRIBUTES, SKATER_INPUT_ATTRIBUTES, select_cpu_side
from nhl94_ai.tasks.defense_setup import (
    _place_object, _rebuild_object_order, _reset_player, init_full_team_defense,
)


OBJECT_BASE = 0xFFB04A
OBJECT_STRIDE = 0x80
NATIVE_TOUCH_PLAYER = 0xFFBF7A
GOALIE_LEVELS = {'low': 1, 'high': 6}
TRAFFIC_KINDS = (
    'none', 'friendly-initial', 'opponent-initial', 'opponent-release', 'pursuit-initial',
    'friendly-between', 'opponent-between', 'pursuit-between',
)


@dataclass(frozen=True)
class CrossingTraffic:
    slot: int
    kind: str
    position: tuple[float, float]
    opponent: bool
    pursuing: bool


def goalie_attributes(level):
    """Neutral hot/cold skill encoding; awareness is a delay, glove left a nibble."""
    if isinstance(level, bool) or not isinstance(level, int) or not 0 <= level <= 6:
        raise ValueError('Goalie skill level must be an integer from zero to six.')
    effective = 5 * level
    return {
        'weight': 64, 'agility': effective, 'speed': effective,
        'defensive_delay': (30 - effective // 2) // 2,
        'puck_control': effective, 'glove_right': effective,
        'stick_left': effective, 'stick_right': effective,
        'glove_left': effective & 15, 'handedness': 0,
    }


def prepare_isolated_crossing(env, *, side=1, direction=1, width=35, depth=216,
                              speed=6500, handedness=0):
    """Keep one controlled carrier and the opposing ordinary CPU goalie on ice."""
    if side not in (1, 2) or direction not in (-1, 1) or handedness not in (0, 1):
        raise ValueError('Use a physical side, signed crossing direction and binary handedness.')
    if not 20 <= width <= 110 or not 160 <= depth <= 230 or not 0 <= speed <= 12000:
        raise ValueError('Crossing start must have bounded width, attacking depth and lateral speed.')
    init_full_team_defense(env)
    env.data.update_ram()
    select_cpu_side(env.data, env.data.lookup_all(), side)
    memory = env.data.memory
    carrier = memory.extract(0xFFC320, '>i2')
    goalie = 11 if side == 1 else 5
    sign = 1 if side == 1 else -1
    inactive = tuple(slot for slot in range(12) if slot not in (carrier, goalie))
    for index, slot in enumerate(inactive):
        point = (-100 + (index % 5) * 50, -sign * (120 + index // 5 * 60))
        _place_object(memory, slot, point)
        base = OBJECT_BASE + slot * OBJECT_STRIDE
        memory.assign(base + 0x34, '>i2', -1)
        memory.assign(base + 0x62, '|u1', memory.extract(base + 0x62, '|u1') & ~8)
    role = memory.extract(OBJECT_BASE + carrier * OBJECT_STRIDE + 0x34, '>i2')
    assignment = 1 if role in (1, 2) else 6 if role == 4 else 4
    position = -direction * width, sign * depth
    _reset_player(memory, carrier, position, 2 if direction > 0 else 6,
                  assignment, True, (0x10, 0x11))
    base = OBJECT_BASE + carrier * OBJECT_STRIDE
    for offset, value in ((0x67, 64), (0x68, 20), (0x69, 20), (0x6C, 30),
                          (0x6D, 20), (0x71, 30), (0x76, handedness)):
        memory.assign(base + offset, '|u1', value)
    memory.assign(base + 0x28, '>i2', direction * speed)
    _reset_player(memory, goalie, (-direction * 12, sign * 250),
                  4 if side == 1 else 0, 0x0E, False)
    _place_object(memory, 14, (position[0] + direction * 10, position[1]))
    memory.assign(0xFFB7AA, '>i2', carrier)
    memory.assign(0xFFBEDA, '>i2', carrier)
    for slot in (carrier, goalie):
        roster = memory.extract(OBJECT_BASE + slot * OBJECT_STRIDE + 0x66, '|u1')
        memory.assign(0xFFC700 + (slot // 6) * 0x364 + roster * 2, '>u2', 4096)
    _rebuild_object_order(memory)
    return carrier, goalie, inactive


def set_goalie_level(env, slot, level):
    """Apply live per-instance ratings after snapshot restoration, before input."""
    if slot not in (5, 11):
        raise ValueError('Full-team goalie slot must be five or eleven.')
    memory = env.data.memory
    attributes = goalie_attributes(level)
    base = OBJECT_BASE + slot * OBJECT_STRIDE
    for name, value in attributes.items():
        memory.assign(base + GOALIE_INPUT_ATTRIBUTES[name][0], '|u1', value)
    for offset in (0x28, 0x2A):
        memory.assign(base + offset, '>i2', 0)
    memory.assign(base + 0x40, '|i1', 0)
    memory.assign(base + 0x43, '|u1', 8)
    return attributes


def set_crossing_velocity(env, carrier, *, direction, lateral, goalward, width=35, depth=216):
    """Change only incoming momentum at fixed current/previous carrier/puck poses."""
    if carrier not in (*range(5), *range(6, 11)) or direction not in (-1, 1):
        raise ValueError('Use a full-team carrier slot and signed crossing direction.')
    if not (math.isfinite(lateral) and math.isfinite(goalward)
            and 0 <= lateral <= 3 and abs(goalward) <= 1):
        raise ValueError('Velocity grid requires lateral 0..3 and goalward -1..1 rink units/frame.')
    if not 20 <= width <= 110 or not 80 <= depth <= 230:
        raise ValueError('Velocity grid geometry is outside the isolated finishing region.')
    sign = 1 if carrier < 6 else -1
    position = -direction * width, sign * depth
    puck = position[0] + direction * 10, position[1]
    memory = env.data.memory
    raw = round(direction * lateral / VELOCITY_SCALE), round(sign * goalward / VELOCITY_SCALE)
    for slot, point in ((carrier, position), (14, puck)):
        _place_object(memory, slot, point)
        for offset, component in zip((0x28, 0x2A), raw):
            memory.assign(OBJECT_BASE + slot * OBJECT_STRIDE + offset, '>i2', component)
    _rebuild_object_order(memory)
    return tuple(component * VELOCITY_SCALE for component in raw)


def crossing_traffic_position(carrier_position, keeper_position, puck_position, kind, direction):
    """Joint placement with enough body clearance, without moving the shooter."""
    if kind not in TRAFFIC_KINDS or kind == 'none' or direction not in (-1, 1):
        raise ValueError('An intervening placement requires a supported traffic kind and signed direction.')
    reference = carrier_position if kind.endswith('-between') else puck_position
    point = ((reference[0] + keeper_position[0]) / 2
             if kind != 'opponent-release' else direction * 20,
             (carrier_position[1] + keeper_position[1]) / 2)
    if any(math.dist(point, actor) < 20 for actor in (carrier_position, keeper_position)):
        raise ValueError('Traffic placement needs at least twenty units of initial actor clearance.')
    return point


def place_crossing_traffic(env, carrier, goalie, *, kind, direction):
    """Activate one collidable idle skater or a native CPU nearest-puck pursuer."""
    if kind not in TRAFFIC_KINDS or direction not in (-1, 1):
        raise ValueError('Use a supported traffic kind and signed crossing direction.')
    if carrier not in (*range(5), *range(6, 11)) or goalie != (11 if carrier < 6 else 5):
        raise ValueError('Traffic requires a full-team carrier and opposing goalie.')
    if kind == 'none':
        return None
    memory = env.data.memory
    def position(slot):
        base = OBJECT_BASE + slot * OBJECT_STRIDE
        return tuple(memory.extract(base + offset, '>i4') / 65536 for offset in (0, 0x14))

    carrier_position, keeper_position, puck_position = position(carrier), position(goalie), position(14)
    opponent = not kind.startswith('friendly')
    first = (6 if carrier < 6 else 0) if opponent else (0 if carrier < 6 else 6)
    slot = next(candidate for candidate in range(first, first + 5) if candidate != carrier)
    point = crossing_traffic_position(carrier_position, keeper_position, puck_position, kind, direction)
    base = OBJECT_BASE + slot * OBJECT_STRIDE
    memory.assign(base + 0x34, '>i2', 3)
    pursuing = kind.startswith('pursuit-')
    _reset_player(memory, slot, point, 4 if carrier < 6 else 0,
                  3 if pursuing else 0, False, (0x11,) if pursuing else ())
    for name, value in (
        ('weight', 64), ('agility', 20), ('speed', 20), ('offensive_delay', 10),
        ('defensive_delay', 7), ('shot_power', 20), ('shot_accuracy', 20), ('passing', 20),
        ('shot_bias', 10), ('stick', 20), ('endurance', 20), ('aggression', 7),
        ('checking', 20), ('handedness', 0),
    ):
        memory.assign(base + SKATER_INPUT_ATTRIBUTES[name][0], '|u1', value)
    roster = memory.extract(base + 0x66, '|u1')
    memory.assign(0xFFC700 + (slot // 6) * 0x364 + roster * 2, '>u2', 4096)
    _rebuild_object_order(memory)
    return CrossingTraffic(slot, kind, point, opponent, pursuing)


def keep_traffic_idle(env, traffic):
    """Suppress autonomous reassignment, not collision-driven physical motion."""
    if traffic is None or traffic.pursuing:
        return
    base = OBJECT_BASE + traffic.slot * OBJECT_STRIDE
    index = env.data.memory.extract(base + 0x36, '>u2')
    if index not in range(8):
        raise AssertionError('Idle traffic has an invalid native assignment index.')
    env.data.memory.assign(base + 0x38 + index, '|u1', 0)


def crossing_touch_player(env):
    """Read ltplayer, distinct from shot/controller-history lastplayer."""
    return env.data.memory.extract(NATIVE_TOUCH_PLAYER, '>i2')
