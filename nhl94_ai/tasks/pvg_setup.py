"""Small seeded offsets around a compatible CPU-goalie shootout save."""
from nhl94_ai.tasks.defense_setup import _place_object, _rebuild_object_order
from nhl94_ai.tasks.setup import _randint_inclusive


PLAYER_POSITION_JITTER = (12, 8)
GOALIE_POSITION_JITTER = (6, 3)
PLAYER_VELOCITY_JITTER = 256


def randomize_pvg_start(env):
    memory, rng = env.data.memory, env.np_random
    player_offset = tuple(_randint_inclusive(rng, -radius, radius)
                          for radius in PLAYER_POSITION_JITTER)
    goalie_offset = tuple(_randint_inclusive(rng, -radius, radius)
                          for radius in GOALIE_POSITION_JITTER)
    velocity_noise = tuple(_randint_inclusive(rng, -PLAYER_VELOCITY_JITTER, PLAYER_VELOCITY_JITTER)
                           for _ in range(2))
    for slot, displacement, noise in (
        (0, player_offset, velocity_noise),
        (11, goalie_offset, (0, 0)),
        (14, player_offset, velocity_noise),
    ):
        base = 0xFFB04A + slot * 0x80
        position = tuple(memory.extract(base + offset, '>i4') / 65536 + delta
                         for offset, delta in zip((0, 0x14), displacement))
        velocity = tuple(memory.extract(base + offset, '>i2') + delta
                         for offset, delta in zip((0x28, 0x2A), noise))
        _place_object(memory, slot, position)
        for offset, speed in zip((0x28, 0x2A), velocity):
            memory.assign(base + offset, '>i2', speed)
    _rebuild_object_order(memory)
