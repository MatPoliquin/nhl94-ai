"""Grounded carrier/CPU-goalie projection in native fixed-point update order."""
from copy import copy
import math

from nhl94_ai.agents.motion import VELOCITY_SCALE, rom_direction, skating, velocity
from nhl94_ai.env.target_control import steering_direction


_DIRECTIONS = ((0, 200), (141, 141), (200, 0), (141, -141),
               (0, -200), (-141, -141), (-200, 0), (-141, 141))


def grounded_step(player, pad, *, goalie=False, stop=False):
    """Integrate friction first, then turn/brake/accelerate for the next frame."""
    future = copy(player)
    vx, vy = (round(component / VELOCITY_SCALE) for component in velocity(player))
    vx, vy = (component - ((component >> 6) or bool(component)) for component in (vx, vy))
    x = player.precise_x if player.precise_x is not None else float(player.x)
    y = player.precise_y if player.precise_y is not None else float(player.y)
    future.precise_x, future.precise_y = x + vx * VELOCITY_SCALE, y + vy * VELOCITY_SCALE
    future.x, future.y = math.floor(future.precise_x), math.floor(future.precise_y)
    phase = player.facing_phase if player.facing_phase is not None else float(
        player.orientation if player.facing is None else player.facing)
    phase = round(phase * 65536) % (8 * 65536)
    heading = phase // 65536
    ux, uy = pad
    wanted = round(math.atan2(ux, uy) * 4 / math.pi) % 8 if ux or uy else 8
    accelerate = wanted < 8
    braking = stop or goalie and wanted == 8
    if not goalie and wanted < 8:
        difference = (wanted - heading) % 8
        if difference == 4:
            direction = rom_direction(vx, vy) if vx or vy else 8
            braking = direction != 8 and (direction - heading + 2) % 8 < 4
            if not braking:
                phase = (phase + (-65536 if player.sprite_flipped_x else 65536)) % (8 * 65536)
            accelerate = False
        elif difference:
            speed_word = (vx * vx + vy * vy) >> 16
            increment = max(0x180, 0x300 - speed_word) * (16 if difference < 4 else -16)
            phase = (phase + increment) % (8 * 65536)
            heading = phase // 65536
            accelerate = speed_word > 2
    if braking:
        vx = int(math.copysign(max(0, abs(vx) - 150), vx))
        vy = int(math.copysign(max(0, abs(vy) - 150), vy))
    elif accelerate:
        factor = (64 - player.weight // 4 + player.agility + player.movement_bonus
                  + (player.agility + 16 if goalie else 0)) // 2
        dx, dy = _DIRECTIONS[wanted if goalie else heading]
        proposed = vx + ((dx * factor) >> 5), vy + ((dy * factor) >> 5)
        _, limit, _ = skating(player)
        if proposed[0] ** 2 + proposed[1] ** 2 <= round((limit / VELOCITY_SCALE) ** 2):
            vx, vy = proposed
    future.facing_phase, future.facing = phase / 65536, phase // 65536
    future.motion_x, future.motion_y = vx * VELOCITY_SCALE, vy * VELOCITY_SCALE
    return future


def skating_route(player, target, frames):
    """Use the same integer feedback and velocity-aware steering as execution."""
    carrier = copy(player)
    yield 0, carrier
    for elapsed in range(1, frames + 1):
        carrier = grounded_step(carrier, steering_direction(carrier, target)[0])
        yield elapsed, carrier
