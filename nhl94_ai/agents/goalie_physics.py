"""Small, integer NHL94 goalie/puck model for independently tested planning.

Sources: Classic's motion/skating models and the local NHL94 disassembly:
playeracc, goalieacc (0x1090A), stopna2, checkpuckcoll,
goaliesave and puckgoalie. No opponent implementation is used here.

This models free flight and upright manual movement near the crease. Contact
is an event boundary, not a simulated complete save/rebound outcome. A C save
has a separately testable momentum-quartering projection; animation-specific
reach, dives, skater collisions and post/board rebounds are not certified.
"""
from dataclasses import dataclass, replace
import math

from nhl94_ai.agents.motion import VELOCITY_SCALE, skating, velocity


FIXED = 65536
PADS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1))


def signed_word(value):
    return (value + 32768) % 65536 - 32768


def friction(value, shift):
    return signed_word(value - ((value >> shift) or bool(value)))


def brake(value, amount):
    return max(0, value - amount) if value >= 0 else min(0, value + amount)


@dataclass(frozen=True, slots=True)
class PuckFlight:
    x: int
    y: int
    z: int
    vx: int
    vy: int
    vz: int
    shift: int = 9

    @classmethod
    def observe(cls, puck):
        values = (puck.motion_x, puck.motion_y, puck.motion_z, puck.height)
        if any(value is None for value in values):
            return None
        return cls(round((puck.precise_x if puck.precise_x is not None else puck.x) * FIXED),
                   round((puck.precise_y if puck.precise_y is not None else puck.y) * FIXED),
                   round(puck.height * FIXED),
                   *(round(value / VELOCITY_SCALE) for value in values[:3]),
                   9 if puck.friction == 511 / 512 else 6)

    def step(self):
        vx, vy = self.vx, self.vy
        if self.z >> 16 == 0:
            vx, vy = friction(vx, self.shift), friction(vy, self.shift)
        z, vz = self.z, self.vz
        if z >= 0 and (z >> 16 or vz):
            vz = signed_word(vz - 102)
            z += vz * 17
            if z < 0:
                z, vz = 0, signed_word(-vz) >> 1
        return PuckFlight(self.x + vx * 17, self.y + vy * 17, z, vx, vy, vz, self.shift)

    @property
    def point(self):
        return self.x / FIXED, self.y / FIXED, self.z / FIXED


@dataclass(frozen=True, slots=True)
class KeeperMotion:
    x: int
    y: int
    vx: int
    vy: int
    acceleration: int
    speed_squared: int
    shift: int = 6
    saving: bool = False

    @classmethod
    def observe(cls, goalie):
        if any(getattr(goalie, name) is None for name in (
                'speed', 'weight', 'agility', 'energy', 'motion_x', 'motion_y')):
            return None
        _, limit, _ = skating(goalie)
        acceleration = (64 - goalie.weight // 4 + 2 * goalie.agility + 16 + goalie.movement_bonus) // 2
        return cls(round((goalie.precise_x if goalie.precise_x is not None else goalie.x) * FIXED),
                   round((goalie.precise_y if goalie.precise_y is not None else goalie.y) * FIXED),
                   *(round(value / VELOCITY_SCALE) for value in velocity(goalie)),
                   acceleration, round((limit / VELOCITY_SCALE) ** 2),
                   9 if (goalie.selection_flags or 0) & 1 else 6)

    def step(self, pad=(0, 0), *, save=False):
        vx, vy = friction(self.vx, self.shift), friction(self.vy, self.shift)
        x, y = self.x + vx * 17, self.y + vy * 17
        saving = self.saving or save
        if save and not self.saving:
            vx, vy = vx >> 2, vy >> 2
        if not saving:
            px, py = x >> 16, y >> 16
            # doinput's manual goalie limits are velocity stops, not clamps.
            blocked = False
            if px > 36 and vx > 0 or px < -36 and vx < 0:
                vx, blocked = 0, True
            if -231 < py <= 231 and vy and (py < 0) != (vy < 0):
                vy, blocked = 0, True
            if not blocked:
                if pad == (0, 0):
                    vx, vy = brake(vx, 2000), brake(vy, 2000)
                else:
                    scale = 141 if all(pad) else 200
                    nx = signed_word(vx + ((pad[0] * scale * self.acceleration) >> 5))
                    ny = signed_word(vy + ((pad[1] * scale * self.acceleration) >> 5))
                    if nx * nx + ny * ny <= self.speed_squared:
                        vx, vy = nx, ny
        return replace(self, x=x, y=y, vx=vx, vy=vy, saving=saving)

    @property
    def point(self):
        return self.x / FIXED, self.y / FIXED


def contact_margin(keeper, puck, *, one_timer=False):
    """ROM upright manual-goalie body test; deliberately excludes stick reach."""
    if puck.z >> 16 > 15:
        return -math.inf
    radius = 12 if one_timer else 15
    dx = (keeper.x >> 16) - (puck.x >> 16)
    dy = (keeper.y >> 16) - (puck.y >> 16)
    return radius - math.hypot(dx, dy)


def free_trajectory(puck, frames, defended_sign):
    """Stop at the goal plane or unsupported board/net geometry."""
    result = []
    for _ in range(frames):
        puck = puck.step()
        result.append(puck)
        if abs(puck.x >> 16) >= 128 or (puck.y >> 16) * defended_sign >= 261:
            break
    return result
