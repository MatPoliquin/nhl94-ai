"""Short-horizon, conservative motion estimates in rink units and emulator frames."""
import math

from nhl94_ai.env.target_control import project_target, route_waypoint
from nhl94_ai.game.motion import (
    VELOCITY_SCALE, _backend, rom_direction, direction, velocity, coast_projection, bounded_projection,
    skate_step, skating, facing, boost_impulse, burst_velocity, stop_projection, check_approach, boost_safe,
    arrival_time, puck_path,
)


LANE_HORIZON = 32
SHOT_SPEED_RANGE = (3.0, 6.0)
BLOCK_RADIUS = 8


def blocks_shot_lane(player, start, goal, delay=0):
    """Conservative low-shot cover, not a promise of a ROM collision.

    Require an existing body-sized block to survive a short release window and
    a range of shot speeds. Never assume an uncontrolled teammate will brake.
    """
    if player.motion_x is None or player.motion_y is None:
        return False
    dx, dy = goal[0] - start[0], goal[1] - start[1]
    length = math.hypot(dx, dy)
    if length < 1:
        return False
    ux, uy = dx / length, dy / length
    px, py = player.x - start[0], player.y - start[1]
    along, across = px * ux + py * uy, px * uy - py * ux
    if not 0 < along < length or abs(across) > BLOCK_RADIUS:
        return False
    vx, vy = velocity(player)
    forward, sideways = vx * ux + vy * uy, vx * uy - vy * ux
    for launch in (delay, delay + 4):
        for speed in SHOT_SPEED_RANGE:
            if speed <= forward:
                return False
            travel = (along + forward * launch) / (speed - forward)
            time = launch + travel
            if not 0 < speed * travel < length or time > LANE_HORIZON:
                return False
            if abs(across + sideways * time) > BLOCK_RADIUS:
                return False
    return True
