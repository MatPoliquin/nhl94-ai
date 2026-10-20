"""Short-horizon, conservative motion estimates in rink units and emulator frames."""
import math

from nhl94_ai.env.target_control import project_target, route_waypoint


VELOCITY_SCALE = 17 / 65536
LANE_HORIZON = 32
SHOT_SPEED_RANGE = (3.0, 6.0)
BLOCK_RADIUS = 8


def _backend(obj):
    from nhl94_ai.game.platforms import get_backend
    return get_backend(getattr(obj, 'platform', 'genesis'))


def rom_direction(dx, dy):
    """Genesis vtoa uses 2:1 sector boundaries, not the pad's nearest-angle boundary."""
    if dx == dy == 0:
        return 8
    if abs(dx) > 2 * abs(dy):
        return 2 if dx > 0 else 6
    if abs(dy) > 2 * abs(dx):
        return 0 if dy > 0 else 4
    return (1 if dx > 0 else 7) if dy > 0 else (3 if dx > 0 else 5)


def direction(player, dx, dy):
    """Console eight-way quantization for a rink displacement."""
    return _backend(player).direction(dx, dy)


def velocity(player):
    if player.motion_x is None or player.motion_y is None:
        if getattr(player, 'platform', 'genesis') != 'genesis':
            from nhl94_ai.game.platforms import get_backend
            scale = 256 * get_backend(player.platform).velocity_scale
            return player.vx * scale, player.vy * scale
        return player.vx * 256 * VELOCITY_SCALE, player.vy * 256 * VELOCITY_SCALE
    return player.motion_x, player.motion_y


def coast_projection(player, frames):
    """ShotMode uses aim, not skating acceleration; retain existing inertia."""
    return _backend(player).coast_projection(player, frames)


def bounded_projection(player, frames, *, boards=True):
    vx, vy = velocity(player)
    point = player.x + vx * frames, player.y + vy * frames
    bounded = project_target(point) if boards else point
    uncertainty = player.projection_uncertainty + math.dist(point, bounded)
    return bounded, uncertainty


def skate_step(player, motion, heading, turning, pad, acceleration, limit):
    """One frame of the console's conservative eight-way skating projection."""
    return _backend(player).skate_step(motion, heading, turning, pad, acceleration, limit)


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


def skating(player, optimistic=False):
    """Console skating acceleration/limit, with explicit conservative unknown ratings."""
    return _backend(player).skating(player, optimistic)


def facing(player):
    direction = player.orientation if player.facing is None else player.facing
    angle = (direction % 8) * math.pi / 4
    return math.sin(angle), math.cos(angle)


def boost_impulse(player):
    return _backend(player).boost_impulse(player)


def burst_velocity(player):
    fx, fy = facing(player)
    vx, vy = velocity(player)
    impulse = boost_impulse(player)
    return vx + fx * impulse, vy + fy * impulse


def stop_projection(player, frames, *, bounded=False):
    """Console hard-brake projection, without reversing velocity."""
    return _backend(player).stop_projection(player, frames, bounded=bounded)


def check_approach(player, carrier):
    """Predict first body contact after a facing-directed burst, within eight frames."""
    vx, vy = burst_velocity(player)
    tx, ty = velocity(carrier)
    dx, dy = carrier.x - player.x, carrier.y - player.y
    rx, ry = tx - vx, ty - vy
    if max(abs(rx), abs(ry)) >= 32768 * VELOCITY_SCALE:
        return None
    speed2, dot = rx * rx + ry * ry, dx * rx + dy * ry
    if speed2 < 0.01 or dot >= 0:
        return None
    distance2 = dx * dx + dy * dy
    discriminant = dot * dot - speed2 * (distance2 - 16**2)
    if discriminant < 0:
        return None
    time = max(0, (-dot - math.sqrt(discriminant)) / speed2)
    if time > 8:
        return None
    time = math.ceil(time)
    separation = dx + rx * time, dy + ry * time
    if math.hypot(*separation) > 16:
        return None
    # checkcx: ((relative raw velocity dot separation) >> 4) >> 8.
    # Ignore accumulated impact; it may decay before this new collision.
    impact = max(0, int(-(rx * separation[0] + ry * separation[1]) / (4096 * VELOCITY_SCALE)))
    return time, (player.x + vx * time, player.y + vy * time), impact


def boost_safe(player, target):
    dx, dy = target[0] - player.x, target[1] - player.y
    distance = math.hypot(dx, dy)
    direction = facing(player)
    vx, vy = velocity(player)
    acceleration, _, energy = skating(player)
    speed = math.hypot(vx, vy) + boost_impulse(player)
    stopping = speed * speed / (2 * acceleration + speed / 64)
    return (energy >= 1024 and distance > max(55, stopping + 12)
            and dx * direction[0] + dy * direction[1] > distance * 0.95
            and (max(abs(player.y), abs(target[1])) < 248
                 or route_waypoint((player.x, player.y), target) == target))


def arrival_time(player, target, *, optimistic=False, boost=False):
    """Reach a stick-sized area, including turning and lateral momentum."""
    return _backend(player).arrival_time(player, target, optimistic=optimistic, boost=boost)


def puck_path(puck, horizon=64, *, sample_interval=4):
    """Console puck trajectory; stops at uncertain wall/net contacts."""
    return _backend(puck).puck_path(puck, horizon, sample_interval=sample_interval)
