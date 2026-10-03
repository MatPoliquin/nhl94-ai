"""Short-horizon, conservative motion estimates in rink units and emulator frames."""
import math

from nhl94_ai.env.target_control import project_target, route_waypoint


VELOCITY_SCALE = 17 / 65536
LANE_HORIZON = 32
SHOT_SPEED_RANGE = (3.0, 6.0)
BLOCK_RADIUS = 8


def velocity(player):
    if player.motion_x is None or player.motion_y is None:
        return player.vx * 256 * VELOCITY_SCALE, player.vy * 256 * VELOCITY_SCALE
    return player.motion_x, player.motion_y


def skate_step(motion, heading, turning, pad, acceleration, limit):
    """One frame of the shared conservative eight-way skating projection."""
    (x, y), (vx, vy) = motion
    ux, uy = pad
    wanted = round(math.atan2(ux, uy) * 4 / math.pi) % 8 if ux or uy else heading
    if wanted != heading:
        turning = turning or 6
        turning -= 1
        if turning == 0:
            difference = (wanted - heading + 4) % 8 - 4
            heading = (heading + (1 if difference > 0 else -1)) % 8
        ux = uy = 0
    length = max(1, math.hypot(ux, uy))
    nx, ny = vx * 63 / 64 + acceleration * ux / length, vy * 63 / 64 + acceleration * uy / length
    if math.hypot(nx, ny) <= max(limit, math.hypot(vx, vy)):
        vx, vy = nx, ny
    else:
        vx, vy = vx * 63 / 64, vy * 63 / 64
    return ((x + vx, y + vy), (vx, vy)), heading, turning


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
    """ROM playeracc/MaxSpeed units, with explicit conservative unknown ratings."""
    speed = player.speed if player.speed is not None else (30 if optimistic else 10)
    agility = player.agility if player.agility is not None else (30 if optimistic else 10)
    weight = player.weight if player.weight is not None else (32 if optimistic else 88)
    energy = player.energy if player.energy is not None else (4096 if optimistic else 2048)
    energy = max(0, min(4096, energy))
    speed = max(0, min(30, speed + player.movement_bonus))
    index = min(15, (speed // 2) * energy // 4096)
    limit = ((20 + index) * 275) ** 2
    if speed % 2 and index < 15:
        limit = (limit + ((21 + index) * 275) ** 2) / 2
    factor = max(1, (64 - weight // 4 + agility + player.movement_bonus) // 2)
    acceleration = (200 * factor // 32) * VELOCITY_SCALE
    return acceleration, math.sqrt(limit) * VELOCITY_SCALE, energy


def facing(player):
    direction = player.orientation if player.facing is None else player.facing
    angle = (direction % 8) * math.pi / 4
    return math.sin(angle), math.cos(angle)


def boost_impulse(player):
    _, _, energy = skating(player)
    return max(0, energy - 204) // 128 * 200 * VELOCITY_SCALE


def burst_velocity(player):
    fx, fy = facing(player)
    vx, vy = velocity(player)
    impulse = boost_impulse(player)
    return vx + fx * impulse, vy + fy * impulse


def stop_projection(player, frames):
    """Skater stopna subtracts 150 raw units per axis, without reversing velocity."""
    x, y = float(player.x), float(player.y)
    vx, vy = velocity(player)
    for _ in range(frames):
        vx = math.copysign(max(0, abs(vx) - 150 * VELOCITY_SCALE), vx) * 63 / 64
        vy = math.copysign(max(0, abs(vy) - 150 * VELOCITY_SCALE), vy) * 63 / 64
        x, y = x + vx, y + vy
    return (x, y), (vx, vy)


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
    target = project_target(target)
    waypoint = (target if max(abs(player.y), abs(target[1])) < 248
                else route_waypoint((player.x, player.y), target))
    dx, dy = waypoint[0] - player.x, waypoint[1] - player.y
    distance = math.hypot(dx, dy)
    route_extra = math.dist(waypoint, target)
    if distance + route_extra < 7:
        return 0.0
    vx, vy = velocity(player)
    acceleration, limit, _ = skating(player, optimistic)
    along = (vx * dx + vy * dy) / max(distance, 1)
    lateral = abs(vx * dy - vy * dx) / max(distance, 1)
    facing_x, facing_y = facing(player)
    turn = 4 * (1 - (facing_x * dx + facing_y * dy) / max(distance, 1))
    if boost and boost_safe(player, waypoint):
        along += boost_impulse(player)
    # Friction reduces acceleration as speed builds. Do not clamp an existing
    # burst/collision velocity to the ordinary acceleration acceptance limit.
    effective = max(acceleration * 0.5, acceleration - max(0, along) / 64)
    distance = max(0, distance + route_extra - 7)
    if along > limit:
        travel = distance / max(limit, along * 0.85)
    else:
        accelerate = max(0, (limit - along) / effective)
        covered = along * accelerate + effective * accelerate * accelerate / 2
        if covered >= distance:
            travel = (-along + math.sqrt(along * along + 2 * effective * distance)) / effective
        else:
            travel = accelerate + (distance - covered) / limit
    return max(0, travel) + turn + min(20, lateral / (2 * acceleration))


def puck_path(puck, horizon=64):
    """Stop at uncertain wall/net contacts; never invent an exact rebound."""
    x, y = float(puck.x), float(puck.y)
    vx, vy = velocity(puck)
    z = max(0, puck.height or 0)
    vz = puck.motion_z or 0
    points = []
    for frame in range(1, horizon + 1):
        if z <= 0:
            vx *= puck.friction
            vy *= puck.friction
        x, y = x + vx, y + vy
        if z > 0 or vz:
            vz -= 102 * VELOCITY_SCALE
            z += vz
            if z < 0:
                z, vz = 0.0, -vz / 2
                if abs(vz) < 0.05:
                    vz = 0
        if math.dist((x, y), project_target((x, y))) > 1e-6:
            break
        if frame % 4 == 0:
            points.append((frame, (x, y), z))
    return points
