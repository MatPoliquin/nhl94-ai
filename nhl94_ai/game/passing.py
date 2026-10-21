"""Pass release, recipient and contact mechanics; tactical admission stays in agents."""
from copy import copy
import math
import numpy as np
from nhl94_ai.game.motion import VELOCITY_SCALE, facing, velocity
from nhl94_ai.game.geometry import aim_pass
from nhl94_ai.env.target_control import project_target

PASS_HORIZON = 48
RELEASE_FRAMES = 4
LIVE_RELEASE_FRAMES = 2


def _backend(obj):
    from nhl94_ai.game.platforms import get_backend
    return get_backend(getattr(obj, 'platform', 'genesis'))

def pad_direction(passer, receiver):
    buttons = np.zeros(12, dtype=np.int8)
    aim_pass(buttons, passer, receiver)
    dx, dy = int(buttons[7] - buttons[6]), int(buttons[4] - buttons[5])
    return round(math.atan2(dx, dy) * 4 / math.pi) % 8 if dx or dy else 8

def selected_receiver(team, puck, passer_slot, direction):
    """Native recipient selection for the puck's console."""
    return _backend(puck).selected_receiver(team, puck, passer_slot, direction)

def pass_launch_view(state, passer):
    """Native launch view on Genesis; None where launch telemetry is unrecovered."""
    return _backend(passer).pass_launch_view(state, passer)

def option_receiver(receiver, option):
    """Effective pre-release origin for consumers of the shared pass forecast."""
    if option.receiver_origin is None:
        return receiver
    future = copy(receiver)
    future.x, future.y, future.motion_x, future.motion_y = option.receiver_origin
    future.precise_x, future.precise_y = float(future.x), float(future.y)
    return future

def pass_speed(passer):
    """Nominal rating-derived speed; live launch vectors include ROM quantization."""
    if getattr(passer, 'platform', 'genesis') != 'genesis':
        from nhl94_ai.game.platforms import get_backend
        return get_backend(passer.platform).estimated_pass_speed
    if passer.passing is None:
        return None
    speed = _nominal_speed(passer.passing, passer.role == 0)
    return speed * 65536 / (16 * 60) * VELOCITY_SCALE

def _nominal_speed(passing, goalie=False):
    speed = 160 + 2 * (8 if goalie else passing)
    return speed + speed // 16 if passing & 1 else speed

def _word(value):
    return (value + 32768) % 65536 - 32768

def _divide(value, divisor):
    quotient = abs(value) // abs(divisor)
    return -quotient if (value < 0) != (divisor < 0) else quotient

def _rom_sqrt(value):
    value &= 0xFFFFFFFF
    if value <= 1600:
        return math.isqrt(value)
    if value > 0xF00000:
        low, high = 0, 65535
        while True:
            middle = (low + high) // 2
            if middle == low:
                return low
            if middle * middle > value:
                high = middle
            else:
                low = middle
    guess = ((value >> 8) + 2) & 65535
    for _ in range(10):
        next_guess = ((value // guess + guess) & 65535) >> 1
        if next_guess == guess:
            break
        guess = next_guess
    return guess

def rom_pass_vector(dx, dy, vx, vy, passing, *, goalie=False):
    """passto's signed word arithmetic and eighth-second lead quantization."""
    speed = _nominal_speed(passing, goalie)
    qx, qy = _word((vx * 240) >> 16), _word((vy * 240) >> 16)
    ax, ay = dx >> 2, dy >> 2
    j = _word(2 * _word(qx * ax + qy * ay))
    k = qx * qx + qy * qy - (speed >> 2)**2
    root = _rom_sqrt(j * j - 4 * _word(k) * _word(ax * ax + ay * ay))
    divisor = _word(k) >> 2 or 1
    time = _divide(_word(root - j), divisor)
    if time < 0:
        time = _divide(_word(-root - j), divisor)
    time = time or 1
    if (time & 65535) > 24:
        time = 24
    result = []
    for distance, velocity in ((dx, qx), (dy, qy)):
        lead = velocity * time >> 1
        register = ((lead & 0xFFFF0000) | ((lead + distance) & 65535)) & 0xFFFFFFFF
        swapped = ((register << 16) | (register >> 16)) & 0xFFFFFFFF
        signed = (swapped + 0x80000000) % 0x100000000 - 0x80000000
        component = _divide(signed, time * 120)
        if not -32768 <= component <= 32767:
            return None
        result.append(component)
    return tuple(result)

def _live_pass_contact(puck, passer, receiver, delay):
    """Genesis sprite hotspot and passto integer launch; used by GenesisBackend."""
    vx, vy = velocity(receiver)
    px, py = velocity(passer)
    start = math.floor(puck.x + px * delay), math.floor(puck.y + py * delay)
    target = (math.floor(receiver.x + vx * delay) + receiver.stick_x,
              math.floor(receiver.y + vy * delay) + receiver.stick_y)
    vector = rom_pass_vector(target[0] - start[0], target[1] - start[1],
                             round(vx / VELOCITY_SCALE), round(vy / VELOCITY_SCALE), passer.passing,
                             goalie=passer.role == 0)
    if vector is None:
        return None
    sx, sy = (component * VELOCITY_SCALE for component in vector)
    closest = math.inf, start, 0
    for frame in range(1, PASS_HORIZON + 1):
        travel = (1 - puck.friction**frame) / (1 - puck.friction) if puck.friction != 1 else frame
        point = start[0] + sx * travel, start[1] + sy * travel
        receiver_point = target[0] + vx * frame, target[1] + vy * frame
        distance = math.dist(point, receiver_point)
        if distance < closest[0]:
            closest = distance, point, frame
    distance, point, frame = closest
    if distance > 14 or math.dist(point, project_target(point)) > 1e-6:
        return None
    return start, point, frame, math.hypot(sx, sy) * puck.friction**frame

def pass_release_frames(receiver, delay=RELEASE_FRAMES):
    """Use the same release timing for launch prediction and collision deadlines."""
    return _backend(receiver).pass_release_frames(receiver, delay)

def pass_contact(puck, passer, receiver, delay=RELEASE_FRAMES):
    """Native or estimated reception contact for the puck's console."""
    return _backend(puck).pass_contact(puck, passer, receiver, delay)

def pass_point_at(puck, receiver, contact, time):
    start, point, flight, _ = contact
    live = receiver.stick_x is not None and receiver.stick_y is not None
    fraction = ((1 - puck.friction**time) / (1 - puck.friction**flight)
                if live and puck.friction != 1 else time / flight)
    return start[0] + (point[0] - start[0]) * fraction, start[1] + (point[1] - start[1]) * fraction

def one_timer_contact_frame(puck, receiver, contact, release):
    """First modeled body/stick contact, not the later closest-approach frame."""
    _, _, flight, _ = contact
    vx, vy = velocity(receiver)
    fx, fy = facing(receiver)
    offset = ((receiver.stick_x, receiver.stick_y) if receiver.stick_x is not None
              and receiver.stick_y is not None else (fx * 10, fy * 10))
    for time in (*range(1, math.ceil(flight)), flight):
        position = pass_point_at(puck, receiver, contact, time)
        body = receiver.x + vx * (release + time), receiver.y + vy * (release + time)
        stick = body[0] + offset[0], body[1] + offset[1]
        if math.dist(position, body) <= 8 or math.dist(position, stick) <= 14:
            return release + time
    return None
