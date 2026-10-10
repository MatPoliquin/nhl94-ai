"""Compare executable finishes using release geometry, never scoring probabilities."""
from dataclasses import dataclass
import math

from nhl94_ai.agents.defense import on_ice
from nhl94_ai.agents.motion import VELOCITY_SCALE, velocity
from nhl94_ai.agents.offense import normal_shot_release_frames, shot_release_in_front
from nhl94_ai.agents.passing import one_timer_contact_frame, option_receiver, pass_point_at, pass_release_frames
from nhl94_ai.agents.skating import grounded_step


@dataclass(frozen=True)
class Finish:
    side: int
    hold_frames: int
    release_frames: float
    point: tuple
    speed: float
    clearance: float
    value: float


def shot_speed(player, hold_frames, *, one_timer=False):
    """doshot's rating/energy scaling with an estimated charged passspeed."""
    power = 15 if player.shot_power is None else player.shot_power
    energy = 4096 if player.energy is None else max(0, min(4096, player.energy))
    charge = 31 if one_timer else 15 + hold_frames
    rating = (power // 2) * energy // 4096
    speed = ((20 + rating) * charge * 0x5249) >> 16
    return max(1.0, speed * 68 * VELOCITY_SCALE)


def _goalie_position(goalie, delay):
    vx, vy = velocity(goalie)
    # Current velocity is evidence over a short interval, not an indefinite path.
    return goalie.x + vx * min(delay, 12), goalie.y + vy * min(delay, 12)


def _score(state, shooter, point, side, release, speed):
    goal = side * 13, state.team2.net.y
    travel = math.dist(point, goal) / speed
    clearance = 40.0
    for team in (state.team1, state.team2):
        for other in (*team.players, team.goalie):
            if other is shooter or not on_ice(other):
                continue
            vx, vy = velocity(other)
            if other is state.team2.goalie:
                start = _goalie_position(other, release)
                end = _goalie_position(other, release + travel)
                radius = 12
                if other.live_anim in (0x250, 0x2A2) and release <= (other.animation_timer or 0):
                    radius = 18
            else:
                start = other.x + vx * release, other.y + vy * release
                end = other.x + vx * (release + travel), other.y + vy * (release + travel)
                radius = 8
            ax, ay = point[0] - start[0], point[1] - start[1]
            dx, dy = goal[0] - end[0] - ax, goal[1] - end[1] - ay
            fraction = max(0, min(1, -(ax * dx + ay * dy) / max(0.01, dx * dx + dy * dy)))
            clearance = min(clearance, math.hypot(ax + dx * fraction, ay + dy * fraction) - radius)
    accuracy = 15 if shooter.shot_accuracy is None else shooter.shot_accuracy
    value = (40 if clearance > 0 else 0) + 2 * max(-24, min(24, clearance)) + accuracy * 0.2 - release * 0.2
    return clearance, value


def normal_finish(state, player, decision_interval):
    """Choose an aim and a C hold that the frame dispatcher can actually emit."""
    choices = []
    for hold in sorted({min(2, decision_interval), decision_interval}):
        if not shot_release_in_front(state, player, hold):
            continue
        release = normal_shot_release_frames(player, hold)
        carrier = player
        for _ in range(release):
            carrier = grounded_step(carrier, (0, 0))
        point = state.puck.x + carrier.x - player.x, state.puck.y + carrier.y - player.y
        speed = shot_speed(player, hold)
        for side in (-1, 1):
            clearance, value = _score(state, player, point, side, release, speed)
            choices.append(Finish(side, hold, release, point, speed, clearance, value))
    return max(choices, key=lambda choice: (choice.value, -choice.release_frames,
                                          choice.side == (-1 if state.team2.goalie.x > 0 else 1)), default=None)


def automatic_aim(state, point, delay):
    """shotdiradj's goalie-relative post comparison for a native one-timer."""
    goalie = state.team2.goalie
    if not on_ice(goalie):
        return 0
    x, y = _goalie_position(goalie, delay)
    vx, vy = (round(component / VELOCITY_SCALE) >> 9 for component in velocity(goalie))
    dx, dy = x + vx - point[0], y + vy - point[1]
    length = max(1, math.isqrt(int(dx * dx + dy * dy + 1)))
    cross = sum(math.trunc((dx * (state.team2.net.y - point[1]) - dy * (post - point[0])) / length)
                for post in (-18, 18))
    if abs(cross) > 44:
        return 0
    return -1 if cross * (1 if state.team2.net.y > 0 else -1) > 0 else 1


def one_timer_finish(state, option):
    """Value the ROM's automatic aim at reception, including the pass delay."""
    if option is None or option.contact is None:
        return None
    shooter = state.team1.players[option.index]
    receiver = option_receiver(shooter, option)
    release = option.release_frames if option.release_frames is not None else pass_release_frames(receiver)
    contact = one_timer_contact_frame(state.puck, receiver, option.contact, release)
    if contact is None:
        return None
    point = pass_point_at(state.puck, receiver, option.contact, contact - release)
    speed = shot_speed(shooter, 0, one_timer=True)
    side = automatic_aim(state, point, contact)
    clearance, value = _score(state, shooter, point, side, contact, speed)
    return Finish(side, 0, contact, point, speed, clearance, value)
