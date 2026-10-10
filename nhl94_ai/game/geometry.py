"""Rink-coordinate input geometry shared by agents and action adapters."""
from nhl94_ai.game.constants import GameConsts as Buttons


def aim_pass(action, passer, receiver):
    """Choose the nearest of eight pad directions, not two independent axes.

    The diagonal boundary is tan(22.5 degrees). Skating toward a point can use
    both axes for any displacement; aiming that way misdirects shallow passes.
    """
    dx, dy = receiver.x - passer.x, receiver.y - passer.y
    action[4:8] = 0
    if abs(dx) > max(3, abs(dy) * 0.41421356237):
        action[Buttons.INPUT_RIGHT if dx > 0 else Buttons.INPUT_LEFT] = 1
    if abs(dy) > max(3, abs(dx) * 0.41421356237):
        action[Buttons.INPUT_UP if dy > 0 else Buttons.INPUT_DOWN] = 1
