"""Rink-coordinate input geometry shared by agents and action adapters."""
from nhl94_ai.game.platforms import get_backend
from nhl94_ai.game.constants import GameConsts as Buttons


def apply_pad(action, pad):
    """Add rink-coordinate directions without clearing existing button presses."""
    dx, dy = pad
    if dx:
        action[Buttons.INPUT_RIGHT if dx > 0 else Buttons.INPUT_LEFT] = 1
    if dy:
        action[Buttons.INPUT_UP if dy > 0 else Buttons.INPUT_DOWN] = 1
    return action


def aim_pass(action, passer, receiver):
    """Write the pass-aim pad direction for the passer's console."""
    get_backend(getattr(passer, 'platform', 'genesis')).aim_pass(action, passer, receiver)
