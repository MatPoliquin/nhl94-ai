"""Rink-coordinate input geometry shared by agents and action adapters."""
from nhl94_ai.game.platforms import get_backend


def aim_pass(action, passer, receiver):
    """Write the pass-aim pad direction for the passer's console."""
    get_backend(getattr(passer, 'platform', 'genesis')).aim_pass(action, passer, receiver)
