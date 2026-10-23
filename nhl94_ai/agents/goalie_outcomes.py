"""Conditional body-contact outcomes from NHL94's puckgoalie routine.

This is a contact model, not a complete save predictor. It marginalizes the
ROM's random threshold without reading RNG state. Stick contacts and animation
geometry must be resolved before using it as a decision evaluator.
"""
from dataclasses import replace

from nhl94_ai.agents.goalie_physics import signed_word


def catch_probability(puck_control, vx, vy, *, animated, close=True, dive=False,
                      unavailable=False):
    """Probability conditional on body contact and a uniform randomd0 draw.

    puckgoalie: min(32767, 2*(3072 + random((PC+2)*512))).
    Upright contacts near the crease divide that threshold by four. Both
    absolute velocity components must fit it. This uses effective live PC.
    """
    if not 0 <= puck_control <= 30:
        raise ValueError('Expected effective goalie puck control in 0..30')
    if dive or unavailable:
        return 0.0
    required = max(abs(vx), abs(vy)) * (4 if close and not animated else 1)
    if required > 32767:
        return 0.0
    minimum_draw = max(0, (required + 1)//2 - 3072)
    bound = (puck_control+2)*512
    # randomd0 returns the high word of a 16-bit uniform value times bound.
    first_uniform = min(65536, (minimum_draw*65536+bound-1)//bound)
    return (65536-first_uniform)/65536


def body_rebound(keeper, puck, *, stick=None):
    """Native body deflection; preserve velocity low bytes, clear vertical speed.

    puckgoalie writes a byte into each big-endian velocity's high byte. The
    equal-Y case uses the animation's stick hotspot; unknown hotspots refuse
    prediction. No post-contact player interactions are invented here.
    """
    dx, dy = (puck.x >> 16)-(keeper.x >> 16), (puck.y >> 16)-(keeper.y >> 16)
    if dy == 0:
        if stick is None:
            return None
        dx, dy = stick
    return replace(puck, vx=signed_word(((dx & 255) << 8) | (puck.vx & 255)),
                   vy=signed_word(((dy & 255) << 8) | (puck.vy & 255)), vz=0)
