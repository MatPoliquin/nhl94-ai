"""Faceoff input belongs only to the native draw participant.

Genesis faceoffinput accepts assignment 0x17. The center's role is 4;
assignment 0x16 belongs to the stationary formation. Geometry is a sanity
check, never a replacement for the native assignment.
"""
import math

import numpy as np

from nhl94_ai.game.constants import GameConsts as Buttons


def faceoff_center(state):
    if not state.engine.clock_stopped or state.engine.puck_owner >= 0 or state.time <= 0:
        return None
    if state.engine.sflags & 0x100:
        return None
    candidates = [state.team1.skater_scnum_base() + i for i, player in enumerate(state.team1.players)
                  if player.assignment == 0x17 and player.role == 4
                  and not player.unavailable & 4 and not player.is_falling
                  and math.dist((player.x, player.y), (state.puck.x, state.puck.y)) <= 28]
    opponents = [player for player in state.team2.players if player.assignment == 0x17
                 and player.role == 4 and math.dist((player.x, player.y), (state.puck.x, state.puck.y)) <= 28]
    return candidates[0] if len(candidates) == 1 and len(opponents) == 1 else None


class FaceoffInput:
    def __init__(self, *, draw_back=False):
        self.controller = None
        self.since = 0
        self.draw_back = draw_back
        self.release_on_handoff = False

    def choose(self, state, slots, frame, *, transient=False, previous=None):
        center = None if transient else faceoff_center(state)
        controller = slots.index(center) if center is not None and center in slots else None
        if controller != self.controller:
            self.controller, self.since = controller, frame
            self.release_on_handoff = (controller is not None and previous is not None
                                       and bool(previous[controller, Buttons.INPUT_B]))
        action = np.zeros(12, dtype=np.int8)
        if controller is not None:
            # Preserve the original cadence/direction. A previously held B must
            # release once before the newly assigned draw owner can press again.
            if frame != self.since or not self.release_on_handoff:
                action[Buttons.INPUT_B] = frame % 8 < 4
                if self.draw_back:
                    action[Buttons.INPUT_UP if state.team1.net.y > state.puck.y else Buttons.INPUT_DOWN] = 1
        return controller, action, center
