"""Classic's two explicit time domains and cached input cadence.

Frames advance on every dispatched native input. Decisions advance only when the
tactical policy runs; special actions can suspend them. A decision deadline must
not be converted to frames by multiplying it by the configured interval.
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ActionScheduler:
    frames: int = 0
    decisions: int = 0
    interval: int = 4
    elapsed: int = 4
    remaining: int = 0
    was_defending: bool = False
    action: np.ndarray = field(default_factory=lambda: np.zeros((1, 12), dtype=np.int8))

    def configure(self, interval):
        if interval < 1:
            raise ValueError('Scripted decision interval must be positive')
        self.interval = interval

    def advance(self, frames):
        self.frames += frames
        return self.frames

    def interrupt(self):
        self.remaining = 0

    def cache(self, action, remaining=None):
        self.action = action
        self.remaining = self.interval if remaining is None else remaining


@dataclass
class ButtonState:
    """Last raw B/C input, including inputs encoded as hockey intents."""
    b_down: bool = False
    c_down: bool = False
