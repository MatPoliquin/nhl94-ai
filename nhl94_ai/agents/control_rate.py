"""Strict Classic observation and raw-button cadence, independent of tactics."""
import numpy as np


def control_interval(args):
    """Zero retains reactive control; positive integers hold that many frames."""
    interval = getattr(args, 'classic_control_interval', 0)
    if isinstance(interval, bool) or not isinstance(interval, int) or interval < 0:
        raise ValueError('classic-control-interval must be a nonnegative integer')
    if interval and getattr(args, 'action_type', 'FILTERED').upper() != 'FILTERED':
        raise ValueError('Strict Classic control requires raw FILTERED buttons')
    return interval


class ControlRateLimit:
    def __init__(self, controller, repeat):
        if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat < 1:
            raise ValueError('Repeat must be a positive integer')
        self.controller, self.repeat = controller, repeat
        self.frames = self.observations = self.remaining = self.changes = 0
        self.action = None
        self.observed = False

    def predict_frame(self, state, frame_skip=4, deterministic=True):
        self.observed = self.remaining == 0
        if self.observed:
            action = self.controller.predict_frame(state, frame_skip, deterministic).copy()
            self.changes += self.action is not None and not np.array_equal(action, self.action)
            self.action = action
            self.observations += 1
            self.remaining = self.repeat
        else:
            self.controller.advance_held_frame()
        self.remaining -= 1
        self.frames += 1
        return self.action.copy()

    def report(self):
        return {'repeat': self.repeat, 'nominal_hz': 60 / self.repeat,
                'input_frames': self.frames, 'controller_observations': self.observations,
                'input_changes': self.changes,
                'scheduler_frames': self.controller.scheduler.frames}
