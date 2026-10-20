"""Deterministic input and controller fixtures shared by recording tests."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np


class InputReplayEnv:
    """Fake emulator whose serialized state depends on every controller input."""
    def __init__(self, buttons):
        self.buttons = buttons
        self.frames, self.value = 0, 0
        self.close = Mock()
        self.em = SimpleNamespace(get_state=self.get_state, set_state=self.set_state)
        self.data = SimpleNamespace(set_variable=Mock(), reset=Mock(), update_ram=Mock(),
                                    lookup_all=self.info)

    def info(self):
        raise NotImplementedError

    def reset(self):
        self.frames, self.value = 0, 0

    def get_state(self):
        return bytes([self.frames, self.value])

    def set_state(self, value):
        self.frames, self.value = value
        return True

    def get_ram(self):
        return np.array([self.frames, self.value], dtype=np.uint8)

    def step(self, buttons):
        self.frames += 1
        self.value = (self.value + sum((index + 1) * int(value) for index, value in enumerate(buttons))) % 256
        return None, 0, False, False, self.info()


def fake_classic():
    return SimpleNamespace(_last_decision='carry', _last_target=(np.int64(1), 2),
                           input_owner='offense', scheduler=SimpleNamespace(decisions=1),
                           offense_diagnostics={'estimate': float('inf')},
                           defense_diagnostics={}, goalie_diagnostics={})


def scripted_viewer(*, quit_after=None):
    count = 0
    def action():
        nonlocal count
        count += 1
        buttons = [0] * 12
        buttons[4 + count // 20 % 4] = 1
        buttons[8] = int(count % 17 < 3)
        buttons[0] = int(count % 31 == 0)
        return buttons
    return SimpleNamespace(before_frame=lambda: quit_after is None or count < quit_after,
                           human_action=action, show=lambda *_: None, close=lambda: None)
