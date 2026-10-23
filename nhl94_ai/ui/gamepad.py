"""Human input for the M30 exposed by the USB receiver as a 6B controller."""

import pygame

from nhl94_ai.game.constants import GameConsts as Buttons


class GamepadInput:
    """Poll the verified 0ca3:0024 layout; ignore unrelated input devices."""

    # Physical button labels, measured on the receiver's raw SDL joystick input.
    BUTTONS = {2: Buttons.INPUT_A, 1: Buttons.INPUT_B,
               5: Buttons.INPUT_C, 3: Buttons.INPUT_X,
               0: Buttons.INPUT_Y, 4: Buttons.INPUT_Z,
               9: Buttons.INPUT_START, 8: Buttons.INPUT_MODE}
    DEADZONE = 0.5

    def __init__(self, *, enabled=True):
        self.enabled = enabled
        self.joystick = None
        self.instance_id = None
        self.connect()

    def connect(self):
        if not self.enabled or self.joystick is not None:
            return
        for index in range(pygame.joystick.get_count()):
            joystick = None
            try:
                joystick = pygame.joystick.Joystick(index)
                guid = joystick.get_guid().lower()
                if (guid[8:16] == 'a30c0000' and guid[16:24] == '24000000'
                        and joystick.get_numbuttons() >= 10 and joystick.get_numaxes() >= 2):
                    self.instance_id = joystick.get_instance_id()
                    self.joystick = joystick
                    print(f'Gamepad connected: {joystick.get_name()} (M30 Genesis button layout).')
                    return
            except pygame.error:
                pass
            if joystick is not None:
                self._release(joystick)

    @staticmethod
    def _release(joystick):
        try:
            joystick.quit()
        except pygame.error:
            pass

    def close(self):
        joystick, self.joystick = self.joystick, None
        self.instance_id = None
        if joystick is not None:
            self._release(joystick)

    def process_event(self, event):
        if event.type == pygame.JOYDEVICEADDED:
            self.connect()
        elif event.type == pygame.JOYDEVICEREMOVED and event.instance_id == self.instance_id:
            self.close()
            self.connect()

    def read(self):
        """Return a fresh native button vector, including releases each frame."""
        actions = [0] * Buttons.INPUT_MAX
        if self.joystick is None:
            return actions
        try:
            for button, action in self.BUTTONS.items():
                actions[action] = int(bool(self.joystick.get_button(button)))
            x, y = self.joystick.get_axis(0), self.joystick.get_axis(1)
            actions[Buttons.INPUT_LEFT] = int(x < -self.DEADZONE)
            actions[Buttons.INPUT_RIGHT] = int(x > self.DEADZONE)
            actions[Buttons.INPUT_UP] = int(y < -self.DEADZONE)
            actions[Buttons.INPUT_DOWN] = int(y > self.DEADZONE)
        except pygame.error:
            self.close()
            return [0] * Buttons.INPUT_MAX
        return actions
