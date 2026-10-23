"""Raw receiver mapping, disconnect recovery and independent human input."""

from collections import defaultdict
import importlib.util
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from nhl94_ai.game.constants import GameConsts as Buttons


@unittest.skipUnless(importlib.util.find_spec('pygame'), 'optional display extra is not installed')
class GamepadTests(unittest.TestCase):
    def setUp(self):
        from nhl94_ai.ui.gamepad import GamepadInput
        self.buttons = set()
        self.axes = [-0.00394, -0.00394, -1.0]
        self.device = Mock()
        self.device.get_guid.return_value = '03001a0da30c00002400000011010000'
        self.device.get_numbuttons.return_value = 10
        self.device.get_numaxes.return_value = 3
        self.device.get_instance_id.return_value = 42
        self.device.get_name.return_value = '6B controller'
        self.device.get_button.side_effect = lambda index: index in self.buttons
        self.device.get_axis.side_effect = lambda index: self.axes[index]
        self.count = patch('pygame.joystick.get_count', return_value=1).start()
        self.open = patch('pygame.joystick.Joystick', return_value=self.device).start()
        self.addCleanup(patch.stopall)
        self.pad = GamepadInput()
        self.addCleanup(self.pad.close)

    def test_measured_buttons_press_and_release_independently(self):
        for raw, action in ((2, Buttons.INPUT_A), (1, Buttons.INPUT_B), (5, Buttons.INPUT_C),
                            (3, Buttons.INPUT_X), (0, Buttons.INPUT_Y), (4, Buttons.INPUT_Z),
                            (9, Buttons.INPUT_START), (8, Buttons.INPUT_MODE)):
            with self.subTest(raw=raw):
                self.buttons.add(raw)
                expected = [0] * 12
                expected[action] = 1
                self.assertEqual(self.pad.read(), expected)
                self.buttons.clear()
                self.assertEqual(self.pad.read(), [0] * 12)

    def test_axes_diagonals_and_center_noise_ignore_third_axis(self):
        for x, y, expected in ((-1, -1, {4, 6}), (1, 1, {5, 7}),
                               (-0.00394, -0.00394, set()), (0.49, -0.49, set())):
            with self.subTest(x=x, y=y):
                self.axes[:2] = [x, y]
                self.assertEqual({i for i, value in enumerate(self.pad.read()) if value}, expected)

    def test_hotplug_uses_instance_id_and_releases_before_reconnecting(self):
        import pygame
        self.buttons.add(5)
        self.pad.process_event(SimpleNamespace(type=pygame.JOYDEVICEREMOVED, instance_id=7))
        self.assertEqual(self.pad.read()[Buttons.INPUT_C], 1)
        self.count.return_value = 0
        self.pad.process_event(SimpleNamespace(type=pygame.JOYDEVICEREMOVED, instance_id=42))
        self.assertEqual(self.pad.read(), [0] * 12)
        self.device.quit.assert_called_once()
        self.count.return_value = 1
        self.buttons.clear()
        self.pad.process_event(SimpleNamespace(type=pygame.JOYDEVICEADDED, device_index=0))
        self.assertEqual(self.pad.instance_id, 42)
        self.buttons.add(1)
        self.assertEqual(self.pad.read()[Buttons.INPUT_B], 1)
        self.pad.process_event(SimpleNamespace(type=pygame.JOYDEVICEADDED, device_index=0))
        self.assertEqual(self.open.call_count, 2)

    def test_poll_failure_discards_partial_pressed_state(self):
        import pygame
        self.buttons.update((2, 5))
        self.device.get_axis.side_effect = pygame.error('disconnected')
        self.assertEqual(self.pad.read(), [0] * 12)
        self.assertIsNone(self.pad.joystick)

    def test_unrelated_device_and_disabled_input_are_ignored(self):
        from nhl94_ai.ui.gamepad import GamepadInput
        self.pad.close()
        self.device.get_guid.return_value = 'unrelated tablet'
        self.pad.connect()
        self.assertIsNone(self.pad.joystick)
        self.open.reset_mock()
        disabled = GamepadInput(enabled=False)
        disabled.connect()
        self.assertEqual(disabled.read(), [0] * 12)
        self.open.assert_not_called()

    def test_keyboard_merge_focus_and_opposing_directions(self):
        from nhl94_ai.ui.debug import NHL94DebugDisplay
        import pygame
        display = NHL94DebugDisplay.__new__(NHL94DebugDisplay)
        display.gamepad = self.pad
        display.input_focused = True
        display.key_action_map = {pygame.K_x: Buttons.INPUT_B, pygame.K_LEFT: Buttons.INPUT_LEFT}
        self.buttons.add(5)
        self.axes[:2] = [1, -1]
        with patch('pygame.event.pump'), patch('pygame.key.get_pressed',
                return_value=defaultdict(bool, {pygame.K_x: True, pygame.K_LEFT: True})):
            actions = display.get_human_input()
            self.assertEqual({i for i, value in enumerate(actions) if value}, {0, 4, 8})
            display.input_focused = False
            self.assertEqual(display.get_human_input(), [0] * 12)
            display.input_focused = True
            self.assertEqual(display.get_human_input(), actions)
