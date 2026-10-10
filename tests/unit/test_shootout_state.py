"""GameState exposes native attempt status without leaking ROM masks to tasks."""
import json
from pathlib import Path
import unittest

from nhl94_ai.game.state import NHL94GameState


class ShootoutStateTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / 'fixtures/NHL94-Genesis-v0.json'
        self.info = json.loads(path.read_text(encoding='utf-8'))
        self.state = NHL94GameState(5)

    def test_new_state_has_no_active_attempt(self):
        self.assertIs(self.state.is_shootout_active, False)

    def test_native_word_decodes_to_a_boolean_each_frame(self):
        for flags, active in ((0x0400, True), (0x2400, True),
                              (0x2000, False), (0x0004, False), (0, False)):
            with self.subTest(flags=flags):
                self.info['ba_ps_flags'] = flags
                self.state.BeginFrame(self.info, [0] * 6)
                self.assertIs(self.state.is_shootout_active, active)
                self.state.EndFrame()

    def test_missing_flag_feedback_clears_previous_attempt_status(self):
        self.info['ba_ps_flags'] = 0x0400
        self.state.BeginFrame(self.info, [0] * 6)
        self.assertIs(self.state.is_shootout_active, True)
        del self.info['ba_ps_flags']
        self.state.BeginFrame(self.info, [0] * 6)
        self.assertIs(self.state.is_shootout_active, False)

    def test_windup_is_not_a_release_and_release_remains_latched(self):
        self.info.update(sflags=0x0800, sflags2=0, ba_ps_flags=0x2400)
        self.state.BeginFrame(self.info, [0, 0, 0, 0, 0, 1])
        self.assertIs(self.state.is_shooting, True)
        self.assertIs(self.state.has_released_shot, False)
        self.info.update(sflags=0, sflags2=0x1000)
        self.state.BeginFrame(self.info, [0] * 6)
        self.assertIs(self.state.is_shooting, False)
        self.assertIs(self.state.has_released_shot, True)
        self.info['sflags2'] = 0
        self.state.BeginFrame(self.info, [0] * 6)
        self.assertIs(self.state.has_released_shot, True)

    def test_low_byte_flags_are_not_a_shot_release(self):
        self.info.update(sflags=0x0008, sflags2=0x0010)
        self.state.BeginFrame(self.info, [0] * 6)
        self.assertIs(self.state.is_shooting, False)
        self.assertIs(self.state.has_released_shot, False)

    def test_c_hold_duration_does_not_wrap_and_clears_on_release(self):
        for _ in range(65):
            self.state.BeginFrame(self.info, [0, 0, 0, 0, 0, 1])
            self.state.EndFrame()
        self.assertIs(self.state.c_pressed, True)
        self.assertEqual(self.state.c_frames_held, 65)
        self.state.BeginFrame(self.info, [0] * 6)
        self.assertIs(self.state.c_pressed, False)
        self.assertEqual(self.state.c_frames_held, 0)
        self.assertEqual(NHL94GameState(5).c_frames_held, 0)


if __name__ == '__main__':
    unittest.main()
