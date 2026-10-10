"""Isolated finishing ratings, controller inputs and event accounting."""
from types import SimpleNamespace
import unittest

from nhl94_ai.evaluation.cross_crease_probe import ShotTiming, summarize
from nhl94_ai.game.constants import GameConsts as Buttons
from nhl94_ai.game.ram import GOALIE_INPUT_ATTRIBUTES
from nhl94_ai.tasks.cross_crease_setup import goalie_attributes, set_goalie_level
from tests.ram_fixture import FixtureMemory


class GoalieRatingTests(unittest.TestCase):
    def test_live_skill_encodings_use_goalie_offsets_and_inverse_awareness_delay(self):
        low, high = goalie_attributes(1), goalie_attributes(6)
        self.assertEqual(low['agility'], 5)
        self.assertEqual(high['agility'], 30)
        self.assertEqual((low['defensive_delay'], high['defensive_delay']), (14, 7))
        self.assertEqual((low['glove_left'], high['glove_left']), (5, 14))
        self.assertEqual(low['weight'], high['weight'])
        self.assertEqual(low['handedness'], high['handedness'])

    def test_profile_applies_live_fields_and_resets_goalie_motion_and_decision(self):
        memory = FixtureMemory({})
        env = SimpleNamespace(data=SimpleNamespace(memory=memory))
        base = 0xFFB04A + 11 * 0x80
        memory.assign(base + 0x28, '>i2', 1500)
        memory.assign(base + 0x40, '|i1', 10)
        attributes = set_goalie_level(env, 11, 6)
        for name, value in attributes.items():
            self.assertEqual(memory.extract(base + GOALIE_INPUT_ATTRIBUTES[name][0], '|u1'), value)
        self.assertEqual(memory.extract(base + 0x28, '>i2'), 0)
        self.assertEqual(memory.extract(base + 0x40, '|i1'), 0)
        self.assertEqual(memory.extract(base + 0x43, '|u1'), 8)

    def test_invalid_levels_and_non_goalie_slots_are_explicit_errors(self):
        for level in (-1, 7, True, 1.5):
            with self.assertRaises(ValueError):
                goalie_attributes(level)
        with self.assertRaises(ValueError):
            set_goalie_level(None, 0, 1)


class ShotTimingTests(unittest.TestCase):
    def test_hold_has_one_fresh_press_and_an_explicit_release(self):
        timing = ShotTiming(4, 12, -1)
        for direction in (-1, 1):
            for frame in range(20):
                action = timing.action(frame, direction)
                self.assertEqual(bool(action[Buttons.INPUT_C]), 4 <= frame < 16)
                aimed = direction if frame < 4 else -direction
                self.assertTrue(action[Buttons.INPUT_RIGHT if aimed > 0 else Buttons.INPUT_LEFT])
                self.assertFalse(action[Buttons.INPUT_B])
                self.assertFalse(action[Buttons.INPUT_UP])
                self.assertFalse(action[Buttons.INPUT_DOWN])

    def test_invalid_timings_are_rejected(self):
        for timing in ((-1, 4, 1), (25, 4, 1), (0, 0, 1), (0, 61, 1), (0, 4, 0)):
            with self.assertRaises(ValueError):
                ShotTiming(*timing)

    def test_requested_shot_windup_release_and_recorded_shot_remain_separate(self):
        rows = [
            dict(goal=False, shot=False, windup_frame=1, release_frame=6,
                 commit_frame=5, goalie_contact_impulses=0, outcome='no-goal'),
            dict(goal=True, shot=True, windup_frame=1, release_frame=37,
                 commit_frame=None, goalie_contact_impulses=0, outcome='goal'),
        ]
        summary = summarize(rows)
        self.assertEqual(summary['goals'], 1)
        self.assertEqual(summary['accepted_windups'], 2)
        self.assertEqual(summary['releases'], 2)
        self.assertEqual(summary['shots'], 1)
        self.assertEqual(summary['pre_release_saves'], 1)


if __name__ == '__main__':
    unittest.main()
