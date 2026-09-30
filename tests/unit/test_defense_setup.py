"""Joint defensive formations and complete reset writes without a ROM."""
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from nhl94_ai.tasks.defense_setup import _sample_formation, _rebuild_object_order, init_full_team_defense
from tests.ram_fixture import FixtureMemory


def reset_env(seed):
    path = Path(__file__).resolve().parents[1] / 'fixtures/NHL94-Genesis-v0.json'
    info = json.loads(path.read_text(encoding='utf-8'))
    info.update(puck_owner=6, period=0, time=274)
    return SimpleNamespace(data=SimpleNamespace(memory=FixtureMemory(info)),
                           np_random=np.random.default_rng(seed))


class DefenseSetupContracts(unittest.TestCase):
    def test_formations_are_bounded_spaced_and_defendable(self):
        rng = np.random.default_rng(42)
        carriers = []
        for _ in range(1000):
            friendly, opponents, goalies = _sample_formation(rng)
            carrier = opponents[0]
            carriers.append(carrier)
            positions = friendly + opponents + goalies
            self.assertTrue(all(-100 <= x <= 100 and -250 <= y <= 250 for x, y in positions))
            self.assertTrue(all(math.dist(first, second) >= 20
                                for i, first in enumerate(positions) for second in positions[i + 1:]))
            self.assertLess(friendly[0][1], carrier[1])
            self.assertLessEqual(math.dist(friendly[0], carrier), 54)
            self.assertGreaterEqual(carrier[1], -180)
            self.assertLessEqual(carrier[1], -125)
            self.assertLessEqual(abs(goalies[0][0]), 14)
            self.assertEqual(goalies[0][1], -250)
        self.assertGreater(len(set(carriers)), 900)
        self.assertTrue(any(x < -60 for x, _ in carriers))
        self.assertTrue(any(x > 60 for x, _ in carriers))
        self.assertTrue(any(abs(x) < 20 for x, _ in carriers))

    def test_impossible_spacing_fails_instead_of_returning_an_invalid_start(self):
        with patch('nhl94_ai.tasks.defense_setup._MIN_SPACING', 1000):
            with self.assertRaisesRegex(RuntimeError, '128 attempts'):
                _sample_formation(np.random.default_rng(1))

    def test_seed_reproduces_all_reset_writes_including_rom_rng(self):
        first, second, different = reset_env(7), reset_env(7), reset_env(8)
        for env in (first, second, different):
            init_full_team_defense(env)
        self.assertEqual(first.data.memory.buffer, second.data.memory.buffer)
        self.assertNotEqual(first.data.memory.buffer, different.data.memory.buffer)
        self.assertNotEqual(first.data.memory.extract(0xFFD066, '>u4'), 0)

    def test_reset_clears_motion_and_stale_actions_without_changing_roster(self):
        env = reset_env(9)
        memory = env.data.memory
        for slot in range(12):
            base = 0xFFB04A + slot * 0x80
            memory.assign(base + 0x28, '>u4', 0x12345678)
            memory.assign(base + 0x2C, '>i2', 100)
            memory.assign(base + 0x63, '|u1', 6)
            memory.assign(base + 0x64, '|u1', 0x3B)
        before = bytes(memory.buffer)
        init_full_team_defense(env)
        owner = memory.extract(0xFFB7AA, '>i2')
        self.assertIn(owner, range(6, 11))
        self.assertEqual(memory.extract(0xFFBEDA, '>i2'), owner)
        self.assertEqual(memory.extract(0xFFBED8, '>i2'), -1)
        self.assertEqual(memory.extract(0xFFBEE0, '>i2'), -1)
        self.assertEqual(memory.extract(0xFFC320, '>i2'), 1)
        self.assertEqual(memory.extract(0xFFC322, '>i2'), -1)
        self.assertEqual(memory.extract(0xFFC468, '>u2'), 274)
        for slot in (*range(12), 14):
            base = 0xFFB04A + slot * 0x80
            for current, previous, velocity in ((0, 0x1C, 0x28), (0x14, 0x20, 0x2A), (0x18, 0x24, 0x2C)):
                self.assertEqual(memory.extract(base + current, '>i4'), memory.extract(base + previous, '>i4'))
                self.assertEqual(memory.extract(base + velocity, '>i2'), 0)
            self.assertEqual(memory.extract(base + 0x18, '>i4'), 0)
            if slot < 12:
                offset = base & 0xFFFF
                self.assertEqual(memory.buffer[offset + 0x66:offset + 0x80], before[offset + 0x66:offset + 0x80])
                self.assertEqual(memory.extract(base + 0x63, '|u1') & 6, 0)
                self.assertEqual(memory.extract(base + 0x64, '|u1') & 0x3B, 0)
                self.assertEqual(bool(memory.extract(base + 0x62, '|u1') & 8), slot == 1)
        self.assertEqual(memory.extract(0xFFB74A, '>i4'), memory.extract(0xFFB04A + owner * 0x80, '>i4'))
        self.assertEqual(memory.extract(0xFFB75E, '>i4'), memory.extract(0xFFB05E + owner * 0x80, '>i4'))

    def test_roles_and_initial_control_vary_without_curriculum_stages(self):
        owners, primaries = set(), set()
        needs_switch = 0
        for seed in range(80):
            env = reset_env(seed)
            init_full_team_defense(env)
            memory = env.data.memory
            owner = memory.extract(0xFFB7AA, '>i2')
            owners.add(owner)
            primary = next(slot for slot in range(5)
                           if memory.extract(0xFFB04A + slot * 0x80 + 0x3F, '|u1') == 0x11)
            primaries.add(primary)
            needs_switch += primary != 1
            base = 0xFFB04A + owner * 0x80
            self.assertEqual(memory.extract(base + 0x36, '>u2'), 6)
            self.assertEqual(memory.extract(base + 0x3E, '|u1'), 0x10)
            self.assertEqual(memory.extract(base + 0x3F, '|u1'), 0x11)
            self.assertEqual(memory.extract(base + 0x40, '|u1'), 12)
            self.assertEqual(memory.extract(base + 0x62, '|u1') & 2, 0)
        self.assertEqual(owners, set(range(6, 11)))
        self.assertEqual(primaries, set(range(5)))
        self.assertGreater(needs_switch, 5)
        self.assertLess(needs_switch, 40)

    def test_each_team_retains_one_nearest_assignment_for_possession_handoffs(self):
        env = reset_env(0)
        init_full_team_defense(env)
        memory = env.data.memory
        for side in (0, 6):
            assignments = [
                memory.extract(0xFFB04A + slot * 0x80 + 0x38 + index, '|u1')
                for slot in range(side, side + 6) for index in range(8)
            ]
            self.assertEqual(assignments.count(0x11), 1)
            self.assertEqual(assignments.count(0x10), 1 if side == 6 else 0)

    def test_assignment_reset_preserves_collision_radii_and_clears_wall_contacts(self):
        env = reset_env(9)
        memory = env.data.memory
        for slot in (*range(12), 14):
            base = 0xFFB04A + slot * 0x80
            memory.assign(base + 0x4A, '>u2', slot + 1)
            memory.assign(base + 0x4C, '>u2', slot + 2)
            memory.assign(base + 0x4E, '>u4', 0x12345678)
        init_full_team_defense(env)
        for slot in (*range(12), 14):
            base = 0xFFB04A + slot * 0x80
            self.assertEqual(memory.extract(base + 0x4A, '>u2'), slot + 1)
            self.assertEqual(memory.extract(base + 0x4C, '>u2'), slot + 2)
            self.assertEqual(memory.extract(base + 0x4E, '>u4'), 0)

    def test_reset_rebuilds_every_collision_coordinate_and_inverse_sort_index(self):
        env = reset_env(0)
        memory = env.data.memory
        init_full_team_defense(env)
        for horizontal in (False, True):
            memory.assign(0xFFC2EC, '|u1', 0x80 if horizontal else 0)
            if horizontal:
                _rebuild_object_order(memory)
            order = [memory.extract(0xFFB88A + index, '|u1') // 2 for index in range(16)]
            self.assertEqual(set(order), set(range(16)))
            coordinates = []
            for index, slot in enumerate(order):
                self.assertEqual(memory.extract(0xFFB86A + slot * 2, '>u2'), index)
                actual = memory.extract(0xFFB04A + slot * 0x80 + (0 if horizontal else 0x14), '>i2')
                self.assertEqual(memory.extract(0xFFB84A + slot * 2, '>i2'), actual)
                coordinates.append(actual)
            self.assertEqual(coordinates, sorted(coordinates))

    def test_stationary_puck_has_no_stale_goal_crossings_or_expired_timeout(self):
        env = reset_env(0)
        memory = env.data.memory
        for address in (0xFFBEE6, 0xFFBEEA):
            memory.assign(address, '>i2', 16)
            memory.assign(address + 2, '>i2', 3)
        init_full_team_defense(env)
        self.assertEqual(memory.extract(0xFFB78C, '>u2'), 120)
        for address in (0xFFBEE6, 0xFFBEEA):
            self.assertEqual(memory.extract(address, '>i2'), 0)
            self.assertEqual(memory.extract(address + 2, '>i2'), -1)

    def test_second_controller_selection_and_flags_are_preserved(self):
        for controlled in (6, 11):
            with self.subTest(controlled=controlled):
                env = reset_env(2)
                memory = env.data.memory
                memory.assign(0xFFC322, '>i2', controlled)
                memory.assign(0xFFC32A, '>u2', 2)
                init_full_team_defense(env)
                self.assertEqual(memory.extract(0xFFC322, '>i2'), controlled)
                for slot in range(6, 12):
                    flags = memory.extract(0xFFB04A + slot * 0x80 + 0x62, '|u1')
                    self.assertEqual(bool(flags & 8), slot == controlled)

    def test_invalid_base_save_fails_before_any_writes(self):
        for address, kind, value in (
            (0xFFC466, '>u2', 1), (0xFFC468, '>u2', 200), (0xFFC328, '>u2', 2),
            (0xFFC320, '>i2', 5), (0xFFB7AA, '>i2', -256),
            (0xFFB04A + 3 * 0x80 + 0x34, '>i2', -1),
            (0xFFB04A + 5 * 0x80 + 0x34, '>i2', 4),
        ):
            with self.subTest(address=address, value=value):
                env = reset_env(0)
                env.data.memory.assign(address, kind, value)
                before = bytes(env.data.memory.buffer)
                with self.assertRaisesRegex(ValueError, 'live first-period 5-on-5'):
                    init_full_team_defense(env)
                self.assertEqual(env.data.memory.buffer, before)


if __name__ == '__main__':
    unittest.main()
