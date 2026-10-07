"""Independent momentum encodings and safe-goal velocity-map summaries."""
from types import SimpleNamespace
import unittest

from nhl94_ai.agents.motion import VELOCITY_SCALE
from nhl94_ai.evaluation.cross_crease_velocity import build_parser, safe_summary, summarize_grid, validate_grid
from nhl94_ai.tasks.cross_crease_setup import OBJECT_BASE, OBJECT_STRIDE, set_crossing_velocity
from tests.ram_fixture import FixtureMemory


class MomentumEncodingTests(unittest.TestCase):
    def test_both_axes_and_ends_preserve_fixed_current_previous_geometry(self):
        for carrier in (1, 7):
            for direction in (-1, 1):
                memory = FixtureMemory({})
                env = SimpleNamespace(data=SimpleNamespace(memory=memory))
                sign = 1 if carrier < 6 else -1
                actual = set_crossing_velocity(env, carrier, direction=direction, lateral=1.5, goalward=-0.25)
                self.assertLessEqual(abs(actual[0] - direction * 1.5), VELOCITY_SCALE / 2)
                self.assertLessEqual(abs(actual[1] + sign * 0.25), VELOCITY_SCALE / 2)
                for slot, x in ((carrier, -direction * 35), (14, -direction * 25)):
                    base = OBJECT_BASE + slot * OBJECT_STRIDE
                    self.assertEqual(memory.extract(base, '>i4'), x * 65536)
                    self.assertEqual(memory.extract(base + 0x1C, '>i4'), x * 65536)
                    self.assertEqual(memory.extract(base + 0x14, '>i4'), sign * 216 * 65536)
                    self.assertEqual(memory.extract(base + 0x20, '>i4'), sign * 216 * 65536)
                    self.assertEqual(memory.extract(base + 0x28, '>i2') * VELOCITY_SCALE, actual[0])
                    self.assertEqual(memory.extract(base + 0x2A, '>i2') * VELOCITY_SCALE, actual[1])

    def test_changing_velocity_does_not_change_positions_or_unrelated_actor_state(self):
        memory = FixtureMemory({})
        env = SimpleNamespace(data=SimpleNamespace(memory=memory))
        goalie = OBJECT_BASE + 11 * OBJECT_STRIDE
        memory.assign(goalie + 0x68, '|u1', 30)
        poses = []
        for lateral, goalward in ((0, 0), (0.5, -1), (3, 1)):
            set_crossing_velocity(env, 1, direction=1, lateral=lateral, goalward=goalward)
            poses.append(tuple(memory.extract(OBJECT_BASE + slot * OBJECT_STRIDE + offset, '>i4')
                               for slot in (1, 14) for offset in (0, 0x14, 0x1C, 0x20)))
            self.assertEqual(memory.extract(goalie + 0x68, '|u1'), 30)
        self.assertEqual(len(set(poses)), 1)

    def test_invalid_grid_momentum_has_explicit_errors(self):
        for lateral, goalward in ((-0.1, 0), (3.1, 0), (1, 1.1), (float('nan'), 0), (1, float('inf'))):
            with self.assertRaises(ValueError):
                set_crossing_velocity(None, 1, direction=1, lateral=lateral, goalward=goalward)
        with self.assertRaises(ValueError):
            set_crossing_velocity(None, 11, direction=1, lateral=1, goalward=0)


class VelocityMapTests(unittest.TestCase):
    def test_grid_rejects_duplicate_nonfinite_and_out_of_range_values(self):
        parser = build_parser()
        for field, values in (('lateral', [1, 1]), ('lateral', [float('nan')]),
                              ('goalward', [1.1]), ('goalward', [])):
            args = parser.parse_args([])
            setattr(args, field, values)
            with self.assertRaises(ValueError):
                validate_grid(args)

    def test_contact_goal_is_not_a_safe_goal_or_reliable_cell(self):
        rows = [
            dict(goal=True, shot=True, windup_frame=1, release_frame=37, commit_frame=3,
                 goalie_contact_impulses=1, controller_metrics={'attempts': 1}, outcome='goal'),
            dict(goal=True, shot=True, windup_frame=1, release_frame=37, commit_frame=3,
                 goalie_contact_impulses=0, controller_metrics={'attempts': 1}, outcome='goal'),
        ]
        summary = safe_summary(rows)
        self.assertEqual(summary['goals'], 2)
        self.assertEqual(summary['safe_goals'], 1)
        self.assertEqual(summary['contact_trials'], 1)
        self.assertEqual(summary['crossing_goals'], 2)
        self.assertFalse(summary['all_observed_trials_scored_safely'])
        self.assertTrue(safe_summary(rows[1:])['all_observed_trials_scored_safely'])

    def test_attempted_c_and_native_windup_are_not_goals(self):
        row = dict(goal=False, shot=False, windup_frame=1, release_frame=None, commit_frame=None,
                   first_c=0, c_frames=24, goalie_contact_impulses=0,
                   controller_metrics={'attempts': 1}, outcome='unreleased')
        summary = safe_summary([row])
        self.assertEqual(summary['crossing_attempts'], 1)
        self.assertEqual(summary['accepted_windups'], 1)
        self.assertEqual(summary['safe_goals'], 0)
        self.assertEqual(summary['crossing_goals'], 0)
        self.assertFalse(summary['all_observed_trials_scored_safely'])

    def test_cells_separate_lateral_and_goalward_axes_and_profiles(self):
        args = build_parser().parse_args(['--lateral', '1', '2', '--goalward', '-0.5', '0.5',
                                          '--goalies', 'high', '--policies', 'held'])
        rows = [dict(goal=True, shot=True, windup_frame=1, release_frame=37, commit_frame=3,
                     goalie_contact_impulses=0, controller_metrics={}, outcome='goal',
                     lateral=1, goalward=-0.5, goalie_rating='high', policy='held')]
        grid = summarize_grid(rows, args)
        self.assertEqual([(x['lateral'], x['goalward']) for x in grid],
                         [(1, -0.5), (2, -0.5), (1, 0.5), (2, 0.5)])
        self.assertEqual(grid[0]['results']['high']['held']['safe_goals'], 1)
        self.assertEqual(grid[1]['results']['high']['held']['attempts'], 0)


if __name__ == '__main__':
    unittest.main()
