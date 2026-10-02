"""Task-coordinate boundaries must constrain the requested target, not just execution."""
from dataclasses import replace
import unittest

import numpy as np

from nhl94_ai.artifacts import run_metadata, validate_schema
from nhl94_ai.env.target_control import (
    TargetPositionController, normalize_position, target_position, target_schema,
    validate_target_bounds,
)
from nhl94_ai.tasks.registry import TASKS, get_task, register_task
from tests.unit.test_target_control import control_frame, target_env
from tests.unit.test_target_interfaces import target_args


class TargetBoundsTests(unittest.TestCase):
    def test_defense_outputs_cover_our_zone_not_center_ice(self):
        bounds = get_task('DefenseZone').target_bounds
        self.assertEqual(bounds, (-120, 120, -270, -88))
        for action, expected in (
            ((-1, -1), (-120, -270)), ((1, 1), (120, -88)),
            ((-1, 1), (-120, -88)), ((1, -1), (120, -270)),
            ((0, 0), (0, -179)), ((0, 0.5), (0, -133.5)),
        ):
            self.assertEqual(target_position(action, bounds), expected)
            np.testing.assert_array_equal(normalize_position(expected, bounds), action)

    def test_entire_output_grid_and_projected_destinations_remain_defensive(self):
        env, state, info = control_frame()
        self.addCleanup(env.close)
        bounds = get_task('DefenseZone').target_bounds
        controller = TargetPositionController(bounds=bounds)
        controller.observe(state, info)
        for x in np.linspace(-1, 1, 25):
            for y in np.linspace(-1, 1, 25):
                controller.step([x, y], state, info)
                diagnostics = controller.diagnostics(state)
                for field in ('target', 'destination'):
                    point = diagnostics[field]
                    self.assertLessEqual(-120, point[0])
                    self.assertLessEqual(point[0], 120)
                    self.assertLessEqual(-270, point[1])
                    self.assertLessEqual(point[1], -88)
                np.testing.assert_array_equal(controller.observation()[1:3], np.asarray([x, y], dtype=np.float32))

    def test_wrapper_reports_the_same_bounds_and_world_target_as_the_controller(self):
        env = target_env()
        self.addCleanup(env.close)
        env.reset(seed=0)
        observation, _, _, _, info = env.step([0, 0])
        self.assertEqual(observation.shape, (322,))
        self.assertEqual(info['target_control']['target'], [0, -179])
        self.assertEqual(info['target_control']['destination'], (0, -179))
        self.assertEqual(info['target_control']['bounds'], (-120, 120, -270, -88))

    def test_unbounded_tasks_keep_the_original_mapping(self):
        name = 'UnboundedTargetTest'
        register_task(name, replace(get_task('DefenseZone'), target_bounds=None))
        self.addCleanup(TASKS.pop, name)
        env = target_env(name)
        self.addCleanup(env.close)
        env.reset(seed=0)
        for action, expected in (([0, 0], [0, 0]), ([0.25, 0.25], [30, 67.5])):
            _, _, _, _, info = env.step(action)
            self.assertEqual(info['target_control']['target'], expected)
            self.assertIsNone(info['target_control']['bounds'])
        schema = run_metadata(target_args(rf=name), {})['schema']['target_controller']
        self.assertEqual(schema, target_schema('defense'))
        self.assertEqual(schema['coordinates'], 'absolute-world-xy-v1')
        self.assertNotIn('bounds', schema)

    def test_invalid_bounds_fail_explicitly(self):
        for bounds in ([0, 1], [-120, 120, np.nan, 0], [-121, 120, -270, -88],
                       [-120, 120, -88, -270], [0, 0, -270, -88]):
            with self.subTest(bounds=bounds), self.assertRaisesRegex(ValueError, 'Target bounds'):
                validate_target_bounds(bounds)

    def test_old_full_rink_and_new_zone_contracts_are_not_interchangeable(self):
        bounded = run_metadata(target_args(), {})
        legacy = run_metadata(target_args(), {})
        legacy['schema']['target_controller'] = target_schema('defense')
        self.assertEqual(bounded['schema']['target_controller']['bounds'], [-120, 120, -270, -88])
        self.assertEqual(bounded['schema']['target_controller']['coordinates'], 'task-bounded-world-xy-v1')
        for actual, expected in ((bounded, legacy), (legacy, bounded)):
            with self.assertRaisesRegex(ValueError, 'target_controller'):
                validate_schema(actual, expected)


if __name__ == '__main__':
    unittest.main()
