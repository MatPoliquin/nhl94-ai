"""Target execution and opt-in wrapper contracts, without a ROM."""
import copy
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from nhl94_ai.env.observation import NHL94Observation2PEnv
from nhl94_ai.env.target_control import (
    CONTROLLER_FIELDS, SETTINGS, TargetPositionController,
    _clear_segment, project_target, route_waypoint, validate_target,
)
from nhl94_ai.env.wrappers import StochasticFrameSkip
from nhl94_ai.game.constants import GameConsts as Buttons
from nhl94_ai.game.ram import register_target_state
from tests.unit.test_environment_contracts import FixtureEnv, wrapped


class TargetFixtureEnv(FixtureEnv):
    def __init__(self):
        super().__init__()
        self.info.update(puck_owner=6, period=0, time=274)
        self.actions = []

    def _register(self, name, definition):
        memory = self.data.memory
        memory.fields[name] = definition['address'], definition['type']
        self.info[name] = memory.extract(definition['address'], definition['type'])

    def reset(self, **kwargs):
        result = super().reset(**kwargs)
        self.data.set_variable = self._register
        self.data.lookup_all = self.info.copy
        register_target_state(self)
        self.actions = []
        return result

    def step(self, action):
        self.actions.append(np.asarray(action).copy())
        return super().step(action)


def target_env(task_name='DefenseZone'):
    args = SimpleNamespace(env='NHL94-Genesis-v0', nn='MlpPolicy', rf=task_name,
                           action_type='TARGET_POSITION', seq_len=16, alg='ppo2', num_players=1,
                           hyperparams_dict={})
    return NHL94Observation2PEnv(TargetFixtureEnv(), args, 1, task_name)


def control_frame():
    env = target_env()
    env.reset(seed=0)
    state, info = env.game_state, env.target_info
    for index, player in enumerate(state.team1.players):
        player.x, player.y = -100 + index * 40, 180
        player.motion_x = player.motion_y = 0.0
        player.is_falling = 0.0
        info[f'target_flags_{index}'] = 0
        info[f'target_unavailable_{index}'] = 0
        info[f'target_facing_{index}'] = 0
    state.team1.players[1].x = state.team1.players[1].y = 0
    state.puck.x, state.puck.y = 100, -100
    state.action = [0] * 6
    # Mechanics tests use unrestricted ice; wrapper tests retain DefenseZone bounds.
    env.target_controller = TargetPositionController()
    env.target_controller.observe(state, info)
    return env, state, info


class TargetGeometryTests(unittest.TestCase):
    def test_continuous_values_are_preserved_and_invalid_actions_fail(self):
        np.testing.assert_array_equal(validate_target([0.25, -0.125]), [0.25, -0.125])
        for value in ([0], [0, 0, 0], [[0, 0]], [np.nan, 0], [0, np.inf], [1.01, 0]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_target(value)

    def test_projection_is_bounded_idempotent_and_outside_nets(self):
        for x in np.linspace(-150, 150, 25):
            for y in np.linspace(-310, 310, 25):
                point = project_target((x, y))
                self.assertLessEqual(abs(point[0]), 120)
                self.assertLessEqual(abs(point[1]), 270)
                self.assertFalse(abs(point[0]) < 28 and abs(point[1]) > 254)
                np.testing.assert_allclose(project_target(point), point)
        self.assertEqual(project_target((0, -180)), (0, -180))

    def test_routes_do_not_cross_nets_and_targets_are_not_changed(self):
        for sign in (-1, 1):
            start, target = (-60, sign * 265), (60, sign * 265)
            for _ in range(4):
                waypoint = route_waypoint(start, target)
                self.assertTrue(_clear_segment(start, waypoint))
                start = waypoint
                if start == target:
                    break
            self.assertEqual(start, target)
        self.assertEqual(route_waypoint((0, 0), (40, 40)), (40, 40))
        self.assertEqual(route_waypoint((0, 260), (0, 0)), (0, 254))


class TargetControllerTests(unittest.TestCase):
    def setUp(self):
        self.env, self.state, self.info = control_frame()
        self.addCleanup(self.env.close)
        self.controller = self.env.target_controller

    def step(self, target):
        action = self.controller.step(target, self.state, self.info)
        self.state.action = [bool(action[index]) for index in (4, 5, 6, 7, 0, 8)]
        return action

    def test_target_replacement_does_not_reset_cooldowns(self):
        self.controller.boost_cooldown = 20
        self.step([0.2, 0])
        first = self.controller.destination
        self.step([-0.2, 0])
        self.assertEqual(first, (24.0, 0.0))
        self.assertEqual(self.controller.destination, (-24.0, 0.0))
        self.assertEqual(self.controller.boost_cooldown, 18)

    def test_steering_brakes_and_holds_without_chasing_puck(self):
        player = self.state.team1.players[1]
        self.assertTrue(self.step([0.25, 0])[Buttons.INPUT_RIGHT])
        player.x, player.motion_x = 29, 1.5
        action = self.step([0.25, 0])
        self.assertTrue(action[Buttons.INPUT_LEFT])
        self.assertEqual(self.controller.mode, 'braking')
        player.x, player.motion_x = 30, 0
        self.assertFalse(np.any(self.step([0.25, 0])))
        self.assertEqual(self.controller.mode, 'holding')
        self.assertEqual(self.controller.destination, (30.0, 0.0))

    def test_boost_requires_correct_facing_and_release(self):
        self.info['target_facing_1'] = 4  # Facing down, target above.
        self.assertFalse(self.step([0, 0.3])[Buttons.INPUT_C])
        self.info['target_facing_1'] = 0
        self.assertTrue(self.step([0, 0.3])[Buttons.INPUT_C])
        for _ in range(SETTINGS.boost_cooldown - 1):
            self.assertFalse(self.step([0, 0.3])[Buttons.INPUT_C])
        self.assertTrue(self.step([0, 0.3])[Buttons.INPUT_C])

    def test_poke_is_local_and_does_not_override_target(self):
        self.state.puck.x, self.state.puck.y = 0, 12
        self.assertTrue(self.step([0.25, 0])[Buttons.INPUT_B])
        self.assertEqual(self.controller.mode, 'poke-request')
        self.assertEqual(self.controller.destination, (30.0, 0.0))
        self.assertFalse(self.step([0.25, 0])[Buttons.INPUT_B])
        self.controller.poke_cooldown = 0
        self.state.puck.y = -12
        self.assertFalse(self.step([0.25, 0])[Buttons.INPUT_B])

    def test_possession_suppresses_defensive_buttons(self):
        for owner in (0, 1, 5):
            self.state.engine.puck_owner = owner
            self.assertFalse(np.any(self.step([0.8, 0.8])))
            self.assertEqual(self.controller.mode, 'possession')

    def test_switches_are_requests_not_assumed_successes_and_are_bounded(self):
        self.state.team1.players[0].x, self.state.team1.players[0].y = 90, 0
        self.state.puck.x, self.state.puck.y = 90, 0
        self.assertTrue(self.step([0.8, 0])[Buttons.INPUT_B])
        self.assertEqual(self.controller.actual_slot, 1)
        self.assertEqual(self.controller.desired_slot, 0)
        presses = 1
        for _ in range(SETTINGS.switch_cooldown * 4):
            presses += int(self.step([0.8, 0])[Buttons.INPUT_B])
        self.assertEqual(presses, SETTINGS.max_switch_attempts)
        self.info['p1_control_slot'] = 0
        diagnostics = self.controller.observe(self.state, self.info)
        self.assertEqual(diagnostics['actual_slot'], 0)
        self.assertEqual(diagnostics['switches'], 1)
        self.assertEqual(self.controller.switch_attempts, 0)

    def test_does_not_switch_when_rom_likely_selects_wrong_player(self):
        self.state.team1.players[0].x, self.state.team1.players[0].y = 90, 0
        self.state.puck.x, self.state.puck.y = self.state.team1.players[4].x, self.state.team1.players[4].y
        self.assertFalse(self.step([0.8, 0])[Buttons.INPUT_B])
        self.assertEqual(self.controller.desired_slot, 0)

    def test_ineligible_skaters_and_goalie_are_not_steered(self):
        self.state.team1.players[0].x, self.state.team1.players[0].y = 90, 0
        self.info['target_unavailable_0'] = 4
        self.step([0.8, 0])
        self.assertNotEqual(self.controller.desired_slot, 0)
        self.info['p1_control_slot'] = 5
        self.controller.observe(self.state, self.info)
        action = self.step([0.8, 0])
        self.assertFalse(np.any(action[4:8]))
        self.assertFalse(action[Buttons.INPUT_C])

    def test_small_changes_do_not_cause_switch_thrashing(self):
        self.state.team1.players[0].x, self.state.team1.players[0].y = 1, 0
        for x in (0.1, 0.09, 0.11, 0.08):
            self.assertFalse(self.step([x, 0])[Buttons.INPUT_B])
            self.assertEqual(self.controller.desired_slot, 1)

    def test_reset_clears_all_policy_relevant_state(self):
        self.step([0.25, 0])
        self.controller.reset()
        np.testing.assert_array_equal(self.controller.observation(), np.zeros(len(CONTROLLER_FIELDS)))
        self.assertIsNone(self.controller.actual_slot)
        with self.assertRaisesRegex(RuntimeError, 'reset frame'):
            self.step([0, 0])


class TargetEnvironmentTests(unittest.TestCase):
    def test_shape_prefix_reset_and_fractional_actions(self):
        env = target_env()
        self.addCleanup(env.close)
        observation, info = env.reset(seed=0)
        self.assertEqual(observation.shape, (322,))
        self.assertEqual(env.action_space.shape, (2,))
        np.testing.assert_array_equal(observation[:310], np.asarray(env.encoder.encode(env.game_state), dtype=np.float32))
        np.testing.assert_array_equal(observation[-12:], np.zeros(12))
        self.assertEqual(info['target_control']['actual_slot'], 1)
        observation, _, _, _, info = env.step(np.array([0.125, -0.25], dtype=np.float32))
        self.assertEqual(info['target_control']['target'], [15.0, -201.75])
        np.testing.assert_array_equal(observation[-12:-9], [1, 0.125, -0.25])
        self.assertEqual(env.env.actions[-1].shape, (12,))
        env.reset(seed=0)
        self.assertIsNone(env.target_controller.target)

    def test_controller_runs_every_frame_while_target_updates_every_decision(self):
        inner = target_env()
        env = StochasticFrameSkip(inner, 4, -1)
        self.addCleanup(env.close)
        env.reset(seed=0)
        for target in ([0.125, 0.25], [-0.25, -0.125]):
            _, _, _, _, info = env.step(target)
            self.assertEqual(info['target_control']['target'], [target[0] * 120, -179 + target[1] * 91])
        self.assertEqual(inner.target_controller.frames, 8)
        self.assertEqual(len(inner.env.actions), 9)

    def test_authoritative_control_overrides_stale_stars_only_in_target_mode(self):
        env = target_env()
        self.addCleanup(env.close)
        env.reset(seed=0)
        env.env.info['p1_control_slot'] = 3
        _, _, _, _, info = env.step([0, 0])
        self.assertEqual(env.game_state.team1.control, 4)
        self.assertEqual(info['target_control']['actual_slot'], 3)
        legacy = wrapped()
        self.addCleanup(legacy.close)
        observation, _ = legacy.reset()
        self.assertEqual(len(observation), 310)
        self.assertIsNone(legacy.target_controller)

    def test_shots_end_target_episodes_and_override_same_frame_outcomes(self):
        for frame_skip in (1, 4, 10):
            for outcome in ('shot', 'save', 'recovery', 'goal'):
                with self.subTest(frame_skip=frame_skip, outcome=outcome):
                    inner = target_env()
                    env = StochasticFrameSkip(inner, frame_skip, -1)
                    self.addCleanup(env.close)
                    env.reset(seed=0)
                    inner.env.info.update(puck_owner=-256)
                    inner.env.info['p2_shots'] += 1
                    if outcome == 'recovery':
                        inner.env.info['puck_owner'] = 0
                    elif outcome == 'save':
                        inner.env.info['puck_owner'] = 5
                    elif outcome == 'goal':
                        inner.env.info['p2_score'] += 1
                    _, reward, done, _, info = env.step([0, 0])
                    self.assertEqual(reward, -1.0)
                    self.assertTrue(done)
                    self.assertEqual(info['target_control']['frame'], 1)

    def test_shot_during_repeated_action_stops_on_its_exact_frame(self):
        for action_type in ('FILTERED', 'TARGET_POSITION'):
            for frame_skip in (1, 4, 10):
                with self.subTest(action_type=action_type, frame_skip=frame_skip):
                    inner = target_env() if action_type == 'TARGET_POSITION' else wrapped(rf='DefenseZone')
                    env = StochasticFrameSkip(inner, frame_skip, -1)
                    self.addCleanup(env.close)
                    env.reset(seed=0)
                    delay = max(1, frame_skip // 2)
                    shot_frame = inner.env.steps + delay

                    def step(action, base=inner.env, original_step=inner.env.step, shot_at=shot_frame):
                        if base.steps + 1 == shot_at:
                            base.info['p2_shots'] += 1
                        return original_step(action)

                    action = [0, 0] if action_type == 'TARGET_POSITION' else np.zeros(12, dtype=np.int8)
                    with patch.object(inner.env, 'step', side_effect=step):
                        _, reward, done, _, info = env.step(action)
                    self.assertEqual(reward, -1.0)
                    self.assertTrue(done)
                    self.assertEqual(inner.env.steps, shot_frame)
                    if action_type == 'TARGET_POSITION':
                        self.assertEqual(info['target_control']['frame'], delay)

    def test_goalie_possession_continues_with_or_without_selected_player(self):
        for selected in (-1, 1, 5):
            with self.subTest(selected=selected):
                inner = target_env()
                env = StochasticFrameSkip(inner, 4, -1)
                self.addCleanup(env.close)
                env.reset(seed=0)
                inner.env.info.update(puck_owner=5, p1_control_slot=selected)
                for frame in (4, 8):
                    observation, reward, done, _, info = env.step([0, 0])
                    self.assertEqual(reward, 0.0)
                    self.assertFalse(done)
                    self.assertEqual(info['target_control']['frame'], frame)
                    self.assertEqual(info['target_control']['mode'], 'possession')
                    self.assertFalse(np.any(info['target_control']['buttons']))
                    self.assertTrue(np.all(np.isfinite(observation)))
                inner.env.info['puck_owner'] = 2
                _, reward, done, _, info = env.step([0, 0])
                self.assertEqual(reward, 1.0)
                self.assertTrue(done)
                self.assertEqual(info['target_control']['frame'], 9)

    def test_missing_ram_fails_explicitly(self):
        env, state, info = control_frame()
        self.addCleanup(env.close)
        del info['p1_live_vel_x']
        with self.assertRaisesRegex(ValueError, 'authoritative RAM'):
            env.target_controller.observe(state, info)

    def test_target_frame_skip_stops_at_truncation(self):
        inner = target_env()
        inner.env.truncate = True
        env = StochasticFrameSkip(inner, 4, -1, stop_on_truncation=True)
        self.addCleanup(env.close)
        env.reset(seed=0)
        _, _, _, truncated, info = env.step([0, 0])
        self.assertTrue(truncated)
        self.assertEqual(info['target_control']['frame'], 1)

    def test_target_observations_do_not_report_stale_star_possession(self):
        env = target_env()
        self.addCleanup(env.close)
        env.reset(seed=0)
        info = env.env.info
        info.update(puck_owner=-256, fullstar_x=info['p1_x'], fullstar_y=info['p1_y'])
        env.step([0, 0])
        self.assertFalse(env.game_state.team1.player_haspuck)
        self.assertFalse(env.game_state.team1.goalie_haspuck)
        self.assertFalse(any(player.has_puck for player in env.game_state.team1.players))


if __name__ == '__main__':
    unittest.main()
