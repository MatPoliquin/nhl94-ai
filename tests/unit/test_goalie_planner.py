"""Planner ownership, bounded work, uncertainty and independent physics contracts."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.agents.goalie import GoalieController, goalie_target
from nhl94_ai.agents.goalie_physics import KeeperMotion, PuckFlight, contact_margin
from nhl94_ai.agents.goalie_planner import MAX_STEPS, MODES, PlanningGoalie, search
from nhl94_ai.game.constants import GameConsts as Buttons
from tests.support.defense import goalie_state


class PhysicsTests(unittest.TestCase):
    def test_manual_neutral_stops_without_reversing_and_keeps_integration_order(self):
        model = KeeperMotion(0, -246*65536, 3000, -1000, 50, 8000**2)
        result = model.step()
        self.assertEqual(result.vy, 0)
        self.assertGreater(result.x, model.x)
        self.assertGreaterEqual(result.vx, 0)
        self.assertEqual(result.step().vx, 0)
        self.assertEqual(model.vx, 3000)

    def test_airborne_puck_does_not_receive_ice_friction(self):
        air = PuckFlight(0, 0, 10*65536, 1000, -2000, 0).step()
        ice = PuckFlight(0, 0, 0, 1000, -2000, 0).step()
        self.assertEqual((air.vx, air.vy), (1000, -2000))
        self.assertLess(abs(ice.vy), abs(air.vy))
        self.assertLess(air.vz, 0)

    def test_contact_distinguishes_one_timer_radius_and_height(self):
        keeper = KeeperMotion(0, 0, 0, 0, 50, 8000**2)
        puck = PuckFlight(14*65536, 0, 0, 0, 0, 0)
        self.assertGreater(contact_margin(keeper, puck), 0)
        self.assertLess(contact_margin(keeper, puck, one_timer=True), 0)
        self.assertLess(contact_margin(keeper, PuckFlight(0, 0, 16*65536, 0, 0, 0)), 0)

    def test_depleted_goalie_has_lower_native_speed_limit(self):
        state = goalie_state(True)
        full = KeeperMotion.observe(state.team1.goalie)
        state.team1.goalie.energy = 1024
        tired = KeeperMotion.observe(state.team1.goalie)
        self.assertLess(tired.speed_squared, full.speed_squared)
        self.assertEqual(tired.acceleration, full.acceleration)


class PlannerTests(unittest.TestCase):
    def test_explicit_classic_and_planner_require_supported_controls(self):
        self.assertIs(type(ClassicAIV1Model(SimpleNamespace(
            goalie_policy='selective', goalie_positioning='classic')).goalie), GoalieController)
        for mode in MODES:
            agent = ClassicAIV1Model(SimpleNamespace(goalie_policy='selective', goalie_planner=mode))
            self.assertIsInstance(agent.goalie, PlanningGoalie)
        for settings in ({'goalie_policy': 'off'}, {'goalie_decision_interval': 4},
                         {'goalie_save_timing': 'contact'}, {'classic_control_interval': 4}):
            args = {'goalie_policy': 'selective', 'goalie_planner': 'search-24', **settings}
            with self.assertRaises(ValueError):
                ClassicAIV1Model(SimpleNamespace(**args))

    def test_search_is_deterministic_bounded_and_does_not_modify_observation(self):
        state = goalie_state(True)
        state.puck.y, state.puck.x, state.puck.height = -190, 14, 0
        state.puck.motion_y, state.puck.motion_z = -4, 0
        keeper, puck = KeeperMotion.observe(state.team1.goalie), PuckFlight.observe(state.puck)
        for horizon in (12, 24, 48):
            result = search(keeper, puck, True, -1, 0, horizon=horizon)
            self.assertEqual(result, search(keeper, puck, True, -1, 0, horizon=horizon))
            self.assertLessEqual(result[1]['steps'], MAX_STEPS)
            self.assertLessEqual(result[1]['horizon'], horizon)
        self.assertEqual(state.puck.y, -190)

    def test_preview_does_not_record_save_that_search_replaces(self):
        state = goalie_state(True)
        state.engine.puck_owner = -256
        state.puck.x, state.puck.y = 14, -215
        state.puck.motion_y, state.puck.motion_z, state.puck.height = -4, 0, 0
        state.team1.goalie.motion_x = -1.5
        for team in (state.team1, state.team2):
            for player in team.players:
                player.x, player.y = 100, 100
                player.motion_x = player.motion_y = 0
        manager = PlanningGoalie()
        plan = goalie_target(state)
        baseline = GoalieController('selective')
        self.assertTrue(baseline._position(state, plan)[Buttons.INPUT_C])
        with patch('nhl94_ai.agents.goalie_planner.search', return_value=(2, {'status': 'test'})):
            action = manager._position(state, plan)
        self.assertTrue(action[Buttons.INPUT_RIGHT])
        self.assertIsNone(manager.pending_save)
        self.assertEqual(manager.save_at, 0)
        self.assertEqual(manager.metrics['save_requests'], 0)

    def test_unsupported_state_uses_existing_action_and_lifecycle(self):
        state = goalie_state(True)
        state.puck.motion_z = None
        original, planner = GoalieController('selective'), PlanningGoalie()
        action = original.step(deepcopy(state))
        np.testing.assert_array_equal(action, planner.step(deepcopy(state)))
        self.assertEqual(original.phase, planner.phase)


if __name__ == '__main__':
    unittest.main()
