"""Common-time support forecasts and read-only opportunity diagnostics."""
from copy import deepcopy
import unittest

import numpy as np

from nhl94_ai.agents.teammates import TeammateCoordinator
from nhl94_ai.agents.teammate_support import SupportPlanner, support_forecast
from nhl94_ai.evaluation.teammate_opportunities import opportunity_diagnostics
from nhl94_ai.game.ram import teammate_controller_info
from nhl94_ai.game.state import NHL94GameState
from tests.support.teammates import team_args, team_info


class SupportForecastTests(unittest.TestCase):
    @staticmethod
    def state():
        state = NHL94GameState(5)
        state.BeginFrame(teammate_controller_info(team_info()), [0] * 6)
        for team in (state.team1, state.team2):
            for player in [*team.players, team.goalie]:
                player.weight, player.agility, player.speed, player.energy = 56, 20, 20, 30
                player.motion_x, player.motion_y = 1.5, 1.0
                player.steering = 8
                player.precise_x, player.precise_y = float(player.x), float(player.y)
        return state

    def test_receiver_reaches_partial_target_with_real_velocity_and_all_actors_advance(self):
        state = self.state()
        before = deepcopy(state)
        support = state.team1.players[1]
        target = (support.x - 60, support.y + 80)
        future = support_forecast(state, 1, target)
        receiver = future.team1.players[1]
        self.assertNotEqual((receiver.x, receiver.y), target)
        self.assertGreater(abs(receiver.motion_x) + abs(receiver.motion_y), 0)
        for original_team, team in ((state.team1, future.team1), (state.team2, future.team2)):
            for original, projected in zip([*original_team.players, original_team.goalie],
                                           [*team.players, team.goalie]):
                self.assertIsNot(original, projected)
                self.assertNotEqual((original.x, original.y), (projected.x, projected.y))
                self.assertAlmostEqual(projected.x, projected.precise_x, delta=1)
        for axis in ('x', 'y'):
            self.assertEqual(getattr(future.puck, axis) - getattr(state.puck, axis),
                             getattr(future.team1.players[0], axis) - getattr(state.team1.players[0], axis))
        self.assertEqual(vars(state.team1.players[1]), vars(before.team1.players[1]))
        self.assertEqual(vars(state.puck), vars(before.puck))

    def test_coast_and_brake_preserve_distinct_reachable_motion(self):
        state = self.state()
        support = state.team1.players[1]
        support.motion_x, support.motion_y = 0, 3
        support.facing_phase, support.facing = 0, 0
        target = (support.x, support.y)
        coast = support_forecast(state, 1, target, coast=True).team1.players[1]
        brake = support_forecast(state, 1, target).team1.players[1]
        self.assertNotEqual((coast.x, coast.y, coast.motion_x, coast.motion_y),
                            (brake.x, brake.y, brake.motion_x, brake.motion_y))
        self.assertGreater(coast.motion_y, 0)

    def test_ineligible_receiver_immediately_invalidates_persistent_target(self):
        state = self.state()
        planner = SupportPlanner()
        planner.choose(state, 1, 1, 0)
        state.team1.players[1].unavailable |= 4
        _, role = planner.choose(state, 1, 2, 0)
        self.assertEqual(role, 'coverage')
        self.assertIsNone(planner.key)
        self.assertFalse(planner.coast)
        state.team1.players[1].unavailable = 0
        state.engine.puck_owner = 7
        _, role = planner.choose(state, 1, 3, 0)
        self.assertEqual(role, 'coverage')
        self.assertNotEqual(planner.diagnostics['phase'], 'unavailable')

    def test_diagnostics_preserve_next_decision_and_expose_candidate_population(self):
        agent = TeammateCoordinator(team_args())
        info = team_info()
        for _ in range(8):
            result = agent.act(info)
        control = deepcopy(agent)
        snapshot = deepcopy(result.diagnostics['teammates'])
        diagnostics = opportunity_diagnostics(agent.states[0], snapshot, np.zeros((2, 12)))
        self.assertEqual(len(diagnostics['candidates']), 8)
        self.assertEqual(snapshot, result.diagnostics['teammates'])
        np.testing.assert_array_equal(agent.act(info).action, control.act(info).action)
        self.assertEqual(agent.last_diagnostics, control.last_diagnostics)


if __name__ == '__main__':
    unittest.main()
