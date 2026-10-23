"""Native period numbering and physical goal ends through controller views."""
import json
from pathlib import Path
import unittest

from nhl94_ai.agents.goalie import goalie_target
from nhl94_ai.evaluation.benchmark import away_view
from nhl94_ai.game.state import NHL94GameState


class PeriodDirectionTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / 'fixtures/NHL94-Genesis-v0.json'
        self.info = json.loads(path.read_text(encoding='utf-8'))
        self.state = NHL94GameState(5)

    def test_native_zero_based_periods_change_ends_and_reset(self):
        for native, home_y in ((0, -264), (1, 264), (2, -264), (3, 264), (0, -264)):
            with self.subTest(native=native):
                self.state.BeginFrame({**self.info, 'period': native}, [0] * 6)
                self.assertEqual(self.state.period, native + 1)
                self.assertEqual(self.state.team1.net.y, home_y)
                self.assertEqual(self.state.team2.net.y, -home_y)
                for team in (self.state.team1, self.state.team2):
                    player = team.get_controlled_player()
                    self.assertEqual(team.net.rel_controlled_y, team.net.y - player.y)

    def test_missing_period_defaults_to_first_period(self):
        for native in (None, 0):
            self.state.BeginFrame({**self.info, 'period': native}, [0] * 6)
            self.assertEqual(self.state.period, 1)
            self.assertEqual(self.state.team1.net.y, -264)
        self.state.BeginFrame({k: v for k, v in self.info.items() if k != 'period'}, [0] * 6)
        self.assertEqual(self.state.period, 1)
        self.assertEqual(self.state.team1.net.y, -264)

    def test_recomputing_away_view_keeps_the_physical_team_goal(self):
        for native, away_y in ((0, 264), (1, -264), (2, 264)):
            self.state.BeginFrame({**self.info, 'period': native}, [0] * 6)
            view = away_view(self.state)
            view.update_nets()
            self.assertEqual(view.team1.controller, 2)
            self.assertEqual(view.team1.net.y, away_y)
            self.assertEqual(self.state.team1.net.y, -away_y)

    def test_goalie_plan_defends_the_current_end_on_both_sides(self):
        for native in (0, 1, 2):
            self.state.BeginFrame({**self.info, 'period': native}, [0] * 6)
            for side in (1, 2):
                with self.subTest(native=native, side=side):
                    view = self.state if side == 1 else away_view(self.state)
                    sign = (-1 if side == 1 else 1) * (-1 if native % 2 else 1)
                    view.team1.goalie.x, view.team1.goalie.y = 0, sign * 246
                    view.puck.x, view.puck.y = 0, sign * 200
                    view.puck.height = 0
                    view.puck.motion_x = view.puck.motion_y = view.puck.motion_z = 0
                    plan = goalie_target(view)
                    self.assertGreaterEqual(plan.target[1] * sign, 236)
                    self.assertLessEqual(plan.target[1] * sign, 254)


if __name__ == '__main__':
    unittest.main()
