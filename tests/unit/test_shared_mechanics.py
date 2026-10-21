"""Physical ownership and forecast isolation across tactical consumers."""
import ast
from copy import deepcopy
from pathlib import Path
import unittest

from nhl94_ai.game.actors import controllable_skater, observable_body
from nhl94_ai.game.forecast import clone_forecast, project_body
from tests.support.offense import offense_state


class SharedMechanicsContracts(unittest.TestCase):
    def test_observable_goalie_is_distinct_from_legacy_controllable_skater(self):
        goalie = offense_state().team1.goalie
        goalie.role = 0
        self.assertTrue(observable_body(goalie))
        self.assertFalse(controllable_skater(goalie))
        goalie.role = -1
        self.assertFalse(observable_body(goalie))

    def test_forecast_bodies_are_isolated_and_precise_positions_follow_projection(self):
        state = offense_state()
        before = deepcopy(state)
        future = clone_forecast(state)
        self.assertIs(future.engine, state.engine)
        for old, new in ((state.team1, future.team1), (state.team2, future.team2)):
            self.assertIsNot(old, new)
            for source, actor in zip((*old.players, old.goalie), (*new.players, new.goalie)):
                self.assertIsNot(source, actor)
                project_body(actor, 4, boards=actor is not new.goalie)
                self.assertEqual((actor.precise_x, actor.precise_y), (float(actor.x), float(actor.y)))
        future.puck.x += 50
        self.assertIsNot(future.puck, state.puck)
        for old, new in ((before.team1, state.team1), (before.team2, state.team2)):
            for a, b in zip((*old.players, old.goalie), (*new.players, new.goalie)):
                self.assertEqual(vars(a), vars(b))
        self.assertEqual(vars(before.puck), vars(state.puck))

    def test_platform_and_physical_modules_do_not_import_tactics(self):
        root = Path(__file__).resolve().parents[2] / 'nhl94_ai/game'
        paths = [*root.joinpath('platforms').glob('*.py'),
                 *(root / name for name in ('actors.py', 'forecast.py', 'motion.py',
                                            'passing.py', 'shooting.py', 'skating.py'))]
        for path in paths:
            for node in ast.walk(ast.parse(path.read_text())):
                imports = ([node.module] if isinstance(node, ast.ImportFrom) else
                           [alias.name for alias in node.names] if isinstance(node, ast.Import) else [])
                self.assertFalse(any(name and name.startswith('nhl94_ai.agents') for name in imports), path)
