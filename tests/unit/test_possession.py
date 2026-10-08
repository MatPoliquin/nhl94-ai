"""Default attack restoration and truthful carry uncertainty on a live fixture."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.agents.offense import OffenseController
from nhl94_ai.agents.possession import PossessionOffenseController
from nhl94_ai.evaluation.play import parse_cmdline
from nhl94_ai.evaluation.runner import build_parser, run
from nhl94_ai.game.state import NHL94GameState, Player
from tests.unit.test_classic_offense import offense_state


def retreat_state():
    path = Path(__file__).parents[1] / 'fixtures/classic-carry-retreat.json'
    data = json.loads(path.read_text(encoding='utf-8'))
    state = NHL94GameState(5)
    state.puck = Player(**data['puck'])
    for key, value in data['engine'].items():
        setattr(state.engine, key, value)
    for name in ('team1', 'team2'):
        team = getattr(state, name)
        row = data[name]
        for key in ('controller', 'control', 'defense_control'):
            setattr(team, key, row[key])
        for key, value in row['net'].items():
            setattr(team.net, key, value)
        team.goalie = Player(**row['goalie'])
        team.players = [Player(**player) for player in row['players']]
    return state, data


class PossessionTests(unittest.TestCase):
    def test_live_forward_route_is_not_replaced_by_uncertainty_driven_retreat(self):
        state, recorded = retreat_state()
        player = state.team1.get_player_by_scnum(state.engine.puck_owner)
        snapshot = deepcopy(vars(player))
        target, details = ClassicAIV1Model().offense.fallback_carry(state, player)
        self.assertEqual(target, tuple(recorded['baseline_carry'][0]))
        self.assertGreater(details['carry_progress'], 4)
        self.assertTrue(details['carry_estimated_clear'])
        self.assertIsNone(details['carry_safe'])
        self.assertEqual(details['carry_shot_value'], 0)
        self.assertEqual(vars(player), snapshot)
        experimental = ClassicAIV1Model(parse_cmdline(['--nn=ClassicAIV1', '--offense-lookahead']))
        target, details = experimental.offense.fallback_carry(state, player)
        self.assertEqual(target, tuple(recorded['current'][0]))
        self.assertFalse(details['carry_safe'])
        self.assertLess(details['carry_progress'], 0)

    def test_direct_obstacle_is_not_reported_as_estimated_clear(self):
        state = offense_state()
        player = state.team1.players[0]
        opponent = state.team2.players[0]
        opponent.x, opponent.y = player.x, player.y
        controller = PossessionOffenseController()
        _, details = controller._estimated_carry(state, player, controller.carry_target(state, player))
        self.assertFalse(details['carry_estimated_clear'])
        self.assertIsNone(details['carry_safe'])

    def test_lookahead_is_explicit_and_existing_risk_experiments_keep_it(self):
        self.assertIsInstance(ClassicAIV1Model().offense, PossessionOffenseController)
        model = ClassicAIV1Model(SimpleNamespace(offense_lookahead=True))
        self.assertIs(type(model.offense), OffenseController)
        state, _ = retreat_state()
        player = state.team1.get_player_by_scnum(state.engine.puck_owner)
        for flag in ('allow_uncertified', 'chance_creation'):
            options = {flag: True}
            default = PossessionOffenseController(**options)
            experimental = OffenseController(**options)
            self.assertEqual(default.fallback_carry(state, player), experimental.fallback_carry(state, player))

    def test_learned_evaluation_rejects_the_classic_only_option(self):
        arguments = build_parser().parse_args(['--model=model.zip', '--offense-lookahead'])
        with self.assertRaisesRegex(ValueError, 'not a learned policy'):
            run(arguments)

    def test_reception_observation_preserves_the_selected_policy_cadence(self):
        for lookahead in (False, True):
            model = ClassicAIV1Model(SimpleNamespace(offense_lookahead=lookahead))
            state = offense_state()
            model.offense.pending = dict(passer=0, receiver=1, launched=True,
                                         flight_observed=True, deadline=100)
            model._frame_remaining = 2
            state.engine.puck_owner = 1
            state.team1.defense_control, state.team1.control = 1, 2
            model._observe_ordinary_pass(state)
            self.assertIsNone(model.offense.pending)
            self.assertEqual(model.offense.last_pass['outcome'], 'received')
            self.assertEqual(model._frame_remaining, 0 if lookahead else 2)


if __name__ == '__main__':
    unittest.main()
