"""Isolation, feasible parameterization and rating-aware decision contracts."""
from dataclasses import asdict, replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from nhl94_ai.agents.classic_tuning import DEFAULT, current, decode, encode, load_params, scope, validate
from nhl94_ai.agents.offense import OffenseController, ordinary_shot_conditions
from nhl94_ai.agents.possession import PossessionOffenseController


class ClassicTuningTests(unittest.TestCase):
    def test_scope_restores_opponent_defaults_even_after_exception(self):
        tuned = replace(DEFAULT, carry_lane_offset=50)
        with self.assertRaisesRegex(RuntimeError, 'interrupted'):
            with scope(tuned):
                self.assertIs(current(), tuned)
                with scope(DEFAULT):
                    self.assertIs(current(), DEFAULT)
                self.assertIs(current(), tuned)
                raise RuntimeError('interrupted')
        self.assertIs(current(), DEFAULT)

    def test_parameter_roundtrip_and_invalid_artifacts(self):
        self.assertEqual(load_params(asdict(DEFAULT)), DEFAULT)
        for name, value in asdict(decode(encode())).items():
            self.assertAlmostEqual(value, getattr(DEFAULT, name))
        for value in (float('nan'), float('inf'), -1, 231, True):
            with self.assertRaises(ValueError):
                validate(replace(DEFAULT, normal_shot_depth=value))
        with self.assertRaises(ValueError):
            validate(replace(DEFAULT, normal_shot_depth=230, carry_target_depth=225))
        with self.assertRaises(ValueError):
            load_params({**asdict(DEFAULT), 'unexpected': 1})

    @patch('nhl94_ai.agents.offense.goalie_contact_time', return_value=100)
    @patch('nhl94_ai.agents.offense.shot_release_in_front', return_value=True)
    def test_shot_trigger_preserves_accuracy_and_attack_direction(self, *unused):
        for sign in (-1, 1):
            state = SimpleNamespace(team1=SimpleNamespace(net=SimpleNamespace(y=-264*sign)),
                                    team2=SimpleNamespace(net=SimpleNamespace(y=264*sign), goalie=None))
            good = SimpleNamespace(x=0, y=220*sign, shot_accuracy=20)
            poor = SimpleNamespace(x=0, y=220*sign, shot_accuracy=5)
            self.assertTrue(ordinary_shot_conditions(state, good, 4, None)[0])
            self.assertFalse(ordinary_shot_conditions(state, poor, 4, None)[0])
            with scope(replace(DEFAULT, shot_accuracy_depth_scale=0.1)):
                self.assertTrue(ordinary_shot_conditions(state, poor, 4, None)[0])
            with scope(replace(DEFAULT, normal_shot_depth=223)):
                self.assertFalse(ordinary_shot_conditions(state, good, 4, None)[0])

    @patch('nhl94_ai.agents.offense.goalie_contact_time', return_value=100)
    @patch('nhl94_ai.agents.offense.shot_release_in_front', return_value=False)
    def test_tuning_cannot_override_release_geometry(self, *unused):
        state = SimpleNamespace(team1=SimpleNamespace(net=SimpleNamespace(y=-264)),
                                team2=SimpleNamespace(net=SimpleNamespace(y=264), goalie=None))
        player = SimpleNamespace(x=0, y=270, shot_accuracy=30)
        with scope(replace(DEFAULT, normal_shot_depth=205)):
            self.assertEqual(ordinary_shot_conditions(state, player, 4, object()), (False, False))

    def test_carry_and_position_pass_preferences_are_local(self):
        state = SimpleNamespace(team1=SimpleNamespace(net=SimpleNamespace(y=-264)),
                                team2=SimpleNamespace(net=SimpleNamespace(y=264),
                                                      goalie=SimpleNamespace(x=10)))
        player = SimpleNamespace(x=20, y=180)
        option = SimpleNamespace(shot_value=20)
        self.assertEqual(OffenseController.carry_target(state, player), (-35, 235))
        self.assertFalse(PossessionOffenseController._immediate_gain(option, player, 1, 'position', 10))
        with scope(replace(DEFAULT, carry_lane_offset=45, position_pass_min_improvement=5)):
            self.assertEqual(OffenseController.carry_target(state, player), (-45, 235))
            self.assertTrue(PossessionOffenseController._immediate_gain(option, player, 1, 'position', 10))
        self.assertEqual(OffenseController.carry_target(state, player), (-35, 235))


if __name__ == '__main__':
    unittest.main()
