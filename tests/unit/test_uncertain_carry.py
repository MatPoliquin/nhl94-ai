"""Uncertified is not safe, blocked geometry or a fabricated probability."""
from copy import deepcopy
import unittest
from unittest.mock import patch

from nhl94_ai.agents.carry import CARRY_FRAMES, carry_path, forecast_carry, risk_tracks
from nhl94_ai.agents.offense import OffenseController, carry_value, carry_future
from nhl94_ai.agents.skating import grounded_step
from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.evaluation.play import parse_cmdline
from nhl94_ai.evaluation.runner import build_parser as evaluation_parser, run as evaluate
from tests.unit.test_carry_continuations import wing_reception_state
from tests.unit.test_classic_offense import offense_state


class UncertainCarryTests(unittest.TestCase):
    def test_uncertified_attacking_routes_are_opt_in(self):
        state = wing_reception_state()
        state.team2.players[0].x, state.team2.players[0].motion_x = 80, 0
        _, default = OffenseController(one_timers=False).passes(state, 'position')
        self.assertEqual(default[0]['continuation_status'], 'no-viable-continuation')
        self.assertNotIn('continuation_value', default[0])
        arguments = parse_cmdline(['--nn=ClassicAIV1', '--uncertain-carry'])
        self.assertTrue(ClassicAIV1Model(arguments).offense.allow_uncertified)
        self.assertFalse(ClassicAIV1Model().offense.allow_uncertified)

    def test_learned_evaluation_cannot_silently_ignore_the_classic_flag(self):
        arguments = evaluation_parser().parse_args(['--model=model.zip', '--uncertain-carry'])
        with self.assertRaisesRegex(ValueError, 'not a learned policy'):
            evaluate(arguments)

    def test_uncertified_receiver_retains_discounted_attacking_value(self):
        state = wing_reception_state()
        defender = state.team2.players[0]
        defender.x, defender.motion_x = 80, 0
        controller = OffenseController(one_timers=False, allow_uncertified=True)
        options, diagnostics = controller.passes(state, 'position')
        self.assertTrue(options)
        option = options[0]
        self.assertFalse(option.continuation_safe)
        self.assertGreater(option.continuation_risk, 0)
        self.assertGreater(option.continuation_value, 0)
        self.assertLess(option.continuation_value, option.continuation_position_value * 0.6)
        self.assertEqual(option.continuation_finish_value, 0)
        self.assertEqual(diagnostics[0]['continuation_status'], 'uncertified-clear-samples')

    def test_direct_body_and_wall_hazards_remain_ineligible(self):
        state = offense_state()
        player = state.team1.players[0]
        defender = state.team2.players[0]
        defender.x, defender.y = player.x, player.y
        _, blocked = forecast_carry(state, player, (0, 235), assess_uncertainty=True)
        self.assertFalse(blocked['carry_viable'])
        self.assertEqual(blocked['carry_status'], 'projected-body-contact')
        player.x, player.y, player.motion_x, player.facing = 119, 0, 2, 2
        _, wall = forecast_carry(state, player, (120, 24), assess_uncertainty=True)
        self.assertFalse(wall['carry_viable'])
        self.assertEqual(wall['carry_status'], 'unbounded-route')

    def test_missing_defender_physics_does_not_invent_low_risk(self):
        state = wing_reception_state()
        state.team2.players[0].x, state.team2.players[0].motion_x = 80, None
        player = state.team1.players[1]
        _, details = forecast_carry(state, player, (35, 235), assess_uncertainty=True)
        self.assertFalse(details['carry_safe'])
        self.assertGreater(details['carry_clearance'], 0)
        self.assertFalse(details['carry_viable'])
        self.assertEqual(details['carry_status'], 'unmeasured-defender-risk')

    def test_a_modeled_burst_contact_before_replanning_is_not_admitted(self):
        state = offense_state()
        player = state.team1.players[0]
        player.x, player.y, player.facing = 0, 180, 0
        state.puck.x, state.puck.y = 0, 190
        for team in (state.team1, state.team2):
            for other in team.players:
                if other is not player:
                    other.role = -1
        defender = state.team2.players[0]
        defender.role, defender.x, defender.y, defender.facing = 4, 22, 180, 6
        _, details = forecast_carry(state, player, (0, 235), assess_uncertainty=True)
        self.assertGreater(details['carry_clearance'], 0)
        self.assertFalse(details['carry_safe'])
        self.assertFalse(details['carry_viable'])
        self.assertEqual(details['carry_status'], 'imminent-modeled-contact')
        self.assertLessEqual(details['carry_sampled_contact_frame'], 4)

    def test_risk_tracks_cache_physics_not_player_identity(self):
        state = offense_state()
        player = state.team2.players[0]
        cache = {}
        before = deepcopy(vars(player))
        initial = risk_tracks(player, 18, cache)
        self.assertIs(initial, risk_tracks(player, 18, cache))
        self.assertEqual(vars(player), before)
        player.motion_x = 1
        self.assertIsNot(initial, risk_tracks(player, 18, cache))

    def test_risk_changes_value_but_never_the_safety_certificate(self):
        details = {'carry_safe': False, 'carry_position_value': 40,
                   'carry_shot_value': 0, 'carry_risk': 0.25}
        self.assertAlmostEqual(carry_value(details), 16.425)
        self.assertFalse(details['carry_safe'])

    def test_zero_opportunity_still_pays_turnover_risk(self):
        state = offense_state()
        player = state.team1.players[0]
        safe_target, risk_target = (player.x - 48, player.y), (player.x + 12, player.y + 24)
        def option(_state, _player, target):
            details = dict(carry_bounded=True, carry_clearance=30, carry_pressure=-8,
                           carry_shot_value=0, carry_position_value=0, carry_progress=0,
                           carry_safe=False, carry_viable=False, carry_risk=1.0)
            if target == safe_target:
                details.update(carry_safe=True, carry_viable=True, carry_risk=0.0, carry_pressure=4)
            if target == risk_target:
                details.update(carry_viable=True, carry_risk=16 / 21,
                               carry_progress=18, carry_sampled_contact_frame=5)
            return target, details
        controller = OffenseController(allow_uncertified=True)
        with patch.object(controller, '_carry_option', side_effect=option):
            selected, details = controller.fallback_carry(state, player)
        self.assertEqual(selected, safe_target)
        self.assertTrue(details['carry_safe'])

    def test_next_input_cannot_remove_contact_one_frame_after_replan(self):
        state = offense_state()
        player = state.team1.players[0]
        player.x, player.y, player.motion_x, player.facing = 0, 180, 2, 2
        path = carry_path(player, (40, 200))
        positions = {(grounded_step(path[4], pad).x, grounded_step(path[4], pad).y)
                     for pad in ((-1, 0), (0, 0), (1, 1), (0, -1))}
        self.assertEqual(len(positions), 1)
        controller = OffenseController(allow_uncertified=True)
        with patch('nhl94_ai.agents.offense.assess_path_risk', return_value={
                'carry_status': 'uncertified-modeled-contact', 'carry_sampled_contact_frame': 5}), \
                patch('nhl94_ai.agents.offense.body_clearance', return_value=20):
            self.assertIsNone(controller._threat_exit(state, player, path, 5, state, 0))

    def test_escape_is_a_native_prefix_and_suffix_not_a_teleport(self):
        state = offense_state()
        player = state.team1.players[0]
        player.x, player.y, player.motion_x, player.facing = 0, 160, 1, 2
        path = carry_path(player, (30, 190))
        seen = []
        def sampled(_state, _player, candidate, *_args):
            seen.append(candidate)
            return {'carry_status': 'uncertified-clear-samples', 'carry_sampled_contact_frame': None}
        controller = OffenseController(allow_uncertified=True)
        with patch('nhl94_ai.agents.offense.assess_path_risk', side_effect=sampled):
            result = controller._threat_exit(state, player, path, 12, state, 0)
        self.assertEqual(result['carry_exit_frame'], 4)
        self.assertEqual(len(seen[0]), CARRY_FRAMES + 1)
        self.assertEqual(seen[0][:5], path[:5])
        self.assertEqual(seen[0][5].precise_x, grounded_step(path[4], (0, 0)).precise_x)

    def test_short_projection_advances_other_bodies_by_actual_replan_time(self):
        state = offense_state()
        player = state.team1.players[0]
        other = state.team1.players[1]
        other.motion_y = 1
        endpoint = carry_path(player, (0, 0))[4]
        future = carry_future(state, player, endpoint, frames=4)
        self.assertEqual(future.team1.players[1].y, other.y + 4)


if __name__ == '__main__':
    unittest.main()
