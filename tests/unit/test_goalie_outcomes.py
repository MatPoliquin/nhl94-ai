"""Native outcome invariants, including rating dependence and byte writes."""
import unittest
from copy import deepcopy
from types import SimpleNamespace

from nhl94_ai.agents.goalie_outcomes import body_rebound, catch_probability
from nhl94_ai.agents.goalie_physics import KeeperMotion, PuckFlight
from nhl94_ai.agents.goalie_physics import free_trajectory
from nhl94_ai.agents.goalie_outcome_planner import (
    OutcomeBranch, OutcomeGoalie, adaptive_incumbent, extend, outcome_search,
)
from nhl94_ai.agents.goalie_planner import SAVE
from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from tests.support.defense import goalie_state


class ContactOutcomeTests(unittest.TestCase):
    def test_fast_shot_can_be_held_in_animation_but_not_upright(self):
        self.assertEqual(catch_probability(16, 0, -14000, animated=False), 0)
        chance = catch_probability(16, 0, -14000, animated=True)
        self.assertGreater(chance, .5)
        self.assertLess(chance, .6)

    def test_effective_rating_changes_catch_probability(self):
        chances = [catch_probability(pc, 0, -14000, animated=True) for pc in (4, 16, 28)]
        self.assertEqual(chances, sorted(chances))
        self.assertEqual(chances[0], 0)
        self.assertGreater(chances[-1], chances[1])

    def test_velocity_sign_and_axis_symmetry(self):
        expected = catch_probability(16, 100, -14000, animated=True)
        for vx, vy in ((-100, 14000), (14000, -100), (-14000, 100)):
            self.assertEqual(catch_probability(16, vx, vy, animated=True), expected)

    def test_dive_and_unavailable_never_glue_body_contact(self):
        self.assertEqual(catch_probability(30, 0, 0, animated=True, dive=True), 0)
        self.assertEqual(catch_probability(30, 0, 0, animated=True, unavailable=True), 0)
        self.assertEqual(catch_probability(0, 0, 6000, animated=True), 1)

    def test_rebound_retains_low_bytes_and_points_away_from_body(self):
        keeper = KeeperMotion(0, -246*65536, 0, 0, 50, 8000**2)
        puck = PuckFlight(-8*65536, -234*65536, 10*65536, 157, -14111, -300)
        rebound = body_rebound(keeper, puck)
        self.assertLess(rebound.vx, 0)
        self.assertGreater(rebound.vy, 0)
        self.assertEqual(rebound.vx & 255, puck.vx & 255)
        self.assertEqual(rebound.vy & 255, puck.vy & 255)
        self.assertEqual(rebound.vz, 0)
        self.assertEqual(rebound.z, puck.z)

    def test_equal_depth_requires_hotspot(self):
        keeper = KeeperMotion(0, 0, 0, 0, 50, 8000**2)
        puck = PuckFlight(2*65536, 0, 0, 0, 0, 0)
        self.assertIsNone(body_rebound(keeper, puck))
        self.assertIsNotNone(body_rebound(keeper, puck, stick=(0, 8)))


class OutcomePlannerTests(unittest.TestCase):
    def test_live_puck_control_is_decoded_and_cleared_when_telemetry_disappears(self):
        state = goalie_state(True)
        info = {'defense_control1': 5, 'defense_team1': 1, 'g1_control_puck_control': 23}
        state.team1._load_defense_fields(info)
        self.assertEqual(state.team1.goalie.puck_control, 23)
        state.team1._load_defense_fields({})
        self.assertIsNone(state.team1.goalie.puck_control)

    def test_unknown_rating_refuses_outcome_prediction(self):
        state = goalie_state(True)
        state.team1.goalie.puck_control = None
        keeper, puck = KeeperMotion.observe(state.team1.goalie), PuckFlight.observe(state.puck)
        command, detail = outcome_search(state, OutcomeGoalie(), keeper, puck, 0)
        self.assertIsNone(command)
        self.assertEqual(detail['status'], 'unsupported-live-puck-control')

    def test_contact_ends_each_scenario_instead_of_improving_margin_after_collision(self):
        keeper = KeeperMotion(0, -246*65536, 0, 0, 50, 8000**2)
        puck = PuckFlight(0, -230*65536, 12*65536, 0, -14000, 0)
        flight = [puck, *free_trajectory(puck, 49, -1)]
        first = extend(OutcomeBranch(keeper), [SAVE]*4, 0, flight, False, 16, -1)
        self.assertIsNotNone(first)
        self.assertTrue(all(event is not None for event in first.contacts))
        continuation = extend(first, [2]*20, 4, flight, False, 16, -1)
        self.assertEqual(first, continuation)

    def test_incumbent_accounts_for_classic_saving_later(self):
        state = goalie_state(True)
        state.engine.puck_owner = -256
        state.puck.x, state.puck.y, state.puck.height = 0, -170, 20
        state.puck.motion_x, state.puck.motion_y, state.puck.motion_z = 0, -4, 0
        manager = OutcomeGoalie()
        keeper, puck = KeeperMotion.observe(state.team1.goalie), PuckFlight.observe(state.puck)
        flight = [puck, *free_trajectory(puck, 49, -1)]
        before = deepcopy(state.puck)
        sequence = adaptive_incumbent(state, manager, keeper, flight, 0, min(24, len(flight)-1))
        self.assertEqual(sequence[0], 0)
        self.assertIn(SAVE, sequence[1:])
        self.assertEqual(state.puck, before)
        self.assertIsNone(manager.pending_save)

    def test_stick_contact_requires_fallback(self):
        keeper = KeeperMotion(0, -246*65536, 0, 0, 50, 8000**2)
        puck = PuckFlight(0, -230*65536, 0, 0, -14000, 0)
        flight = [puck, *free_trajectory(puck, 49, -1)]
        self.assertIsNone(extend(OutcomeBranch(keeper), [SAVE]*4, 0, flight, False, 16, -1))

    def test_outcome_planner_is_explicitly_opt_in(self):
        agent = ClassicAIV1Model(SimpleNamespace(goalie_policy='selective', goalie_planner='outcome-24'))
        self.assertIsInstance(agent.goalie, OutcomeGoalie)
        default = ClassicAIV1Model(SimpleNamespace(goalie_policy='selective'))
        self.assertNotIsInstance(default.goalie, OutcomeGoalie)


if __name__ == '__main__':
    unittest.main()
