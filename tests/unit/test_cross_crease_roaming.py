"""Openings are optional; traffic must not force the specialist's crossing."""
import math
import unittest

from nhl94_ai.agents.cross_crease_roaming import RoamingCrossCrease
from nhl94_ai.agents.skating import grounded_step
from nhl94_ai.game.constants import GameConsts as Buttons
from tests.support.offense import crossing_state


class RoamingCrossCreaseTests(unittest.TestCase):
    def test_open_moving_opportunity_does_not_require_a_setup_waypoint(self):
        for direction in (-1, 1):
            for attack in (-1, 1):
                state = crossing_state(direction=direction, attack=attack)
                player = state.team1.players[0]
                player.x, player.y = -direction * 25, attack * 226
                state.puck.x, state.puck.y = -direction * 8, player.y
                model = RoamingCrossCrease()
                action = model.step(state, 0)
                self.assertTrue(action[Buttons.INPUT_C])
                self.assertEqual(model.phase, 'shot')

    def test_crowded_crossing_is_optional_and_is_reconsidered_when_traffic_moves(self):
        state = crossing_state()
        player = state.team1.players[0]
        player.x, player.y = -60, 218
        state.puck.x, state.puck.y = -50, 218
        defender = state.team2.players[0]
        defender.x, defender.y = -5, 218
        defender.motion_x = defender.motion_y = 0
        model = RoamingCrossCrease()
        action = model.step(state, 0)
        self.assertEqual(model.phase, 'roam')
        self.assertFalse(action[Buttons.INPUT_C])
        self.assertFalse(action[Buttons.INPUT_B])
        self.assertIsNotNone(model.target)
        defender.y = 120
        model.step(state, 4)
        self.assertEqual(model.phase, 'cross')

    def test_new_traffic_can_cancel_a_crossing_before_windup(self):
        state = crossing_state()
        player = state.team1.players[0]
        player.x, player.y = -60, 218
        state.puck.x, state.puck.y = -50, 218
        model = RoamingCrossCrease()
        model.step(state, 0)
        self.assertEqual(model.phase, 'cross')
        defender = state.team2.players[0]
        defender.x, defender.y = -5, 218
        defender.motion_x = defender.motion_y = 0
        action = model.step(state, 4)
        self.assertEqual(model.phase, 'roam')
        self.assertFalse(action[Buttons.INPUT_C])

    def test_roaming_can_target_space_farther_from_goal(self):
        state = crossing_state()
        player = state.team1.players[0]
        player.x, player.y = -70, 218
        for defender, x in zip(state.team2.players, (-90, -55, -20, 20, 60)):
            defender.x, defender.y = x, 225
            defender.motion_x = defender.motion_y = 0
        model = RoamingCrossCrease()
        model.step(state, 0)
        self.assertEqual(model.phase, 'roam')
        self.assertLess(model.target[1], player.y)

    def test_windup_must_not_carry_the_puck_wide_before_release(self):
        state = crossing_state(direction=-1)
        player = state.team1.players[0]
        player.x, player.y = -20, 226
        player.motion_x, player.motion_y = -1.2, 0
        state.puck.x, state.puck.y = -35, 226
        self.assertTrue(RoamingCrossCrease(select_release=False, prepare_release=False).step(state, 0)[Buttons.INPUT_C])
        model = RoamingCrossCrease()
        self.assertFalse(model.step(state, 0)[Buttons.INPUT_C])
        self.assertEqual(model.diagnostics['finish'], 'too-wide')

    def test_drifting_away_can_make_a_currently_close_shot_too_far(self):
        for attack in (-1, 1):
            state = crossing_state(attack=attack)
            player = state.team1.players[0]
            player.x, player.y = -35, attack * 201
            player.motion_x, player.motion_y = 1.6, -attack
            state.puck.x, state.puck.y = -18, attack * 198
            model = RoamingCrossCrease()
            self.assertTrue(RoamingCrossCrease(select_release=False, prepare_release=False).step(state, 0)[Buttons.INPUT_C])
            self.assertFalse(model.step(state, 0)[Buttons.INPUT_C])
            self.assertEqual(model.diagnostics['finish'], 'too-far')

    def test_incoming_momentum_can_make_a_farther_start_a_useful_shot(self):
        for attack in (-1, 1):
            state = crossing_state(attack=attack)
            player = state.team1.players[0]
            player.x, player.y = -35, attack * 190
            player.motion_x, player.motion_y = 1.6, attack * 1.5
            state.puck.x, state.puck.y = -18, attack * 195
            model = RoamingCrossCrease()
            self.assertTrue(model.step(state, 0)[Buttons.INPUT_C])
            self.assertEqual(model.diagnostics['finish'], 'ready')

    def test_steering_gives_a_teammate_in_the_path_more_room(self):
        state = crossing_state()
        player = state.team1.players[0]
        player.x, player.y = -35, 218
        for other in (*state.team1.players[1:], *state.team2.players, state.team2.goalie):
            other.x, other.y = 120, -180
            other.motion_x = other.motion_y = 0
        teammate = state.team1.players[1]
        teammate.x, teammate.y = 0, 218
        clearances = []
        for enabled in (False, True):
            model = RoamingCrossCrease(traffic_aware=enabled)
            model.phase = 'cross'
            pad = model._skate(state, player, (95, 218), 0)
            future, clearance = player, math.inf
            for _ in range(24):
                future = grounded_step(future, pad)
                clearance = min(clearance, math.dist((future.x, future.y), (teammate.x, teammate.y)))
            clearances.append(clearance)
        self.assertGreater(clearances[1], clearances[0] + 4)


if __name__ == '__main__':
    unittest.main()
