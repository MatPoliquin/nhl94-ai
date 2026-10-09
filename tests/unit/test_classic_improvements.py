"""Regression cases for executable Classic decisions and safe releases."""
from types import SimpleNamespace
from copy import deepcopy
import unittest
from unittest.mock import patch

from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.agents.carry import carry_path, body_clearance
from nhl94_ai.agents.defense import DefenseController
from nhl94_ai.agents.finishing import normal_finish
from nhl94_ai.agents.offense import OffenseController, carry_clearance, projected_state
from nhl94_ai.agents.passing import pass_launch_view, selected_receiver
from nhl94_ai.env.intents import HOCKEY_INTENT_NOOP, HOCKEY_INTENT_PASS_START
from nhl94_ai.game.constants import GameConsts as Buttons
from tests.unit.test_classic_offense import offense_state
from tests.unit.test_classic_defense import defense_state


def outlet_state():
    state = offense_state()
    state.engine.puck_owner = state.team1.defense_control = 5
    state.puck.x, state.puck.y = 0, -238
    for player, point in zip(state.team1.players, (
            (0, -200), (-65, -160), (65, -130), (110, -40), (-110, 30))):
        player.x, player.y = point
    state.team1.players[0].unavailable = 4
    for player in state.team2.players:
        player.y = 180
    return state


class DefaultOutletTests(unittest.TestCase):
    def test_unavailable_nearest_receiver_is_skipped_and_release_is_tracked(self):
        for schema in ('FILTERED', 'HOCKEY_INTENT_DPAD'):
            with self.subTest(schema=schema):
                state = outlet_state()
                model = ClassicAIV1Model(SimpleNamespace(action_type=schema))
                action = model.predict_frame(state)[0]
                self.assertEqual(model._last_decision, 'goalie-outlet')
                receiver = model.offense.pass_action.pending.receiver
                self.assertNotEqual(receiver, 0)
                if schema == 'FILTERED':
                    self.assertEqual(action[Buttons.INPUT_B], 1)
                else:
                    self.assertEqual(action[0], HOCKEY_INTENT_PASS_START + receiver)
                state.team1.pass_attempts = 1
                state.engine.pass_target = receiver
                state.engine.puck_owner = -256
                for _ in range(4):
                    action = model.predict_frame(state)[0]
                self.assertTrue(model.offense.pass_action.pending.launched)
                self.assertEqual(model.offense.pass_action.pending.actual_receiver, receiver)
                self.assertEqual(action[Buttons.INPUT_B] if schema == 'FILTERED' else action[0],
                                 0 if schema == 'FILTERED' else HOCKEY_INTENT_NOOP)
                state.engine.puck_owner = state.team1.defense_control = receiver
                model.predict_frame(state)
                self.assertIsNone(model.offense.pass_action.pending)
                self.assertEqual(model.offense.pass_action.last_pass['outcome'], 'received')

    def test_no_eligible_outlet_holds_for_automatic_cover(self):
        state = outlet_state()
        for player in state.team1.players:
            player.unavailable = 4
        model = ClassicAIV1Model()
        self.assertFalse(model.predict_frame(state).any())
        self.assertEqual(model._last_decision, 'goalie-hold')
        self.assertIsNone(model.offense.pass_action.pending)

    def test_outlet_requires_a_new_b_edge(self):
        model = ClassicAIV1Model()
        model.buttons.b_down = True
        self.assertFalse(model.predict_frame(outlet_state()).any())
        self.assertIsNone(model.offense.pass_action.pending)

    def test_one_frame_decisions_hold_b_through_native_direction_selection(self):
        state = outlet_state()
        model = ClassicAIV1Model()
        self.assertTrue(model.predict_frame(state, frame_skip=1)[0, Buttons.INPUT_B])
        self.assertTrue(model.predict_frame(state, frame_skip=1)[0, Buttons.INPUT_B])
        state.engine.puck_owner = -256
        state.team1.pass_attempts = 1
        state.engine.pass_target = model.offense.pass_action.pending.receiver
        self.assertFalse(model.predict_frame(state, frame_skip=1)[0, Buttons.INPUT_B])


class PassTimingTests(unittest.TestCase):
    def test_competing_receivers_use_native_slot_order_and_puck_attachment(self):
        # Native regression: the current-position selector requests 2, but the ROM selects 0.
        state = offense_state()
        state.engine.puck_owner = state.team1.defense_control = 1
        passer = state.team1.players[1]
        passer.x, passer.y, passer.stick_x, passer.stick_y = 0, -100, 17, -3
        state.puck.x, state.puck.y = 0, -90
        for index, point, motion in (
                (0, (70, -38), (-1.477020263671875, -0.3789825439453125)),
                (2, (32, -26), (1.2803955078125, -0.597137451171875)),
                (3, (-110, -180), (0, 0)), (4, (110, -210), (0, 0))):
            player = state.team1.players[index]
            player.x, player.y = point
            player.motion_x, player.motion_y = motion
        before = deepcopy(state)
        team, puck = pass_launch_view(state, passer)
        self.assertEqual(selected_receiver(state.team1, state.puck, 1, 1), 2)
        self.assertEqual(selected_receiver(team, puck, 1, 1), 0)
        self.assertEqual((puck.x, puck.y), (4, -94))
        self.assertEqual((team.players[0].x, team.players[0].y), (67, -39))
        self.assertEqual((team.players[2].x, team.players[2].y), (33, -27))
        self.assertEqual(state.team1.players, before.team1.players)
        self.assertEqual(state.puck, before.puck)


class CarryExecutionTests(unittest.TestCase):
    def test_cut_cannot_cross_a_teammate_between_sampled_frames(self):
        state = offense_state()
        player = state.team1.players[0]
        player.x, player.y, player.facing, player.facing_phase = 0, 180, 1, 1.3514783172377647
        player.motion_x, player.motion_y = 1.8619486968767651, -2.705221194547588
        for other in state.team1.players[1:] + state.team2.players:
            other.role = -1
        state.team1.players[1].role = 4
        state.team1.players[1].x, state.team1.players[1].y = -4, 163
        state.team1.players[1].motion_x, state.team1.players[1].motion_y = -1.509014282657394, -0.38582450206101626
        state.team1.goalie.role = state.team2.goalie.role = -1
        target = (26, 188)
        self.assertGreater(carry_clearance(state, player, target), 0)
        self.assertLess(body_clearance(state, player, carry_path(player, target, 4)), 0)
        controller = OffenseController(decision_interval=4)
        controller.carry_motion = True
        controller.feint_target, controller.feint_until = target, 100
        self.assertIsNone(controller._continue_cut(state, player, 1))
        self.assertIsNone(controller.feint_target)

    def test_cut_projection_retains_fractional_facing_and_actual_cadence(self):
        state = offense_state()
        player = state.team1.players[0]
        player.facing_phase = 1.75
        player.motion_x, player.motion_y = 1.4, 0.8
        future = projected_state(state, (-20, -80), frames=5, decision_interval=5)
        expected = carry_path(player, (-20, -80), 5, decision_interval=5)[-1]
        actual = future.team1.players[0]
        self.assertEqual((actual.precise_x, actual.precise_y, actual.facing_phase),
                         (expected.precise_x, expected.precise_y, expected.facing_phase))
        self.assertNotEqual(actual.facing_phase, int(actual.facing_phase))


class FinishExecutionTests(unittest.TestCase):
    def test_moving_goalie_can_change_both_aim_and_hold(self):
        state = offense_state()
        player = state.team1.players[0]
        player.x, player.y, player.shot_power = 0, 220, 20
        player.shot_offsets_y = (-5, 5)
        state.puck.x, state.puck.y = 0, 225
        for other in state.team1.players[1:] + state.team2.players:
            other.role = -1
        state.team2.goalie.x = 10
        static = normal_finish(state, player, 4)
        state.team2.goalie.motion_x = -2
        moving = normal_finish(state, player, 4)
        self.assertEqual(static.side, -1)
        self.assertEqual(moving.side, 1)
        self.assertGreater(moving.clearance, 0)

    def test_short_hold_releases_c_inside_the_four_frame_decision(self):
        for schema in ('FILTERED', 'HOCKEY_INTENT_DPAD'):
            with self.subTest(schema=schema):
                state = offense_state()
                player = state.team1.players[0]
                player.x, player.y = state.puck.x, 225
                state.puck.y = 230
                state.team2.goalie.x = -80
                model = ClassicAIV1Model(SimpleNamespace(
                    action_type=schema, one_timers=False, classic_refinements=['finishing']))
                choice = normal_finish(state, player, 4)
                with patch('nhl94_ai.agents.classic_v1.evaluate_finish', return_value=choice):
                    actions = [model.predict_frame(state)[0] for _ in range(4)]
                c = [int(row[Buttons.INPUT_C]) if schema == 'FILTERED' else int(row[0] == 2) for row in actions]
                self.assertEqual(c, [1] * choice.hold_frames + [0] * (4 - choice.hold_frames))


class InterceptionExecutionTests(unittest.TestCase):
    def test_optimistic_arrival_does_not_admit_an_unreachable_deadline(self):
        state = defense_state()
        player = state.team1.players[0]
        player.facing_phase, player.motion_x = 0.7, 2
        controller = DefenseController(verify_interceptions=True)
        controller.boost_at = 1000
        target = (40, -160)
        self.assertLess(controller._cost(state, 0, player, target) + 5, 56)
        self.assertIsNone(controller._skating_arrival(state, 0, player, target, 56))

    def test_interception_selects_the_skater_whose_arrival_was_verified(self):
        state = defense_state()
        state.engine.puck_owner = -256
        state.puck.x, state.puck.y = 0, -200
        for player in state.team2.players:
            player.y = 180
        controller = DefenseController(verify_interceptions=True)
        controller.step(state)
        self.assertEqual(controller.plan.mode, 'recover-safe')
        self.assertEqual(controller.desired_slot, controller.plan.recovery_slot)
        self.assertLessEqual(controller.plan.skating_arrival + 5, controller.plan.puck_arrival)
