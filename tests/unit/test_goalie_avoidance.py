"""Goalie clearance, braking, finishing and authoritative goalie possession."""
from types import SimpleNamespace
import math
import unittest
from unittest.mock import patch

from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.agents.defense import DefenseController
from nhl94_ai.agents.motion import stop_projection
from nhl94_ai.agents.offense import (
    FEINT_FRAMES, GOALIE_HORIZON, OffenseController, _goalie_clearance,
    goalie_avoidance, goalie_contact_time, projected_state,
)
from nhl94_ai.env.actions import HockeyActionController
from nhl94_ai.game.constants import GameConsts as Buttons
from tests.unit.test_classic_offense import live_one_timer_state, offense_state


def approach_state(*, away=False):
    state = offense_state()
    player = state.team1.players[0]
    player.x, player.y, player.motion_y = 0, 180, 1.6
    state.team2.goalie.x, state.team2.goalie.y = 0, 228
    state.puck.x, state.puck.y = 0, 190
    for other in (*state.team1.players[1:], *state.team2.players):
        other.x, other.y = 110, -180
    if away:
        state.team1.controller, state.team2.controller = 2, 1
        state.team1.defense_control = state.engine.puck_owner = 6
        for team in (state.team1, state.team2):
            team.net.y *= -1
            for other in (*team.players, team.goalie):
                other.y *= -1
                other.motion_y *= -1
                other.facing = (other.facing + 4) % 8
        state.puck.y *= -1
    return state


def buttons_for(model, state):
    action = model.predict_frame(state)[0]
    processor = HockeyActionController(SimpleNamespace(
        action_type=model.args.action_type, game_state=state))
    return processor._process_action(action, processor._new_action_state())[0]


def conflicted_setup_state(*, away=False):
    state = live_one_timer_state(away=away)
    sign = -1 if away else 1
    passer, receiver = state.team1.players[:2]
    passer.x, passer.facing = 0, 5 if away else 1
    receiver.x, receiver.y, receiver.motion_y = sign * 35, sign * 205, 0
    blocker = state.team1.players[2]
    blocker.x, blocker.y, blocker.unavailable = sign * 15, sign * 209, 4
    state.puck.x = 0
    state.team2.players[0].x, state.team2.players[0].y = sign * 70, sign * 245
    state.team2.goalie.x, state.team2.goalie.y = 0, sign * 220
    return state


class GoalieAvoidanceTests(unittest.TestCase):
    def test_opt_in_native_carry_does_not_inherit_the_legacy_goalie_veto(self):
        state = approach_state()
        player = state.team1.players[0]
        player.x, player.y, player.facing, player.facing_phase = 48, 190, 5, 5.0
        player.motion_x, player.motion_y = -1, 1
        state.team2.goalie.x, state.team2.goalie.y = 0, 250
        target = (-35, 235)
        self.assertLess(_goalie_clearance(player, state.team2.goalie, target), 32)
        self.assertGreater(_goalie_clearance(player, state.team2.goalie, target, decision_interval=4), 32)
        self.assertIsNone(goalie_avoidance(state, player, target, decision_interval=4))
        actual, _ = OffenseController(one_timers=False)._carry_option(state, player, target)
        self.assertEqual(actual, target)
        model = ClassicAIV1Model(SimpleNamespace(
            action_type='FILTERED', one_timers=False, offense_lookahead=True))
        model._steer([0] * 12, player, *target, state)
        self.assertEqual(model._last_target, target)

    def test_stop_projection_reduces_inertia_without_instant_reversal(self):
        player = approach_state().team1.players[0]
        position, motion = stop_projection(player, 4)
        self.assertGreater(position[1], player.y)
        self.assertGreater(motion[1], 0)
        self.assertLess(motion[1], player.motion_y)
        player.motion_x, player.motion_y = -0.01, 0.01
        position, motion = stop_projection(player, 4)
        self.assertEqual(motion, (0, 0))
        self.assertEqual(position, (player.x, player.y))

    def test_moving_goalie_changes_collision_deadline(self):
        state = approach_state()
        player, goalie = state.team1.players[0], state.team2.goalie
        self.assertLess(goalie_contact_time(player, goalie), GOALIE_HORIZON)
        goalie.motion_y = 2
        self.assertEqual(goalie_contact_time(player, goalie), math.inf)
        player.motion_y = 0
        goalie.x, goalie.y, goalie.motion_x, goalie.motion_y = 45, 190, -2, 0
        self.assertLess(goalie_contact_time(player, goalie), GOALIE_HORIZON)

    def test_carrier_brakes_before_committing_to_a_collision_locked_shot(self):
        for away in (False, True):
            for schema in ('FILTERED', 'HOCKEY_INTENT_DPAD'):
                with self.subTest(away=away, schema=schema):
                    state = approach_state(away=away)
                    model = ClassicAIV1Model(SimpleNamespace(action_type=schema, one_timers=False))
                    action = buttons_for(model, state)
                    self.assertEqual(model._last_decision, 'goalie-avoid')
                    self.assertFalse(action[Buttons.INPUT_B] or action[Buttons.INPUT_C])
                    self.assertTrue(action[Buttons.INPUT_UP if away else Buttons.INPUT_DOWN])
                    player = state.team1.players[0]
                    self.assertLess(model.offense_diagnostics['target'][1] * (-1 if away else 1), abs(player.y))

    def test_safe_close_shot_still_fires_in_both_directions_and_formats(self):
        for away in (False, True):
            for schema in ('FILTERED', 'HOCKEY_INTENT_DPAD'):
                with self.subTest(away=away, schema=schema):
                    state = approach_state(away=away)
                    player, goalie = state.team1.players[0], state.team2.goalie
                    player.x, player.y, player.motion_y = -30, -225 if away else 225, 0
                    goalie.x, goalie.y = 12, -250 if away else 250
                    model = ClassicAIV1Model(SimpleNamespace(action_type=schema, one_timers=False))
                    action = buttons_for(model, state)
                    self.assertEqual(model._last_decision, 'shoot')
                    self.assertTrue(action[Buttons.INPUT_C])

    def test_early_goalie_finish_cannot_release_after_coasting_behind_the_net(self):
        state = approach_state()
        player = state.team1.players[0]
        player.x, player.y, player.motion_y = 40, 214, 8
        state.puck.x, state.puck.y = 40, 224
        state.team2.goalie.x, state.team2.goalie.y = 0, 250
        self.assertGreater(goalie_contact_time(player, state.team2.goalie, clearance=24), GOALIE_HORIZON)
        self.assertIsNotNone(goalie_avoidance(state, player, OffenseController.carry_target(state, player)))
        model = ClassicAIV1Model(SimpleNamespace(action_type='FILTERED', one_timers=False))
        action = buttons_for(model, state)
        self.assertNotEqual(model._last_decision, 'shoot')
        self.assertFalse(action[Buttons.INPUT_C])

    def test_setup_plans_the_actual_goalie_safe_route_in_both_directions_and_formats(self):
        for away in (False, True):
            for schema in ('FILTERED', 'HOCKEY_INTENT_DPAD'):
                with self.subTest(away=away, schema=schema):
                    state = conflicted_setup_state(away=away)
                    planner = OffenseController()
                    self.assertFalse(planner.passes(state, 'one-timer')[0])
                    plan = planner.choose(state, 1)
                    self.assertIsNotNone(plan, planner.diagnostics)
                    self.assertEqual(plan[0], 'one-timer-setup')
                    player = state.team1.players[0]
                    escape = goalie_avoidance(state, player, plan[1])
                    self.assertTrue(escape is None or escape[0] == plan[1] and escape[1]['goalie_avoidance_safe'])
                    future = projected_state(state, plan[1])
                    self.assertTrue(planner.passes(future, 'one-timer')[0])
                    model = ClassicAIV1Model(SimpleNamespace(action_type=schema))
                    action = buttons_for(model, state)
                    self.assertEqual(model._last_decision, 'one-timer-setup')
                    self.assertEqual(model._last_target, plan[1])
                    self.assertEqual(model.offense.feint_target, plan[1])
                    self.assertFalse(action[Buttons.INPUT_B] or action[Buttons.INPUT_C])

    def test_safe_setup_takes_priority_over_early_goalie_finishing(self):
        for schema in ('FILTERED', 'HOCKEY_INTENT_DPAD'):
            with self.subTest(schema=schema):
                state = approach_state()
                player = state.team1.players[0]
                player.motion_y = 0
                state.team2.goalie.y = 215
                target = (48, player.y)
                self.assertIsNone(goalie_avoidance(state, player, target))
                self.assertIsNotNone(goalie_avoidance(state, player, OffenseController.carry_target(state, player)))
                model = ClassicAIV1Model(SimpleNamespace(action_type=schema))
                with patch.object(model.offense, 'choose', return_value=('one-timer-setup', target, None)), \
                        patch.object(model, '_one_timer_target', return_value=None):
                    action = buttons_for(model, state)
                self.assertEqual(model._last_decision, 'one-timer-setup')
                self.assertEqual(model._last_target, target)
                self.assertFalse(action[Buttons.INPUT_B] or action[Buttons.INPUT_C])

    def test_safe_one_timer_is_not_hidden_by_an_unsafe_close_shot(self):
        state = live_one_timer_state()
        passer, receiver = state.team1.players[:2]
        passer.x, passer.y, passer.motion_y = -30, 225, 1.2
        receiver.y, receiver.motion_y = 205, 0
        state.puck.x, state.puck.y = -30, 235
        state.team2.goalie.x, state.team2.goalie.y = -30, 260
        self.assertLess(goalie_contact_time(passer, state.team2.goalie, clearance=24), GOALIE_HORIZON)
        model = ClassicAIV1Model()
        self.assertTrue(model.offense.passes(state, 'one-timer')[0])
        action = model.predict_frame(state)[0]
        self.assertEqual(model._last_decision, 'one-timer-pass')
        self.assertTrue(action[Buttons.INPUT_B])
        self.assertFalse(action[Buttons.INPUT_C])

    def test_braking_projection_is_shared_by_goalie_safety_and_opportunity_planning(self):
        state = approach_state()
        player = state.team1.players[0]
        expected, motion = stop_projection(player, FEINT_FRAMES)
        future = projected_state(state, (player.x, player.y - 48))
        self.assertIsNotNone(future)
        mover = future.team1.get_player_by_scnum(state.engine.puck_owner)
        self.assertEqual((mover.x, mover.y), expected)
        self.assertEqual((mover.motion_x, mover.motion_y), motion)

    def test_goalie_reroute_cancels_instead_of_silently_replacing_a_live_setup(self):
        state = conflicted_setup_state()
        controller = OffenseController()
        controller.feint_mode = 'one-timer-setup'
        controller.feint_target, controller.feint_until = (26, 188), 18
        self.assertIsNone(controller._continue_cut(state, state.team1.players[0], 4))
        self.assertIsNone(controller.feint_target)
        self.assertIn('discard predicted opportunity', controller.diagnostics['cut_cancelled'])

    def test_windup_aim_waits_for_recorded_shot_not_just_a_loose_puck(self):
        state = approach_state()
        state.engine.puck_owner = -256
        state.team1.stats.shots = 3
        model = ClassicAIV1Model()
        model._shot_until, model._shots_before, model._shot_slot = 10, 3, 0
        action = model.predict_game_state(state)[0]
        self.assertEqual(model._last_decision, 'shot-follow-through')
        self.assertTrue(action[Buttons.INPUT_RIGHT])
        self.assertFalse(action[Buttons.INPUT_LEFT] or action[Buttons.INPUT_DOWN])
        state.team1.stats.shots = 4
        action = model.predict_game_state(state)[0]
        self.assertEqual(model._last_decision, 'goalie-avoid')
        self.assertFalse(action[Buttons.INPUT_B] or action[Buttons.INPUT_C])
        self.assertTrue(action[Buttons.INPUT_DOWN])

    def test_another_shooters_counter_does_not_release_windup_aim(self):
        state = approach_state()
        state.engine.puck_owner, state.engine.shot_player = -256, 1
        state.team1.stats.shots = 4
        model = ClassicAIV1Model()
        model._shot_until, model._shots_before, model._shot_slot = 10, 3, 0
        action = model.predict_game_state(state)[0]
        self.assertEqual(model._last_decision, 'shot-follow-through')
        self.assertTrue(action[Buttons.INPUT_RIGHT])

    def test_pass_release_aim_is_untouched_but_launched_pass_can_clear_goalie(self):
        state = approach_state()
        model = ClassicAIV1Model()
        model.offense.pending = {
            'passer': 0, 'receiver': 1, 'purpose': 'advance', 'frame': 0, 'deadline': 80,
            'point': (-80, -30), 'passes_before': 0, 'actual_receiver': None, 'launched': False,
            'flight_observed': False,
        }
        self.assertFalse(model.predict_game_state(state).any())
        state.engine.puck_owner, state.engine.last_puck_player = -256, 0
        state.team1.pass_attempts, state.engine.pass_target = 1, 1
        action = model.predict_game_state(state)[0]
        self.assertEqual(model._last_decision, 'goalie-avoid')
        self.assertTrue(action[Buttons.INPUT_DOWN])
        self.assertFalse(action[Buttons.INPUT_B] or action[Buttons.INPUT_C])
        self.assertIsNotNone(model.offense.pending)

    def test_existing_cut_is_cancelled_when_goalie_moves_into_it(self):
        state = approach_state()
        player = state.team1.players[0]
        player.motion_y = 0
        controller = OffenseController()
        controller.feint_target, controller.feint_until = (26, 188), 18
        state.team2.goalie.x, state.team2.goalie.y = 0, 185
        self.assertIsNone(controller._continue_cut(state, player, 4))
        self.assertIsNone(controller.feint_target)

    def test_already_overlapping_goalie_is_not_reported_as_a_safe_escape(self):
        state = approach_state()
        player = state.team1.players[0]
        player.y = 220
        point, details = goalie_avoidance(state, player, (-35, 235))
        self.assertFalse(details['goalie_avoidance_safe'])
        self.assertIn('unavoidable', details['reason'])
        self.assertTrue(all(math.isfinite(coordinate) for coordinate in point))
        self.assertIsNone(OffenseController._cut_target(state, player, (-35, 235)))

    def test_clear_routes_and_own_goalie_are_not_redirected(self):
        state = approach_state()
        state.team2.goalie.x = 110
        self.assertIsNone(goalie_avoidance(state, state.team1.players[0], (-35, 235)))
        self.assertIsNone(goalie_avoidance(state, state.team1.goalie, (0, 235)))

    def test_authoritative_opponent_goalie_owner_covers_outlet_instead_of_puck(self):
        for away in (False, True):
            for count in (1, 2, 5):
                with self.subTest(away=away, count=count):
                    state = approach_state(away=away)
                    state.team2.players = state.team2.players[:count]
                    state.team2.num_players = count
                    slot = state.team2.goalie_scnum()
                    state.team2.defense_goalie = slot
                    state.engine.puck_owner = slot
                    state.puck.x, state.puck.y = state.team2.goalie.x, state.team2.goalie.y
                    controller = DefenseController()
                    plan = controller.choose_target(state)
                    self.assertEqual(plan.mode, 'deny-goalie-outlet')
                    self.assertNotEqual(plan.threat_slot, slot)
                    self.assertGreater(math.dist(plan.target, (state.puck.x, state.puck.y)), 40)
                    action = controller.step(state)
                    self.assertNotEqual(controller.diagnostics['mode'], 'poke-request')
                    self.assertEqual(controller.diagnostics['check']['status'], 'no-active-carrier')
                    self.assertFalse(action[Buttons.INPUT_C])


if __name__ == '__main__':
    unittest.main()
