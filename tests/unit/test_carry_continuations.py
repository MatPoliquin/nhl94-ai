"""Ordinary carry physics, receiving continuations and live score diagnostics."""
from copy import deepcopy
from dataclasses import replace
import math
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from nhl94_ai.agents.carry import (
    CARRY_FRAMES, carry_pad, carry_path, forecast_carry, interception_time, reach_envelopes,
)
from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.agents.offense import (
    OffenseController, carry_future, ordinary_shot_conditions, reception_frames, reception_state,
)
from nhl94_ai.agents.passing import evaluate_pass
from nhl94_ai.agents.skating import grounded_step
from nhl94_ai.env.actions import HockeyActionController
from nhl94_ai.game.constants import GameConsts as Buttons
from tests.unit.test_classic_offense import offense_state


def wing_reception_state(*, carrier_x=80, receiver_x=105):
    state = offense_state()
    passer, receiver = state.team1.players[:2]
    passer.x, passer.y, passer.facing = carrier_x, 145, 0
    state.puck.x, state.puck.y = carrier_x, 155
    receiver.x, receiver.y, receiver.facing = receiver_x, 180, 6
    receiver.motion_x, receiver.motion_y = -1.5, 0
    for team in (state.team1, state.team2):
        for player in team.players:
            if player is not passer and player is not receiver:
                player.role = -1
    defender = state.team2.players[0]
    defender.role, defender.x, defender.y, defender.motion_x = 4, 0, 230, -2
    state.team2.goalie.x = -15
    return state


class CarryForecastTests(unittest.TestCase):
    def test_alternative_steering_cannot_be_certified_as_four_frames_safe(self):
        state = offense_state()
        player, defender = state.team1.players[0], state.team2.players[0]
        player.x, player.y, player.facing = 0, 180, 0
        state.puck.x, state.puck.y = 0, 190
        for team in (state.team1, state.team2):
            for other in (*team.players, team.goalie):
                if other is not player:
                    other.role = -1
        defender.role, defender.x, defender.y = 4, -9, 216
        defender.motion_x, defender.motion_y = -0.5944902118335285, -1.7344195104050097
        defender.facing, defender.facing_phase = 6, 6.210530207564739
        path, details = forecast_carry(state, player, (0, 235))
        endpoint = path[-1].x, path[-1].y
        for _ in range(18):
            defender = grounded_step(defender, (1, 0))
        self.assertLessEqual(math.dist((defender.x, defender.y), endpoint), 16)
        self.assertFalse(details['carry_safe'])
        self.assertLess(details['carry_pressure'], 3)

    def test_reach_envelope_contains_adversarial_steering_and_braking(self):
        rng = random.Random(94)
        defender = offense_state().team2.players[0]
        for _ in range(32):
            defender.facing_phase = rng.uniform(0, 8)
            defender.facing = int(defender.facing_phase)
            defender.motion_x, defender.motion_y = rng.uniform(-3, 3), rng.uniform(-3, 3)
            envelopes = reach_envelopes(defender, 24)
            future = defender
            for elapsed in range(1, 25):
                future = grounded_step(future, (rng.randrange(-1, 2), rng.randrange(-1, 2)))
                centers, radii = envelopes[elapsed]
                self.assertLessEqual(math.dist((future.x, future.y), centers[0]), radii[0] + 1e-9)

    def test_delayed_and_repeated_bursts_are_inside_the_reach_envelope(self):
        from nhl94_ai.agents.motion import burst_velocity
        defender = offense_state().team2.players[0]
        envelopes = reach_envelopes(defender, 56)
        future = defender
        for elapsed in range(56):
            if elapsed in (4, 30):
                future.motion_x, future.motion_y = burst_velocity(future)
            future = grounded_step(future, (-1 if elapsed < 24 else 1, -1))
            centers, radii = envelopes[elapsed + 1]
            self.assertLessEqual(math.dist((future.x, future.y), centers[0]), radii[0] + 1e-9)

    def test_unknown_or_forced_full_energy_does_not_underestimate_a_burst(self):
        state = offense_state()
        defender = state.team2.players[0]
        defender.x, defender.y, defender.energy, defender.facing = 24, 150, 204, 6
        known = interception_time(state, defender, (0, 150), 21, body=True)
        defender.burst_full_energy = True
        forced = interception_time(state, defender, (0, 150), 21, body=True)
        defender.burst_full_energy = None
        self.assertEqual(interception_time(state, defender, (0, 150), 21, body=True), forced)
        self.assertLess(forced, known)

    def test_projection_holds_the_same_buttons_as_the_controller(self):
        state = offense_state()
        player = state.team1.players[0]
        player.x, player.y, player.motion_x, player.facing = 0, 180, 1.0, 2
        before = deepcopy(vars(player))
        target = (12, 204)
        for interval in (1, 4, 10):
            with self.subTest(interval=interval):
                path = carry_path(player, target, decision_interval=interval)
                expected = player
                for elapsed in range(CARRY_FRAMES):
                    if elapsed % interval == 0:
                        pad = carry_pad(expected, target)
                    expected = grounded_step(expected, pad)
                    self.assertEqual(vars(path[elapsed + 1]), vars(expected))
        self.assertEqual(vars(player), before)

    def test_defender_facing_and_available_boost_change_interception_time(self):
        state = offense_state()
        defender = state.team2.players[0]
        defender.x, defender.y = 24, 150
        defender.facing = defender.facing_phase = 6
        toward = interception_time(state, defender, (0, 150), 21)
        defender.facing = defender.facing_phase = 2
        away = interception_time(state, defender, (0, 150), 21)
        self.assertLessEqual(toward, away)
        defender.facing = defender.facing_phase = 6
        defender.energy = 204
        self.assertGreater(interception_time(state, defender, (0, 150), 21), toward)

    def test_unavailable_defender_is_not_assumed_unavailable_for_the_whole_horizon(self):
        state = offense_state()
        player, defender = state.team1.players[0], state.team2.players[0]
        defender.x, defender.y, defender.unavailable = player.x, player.y, 4
        self.assertEqual(interception_time(state, defender, (player.x, player.y), 21), 0)
        _, details = forecast_carry(state, player, (0, 0))
        self.assertFalse(details['carry_safe'])
        self.assertLess(details['carry_clearance'], 0)

    def test_boosted_body_contact_is_earlier_than_puck_collection_reach(self):
        state = offense_state()
        defender = state.team2.players[0]
        defender.x, defender.y, defender.facing = 24, 150, 6
        body = interception_time(state, defender, (0, 150), 21, body=True)
        puck = interception_time(state, defender, (0, 150), 21)
        self.assertLess(body, puck)

    def test_unknown_defender_ratings_report_the_conservative_model(self):
        state = offense_state()
        state.team2.players[0].weight = None
        _, details = forecast_carry(state, state.team1.players[0], (0, -80))
        self.assertEqual(details['carry_pressure_model'], 'conservative-missing-defender-feedback')

    def test_future_receiver_pressure_includes_defender_reaction_during_the_pass(self):
        original = offense_state()
        player = original.team1.players[0]
        player.x, player.y, player.motion_y = 0, 150, 0.5
        original.puck.x, original.puck.y = 0, 160
        for team in (original.team1, original.team2):
            for other in team.players:
                if other is not player:
                    other.role = -1
        defender = original.team2.players[0]
        defender.role, defender.x, defender.y, defender.facing = 4, 24, 160, 6
        future = deepcopy(original)
        future.team2.players[0].x, future.team2.players[0].y = 100, -100
        _, without = forecast_carry(future, future.team1.players[0], (0, 235))
        _, with_reaction = forecast_carry(
            future, future.team1.players[0], (0, 235), pressure_state=original, pressure_delay=6)
        self.assertTrue(without['carry_safe'])
        self.assertFalse(with_reaction['carry_safe'])
        self.assertGreater(with_reaction['carry_clearance'], 0)

    def test_fractional_contact_time_has_a_bounded_integer_reach_horizon(self):
        state = offense_state()
        defender = state.team2.players[0]
        self.assertGreaterEqual(interception_time(state, defender, (0, 150), 21.5), 0)

    def test_wall_contact_is_not_approved_as_a_small_adjustment(self):
        state = offense_state()
        player = state.team1.players[0]
        player.x, player.y, player.motion_x, player.facing = 119, 0, 2, 2
        path, details = forecast_carry(state, player, (120, 24))
        self.assertIsNone(path)
        self.assertFalse(details['carry_safe'])
        self.assertFalse(details['carry_bounded'])


class ReceiverContinuationTests(unittest.TestCase):
    def test_non_executable_endpoint_gets_position_credit_but_no_finishing_credit(self):
        state = wing_reception_state()
        controller = OffenseController(one_timers=False)
        option = controller.passes(state, 'position')[0][0]
        self.assertGreater(option.continuation_position_value, 0)
        self.assertEqual(option.continuation_finish_value, 0)
        self.assertAlmostEqual(option.continuation_value, option.continuation_position_value * 0.6)
        future = reception_state(state, option)
        receiver = future.team1.players[option.index]
        endpoint = carry_path(receiver, option.continuation_target)[-1]
        snapshot = carry_future(future, receiver, endpoint)
        model = ClassicAIV1Model(SimpleNamespace(action_type='FILTERED', one_timers=False))
        model.predict_frame(snapshot)
        self.assertNotEqual(model._last_decision, 'shoot')

    def test_ready_endpoint_gets_finish_credit_only_after_safe_release(self):
        state = offense_state()
        player = state.team1.players[0]
        player.x, player.y, player.motion_y = -30, 212, 0.8
        state.puck.x, state.puck.y = -30, 222
        state.team2.goalie.x = 15
        for team in (state.team1, state.team2):
            for other in team.players:
                if other is not player:
                    other.role = -1
        controller = OffenseController(one_timers=False)
        _, details = controller._carry_option(state, player, (-35, 235))
        self.assertGreater(details['carry_shot_value'], 0)
        self.assertGreater(details['carry_position_value'], details['carry_shot_value'])
        self.assertEqual(details['carry_release_frames'], 6)

    def test_a_late_body_collision_removes_finishing_credit(self):
        state = offense_state()
        player = state.team1.players[0]
        player.x, player.y = -30, 225
        state.puck.x, state.puck.y = -30, 235
        state.team2.goalie.x = 15
        for team in (state.team1, state.team2):
            for other in team.players:
                if other is not player:
                    other.role = -1
        friend = state.team1.players[1]
        friend.role, friend.x, friend.y, friend.motion_y = 4, -30, 205, 1
        self.assertTrue(ordinary_shot_conditions(state, player, 4, None)[0])
        self.assertGreater(math.dist((player.x, player.y), (friend.x, friend.y)), 16)
        value, release = OffenseController()._finish_value(state, player)
        self.assertEqual(value, 0)
        self.assertEqual(release, 6)

    def test_early_shot_credit_does_not_assume_priority_over_one_timer_preparation(self):
        from tests.unit.test_goalie_avoidance import approach_state
        state = approach_state()
        player = state.team1.players[0]
        player.motion_y = 0
        state.team2.goalie.y = 215
        for team in (state.team1, state.team2):
            for other in team.players:
                if other is not player:
                    other.role = -1
        self.assertEqual(OffenseController(one_timers=True)._finish_value(state, player), (0.0, 0))
        self.assertGreater(OffenseController(one_timers=False)._finish_value(state, player)[0], 0)

    def test_idle_nearby_defender_cannot_be_assumed_idle_for_a_receiver_continuation(self):
        state = wing_reception_state()
        state.team2.players[0].x, state.team2.players[0].motion_x = 80, 0
        _, details = OffenseController(one_timers=False, allow_uncertified=True).passes(state, 'position')
        self.assertFalse(details[0]['continuation_safe'])
        self.assertGreater(details[0]['continuation_risk'], 0)
        self.assertEqual(details[0]['continuation_status'], 'uncertified-clear-samples')
        self.assertGreater(details[0]['continuation_value'], 0)

    def test_wide_receiver_is_valued_for_carrying_into_a_shot(self):
        state = wing_reception_state()
        passer, receiver = state.team1.players[:2]
        immediate, _ = evaluate_pass(state, passer, 1, receiver, 'position')
        self.assertIsNotNone(immediate)
        self.assertEqual(immediate.shot_value, 0)
        controller = OffenseController(one_timers=False)
        plan = controller.choose(state, 1)
        self.assertEqual(plan[0], 'position-pass', controller.diagnostics)
        option = plan[2]
        self.assertGreater(option.continuation_value, controller.diagnostics['retained_value'] + 12)
        self.assertEqual(option.continuation_frames, CARRY_FRAMES)
        self.assertGreater(option.value, immediate.value)
        self.assertTrue(controller.diagnostics['candidates'][0]['worthwhile'])

    def test_receiver_skating_ratings_affect_the_continuation_not_an_arbitrary_bonus(self):
        state = wing_reception_state()
        receiver = state.team1.players[1]
        controller = OffenseController(one_timers=False)
        fast = controller.passes(state, 'position')[0][0]
        receiver.speed, receiver.agility = 0, 0
        slow = controller.passes(state, 'position')[0][0]
        self.assertEqual(fast.shot_value, slow.shot_value)
        self.assertGreater(fast.continuation_value, slow.continuation_value)
        self.assertGreater(fast.value, slow.value)

    def test_keeping_the_puck_wins_when_the_carrier_has_the_better_continuation(self):
        state = wing_reception_state(carrier_x=65)
        controller = OffenseController(one_timers=False)
        plan = controller.choose(state, 1)
        self.assertNotEqual(plan[0], 'position-pass')
        candidate = controller.diagnostics['candidates'][0]
        self.assertEqual(candidate['status'], 'safe')
        self.assertFalse(candidate['worthwhile'])
        self.assertGreater(controller.diagnostics['retained_value'], candidate['continuation_value'] - 12)

    def test_reception_preserves_body_pose_momentum_facing_and_original_state(self):
        state = wing_reception_state()
        receiver = state.team1.players[1]
        receiver.precise_x, receiver.precise_y = receiver.x + 0.25, receiver.y + 0.5
        state.team2.players[0].precise_x = 80.25
        option, _ = evaluate_pass(state, state.team1.players[0], 1, receiver, 'position')
        before = deepcopy(vars(receiver))
        future = reception_state(state, option)
        elapsed = reception_frames(state, option)
        self.assertLess(elapsed, option.flight_frames + 4)
        projected = future.team1.players[1]
        self.assertAlmostEqual(projected.x, receiver.x + receiver.motion_x * elapsed)
        self.assertNotEqual((projected.x, projected.y), option.point)
        self.assertEqual((projected.motion_x, projected.motion_y, projected.facing),
                         (receiver.motion_x, receiver.motion_y, receiver.facing))
        self.assertEqual((projected.precise_x, projected.precise_y), (projected.x, projected.y))
        self.assertEqual(future.engine.puck_owner, option.slot)
        self.assertEqual(state.engine.puck_owner, 0)
        self.assertEqual(vars(receiver), before)

    def test_continuation_cannot_rescue_an_intercepted_pass(self):
        state = wing_reception_state()
        blocker = state.team2.players[0]
        blocker.x, blocker.y = 85, 165
        choices, details = OffenseController(one_timers=False).passes(state, 'position')
        self.assertFalse(choices)
        self.assertNotIn('continuation_value', details[0])
        self.assertNotEqual(details[0]['status'], 'safe')

    def test_uncertain_contact_does_not_crash_or_invent_a_carry_opportunity(self):
        state = wing_reception_state()
        option, _ = evaluate_pass(state, state.team1.players[0], 1, state.team1.players[1], 'position')
        with patch('nhl94_ai.agents.offense.one_timer_contact_frame', return_value=None):
            result, details = OffenseController()._receiver_continuation(state, option)
            self.assertIsNone(reception_frames(state, option))
            with self.assertRaisesRegex(ValueError, 'without modeled body/stick contact'):
                reception_state(state, option)
        self.assertIs(result, option)
        self.assertEqual(result.continuation_value, 0)
        self.assertEqual(details, {'continuation_status': 'no-modeled-reception-contact'})

    def test_one_timer_score_does_not_gain_a_carry_continuation(self):
        from tests.unit.test_classic_offense import live_one_timer_state
        choices, details = OffenseController().passes(live_one_timer_state(), 'one-timer')
        self.assertTrue(choices, details)
        self.assertEqual(choices[0].continuation_value, 0)
        self.assertEqual(choices[0].continuation_frames, 0)

    def test_pass_reception_replans_from_the_actual_receiver_at_frame_cadence(self):
        for schema in ('FILTERED', 'HOCKEY_INTENT_DPAD'):
            for interval in (4, 10):
                with self.subTest(schema=schema, interval=interval):
                    state = wing_reception_state()
                    model = ClassicAIV1Model(SimpleNamespace(action_type=schema, one_timers=False))
                    action = model.predict_frame(state, interval)[0]
                    processor = HockeyActionController(SimpleNamespace(action_type=schema, game_state=state))
                    buttons = processor._process_action(action, processor._new_action_state())[0]
                    self.assertTrue(buttons[Buttons.INPUT_B])
                    self.assertIsNotNone(model.offense.pending)
                    state.engine.puck_owner = -256
                    model.predict_frame(state, interval)
                    state.engine.puck_owner = 2
                    state.team1.defense_control, state.team1.control = 2, 3
                    state.team1.players[2].role = 4
                    state.team1.players[2].x, state.team1.players[2].y = 70, 180
                    model.predict_frame(state, interval)
                    self.assertIsNone(model.offense.pending)
                    self.assertEqual(model.offense.last_pass['outcome'], 'other-receiver')
                    self.assertEqual(model.offense_diagnostics['actual_slot'], 2)
                    self.assertNotIn(model._last_decision, ('pass-release', 'pass-flight'))

    def test_better_future_shot_can_pass_the_positional_eligibility_filter(self):
        state = wing_reception_state()
        option, _ = evaluate_pass(state, state.team1.players[0], 1, state.team1.players[1], 'position')
        self.assertFalse(OffenseController._worthwhile(option, state.team1.players[0], 1, 'position', 0))
        improved = replace(option, continuation_value=30)
        self.assertTrue(OffenseController._worthwhile(improved, state.team1.players[0], 1, 'position', 0))

    def test_scores_include_rejected_teammates_even_on_a_breakaway(self):
        state = offense_state()
        for opponent in state.team2.players:
            opponent.y = -240
        model = ClassicAIV1Model()
        model.predict_frame(state)
        self.assertEqual(model._last_decision, 'carry-breakaway')
        scores = model.offense_diagnostics['teammate_scores']
        self.assertEqual([row['slot'] for row in scores], [1, 2, 3, 4])
        self.assertTrue(any(row['pass']['status'] == 'safe' for row in scores))
        self.assertTrue(any(row['pass']['status'] != 'safe' for row in scores))
        self.assertTrue(all('value' not in row['pass'] for row in scores if row['pass']['status'] != 'safe'))


if __name__ == '__main__':
    unittest.main()
