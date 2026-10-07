"""Progression, moving-pass safety, bounded feints and outcome attribution."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from nhl94_ai.agents.base import AgentInput
from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.agents.carry import forecast_carry
from nhl94_ai.agents.motion import VELOCITY_SCALE
from nhl94_ai.agents.offense import FEINT_FRAMES, OffenseController, carry_clear, carry_projection, projected_state
from nhl94_ai.agents.passing import (
    PassOption, evaluate_pass, pad_direction, pass_contact, pass_speed, rom_direction, selected_receiver,
)
from nhl94_ai.agents.registry import create_scripted
from nhl94_ai.env.actions import HockeyActionController
from nhl94_ai.evaluation.benchmark import away_view
from nhl94_ai.evaluation.offense_metrics import OffenseMetrics
from nhl94_ai.game.constants import GameConsts as Buttons
from nhl94_ai.game.ram import register_defense_state
from nhl94_ai.game.state import NHL94GameState


def offense_state():
    state = NHL94GameState(5)
    state.team1.net.y, state.team2.net.y = -264, 264
    state.team1.control, state.team1.defense_control = 1, 0
    state.engine.puck_owner, state.engine.puck_owner_known = 0, True
    state.engine.offsides_enabled = False
    state.team1.pass_attempts = 0
    for team in (state.team1, state.team2):
        for player in (*team.players, team.goalie):
            player.motion_x = player.motion_y = 0
            player.speed, player.agility, player.weight = 20, 20, 64
            player.passing, player.energy, player.stick = 20, 4096, 20
            player.burst_full_energy = False
            player.role, player.facing, player.selection_flags = 4, 0, 0
            player.shot_accuracy = 20
        team.goalie.role = 0
    for player, point in zip(state.team1.players, ((0, -130), (-80, -30), (100, -210), (110, -240), (70, -220))):
        player.x, player.y = point
    for player, point in zip(state.team2.players, ((40, -75), (110, 80), (110, 140), (90, 180), (100, 220))):
        player.x, player.y = point
    state.team1.players[0].selection_flags = 8
    state.team1.goalie.y, state.team2.goalie.y = -250, 250
    state.puck.x, state.puck.y = 0, -120
    state.puck.motion_x = state.puck.motion_y = 0
    state.puck.height = 0
    return state


def live_one_timer_state(*, away=False):
    state = offense_state()
    passer, receiver = state.team1.players[:2]
    passer.x, passer.y, passer.facing = -50, 180, 1
    receiver.x, receiver.y, receiver.motion_y = 35, 160, 0.7
    state.puck.x, state.puck.y = -50, 190
    for other in (*state.team1.players[2:], *state.team2.players):
        other.x, other.y = 100, -180
    if away:
        for team in (state.team1, state.team2):
            team.controller = 3 - team.controller
            team.defense_control = team.skater_scnum_base()
            team.net.y = -team.net.y
            for player in (*team.players, team.goalie):
                player.x, player.y = -player.x, -player.y
                player.motion_x, player.motion_y = -player.motion_x, -player.motion_y
                player.facing = (player.facing + 4) % 8
        state.engine.puck_owner = 6
        state.puck.x, state.puck.y = -state.puck.x, -state.puck.y
    return state


class PassEstimateTests(unittest.TestCase):
    def setUp(self):
        self.state = offense_state()
        self.passer, self.receiver = self.state.team1.players[:2]

    def _evaluate(self, purpose='advance'):
        return evaluate_pass(self.state, self.passer, 1, self.receiver, purpose)

    def test_inactive_friendly_blocker_and_unknown_opponent_motion_do_not_veto_a_pass(self):
        baseline, _ = self._evaluate()
        self.assertIsNotNone(baseline)
        friend = self.state.team1.players[2]
        friend.role, friend.x, friend.y = -1, -40, -70
        opponent = self.state.team2.players[0]
        opponent.role = -1
        opponent.motion_x = opponent.motion_y = None
        option, details = self._evaluate()
        self.assertIsNotNone(option, details)
        self.assertEqual(option.bypassed, 0)

    def test_inactive_skater_does_not_block_shots_but_a_fallen_on_ice_skater_does(self):
        from nhl94_ai.agents.passing import shot_value
        self.passer.x, self.passer.y = 0, 200
        self.state.team2.goalie.x = -70
        for other in self.state.team2.players:
            other.x, other.y = 100, -180
        baseline = shot_value(self.state, self.passer)
        blocker = self.state.team2.players[0]
        blocker.role, blocker.x, blocker.y = -1, 0, 230
        self.assertEqual(shot_value(self.state, self.passer), baseline)
        blocker.role, blocker.is_falling, blocker.unavailable = 4, 1, 4
        self.assertLess(shot_value(self.state, self.passer), baseline)

    def test_inactive_projection_retains_off_ice_position_without_board_uncertainty(self):
        other = self.state.team2.players[0]
        other.role, other.x, other.motion_x = -1, 119, 4
        future = projected_state(self.state, (-20, -80))
        self.assertIsNotNone(future)
        self.assertEqual(future.team2.players[0].x, 119)
        self.assertEqual(future.team2.players[0].projection_uncertainty, 0)

    def test_passing_rating_is_speed_not_random_accuracy(self):
        self.passer.passing = 0
        slow = pass_speed(self.passer)
        self.passer.passing = 30
        self.assertGreater(pass_speed(self.passer), slow)
        self.passer.passing = 29
        odd = pass_speed(self.passer)
        self.passer.passing = 30
        self.assertGreater(odd, pass_speed(self.passer))
        self.passer.passing = None
        self.assertIsNone(pass_speed(self.passer))

    def test_rom_direction_boundaries_and_later_slot_ties(self):
        self.assertEqual(rom_direction(100, 49), 2)
        self.assertEqual(rom_direction(100, 50), 1)
        self.assertEqual(rom_direction(-100, -50), 5)
        self.assertEqual(rom_direction(0, 0), 8)
        direction = pad_direction(self.passer, self.receiver)
        self.assertEqual(selected_receiver(self.state.team1, self.state.puck, 0, direction), 1)
        self.state.team1.players[2].x, self.state.team1.players[2].y = self.receiver.x, self.receiver.y
        self.assertEqual(selected_receiver(self.state.team1, self.state.puck, 0, direction), 2)

    def test_unobstructed_advancement_predicts_space_at_reception(self):
        option, details = self._evaluate()
        self.assertIsNotNone(option, details)
        self.assertEqual(details['status'], 'safe')
        self.assertGreater(option.forward_gain, 80)
        self.assertGreaterEqual(option.margin, 3)
        self.assertEqual(option.robustness, 1)

    def test_receiver_motion_changes_contact_time_and_point(self):
        stationary = pass_contact(self.state.puck, self.passer, self.receiver)
        self.receiver.motion_x = -0.5
        moving = pass_contact(self.state.puck, self.passer, self.receiver)
        self.assertIsNotNone(moving)
        self.assertLess(moving[1][0], stationary[1][0])
        self.assertGreater(moving[2], stationary[2])

    def test_static_clear_lane_can_become_a_moving_interception(self):
        opponent = self.state.team2.players[0]
        opponent.x, opponent.y = 0, -70
        opponent.motion_x = -2
        option, details = self._evaluate()
        self.assertIsNone(option)
        self.assertIn(details['status'], ('moving-interception', 'contested-reception-or-lane'))

    def test_clear_lane_but_contested_receiver_is_rejected(self):
        self.state.team2.players[0].x, self.state.team2.players[0].y = -90, -15
        option, details = self._evaluate()
        self.assertIsNone(option)
        self.assertIn(details['status'], ('moving-interception', 'contested-reception-or-lane'))

    def test_wrong_recipient_is_rejected_before_risk_estimation(self):
        self.state.team1.players[2].x, self.state.team1.players[2].y = -40, -75
        option, details = self._evaluate()
        self.assertIsNone(option)
        self.assertEqual(details['status'], 'different-rom-recipient')

    def test_unknown_motion_and_ratings_are_explicitly_unsafe(self):
        for player, field in ((self.passer, 'passing'), (self.receiver, 'motion_x'), (self.receiver, 'energy')):
            before = getattr(player, field)
            setattr(player, field, None)
            option, details = self._evaluate()
            self.assertIsNone(option)
            self.assertEqual(details['status'], 'missing-feedback')
            setattr(player, field, before)
        self.state.team2.players[0].motion_x = None
        self.assertEqual(self._evaluate()[1]['status'], 'unknown-opponent-motion')

    def test_forward_progress_and_receiving_speed_are_hard_gates(self):
        self.receiver.x, self.receiver.y = -80, -125
        self.assertEqual(self._evaluate()[1]['status'], 'insufficient-progress')
        self.receiver.x, self.receiver.y = -80, -30
        self.passer.passing, self.receiver.stick = 30, 0
        self.assertEqual(self._evaluate()[1]['status'], 'too-fast-to-control')

    def test_ordinary_catch_speed_uses_live_stick_without_energy_scaling(self):
        self.passer.passing = 30
        for purpose in ('advance', 'position'):
            for stick in (0, 20, 30):
                self.receiver.stick = stick
                self.receiver.energy = 4096
                expected = self._evaluate(purpose)
                for energy in (0, 1024, 2048, 4096):
                    with self.subTest(purpose=purpose, stick=stick, energy=energy):
                        self.receiver.energy = energy
                        result = self._evaluate(purpose)
                        self.assertEqual(result, expected)
                        self.assertEqual(result[1]['status'], 'too-fast-to-control' if stick == 0 else 'safe')

    def test_ordinary_catch_accepts_exact_rom_speed_threshold(self):
        start, point, flight, _ = pass_contact(self.state.puck, self.passer, self.receiver)
        for purpose in ('advance', 'position'):
            for stick in (0, 15, 30):
                self.receiver.stick = stick
                for energy in (1024, 4096):
                    self.receiver.energy = energy
                    for difference in (-1, 0, 1):
                        with self.subTest(purpose=purpose, stick=stick, energy=energy, difference=difference):
                            speed = (13000 + 350 * stick + difference) * VELOCITY_SCALE
                            with patch('nhl94_ai.agents.passing.pass_contact', return_value=(start, point, flight, speed)):
                                option, details = self._evaluate(purpose)
                            if difference > 0:
                                self.assertIsNone(option)
                                self.assertEqual(details['status'], 'too-fast-to-control')
                            else:
                                self.assertIsNotNone(option, details)

    def test_uncertainty_is_reproducible_not_a_random_pass_permission(self):
        first = self._evaluate()
        np.random.seed(999)
        self.assertEqual(first, self._evaluate())
        for _ in range(3):
            self.state.team2.players[0].x, self.state.team2.players[0].y = -40, -70
            self.assertIsNone(self._evaluate()[0])

    def test_registered_passing_is_optional_and_skips_unused_slots(self):
        env = Mock()
        register_defense_state(env, 2)
        fields = dict(call.args for call in env.data.set_variable.call_args_list)
        self.assertEqual(fields['defense_7_passing'], {'address': 0xFFB04A + 7 * 0x80 + 0x6E, 'type': '|u1'})
        self.assertNotIn('defense_2_passing', fields)
        self.assertNotIn('p1_passing', fields)

    def test_airborne_puck_and_enabled_offsides_are_rejected(self):
        self.state.puck.height = 20
        self.assertEqual(self._evaluate()[1]['status'], 'airborne-or-unsafe-route')
        self.state.puck.height = 0
        self.passer.y, self.state.puck.y, self.receiver.y = 0, 10, 105
        self.state.engine.offsides_enabled = True
        self.assertEqual(self._evaluate()[1]['status'], 'offside-receiver')
        self.state.engine.offsides_enabled = None
        self.assertEqual(self._evaluate()[1]['status'], 'unknown-offside-rule')

    def test_predicted_pass_start_inside_net_rejected_without_route_search(self):
        self.passer.x, self.passer.y = 28, -254
        self.passer.motion_x, self.passer.motion_y = -0.01, -0.26
        self.state.puck.x, self.state.puck.y = 28, -254
        self.receiver.x, self.receiver.y = -80, -230
        option, details = self._evaluate()
        self.assertIsNone(option)
        self.assertEqual(details['status'], 'airborne-or-unsafe-route')

class OneTimerPassEstimateTests(unittest.TestCase):
    def test_live_contact_deadline_covers_the_flight_at_all_decision_intervals(self):
        for interval in (1, 4, 10):
            with self.subTest(interval=interval):
                state = live_one_timer_state()
                model = ClassicAIV1Model()
                choices, details = model.offense.passes(state, 'one-timer')
                self.assertTrue(choices, details)
                model.predict_frame(state, frame_skip=interval)
                self.assertIsNotNone(model._one_timer)
                self.assertGreater(model._one_timer[2] - model._one_timer_started,
                                   choices[0].flight_frames + 4)
                self.assertLess(model._one_timer[2] - model._one_timer_started,
                                choices[0].flight_frames + 25)
                state.engine.puck_owner = -256
                for _ in range(25):
                    model.predict_frame(state, frame_skip=interval)
                self.assertIsNotNone(model._one_timer)

    def test_cue_eligibility_accounts_for_the_configured_decision_interval(self):
        state = live_one_timer_state()
        passer, receiver = state.team1.players[:2]
        receiver.y, receiver.motion_y = 205, 0
        receiver.stick_x = receiver.stick_y = 0
        for flight, allowed in ((8, False), (8.01, False), (10.5, True)):
            with self.subTest(flight=flight):
                with patch('nhl94_ai.agents.passing.pass_contact',
                           return_value=((-50, 190), (35, 205), flight, 2)):
                    option, details = evaluate_pass(
                        state, passer, 1, receiver, 'one-timer', decision_interval=10)
                self.assertEqual(details['cue_frame'], 10)
                if allowed:
                    self.assertIsNotNone(option, details)
                else:
                    self.assertIsNone(option)
                    self.assertEqual(details['status'], 'too-short-for-one-timer-cue')

    def test_one_timer_uses_future_reception_depth_in_both_attacking_directions(self):
        for away in (False, True):
            with self.subTest(away=away):
                state = live_one_timer_state(away=away)
                passer, receiver = state.team1.players[:2]
                sign = -1 if away else 1
                self.assertLess(receiver.y * sign, 175)
                choices, details = OffenseController().passes(state, 'one-timer')
                self.assertTrue(choices, details)
                self.assertGreaterEqual(choices[0].point[1] * sign, 175)
                model = ClassicAIV1Model()
                action = model.predict_frame(state)[0]
                self.assertEqual(model._last_decision, 'one-timer-pass')
                self.assertTrue(action[Buttons.INPUT_B])
                self.assertEqual(model._one_timer[1], state.team1.skater_scnum_base() + 1)

    def test_wide_receiver_entering_the_slot_is_judged_at_contact(self):
        state = live_one_timer_state()
        receiver = state.team1.players[1]
        receiver.x, receiver.motion_x = 90, -1.2
        choices, details = OffenseController().passes(state, 'one-timer')
        self.assertTrue(choices, details)
        self.assertLessEqual(abs(choices[0].point[0]), 70)
        self.assertGreaterEqual(choices[0].point[1], 175)

    def test_pass_arriving_before_the_c_cue_is_not_a_deliberate_one_timer(self):
        state = live_one_timer_state()
        passer, receiver = state.team1.players[:2]
        passer.x, passer.y = 0, 200
        receiver.x, receiver.y, receiver.motion_y = 4, 200, 0
        state.puck.x, state.puck.y = 0, 200
        option, details = evaluate_pass(state, passer, 1, receiver, 'one-timer')
        self.assertIsNone(option)
        self.assertEqual(details['status'], 'too-short-for-one-timer-cue')

    def test_drifting_out_of_shooting_range_is_not_a_one_timer_opportunity(self):
        for axis in ('x', 'y'):
            with self.subTest(axis=axis):
                state = live_one_timer_state()
                receiver = state.team1.players[1]
                receiver.y, receiver.motion_y = 190, 0
                setattr(receiver, 'motion_' + axis, 1.5 if axis == 'x' else -1.6)
                choices, details = OffenseController().passes(state, 'one-timer')
                self.assertFalse(choices)
                self.assertIn('outside-one-timer-window', [detail['status'] for detail in details])

    def test_same_side_one_timer_requires_a_stronger_receiving_position(self):
        state = live_one_timer_state()
        passer, receiver = state.team1.players[:2]
        passer.x, passer.y, passer.shot_accuracy = 55, 150, 0
        receiver.x, receiver.y, receiver.motion_y, receiver.shot_accuracy = 20, 200, 0, 30
        state.puck.x, state.puck.y = 55, 160
        controller = OffenseController()
        self.assertGreater(passer.x * receiver.x, 0)
        choices, details = controller.passes(state, 'one-timer')
        self.assertTrue(choices, details)
        model = ClassicAIV1Model()
        model.predict_frame(state)
        self.assertEqual(model._last_decision, 'one-timer-pass')
        passer.x, passer.y, passer.shot_accuracy = 20, 205, 30
        receiver.x, receiver.y, receiver.shot_accuracy = 55, 195, 0
        state.team2.goalie.x = 35
        state.puck.x, state.puck.y = 20, 215
        choices, details = controller.passes(state, 'one-timer')
        self.assertFalse(choices)
        self.assertIn('weaker-one-timer-position', [detail['status'] for detail in details])

    def test_one_timer_bypasses_normal_catch_speed_but_not_interception_checks(self):
        state = live_one_timer_state()
        passer, receiver = state.team1.players[:2]
        passer.passing, receiver.stick = 30, 0
        receiver.y, receiver.motion_y = 205, 0
        option, details = evaluate_pass(state, passer, 1, receiver, 'position')
        self.assertIsNone(option)
        self.assertEqual(details['status'], 'too-fast-to-control')
        option, details = evaluate_pass(state, passer, 1, receiver, 'one-timer')
        self.assertIsNotNone(option, details)
        opponent = state.team2.players[0]
        opponent.x, opponent.y = -10, 197
        option, details = evaluate_pass(state, passer, 1, receiver, 'one-timer')
        self.assertIsNone(option)
        self.assertEqual(details['status'], 'moving-interception')

    def test_body_contact_between_flight_samples_is_rejected(self):
        state = live_one_timer_state()
        passer, receiver = state.team1.players[:2]
        passer.x, passer.facing = 0, 1
        receiver.x, receiver.y, receiver.motion_y = 35, 205, 0
        state.puck.x = 0
        blocker = state.team1.players[2]
        blocker.x, blocker.y, blocker.unavailable = 15, 207, 4
        option, details = evaluate_pass(state, passer, 1, receiver, 'one-timer')
        self.assertIsNone(option)
        self.assertEqual(details['status'], 'friendly-obstruction')


class OffenseControllerTests(unittest.TestCase):
    def setUp(self):
        self.state = offense_state()
        self.controller = OffenseController()

    def test_advancement_pass_preferred_to_long_carry(self):
        plan = self.controller.choose(self.state, 1)
        self.assertIsNotNone(plan, self.controller.diagnostics)
        self.assertEqual(plan[0], 'advance-pass')
        self.assertEqual(plan[2].slot, 1)

    def test_an_ineligible_top_pass_does_not_hide_an_eligible_second_choice(self):
        for purpose in ('advance', 'position'):
            with self.subTest(purpose=purpose):
                state = offense_state()
                if purpose == 'position':
                    state.team1.players[0].y = 150
                first = PassOption(1, 1, (40, -90), 20, 1, 20, 1, 40, 0, 20, 36)
                second = PassOption(2, 2, (-80, -70), 20, 7, 5.9, 1, 60, 0, 30, 29.9)
                controller = OffenseController()
                details = [{'slot': option.slot, 'status': 'safe'} for option in (first, second)]
                with patch.object(controller, 'passes', return_value=([first, second], details)), \
                        patch.object(controller, 'breakaway', return_value=False), \
                        patch.object(controller, '_feint', return_value=None), \
                        patch('nhl94_ai.agents.offense.shot_value', return_value=15):
                    plan = controller.choose(state, 1)
                self.assertEqual(plan[0], purpose + '-pass')
                self.assertIs(plan[2], second)
                self.assertFalse(details[0]['worthwhile'])
                self.assertTrue(details[1]['worthwhile'])

    def test_breakaway_keeps_puck_and_does_not_trust_flag_alone(self):
        self.state.team1.players[0].is_breakaway = 1
        self.assertFalse(self.controller.breakaway(self.state, self.state.team1.players[0]))
        for opponent in self.state.team2.players:
            opponent.y = -240
        plan = self.controller.choose(self.state, 1)
        self.assertEqual(plan[0], 'carry-breakaway')
        self.assertIsNone(plan[2])
        self.state.team1.players[0].x, self.state.team1.players[0].y = 100, 160
        plan = self.controller.choose(self.state, 2)
        self.assertEqual(plan[0], 'carry-breakaway')
        self.assertLessEqual(abs(plan[1][0]), 35)

    def test_close_finisher_does_not_pass_just_for_more_progress(self):
        player = self.state.team1.players[0]
        player.y = 205
        for opponent in self.state.team2.players:
            opponent.y = -240
        self.assertIsNone(self.controller.choose(self.state, 1))
        self.assertEqual(self.controller.diagnostics['status'], 'close-to-finish')

    def test_live_cut_can_create_and_continue_a_safe_one_timer_setup(self):
        state = live_one_timer_state()
        passer, receiver = state.team1.players[:2]
        passer.x, passer.facing = 0, 1
        receiver.x, receiver.y, receiver.motion_y = 35, 205, 0
        blocker = state.team1.players[2]
        blocker.x, blocker.y, blocker.unavailable = 15, 209, 4
        state.puck.x = 0
        state.team2.players[0].x, state.team2.players[0].y = 70, 245
        state.team2.goalie.x, state.team2.goalie.y = 0, 220
        self.assertFalse(self.controller.passes(state, 'one-timer')[0])
        plan = self.controller.choose(state, 1)
        self.assertIsNotNone(plan, self.controller.diagnostics)
        self.assertEqual(plan[0], 'one-timer-setup', self.controller.diagnostics)
        self.assertGreater(plan[1][0], passer.x)
        self.assertIsNone(plan[2])
        self.assertEqual(self.controller.choose(state, 5)[0], 'one-timer-setup')
        self.assertLessEqual(self.controller.feint_until, 1 + FEINT_FRAMES)
        disabled = OffenseController(one_timers=False)
        plan = disabled.choose(state, 1)
        self.assertTrue(plan is None or plan[0] != 'one-timer-setup')

    def test_pass_release_and_reception_cancel_without_one_timer_c(self):
        for schema in ('FILTERED', 'HOCKEY_INTENT_DPAD'):
            with self.subTest(schema=schema):
                state = deepcopy(self.state)
                model = ClassicAIV1Model(SimpleNamespace(action_type=schema))
                processor = HockeyActionController(SimpleNamespace(action_type=schema, game_state=state))
                macro = processor._new_action_state()
                action = model.predict_frame(state)[0]
                buttons = processor._process_action(action, macro)[0]
                self.assertTrue(buttons[Buttons.INPUT_B])
                self.assertEqual(model._last_decision, 'advance-pass')
                state.engine.puck_owner, state.engine.last_puck_player = -256, 0
                state.engine.pass_target, state.team1.pass_attempts = 1, 1
                for _ in range(4):
                    buttons = processor._process_action(model.predict_frame(state)[0], macro)[0]
                    self.assertFalse(buttons[Buttons.INPUT_C])
                self.assertEqual(model.offense.pending['actual_receiver'], 1)
                state.engine.puck_owner, state.team1.defense_control, state.team1.control = 1, 1, 2
                model.predict_frame(state)
                self.assertIsNone(model.offense.pending)
                self.assertEqual(model.offense.last_pass['outcome'], 'received')

    def test_original_passer_recovery_cancels_ordinary_wait_immediately(self):
        for schema in ('FILTERED', 'HOCKEY_INTENT_DPAD'):
            for interval in (1, 4, 8, 10):
                with self.subTest(schema=schema, interval=interval):
                    state = deepcopy(self.state)
                    model = ClassicAIV1Model(SimpleNamespace(action_type=schema))
                    model.predict_frame(state, interval)
                    self.assertIsNotNone(model.offense.pending)
                    state.engine.puck_owner = -256
                    model.predict_frame(state, interval)
                    state.engine.puck_owner = 0
                    model.predict_frame(state, interval)
                    self.assertIsNone(model.offense.pending)
                    self.assertEqual(model.offense.last_pass['outcome'], 'recovered-by-passer')
                    self.assertNotIn(model._last_decision, ('pass-flight', 'pass-release'))

    def test_a_pass_attempt_counter_does_not_mistake_initial_possession_for_recovery(self):
        state = deepcopy(self.state)
        model = ClassicAIV1Model()
        model.predict_frame(state)
        state.team1.pass_attempts, state.engine.pass_target = 1, 1
        for _ in range(5):
            model.predict_frame(state)
        self.assertIsNotNone(model.offense.pending)
        self.assertTrue(model.offense.pending['launched'])
        self.assertFalse(model.offense.pending['flight_observed'])

    def test_turnover_immediately_cancels_pending_offense(self):
        model = ClassicAIV1Model()
        model.predict_frame(self.state)
        self.assertIsNotNone(model.offense.pending)
        self.state.engine.puck_owner = 6
        model.predict_frame(self.state)
        self.assertIsNone(model.offense.pending)
        self.assertEqual(model.offense_diagnostics, {})
        self.assertTrue(model.defense_diagnostics)

    def test_intent_recipient_mapping_prefers_actual_control_over_stale_star(self):
        self.state.team1.control = 3
        model = ClassicAIV1Model(SimpleNamespace(action_type='HOCKEY_INTENT_DPAD'))
        action = model.predict_frame(self.state)[0]
        processor = HockeyActionController(SimpleNamespace(action_type='HOCKEY_INTENT_DPAD', game_state=self.state))
        self.assertIs(processor._hockey_controlled_player(self.state.team1), self.state.team1.players[0])
        self.assertIs(processor._hockey_pass_target(int(action[0])), self.state.team1.players[1])

    def test_timeout_wrong_receiver_and_stale_target(self):
        option = self.controller.choose(self.state, 0)[2]
        self.controller.start_pass(self.state, option, 0, 'advance-pass')
        self.state.engine.pass_target = 3
        self.controller.observe(self.state, 1)
        self.assertIsNone(self.controller.pending['actual_receiver'])
        self.state.engine.puck_owner = 2
        self.controller.observe(self.state, 2)
        self.assertEqual(self.controller.last_pass['outcome'], 'other-receiver')
        self.state.engine.puck_owner = 0
        self.controller.start_pass(self.state, option, 3, 'advance-pass')
        self.controller.observe(self.state, 100)
        self.assertEqual(self.controller.last_pass['outcome'], 'not-released')

    def test_reset_clears_pending_feint_and_pass(self):
        agent = create_scripted('classic-v1', SimpleNamespace(action_type='FILTERED'))
        output = agent.act(AgentInput(self.state))
        self.assertIn('classic_offense', output.diagnostics)
        self.assertTrue(agent.offense.pending)
        agent.offense.feint_target = (50, 150)
        agent.reset()
        self.assertIsNone(agent.offense.pending)
        self.assertIsNone(agent.offense.feint_target)

    def test_cut_projection_cannot_reverse_momentum_instantly(self):
        player = self.state.team1.players[0]
        player.motion_x = 2
        motion = carry_projection(player, (-30, player.y), 4)
        self.assertGreater(motion[0][0], player.x)
        self.assertGreater(motion[1][0], 0)

    def test_feint_requires_a_gain_and_has_bounded_cooldown(self):
        player = self.state.team1.players[0]
        player.y = 180
        self.state.puck.y = 190
        for opponent in self.state.team2.players:
            opponent.x, opponent.y = 110, -200
        with patch('nhl94_ai.agents.offense.shot_value', return_value=10):
            self.assertIsNone(self.controller._feint(self.state, player, 1, 10))
        with patch('nhl94_ai.agents.offense.shot_value',
                   side_effect=lambda state, mover: 30 if abs(mover.x) > 1 else 10), \
                patch.object(self.controller, 'passes', return_value=([], [])):
            cut = self.controller._feint(self.state, player, 1, 10)
            self.assertIsNotNone(cut)
            self.assertEqual(cut[0], 'feint')
            self.assertEqual(self.controller.feint_until, 1 + FEINT_FRAMES)
            self.assertIsNone(self.controller._feint(self.state, player, 2, 10))

    def test_real_geometry_cut_opens_an_option_not_just_forward_progress(self):
        player = self.state.team1.players[0]
        player.y, self.state.puck.y, self.state.team2.goalie.x = 180, 190, -18
        for opponent in self.state.team2.players:
            opponent.x, opponent.y = 110, -180
        self.state.team2.players[0].x, self.state.team2.players[0].y = -8, 210
        plan = self.controller.choose(self.state, 1)
        self.assertEqual(plan[0], 'feint')
        self.assertGreater(plan[1][0], player.x)
        self.assertEqual(self.controller.diagnostics['reason'], 'open shooting lane')
        self.state.team2.players[0].x = -6
        self.assertIsNone(OffenseController().choose(self.state, 1))

class OffenseSafetyTests(unittest.TestCase):
    def setUp(self):
        self.state = offense_state()
        self.controller = OffenseController()

    def test_remote_board_contact_does_not_veto_a_valid_shooting_cut(self):
        player = self.state.team1.players[0]
        player.y, self.state.puck.y, self.state.team2.goalie.x = 180, 190, -18
        for opponent in self.state.team2.players:
            opponent.x, opponent.y = 110, -180
        self.state.team2.players[0].x, self.state.team2.players[0].y = -8, 210
        remote = self.state.team1.players[3]
        remote.x, remote.y, remote.motion_x = 119, -200, 0.5
        plan = self.controller.choose(self.state, 1)
        self.assertEqual(plan[0], 'feint')
        future = projected_state(self.state, plan[1])
        self.assertIsNotNone(future)
        self.assertEqual(future.team1.players[3].x, 120)
        self.assertGreaterEqual(future.team1.players[3].projection_uncertainty, 8)
        self.assertEqual((remote.x, remote.motion_x, remote.projection_uncertainty), (119, 0.5, 0))

    def test_carrier_braking_is_validated_instead_of_its_unused_linear_projection(self):
        player = self.state.team1.players[0]
        player.x, player.y, player.motion_x, player.facing = 119, 0, 0.2, 2
        future = projected_state(self.state, (71, 0))
        self.assertIsNotNone(future)
        self.assertLessEqual(future.team1.players[0].x, 120)
        player.motion_x = 2
        self.assertIsNone(projected_state(self.state, (71, 0)))

    def test_projected_carry_preserves_the_live_puck_offset(self):
        player = self.state.team1.players[0]
        self.state.puck.x, self.state.puck.y = player.x - 16, player.y + 10
        future = projected_state(self.state, (26, player.y + 8))
        self.assertIsNotNone(future)
        carrier = future.team1.players[0]
        self.assertAlmostEqual(future.puck.x - carrier.x, -16)
        self.assertAlmostEqual(future.puck.y - carrier.y, 10)
        self.assertEqual((self.state.puck.x, self.state.puck.y), (player.x - 16, player.y + 10))

    def test_a_relevant_board_contact_remains_a_conservative_collision_risk(self):
        player = self.state.team1.players[0]
        player.x, player.y = 105, 0
        blocker = self.state.team2.players[0]
        blocker.x, blocker.y, blocker.motion_x = 119, 0, 1
        self.assertFalse(carry_clear(self.state, player, (105, 0)))
        future = projected_state(self.state, (105, 0))
        self.assertIsNotNone(future)
        self.assertGreater(future.team2.players[0].projection_uncertainty, 0)

    def test_an_uncertain_board_receiver_is_rejected_without_vetoing_the_state(self):
        receiver = self.state.team1.players[1]
        receiver.projection_uncertainty = 8
        option, details = evaluate_pass(self.state, self.state.team1.players[0], 1, receiver)
        self.assertIsNone(option)
        self.assertEqual(details['status'], 'uncertain-reception')

    def test_unsafe_default_carry_reports_an_uncertified_escape_on_either_side(self):
        for away in (False, True):
            for schema in ('FILTERED', 'HOCKEY_INTENT_DPAD'):
                with self.subTest(away=away, schema=schema):
                    state = offense_state()
                    player = state.team1.players[0]
                    player.x, player.y, player.motion_x, player.motion_y = 0, 150, -0.5, 0.8
                    state.puck.x, state.puck.y = 0, 150
                    for other in (*state.team1.players[1:], *state.team2.players):
                        other.x, other.y = 110, -180
                    state.team2.players[0].x, state.team2.players[0].y = -20, 167
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
                    preferred = OffenseController.carry_target(state, player)
                    self.assertFalse(carry_clear(state, player, preferred))
                    model = ClassicAIV1Model(SimpleNamespace(
                        action_type=schema, one_timers=False, uncertain_carry=True))
                    action = model.predict_frame(state)[0]
                    self.assertEqual(model._last_decision, 'carry-escape')
                    self.assertNotEqual(model._last_target, preferred)
                    self.assertFalse(forecast_carry(state, player, model._last_target)[1]['carry_safe'])
                    self.assertFalse(model.offense_diagnostics['carry_safe'])
                    self.assertEqual(model.carry_metrics['uncertified-viable'], 1)
                    self.assertTrue(model.offense_diagnostics['carry_viable'])
                    processor = HockeyActionController(SimpleNamespace(action_type=schema, game_state=state))
                    buttons = processor._process_action(action, processor._new_action_state())[0]
                    self.assertFalse(buttons[Buttons.INPUT_B] or buttons[Buttons.INPUT_C])
                    expected = self.controller._carry_option(state, player, model._last_target)[1]
                    self.assertFalse(expected['carry_safe'])
                    self.assertTrue(np.any(buttons[4:8]))

    def test_safe_default_carry_is_preserved(self):
        point, details = self.controller.fallback_carry(self.state, self.state.team1.players[0])
        self.assertEqual(point, self.controller.carry_target(self.state, self.state.team1.players[0]))
        self.assertEqual(details['mode'], 'carry')
        self.assertTrue(details['carry_safe'])

    def test_safe_escape_prefers_an_attacking_opportunity_to_unneeded_retreat(self):
        player = self.state.team1.players[0]
        player.y = 180
        preferred = self.controller.carry_target(self.state, player)
        unsafe = (preferred, dict(carry_safe=False, carry_bounded=True, carry_pressure=0,
                                  carry_clearance=-1, carry_progress=10, carry_shot_value=0))
        retreat = ((0, 132), dict(carry_safe=True, carry_bounded=True, carry_pressure=12,
                                 carry_clearance=16, carry_progress=-12, carry_shot_value=0))
        advance = ((48, 204), dict(carry_safe=True, carry_bounded=True, carry_pressure=3.1,
                                  carry_clearance=5, carry_progress=12, carry_shot_value=30))
        def option(_state, _player, target):
            return unsafe if target == preferred else advance if target == advance[0] else retreat
        with patch.object(self.controller, '_carry_option', side_effect=option):
            point, details = self.controller.fallback_carry(self.state, player)
        self.assertEqual(point, advance[0])
        self.assertTrue(details['carry_safe'])
        self.assertEqual(details['carry_pressure'], 3.1)

    def test_unavoidable_pressure_is_reported_instead_of_approving_a_fallback(self):
        player = self.state.team1.players[0]
        self.state.team2.players[0].x, self.state.team2.players[0].y = player.x, player.y
        _, details = self.controller.fallback_carry(self.state, player)
        self.assertFalse(details['carry_safe'])
        self.assertIn('no viable carry', details['reason'])

    def test_unavoidable_wall_contact_requests_braking_without_claiming_safety(self):
        player = self.state.team1.players[0]
        player.x, player.y, player.motion_x, player.facing = 119, 0, 2, 2
        point, details = self.controller.fallback_carry(self.state, player)
        self.assertLess(point[0], player.x)
        self.assertFalse(details['carry_safe'])
        self.assertFalse(details['carry_bounded'])
        self.assertIn('unavoidable wall or net contact', details['reason'])

    def test_one_skater_and_away_team_keep_valid_slot_mapping(self):
        self.state.team1.players = self.state.team1.players[:1]
        self.state.team1.num_players = 1
        self.assertEqual(self.controller.passes(self.state), ([], []))
        state = offense_state()
        for team in (state.team1, state.team2):
            for player in (*team.players, team.goalie):
                player.y *= -1
                player.facing = (player.facing + 4) % 8
            team.net.y *= -1
        state.puck.y *= -1
        state.team1.controller, state.team2.controller = 2, 1
        state.team1.defense_control, state.engine.puck_owner = 6, 6
        plan = self.controller.choose(state, 1)
        self.assertEqual(plan[2].slot, 7)


class OffenseMetricTests(unittest.TestCase):
    def test_entry_across_pass_and_turnover_count_once(self):
        state, metrics = offense_state(), OffenseMetrics()
        metrics.observe(0, state)
        state.engine.puck_owner = -256
        metrics.observe(5, state)
        state.engine.puck_owner = 1
        state.team1.players[1].y = 100
        metrics.observe(10, state)
        metrics.observe(11, state)
        self.assertEqual(metrics.summary()['entries'], 1)
        self.assertEqual(metrics.entries[0]['elapsed_frames'], 10)
        state.engine.puck_owner = 6
        metrics.observe(12, state)
        metrics.observe(13, state)
        self.assertEqual(metrics.summary()['zone_turnovers'], 1)

    def test_recorded_shot_followup_is_not_reported_as_a_turnover(self):
        state, metrics = offense_state(), OffenseMetrics()
        metrics.observe(0, state)
        state.engine.puck_owner = -1
        state.team1.stats.shots = 1
        metrics.observe(1, state)
        state.engine.puck_owner = 11
        metrics.observe(2, state)
        self.assertEqual(metrics.summary()['possession_losses'], 1)
        self.assertEqual(metrics.summary()['turnovers'], 0)

if __name__ == '__main__':
    unittest.main()
