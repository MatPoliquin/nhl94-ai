"""Conditional forecasts retain native clocks, certification and live pass authority."""
from copy import deepcopy
from functools import partial
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from nhl94_ai.agents.carry import carry_path
from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.agents.offense import OffenseController
from nhl94_ai.agents.responses import response_future, response_step
from nhl94_ai.evaluation.play import parse_cmdline
from nhl94_ai.evaluation.runner import build_parser, run
from nhl94_ai.evaluation.chance_creation_probe import response_comparison
from nhl94_ai.game.ram import register_defense_state
from tests.ram_fixture import FixtureMemory
from tests.unit.test_classic_offense import offense_state


def response_state():
    state = offense_state()
    carrier, receiver = state.team1.players[:2]
    carrier.x, carrier.y, carrier.facing = -50, 160, 0
    receiver.x, receiver.y, receiver.facing = 0, 190, 0
    state.puck.x, state.puck.y = -50, 170
    receiver.assignment, receiver.cpu_target = 6, (0, 170)
    receiver.steering, receiver.steering_timer = 8, 0
    receiver.cpu_response_available, receiver.support_zone = True, 16
    for team in (state.team1, state.team2):
        for player in team.players:
            if player not in (carrier, receiver):
                player.role = -1
    return state


class ChanceCreationTests(unittest.TestCase):
    def test_native_response_summary_exposes_missed_conditional_motion(self):
        common = dict(style='center', width=55, puck_offset='forward-offset', action_type='FILTERED',
                      side=2, decision_interval=4, initial_ram_sha256='a' * 64)
        left = dict(common, direction=-1, actual=(0, 190), predicted=(0, 190))
        right = dict(common, direction=1, actual=(1, 190), predicted=(0, 190))
        result = response_comparison([left, right])
        self.assertEqual(result['missed_signal_pairs'], 1)
        self.assertEqual(result['maximum_delta_error'], 1)
        with self.assertRaisesRegex(ValueError, 'initial state'):
            response_comparison([left, dict(right, initial_ram_sha256='b' * 64)])
        with self.assertRaisesRegex(ValueError, 'complete paired'):
            response_comparison([left])

    def test_flag_is_opt_in_and_learned_evaluation_rejects_it(self):
        args = parse_cmdline(['--nn=ClassicAIV1', '--env=NHL94-Genesis-v0', '--chance-creation'])
        self.assertTrue(ClassicAIV1Model(args).offense.chance_creation)
        self.assertFalse(ClassicAIV1Model().offense.chance_creation)
        with self.assertRaisesRegex(ValueError, 'not a learned policy'):
            run(build_parser().parse_args(['--model=model.zip', '--chance-creation']))
        with self.assertRaisesRegex(ValueError, 'full-team'):
            ClassicAIV1Model(SimpleNamespace(chance_creation=True, env='NHL941on1-Genesis-v0'))

    def test_support_reacts_to_carrier_route_without_mutating_live_state(self):
        state = response_state()
        before = deepcopy(state)
        carrier = state.team1.players[0]
        carrier.x, state.puck.x = -35, -35
        left, details = response_future(state, carry_path(carrier, (-61, 168)))
        right, _ = response_future(state, carry_path(carrier, (-9, 168)))
        self.assertIn(1, details['response_slots'])
        self.assertNotEqual(vars(left.team1.players[1]), vars(right.team1.players[1]))
        self.assertEqual(vars(state.team1.players[1]), vars(before.team1.players[1]))
        self.assertEqual(state.puck.y, before.puck.y)
        self.assertEqual(state.puck.x, -35)

    def test_existing_steering_survives_until_its_native_countdown(self):
        state = response_state()
        receiver = state.team1.players[1]
        receiver.steering_timer, receiver.steering = 5, 2
        future = response_step(receiver, state.puck, sign=1, friendly=True)
        self.assertEqual(future.steering_timer, 4)
        self.assertEqual(future.steering, 2)
        receiver.steering_timer = 0
        future = response_step(receiver, state.puck, sign=1, friendly=True)
        self.assertEqual(future.steering_timer, 11)
        self.assertNotEqual(future.steering, 2)

    def test_unknown_and_zone_changing_support_do_not_get_response_credit(self):
        state = response_state()
        receiver = state.team1.players[1]
        receiver.cpu_response_available = False
        _, details = response_future(state, carry_path(state.team1.players[0], (-24, 168)))
        self.assertNotIn(1, details['response_slots'])
        self.assertIn(1, details['response_unsupported_slots'])
        receiver.cpu_response_available = True
        receiver.support_zone = 8
        _, details = response_future(state, carry_path(state.team1.players[0], (-24, 168)))
        self.assertNotIn(1, details['response_slots'])

    def test_chance_value_never_relaxes_carry_certificate(self):
        controller = OffenseController(chance_creation=True)
        state = response_state()
        with patch.object(controller, '_carry_option', return_value=((0, 180), {'carry_safe': False})), \
                patch.object(controller, '_passing_window', return_value=(100, 1, 'one-timer')), \
                patch('nhl94_ai.agents.offense.response_future') as response:
            self.assertIsNone(controller._create_chance(state, state.team1.players[0], 0))
        response.assert_not_called()

    def test_straight_shooting_improvement_prevents_an_unnecessary_setup(self):
        controller = OffenseController(chance_creation=True)
        state = response_state()
        safety = {'carry_safe': True, 'carry_value_frames': 18}
        with patch.object(controller, '_carry_option', side_effect=lambda _s, _p, t: (t, safety)), \
                patch.object(controller, '_passing_window',
                             side_effect=[(0, None, None), (0, None, None), (60, 1, 'one-timer'),
                                          (0, None, None), (0, None, None), (0, None, None)]), \
                patch('nhl94_ai.agents.offense.response_future',
                      return_value=(state, {'response_slots': [1], 'response_clearance': 20})), \
                patch('nhl94_ai.agents.offense.shot_value', side_effect=(0, 100)):
            self.assertIsNone(controller._create_chance(state, state.team1.players[0], 0))

    def test_native_pursuit_branch_is_explicitly_unsupported_not_misread_as_containment(self):
        state = response_state()
        defender = state.team2.players[0]
        defender.role, defender.assignment = 3, 0x11
        defender.cpu_response_available, defender.cpu_target = True, (240, 0)
        defender.steering_timer, defender.steering = 0, 6
        _, details = response_future(state, carry_path(state.team1.players[0], (-24, 168)))
        self.assertIn(6, details['response_unsupported_slots'])
        with self.assertRaisesRegex(ValueError, 'supported live'):
            response_step(defender, state.puck, sign=-1, friendly=False)

    def test_changed_friendly_carrier_cancels_setup_before_any_pending_pass(self):
        state = response_state()
        controller = OffenseController(chance_creation=True)
        controller.chance_owner, controller.chance_until = 0, 18
        state.engine.puck_owner = 1
        controller.observe(state, 4)
        self.assertIsNone(controller.chance_owner)
        self.assertEqual(controller.chance_until, 0)

    def test_predicted_window_selects_movement_not_a_pass_and_has_a_deadline(self):
        controller = OffenseController(chance_creation=True)
        state = response_state()
        safety = {'carry_safe': True, 'carry_value_frames': 18}
        with patch.object(controller, '_carry_option', side_effect=lambda _s, _p, t: (t, safety)), \
                patch.object(controller, '_passing_window',
                             side_effect=[(0, None, None), (0, None, None),
                                          (60, 1, 'one-timer'), (0, None, None),
                                          (0, None, None), (0, None, None)]), \
                patch('nhl94_ai.agents.offense.response_future',
                      return_value=(state, {'response_slots': [1], 'response_clearance': 20})):
            plan = controller._create_chance(state, state.team1.players[0], 10)
        self.assertEqual(plan[0], 'create-chance')
        self.assertIsNone(plan[2])
        self.assertEqual(controller.chance_until, 28)
        self.assertEqual(controller.diagnostics['chance_window_status'], 'predicted-not-observed')
        self.assertIsNone(controller._create_chance(state, state.team1.players[0], 28))
        self.assertEqual(controller.chance_at, 100)

    def test_default_selector_does_not_evaluate_experimental_responses(self):
        state = offense_state()
        with patch('nhl94_ai.agents.offense.response_future') as response:
            OffenseController().choose(state, 0)
        response.assert_not_called()

    def test_replanning_does_not_extend_the_original_setup_horizon(self):
        controller = OffenseController(chance_creation=True)
        controller.chance_owner, controller.chance_until = 0, 18
        state = response_state()
        safety = {'carry_safe': True, 'carry_value_frames': 18}
        with patch.object(controller, '_carry_option', side_effect=lambda _s, _p, t: (t, safety)), \
                patch.object(controller, '_passing_window', return_value=(0, None, None)), \
                patch('nhl94_ai.agents.offense.response_future',
                      return_value=(state, {'response_slots': [1], 'response_clearance': 20})) as response:
            controller._create_chance(state, state.team1.players[0], 16)
        self.assertTrue(response.call_args_list)
        self.assertTrue(all(len(call.args[1]) == 3 for call in response.call_args_list))
        self.assertEqual(controller.chance_until, 18)

    def test_cached_input_is_interrupted_at_the_exact_setup_deadline_in_both_schemas(self):
        for schema in ('FILTERED', 'HOCKEY_INTENT_DPAD'):
            for interval in (1, 4, 10):
                with self.subTest(schema=schema, interval=interval):
                    state = response_state()
                    model = ClassicAIV1Model(SimpleNamespace(
                        chance_creation=True, one_timers=False, action_type=schema))
                    model.offense.chance_owner, model.offense.chance_until = 0, 19
                    decisions = []
                    def choose(_state, frame, _decisions):
                        _decisions.append(frame)
                        return ('create-chance', (-76, 168), None) if frame < 19 else None
                    with patch.object(model.offense, 'choose', side_effect=partial(choose, _decisions=decisions)):
                        for frame in range(19):
                            model.predict_frame(state, frame_skip=interval)
                            self.assertEqual(model._last_decision == 'create-chance', frame < 18)
                    self.assertIn(19, decisions)
                    self.assertIsNone(model.offense.chance_owner)

    def test_known_pass_retry_deadlines_withhold_unusable_windows(self):
        state = response_state()
        controller = OffenseController(chance_creation=True)
        controller.pass_at, controller.one_timer_at = 24, 96
        with patch.object(controller, 'passes', return_value=([], [])) as passes:
            self.assertEqual(controller._passing_window(state, 23), (0, None, None))
            passes.assert_not_called()
            controller._passing_window(state, 24)
            self.assertEqual(passes.call_args.args[1], 'position')
            self.assertEqual(passes.call_count, 1)
            controller._passing_window(state, 96)
            self.assertEqual(passes.call_count, 3)

    def test_discarded_plan_does_not_leave_an_unexecuted_setup_clock(self):
        state = response_state()
        model = ClassicAIV1Model(SimpleNamespace(chance_creation=True))
        model.offense.chance_owner, model.offense.chance_until = 0, 18
        with patch.object(model, '_decide', return_value=(model._frame_action[0], 0)):
            model._predict_decision(state)
        self.assertIsNone(model.offense.chance_owner)
        self.assertEqual(model.offense.chance_until, 0)

    def test_optional_cpu_feedback_decodes_and_clears_without_neural_schema_changes(self):
        memory = FixtureMemory({})
        def register(name, field):
            memory.fields[name] = field['address'], field['type']
        register_defense_state(SimpleNamespace(data=SimpleNamespace(set_variable=register)), 5)
        base = 0xFFB04A
        for offset, kind, value in ((0x36, '>u2', 3), (0x3B, '|u1', 6),
                                   (0x40, '|i1', 5), (0x42, '|i1', 2), (0x43, '|u1', 8),
                                   (0x44, '>i2', -12), (0x46, '>i2', 170), (0x48, '>i2', 16)):
            memory.assign(base + offset, kind, value)
        info = {name: memory.extract(address, kind) for name, (address, kind) in memory.fields.items()}
        state = offense_state()
        state.team1._load_defense_fields(info)
        player = state.team1.players[0]
        self.assertEqual((player.assignment, player.decision_timer, player.steering_timer), (6, 5, 2))
        self.assertEqual(player.cpu_target, (-12, 170))
        self.assertTrue(player.cpu_response_available)
        state.team1._load_defense_fields({})
        self.assertIsNone(player.assignment)
        self.assertIsNone(player.cpu_target)
        self.assertFalse(player.cpu_response_available)


if __name__ == '__main__':
    unittest.main()
