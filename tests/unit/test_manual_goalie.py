"""Manual goalie targets, bounded exclusive handoffs and optional telemetry."""
from copy import deepcopy
import json
import pickle
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from nhl94_ai.agents.base import AgentInput
from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.agents.goalie import (
    DIVE_ANIMATION, HANDOFF_TIMEOUT, GoalieController, goalie_arrival, goalie_motion,
    goalie_steer, goalie_target,
)
from nhl94_ai.agents.motion import VELOCITY_SCALE
from nhl94_ai.agents.passing import pass_speed, rom_pass_vector
from nhl94_ai.agents.registry import create_scripted
from nhl94_ai.config import EnvironmentConfig
from nhl94_ai.evaluation.cpu_benchmark import build_parser
from nhl94_ai.evaluation.play import parse_cmdline
from nhl94_ai.game.constants import GameConsts as Buttons
from nhl94_ai.game.ram import register_goalie_control
from nhl94_ai.game.state import NHL94GameState
from tests.unit.test_classic_defense import defense_state


def goalie_state(selected=False):
    state = defense_state()
    state.team1.defense_goalie, state.team2.defense_goalie = 5, 11
    state.engine.goalie_modes, state.engine.goalie_hold_counts = (0, 0), (17, 17)
    state.engine.controller_teams = (1, 0)
    state.engine.input_block, state.engine.clock_stopped = 0, False
    state.engine.camera = (0, -200)
    goalie = state.team1.goalie
    goalie.x, goalie.y = 0, -246
    goalie.motion_x = goalie.motion_y = 0
    goalie.role, goalie.facing, goalie.assignment = 0, 0, 14
    goalie.speed, goalie.agility, goalie.weight, goalie.energy = 20, 20, 64, 4096
    goalie.stick, goalie.passing = 8, 6
    goalie.live_anim, goalie.live_state_flags, goalie.selection_flags = 2, 0, 0x80
    state.puck.x, state.puck.y = 10, -120
    state.puck.motion_x, state.puck.motion_y = 0, -1.3
    state.team1.players[0].x, state.team1.players[0].y = -110, 40
    if selected:
        state.team1.defense_control = 5
        goalie.selection_flags |= 8
    return state


class GoalieContractTests(unittest.TestCase):
    def test_registration_uses_each_variants_actual_goalie_and_ring_index(self):
        for count in (1, 2, 5):
            env = Mock()
            register_goalie_control(env, count)
            fields = dict(call.args for call in env.data.set_variable.call_args_list)
            for side, slot in ((1, count), (2, count + 6)):
                base = 0xFFB04A + slot * 0x80
                self.assertEqual(fields[f'g{side}_control_assignment_index'],
                                 {'address': base + 0x36, 'type': '>u2'})
                self.assertEqual(fields[f'g{side}_control_assignment_7'],
                                 {'address': base + 0x3F, 'type': '|u1'})
                self.assertEqual(fields[f'g{side}_control_anim_timer'],
                                 {'address': base + 0x5C, 'type': '|i1'})
            self.assertEqual(fields['goalie_mode_1'], {'address': 0xFFD05A, 'type': '>u2'})
            self.assertNotIn('g1_anim', fields)

    def test_optional_fields_clear_and_do_not_change_legacy_data(self):
        info = json.loads((Path(__file__).resolve().parents[1] /
                           'fixtures/NHL94-Genesis-v0.json').read_text(encoding='utf-8'))
        info.update(defense_control1=2, defense_control2=-1, defense_team1=1, defense_team2=0,
                    goalie_mode_1=0, goalie_mode_2=0, goalie_hold_b_1=7, goalie_hold_b_2=17,
                    goalie_input_block=0, goalie_clock_flags=1,
                    g1_control_anim=0x250, g1_control_assignment_index=3, g1_control_assignment_3=29)
        state = NHL94GameState(5)
        state.BeginFrame(info, [0] * 6)
        self.assertEqual(state.team1.goalie.assignment, 29)
        self.assertEqual(state.team1.goalie.live_anim, 0x250)
        self.assertEqual(state.team1.goalie.anim, info.get('g1_anim', 0))
        self.assertEqual(state.engine.goalie_hold_counts, (7, 17))
        self.assertTrue(state.engine.clock_stopped)
        state.BeginFrame({k: v for k, v in info.items()
                          if not k.startswith(('goalie_', 'g1_control_', 'defense_'))}, [0] * 6)
        self.assertIsNone(state.team1.goalie.assignment)
        self.assertIsNone(state.team1.goalie.live_anim)
        self.assertIsNone(state.engine.goalie_modes)
        self.assertIsNone(state.engine.controller_teams)

    def test_validation_distinguishes_joystick_and_team_modes(self):
        state = goalie_state()
        state.Flip()
        state.engine.controller_teams = (2, 0)
        state.team1.goalie = deepcopy(state.team2.goalie)
        self.assertEqual(GoalieController.validate(state), 0)
        for modes in ((1, 0), (0, 1)):
            state.engine.goalie_modes = modes
            with self.assertRaisesRegex(ValueError, 'Manual goalie settings'):
                GoalieController.validate(state)

    def test_missing_feedback_and_unsupported_action_fail_explicitly(self):
        state = goalie_state()
        state.engine.goalie_modes = None
        with self.assertRaisesRegex(ValueError, 'telemetry'):
            GoalieController('selective').step(state)
        with self.assertRaisesRegex(ValueError, 'FILTERED'):
            ClassicAIV1Model(SimpleNamespace(goalie_policy='selective', action_type='HOCKEY_INTENT_DPAD'))
        args = parse_cmdline(['--env=NHL94-Genesis-v0', '--nn=ClassicAIV1', '--goalie-policy=selective',
                             '--rf=PostPlay'])
        EnvironmentConfig.from_args(args)
        for name, value in (('env', 'NHL941on1-Genesis-v0'), ('nn', 'MlpPolicy'),
                            ('action_type', 'DISCRETE'), ('selfplay', True)):
            broken = deepcopy(args)
            setattr(broken, name, value)
            with self.assertRaises(ValueError):
                EnvironmentConfig.from_args(broken)
        self.assertEqual(build_parser().parse_args([]).matchups,
                         ['penguins-senators', 'senators-penguins', 'nordiques-canadiens'])

    def test_empty_assignment_stack_is_optional_diagnostic_feedback(self):
        state = goalie_state()
        expected = GoalieController('selective').step(deepcopy(state))
        state.team1.goalie.assignment = None
        controller = GoalieController('selective')
        np.testing.assert_array_equal(controller.step(state), expected)
        self.assertIsNone(controller.diagnostics['assignment'])
        state.team1.goalie.motion_x = None
        with self.assertRaisesRegex(ValueError, 'Missing live goalie motion'):
            controller.step(state)


class GoalieGeometryTests(unittest.TestCase):
    def test_targets_remain_in_crease_for_both_ends(self):
        state = goalie_state()
        for sign in (-1, 1):
            state.team1.net.y = sign * 264
            for x, y in ((110, 120), (-110, 270), (0, -200)):
                state.puck.x, state.puck.y = x, y * sign
                target = goalie_target(state).target
                self.assertLessEqual(abs(target[0]), 32)
                self.assertTrue(236 <= target[1] * sign <= 254)

    def test_released_shot_crossing_rejects_wide_away_and_high_paths(self):
        state = goalie_state(True)
        state.engine.puck_owner = -256
        state.puck.x, state.puck.y, state.puck.motion_y = 8, -210, -4
        plan = goalie_target(state)
        self.assertIsNotNone(plan.crossing_frames)
        self.assertAlmostEqual(plan.target[0], 8)
        for x, vy, height in ((80, -4, 0), (8, 4, 0), (8, -4, 80)):
            state.puck.x, state.puck.motion_y, state.puck.height = x, vy, height
            self.assertIsNone(goalie_target(state).crossing_frames)

    def test_incoming_pass_anticipates_receiver_but_rejects_stale_target(self):
        state = goalie_state()
        state.engine.puck_owner, state.engine.last_puck_player, state.engine.pass_target = -256, 6, 7
        state.puck.x, state.puck.y = -60, -175
        state.puck.motion_x, state.puck.motion_y = 4, 0
        receiver = state.team2.players[1]
        receiver.x, receiver.y = 30, -175
        plan = goalie_target(state)
        self.assertIn('incoming receiver', plan.reason)
        self.assertGreater(plan.target[0], 0)
        receiver.y = 20
        self.assertNotIn('incoming receiver', goalie_target(state).reason)

    def test_shot_intercept_uses_live_depth_and_fractional_crossing(self):
        for sign in (-1, 1):
            for depth in (236, 242, 254):
                with self.subTest(sign=sign, depth=depth):
                    state = goalie_state(True)
                    state.team1.net.y = sign * 264
                    state.team1.goalie.y = sign * depth
                    state.engine.puck_owner = -256
                    state.puck.x, state.puck.y = 40, sign * 205
                    state.puck.motion_x, state.puck.motion_y = -2, sign * 4
                    state.puck.friction = 1
                    plan = goalie_target(state)
                    time = (depth - 205) / 4
                    self.assertAlmostEqual(plan.crossing_frames, time)
                    self.assertEqual(plan.target, (40 - 2 * time, sign * depth))
                    self.assertEqual(plan.deadline, (264 - 205) / 4)

    def test_puck_past_goalie_does_not_send_him_back_up_ice(self):
        for sign in (-1, 1):
            state = goalie_state(True)
            state.team1.net.y = sign * 264
            state.team1.goalie.y = sign * 240
            state.engine.puck_owner = -256
            state.puck.x, state.puck.y = 8, sign * 250
            state.puck.motion_y = sign * 4
            plan = goalie_target(state)
            self.assertEqual(plan.target, (8, sign * 250))
            self.assertEqual(plan.crossing_frames, 0)
            self.assertEqual(plan.crossing_x, 8)

    def test_goalie_acceleration_and_braking_are_not_skater_estimates(self):
        goalie = goalie_state().team1.goalie
        accel, limit = goalie_motion(goalie)
        self.assertEqual(accel, (200 * ((64 - 16 + 40 + 16) // 2) // 32) * VELOCITY_SCALE)
        initial = goalie_arrival(goalie, (25, -246))
        goalie.facing = 4
        self.assertEqual(initial, goalie_arrival(goalie, (25, -246)))
        goalie.motion_x = 2
        self.assertLess(goalie_arrival(goalie, (25, -246)), initial)
        goalie.x = 23
        self.assertFalse(goalie_steer(goalie, (25, -246)).any())
        goalie.has_puck = 1
        carrying, carry_limit = goalie_motion(goalie)
        self.assertEqual(carrying, accel / 2)
        self.assertAlmostEqual(carry_limit, limit / 2**0.5)

    def test_goalie_pass_uses_fixed_eight_and_glove_right_parity(self):
        goalie = goalie_state().team1.goalie
        goalie.passing = 6
        even = pass_speed(goalie)
        goalie.passing = 24
        self.assertEqual(pass_speed(goalie), even)
        goalie.passing = 7
        self.assertGreater(pass_speed(goalie), even)
        self.assertEqual(rom_pass_vector(60, 50, 0, 0, 6, goalie=True),
                         rom_pass_vector(60, 50, 0, 0, 24, goalie=True))


class GoalieHandoffTests(unittest.TestCase):
    def test_hold_waits_for_actual_goalie_and_release_then_taps_back(self):
        state = goalie_state()
        manager = GoalieController('always')
        self.assertFalse(manager.step(state).any())
        self.assertFalse(manager.step(state).any())
        action = manager.step(state)
        self.assertEqual(np.flatnonzero(action).tolist(), [Buttons.INPUT_B])
        state.engine.goalie_hold_counts = (0, 17)
        manager.step(state)
        self.assertEqual(manager.metrics['takeovers'], 0)
        state.team1.defense_control = 5
        state.team1.goalie.selection_flags |= 8
        self.assertFalse(manager.step(state).any())
        self.assertEqual(manager.metrics['takeovers'], 1)
        self.assertFalse(manager.step(state).any())
        state.engine.puck_owner = 0
        self.assertEqual(np.flatnonzero(manager.step(state)).tolist(), [Buttons.INPUT_B])
        self.assertTrue(manager.step(state)[Buttons.INPUT_B])
        self.assertFalse(manager.step(state).any())
        state.team1.goalie.selection_flags &= ~8
        self.assertFalse(manager.step(state).any())
        self.assertEqual(manager.metrics['returns'], 0)
        state.team1.defense_control = 2
        self.assertFalse(manager.step(state).any())
        self.assertEqual(manager.metrics['returns'], 1)
        self.assertFalse(manager.step(state).any())
        self.assertIsNone(manager.step(state))

    def test_timeout_reports_failure_and_does_not_hold_forever(self):
        state = goalie_state()
        manager = GoalieController('always')
        manager.step(state)
        manager.step(state)
        manager.step(state)
        for _ in range(HANDOFF_TIMEOUT - 1):
            manager.step(state)
        with self.assertWarnsRegex(RuntimeWarning, 'not confirmed'):
            self.assertFalse(manager.step(state).any())
        self.assertEqual(manager.metrics['handoff_timeouts'], 1)
        manager.step(state)
        self.assertIsNone(manager.step(state))

    def test_recovery_during_B_hold_cancels_without_passing_or_shooting(self):
        state = goalie_state()
        manager = GoalieController('always')
        for _ in range(3):
            manager.step(state)
        state.engine.puck_owner = 0
        self.assertFalse(manager.step(state).any())
        self.assertEqual(manager.metrics['takeover_cancelled'], 1)
        self.assertFalse(manager.step(state).any())
        self.assertIsNone(manager.step(state))

    def test_offscreen_goalie_returns_to_skater_when_threat_is_gone(self):
        state = goalie_state(True)
        state.team1.goalie.live_state_flags = 4
        state.engine.puck_owner = 0
        manager = GoalieController('selective')
        self.assertEqual(np.flatnonzero(manager.step(state)).tolist(), [Buttons.INPUT_B])
        self.assertEqual(manager.phase, 'request-skater')

    def test_takeover_cancels_when_threat_or_manual_availability_disappears(self):
        for change in ('cleared', 'offscreen', 'locked', 'blocked'):
            with self.subTest(change=change):
                state = goalie_state()
                manager = GoalieController('selective')
                for _ in range(3):
                    manager.step(state)
                self.assertEqual(manager.phase, 'request-goalie')
                if change == 'cleared':
                    state.puck.y = 0
                elif change == 'offscreen':
                    state.engine.camera = (0, 0)
                elif change == 'locked':
                    state.team1.goalie.unavailable = 2
                self.assertFalse(manager.step(state, blocked=change == 'blocked').any())
                self.assertEqual(manager.metrics['takeover_cancelled'], 1)
                self.assertFalse(manager.step(state).any())
                self.assertIsNone(manager.step(state))

    def test_takeover_must_finish_before_goalie_contact_not_goal_line(self):
        state = goalie_state()
        state.team1.goalie.x, state.team1.goalie.y = 0, -236
        state.engine.puck_owner = -256
        state.puck.x, state.puck.y, state.puck.motion_y = 0, -210, -1
        state.puck.friction = 1
        plan = goalie_target(state)
        self.assertEqual(plan.crossing_frames, 26)
        self.assertEqual(plan.deadline, 54)
        self.assertIsNone(GoalieController('selective').step(state))

    def test_selective_keeps_useful_skater_and_rejects_late_shot(self):
        state = goalie_state()
        manager = GoalieController('selective')
        self.assertIsNotNone(manager.step(state))
        state = goalie_state()
        state.team1.players[0].x, state.team1.players[0].y = state.puck.x, state.puck.y - 15
        self.assertIsNone(GoalieController('selective').step(state))
        state = goalie_state()
        state.engine.puck_owner = -256
        state.puck.y, state.puck.motion_y = -220, -6
        self.assertIsNone(GoalieController('selective').step(state))
        state = goalie_state()
        self.assertIsNone(GoalieController('selective').step(state, blocked=True))

    def test_fallback_locks_stoppages_and_pulled_goalie_do_not_receive_saves(self):
        state = goalie_state(True)
        state.engine.puck_owner = -256
        state.puck.y, state.puck.motion_y = -220, -5
        for attr, value in (('live_state_flags', 4), ('unavailable', 2), ('selection_flags', 0xA8)):
            broken = deepcopy(state)
            setattr(broken.team1.goalie, attr, value)
            self.assertFalse(GoalieController('selective').step(broken).any())
        state.team1.goalie.role = -1
        self.assertEqual(np.flatnonzero(GoalieController('selective').step(state)).tolist(), [Buttons.INPUT_B])
        state = goalie_state()
        state.engine.clock_stopped = True
        self.assertIsNone(GoalieController('selective').step(state))
        state.team1.defense_control, state.team1.goalie.selection_flags = 5, 0x88
        self.assertFalse(GoalieController('selective').step(state).any())
        self.assertIsNone(GoalieController('selective').step(goalie_state(), blocked=True))

    def test_save_and_dive_are_exclusive_requests_confirmed_by_animation(self):
        for x, expected, button in ((10, 0x1EC, Buttons.INPUT_C), (27, DIVE_ANIMATION, Buttons.INPUT_A)):
            state = goalie_state(True)
            state.engine.puck_owner = -256
            state.puck.x, state.puck.y, state.puck.motion_y = x, -225, -4
            state.team1.goalie.x = -10 if button == Buttons.INPUT_A else 0
            state.puck.motion_x = -0.5 if button == Buttons.INPUT_A else 0
            manager = GoalieController('selective')
            action = manager.step(state)
            self.assertTrue(action[button])
            self.assertFalse(action[Buttons.INPUT_B])
            self.assertFalse(action[Buttons.INPUT_A if button == Buttons.INPUT_C else Buttons.INPUT_C])
            self.assertEqual(manager.metrics['save_animations'] + manager.metrics['dive_animations'], 0)
            state.team1.goalie.live_anim, state.team1.goalie.unavailable = expected, 2
            self.assertFalse(manager.step(state).any())
            self.assertEqual(manager.metrics['save_animations'] + manager.metrics['dive_animations'], 1)

    def test_cpu_fallback_catch_is_not_claimed_as_manual(self):
        state = goalie_state(True)
        manager = GoalieController('selective')
        state.team1.goalie.live_state_flags = 4
        manager.step(state)
        state.engine.puck_owner = 5
        manager.step(state)
        state.team1.goalie.live_state_flags = 0
        manager.step(state)
        self.assertEqual(manager.metrics['controlled_catches'], 0)
        state.engine.puck_owner = -256
        manager.step(state)
        state.engine.puck_owner = 5
        manager.step(state)
        self.assertEqual(manager.metrics['controlled_catches'], 1)

    def test_pending_save_cannot_send_buttons_to_skater_or_after_catch(self):
        for dive in (False, True):
            for change in ('skater', 'catch', 'fallback', 'locked'):
                with self.subTest(dive=dive, change=change):
                    state = goalie_state(True)
                    state.engine.puck_owner = -256
                    state.puck.x, state.puck.y, state.puck.motion_y = 10, -225, -4
                    state.team1.goalie.x = -15 if dive else 0
                    manager = GoalieController('selective')
                    self.assertTrue(manager.step(state)[Buttons.INPUT_A if dive else Buttons.INPUT_C])
                    if change == 'skater':
                        state.team1.defense_control = 2
                        state.team1.goalie.selection_flags &= ~8
                    elif change == 'catch':
                        state.engine.puck_owner = 5
                    elif change == 'fallback':
                        state.team1.goalie.live_state_flags = 4
                    else:
                        state.team1.goalie.unavailable = 2
                    self.assertFalse(manager.step(state).any())
                    self.assertIsNone(manager.pending_save)
                    self.assertEqual(manager.metrics['dive_cancelled' if dive else 'save_cancelled'], 1)

    def test_outlet_waits_for_pass_feedback_and_reception(self):
        state = goalie_state(True)
        state.engine.puck_owner = 5
        state.team1.pass_attempts = 0
        manager = GoalieController('selective')
        option = SimpleNamespace(slot=2, flight_frames=18, value=30)
        with patch('nhl94_ai.agents.goalie.evaluate_pass', return_value=(option, {})):
            for _ in range(8):
                self.assertFalse(manager.step(state).any())
            self.assertTrue(manager.step(state)[Buttons.INPUT_B])
        state.team1.defense_control = 2
        state.team1.pass_attempts = 1
        state.engine.puck_owner = -256
        self.assertFalse(manager.step(state).any())
        self.assertEqual(manager.metrics['outlets_launched'], 1)
        self.assertFalse(manager.step(state).any())
        state.engine.puck_owner = 2
        manager.step(state)
        self.assertEqual(manager.metrics['outlet_receptions'], 1)
        self.assertFalse(manager.step(state).any())
        self.assertIsNone(manager.step(state))

    def test_no_safe_outlet_uses_bounded_A_clear_not_invented_freeze(self):
        state = goalie_state(True)
        state.engine.puck_owner = 5
        manager = GoalieController('selective')
        with patch('nhl94_ai.agents.goalie.evaluate_pass', return_value=(None, {})):
            for _ in range(90):
                self.assertFalse(manager.step(state).any())
            action = manager.step(state)
        self.assertTrue(action[Buttons.INPUT_A])
        self.assertFalse(action[Buttons.INPUT_B])
        self.assertFalse(action[Buttons.INPUT_C])

    def test_per_frame_dispatch_advances_once_and_reset_clears_state(self):
        args = SimpleNamespace(action_type='FILTERED', goalie_policy='always')
        agent = create_scripted('classic-v1', args)
        agent.frame_skip = 4
        output = agent.act(AgentInput(goalie_state()))
        self.assertIn('classic_goalie', output.diagnostics)
        self.assertEqual(agent.goalie.frames, 1)
        self.assertEqual(agent.defense.frames, 1)
        for _ in range(6):
            agent.act(AgentInput(goalie_state()))
        self.assertEqual(agent.goalie.frames, 7)
        self.assertEqual(agent.defense.frames, 7)
        agent.reset()
        self.assertEqual(agent.goalie.frames, 0)
        self.assertEqual(agent.goalie.phase, 'skater')
        legacy = create_scripted('classic-v1', SimpleNamespace(action_type='FILTERED'))
        self.assertNotIn('classic_goalie', legacy.act(AgentInput(defense_state())).diagnostics)


class GoalieTimingTests(unittest.TestCase):
    def test_receiver_contact_before_goalie_plane_does_not_trigger_save_on_the_pass(self):
        state = goalie_state(True)
        state.engine.puck_owner, state.engine.last_puck_player, state.engine.pass_target = -256, 6, 7
        receiver = state.team2.players[1]
        receiver.x, receiver.y, receiver.stick_x, receiver.stick_y = 14, -232, 0, 0
        state.puck.x, state.puck.y, state.puck.motion_y = 14, -215, -4
        state.puck.friction = 1
        plan = goalie_target(state)
        self.assertIsNone(plan.crossing_frames)
        self.assertEqual(plan.receiver, (14, -232))
        self.assertLess(plan.reception_frames, (246 - 215) / 4)
        action = GoalieController('selective').step(state)
        self.assertFalse(action[Buttons.INPUT_C])
        self.assertTrue(action[4:8].any())

    def test_stale_pass_target_at_shooter_does_not_hide_released_shot(self):
        state = goalie_state(True)
        state.engine.puck_owner, state.engine.last_puck_player, state.engine.pass_target = -256, 7, 7
        state.puck.x, state.puck.y, state.puck.motion_y = 14, -215, -4
        state.puck.friction = 1
        self.assertIsNotNone(goalie_target(state).crossing_frames)

    def test_reachable_lateral_alignment_precedes_momentum_killing_C(self):
        for sign in (-1, 1):
            state = goalie_state(True)
            state.team1.net.y, state.team2.net.y = sign * 264, -sign * 264
            state.team1.goalie.y, state.team1.goalie.motion_x = sign * 246, 1.5
            state.engine.puck_owner = -256
            state.puck.x, state.puck.y, state.puck.motion_y = 14, sign * 215, sign * 4
            state.puck.friction = 1
            manager = GoalieController('selective')
            before = pickle.dumps(state)
            action = manager.step(state)
            self.assertTrue(action[Buttons.INPUT_RIGHT])
            self.assertFalse(action[Buttons.INPUT_C])
            self.assertEqual(manager.diagnostics['save_decision'], 'align-before-save')
            self.assertEqual(pickle.dumps(state), before)
            state.team1.goalie.x = 6
            action = manager.step(state)
            self.assertTrue(action[Buttons.INPUT_C])
            self.assertFalse(action[4:8].any())

    def test_unreachable_or_high_shot_keeps_emergency_save(self):
        for height, speed in ((0, -1.5), (16, 1.5)):
            state = goalie_state(True)
            state.engine.puck_owner = -256
            state.team1.goalie.motion_x = speed
            state.puck.x, state.puck.y, state.puck.motion_y, state.puck.height = 14, -215, -4, height
            state.puck.friction = 1
            self.assertTrue(GoalieController('selective').step(state)[Buttons.INPUT_C])

    def test_remaining_B_hold_uses_live_countdown_not_elapsed_guess(self):
        state = goalie_state()
        state.engine.puck_owner = -256
        state.puck.x, state.puck.y, state.puck.motion_y = 0, -236, -1
        state.puck.friction = 1
        state.engine.goalie_hold_counts = (2, 17)
        manager = GoalieController('selective')
        manager.phase = 'request-goalie'
        action = manager.step(state)
        self.assertTrue(action[Buttons.INPUT_B])
        self.assertEqual(manager.diagnostics['remaining_hold_frames'], 4)
        self.assertEqual(manager.phase, 'request-goalie')

    def test_live_countdown_accounts_for_observed_input_tick_spacing(self):
        manager = GoalieController('selective')
        self.assertEqual(manager._remaining_hold(17), 19)
        manager.frames = 1
        manager._remaining_hold(16)
        manager.frames = 3
        self.assertEqual(manager._remaining_hold(15), 32)

    def test_fresh_pass_anticipates_windup_before_possession_releases(self):
        state = goalie_state()
        passer, receiver = state.team2.players[:2]
        passer.x, passer.y, passer.passing = -60, -175, 20
        receiver.x, receiver.y, receiver.stick_x, receiver.stick_y = 30, -175, 0, 0
        state.puck.x, state.puck.y, state.puck.motion_x, state.puck.motion_y = -60, -175, 0, 0
        state.puck.friction = 1
        state.engine.pass_target, state.team2.pass_attempts = 7, 0
        manager = GoalieController('selective')
        manager.step(state)
        self.assertIsNone(manager.diagnostics['receiver'])
        state.team2.pass_attempts = 1
        before = pickle.dumps(state)
        manager.step(state)
        self.assertEqual(manager.diagnostics['receiver'], (30, -175))
        self.assertGreater(manager.diagnostics['reception_frames'], 0)
        self.assertEqual(pickle.dumps(state), before)
        state.engine.pass_target = -1
        manager.step(state)
        self.assertIsNone(manager.pass_windup)

    def test_initial_saved_pass_count_does_not_invent_a_fresh_windup(self):
        state = goalie_state()
        state.team2.pass_attempts, state.engine.pass_target = 7, 7
        manager = GoalieController('selective')
        manager.step(state)
        self.assertIsNone(manager.pass_windup)

    def test_idle_reasons_distinguish_target_holding_and_CPU_fallback(self):
        state = goalie_state(True)
        state.engine.puck_owner = -256
        state.puck.x, state.puck.y, state.puck.motion_y = 0, -215, -4
        state.puck.friction = 1
        manager = GoalieController('selective')
        manager.save_at = 100
        self.assertFalse(manager.step(state).any())
        self.assertEqual(manager.diagnostics['idle_reason'], 'holding-position')
        state.team1.goalie.live_state_flags = 4
        manager.step(state)
        self.assertEqual(manager.diagnostics['idle_reason'], 'cpu-fallback')

    def test_unconfirmed_request_does_not_block_saves_for_forty_frames(self):
        state = goalie_state(True)
        manager = GoalieController('selective')
        manager.pending_save, manager.save_at, manager.frames = ('save', 0), 40, 6
        manager.step(state)
        self.assertIsNone(manager.pending_save)
        self.assertLess(manager.save_at, 40)
        self.assertEqual(manager.metrics['save_unconfirmed'], 1)

    def test_classic_goalie_prediction_cannot_write_environment_or_game_state(self):
        state = goalie_state(True)
        env = Mock()
        model = ClassicAIV1Model(SimpleNamespace(action_type='FILTERED', goalie_policy='selective'), env)
        before = pickle.dumps(state)
        action = model.predict_frame(state)[0]
        self.assertEqual(action.shape, (12,))
        self.assertTrue(np.isin(action, (0, 1)).all())
        self.assertEqual(pickle.dumps(state), before)
        self.assertEqual(env.mock_calls, [])


if __name__ == '__main__':
    unittest.main()
