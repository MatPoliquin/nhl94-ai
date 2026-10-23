"""Predicted-contact one-timer cue during the release phase (opt-in)."""
import unittest
from unittest.mock import patch

import numpy as np

from nhl94_ai.agents.teammate_receiving import SharedReceiving
from nhl94_ai.env.actions import HOCKEY_PASS_PRESS_FRAMES
from nhl94_ai.game.constants import GameConsts as Buttons
from nhl94_ai.game.ram import teammate_controller_info
from nhl94_ai.game.state import NHL94GameState
from tests.support.teammates import team_args, team_info


def anticipated(shared=None, **request):
    state = NHL94GameState(5)
    state.BeginFrame(teammate_controller_info(team_info()), [0] * 6)
    sign = 1 if state.team2.net.y > state.team1.net.y else -1
    receiver = state.team1.players[1]
    receiver.x, receiver.y = 30, 210 * sign
    shared = shared or SharedReceiving(anticipate_cue=True)
    shared.observe(state, [0, 1], 1)
    shared.propose({'passer': 0, 'receiver': 1, 'point': (30, 210 * sign),
                    'frame': 1, 'deadline': 61, 'purpose': 'one-timer', **request},
                   1, [0, 1], state, 1)
    return state, shared


class CueAnticipationTests(unittest.TestCase):
    def test_predicted_cue_fires_once_during_release_inside_the_window(self):
        state, shared = anticipated(flight_frames=8)
        self.assertEqual(shared.pending.predicted_contact, 9)
        with patch('nhl94_ai.agents.teammate_receiving.shot_value', return_value=30):
            early, _, role = shared.receiver_action(state, 2, 8, np.zeros(12))
            self.assertEqual(role, 'receive')
            self.assertFalse(early.any())
            action, target, role = shared.receiver_action(state, 2, 3, np.zeros(12))
            self.assertIsNone(shared.pending.cue)
            shared.accept_applied_action(2, 1, 3, action, np.zeros(12))
            second, _, second_role = shared.receiver_action(state, 2, 4, action)
        self.assertEqual(role, 'one-timer-receiver')
        self.assertEqual(action[Buttons.INPUT_C], 1)
        self.assertEqual(int(action.sum()), 1)
        self.assertEqual(target, (30, 210))
        self.assertEqual(shared.pending.cue, 3)
        self.assertEqual(shared.pending.phase, 'release')
        self.assertEqual(shared.events[-1]['phase'], 'one-timer-cue')
        self.assertEqual(shared.events[-1]['basis'], 'predicted')
        self.assertEqual(second_role, 'receive')
        self.assertFalse(second.any())

    def test_no_cue_without_flight_frames_in_the_request(self):
        state, shared = anticipated()
        self.assertIsNone(shared.pending.predicted_contact)
        action, _, role = shared.receiver_action(state, 2, 3, np.zeros(12))
        self.assertEqual(role, 'receive')
        self.assertFalse(action.any())
        self.assertIsNone(shared.pending.cue)

    def test_flag_off_keeps_stand_still_and_native_endings(self):
        state, shared = anticipated(shared=SharedReceiving(), flight_frames=8)
        self.assertFalse(shared.anticipate_cue)
        self.assertEqual(shared.pending.predicted_contact, 9)
        action, _, role = shared.receiver_action(state, 2, 3, np.zeros(12))
        self.assertEqual(role, 'receive')
        self.assertFalse(action.any())
        self.assertIsNone(shared.pending.cue)
        shared.observe(state, [0, 1], 1 + HOCKEY_PASS_PRESS_FRAMES + 8 + 1)
        self.assertIsNotNone(shared.pending)
        self.assertFalse(any(event['phase'] == 'one-timer-cue' for event in shared.events))

    def test_retarget_resets_cue_and_predicted_contact(self):
        state, shared = anticipated(flight_frames=8)
        with patch('nhl94_ai.agents.teammate_receiving.shot_value', return_value=30):
            action = shared.receiver_action(state, 2, 3, np.zeros(12))[0]
        shared.accept_applied_action(2, 1, 3, action, np.zeros(12))
        self.assertEqual(shared.pending.cue, 3)
        state.engine.puck_owner, state.engine.pass_target = -1, 0
        state.team1.pass_attempts = 1
        shared.observe(state, [0, 1], 4)
        plan = shared.pending
        self.assertEqual(plan.receiver, 0)
        self.assertEqual(plan.phase, 'flight')
        self.assertIsNone(plan.cue)
        self.assertIsNone(plan.predicted_contact)
        state.engine.puck_owner = 7
        shared.observe(state, [0, 1], 5)
        self.assertEqual(shared.events[-1]['reason'], 'intercepted')
        self.assertNotIn('cue_whipt', shared.events[-1])

    def test_unconfirmed_release_ends_after_the_press_window(self):
        state, shared = anticipated(flight_frames=8)
        boundary = 1 + HOCKEY_PASS_PRESS_FRAMES + 8
        shared.observe(state, [0, 1], boundary)
        self.assertIsNotNone(shared.pending)
        shared.observe(state, [0, 1], boundary + 1)
        self.assertIsNone(shared.pending)
        self.assertEqual(shared.events[-1]['reason'], 'release-unconfirmed')

    def test_cued_commitment_ending_before_a_shot_reports_cue_whipt(self):
        state, shared = anticipated(flight_frames=8)
        with patch('nhl94_ai.agents.teammate_receiving.shot_value', return_value=30):
            action = shared.receiver_action(state, 2, 3, np.zeros(12))[0]
        shared.accept_applied_action(2, 1, 3, action, np.zeros(12))
        state.engine.puck_owner = 7
        shared.observe(state, [0, 1], 4)
        self.assertEqual(shared.events[-1]['reason'], 'lost-before-release')
        self.assertTrue(shared.events[-1]['cue_whipt'])

    def test_uncued_ending_omits_cue_whipt(self):
        state, shared = anticipated(flight_frames=8)
        state.engine.puck_owner = 7
        shared.observe(state, [0, 1], 2)
        self.assertEqual(shared.events[-1]['reason'], 'lost-before-release')
        self.assertNotIn('cue_whipt', shared.events[-1])

    def test_live_cue_after_confirmed_release_reports_live_basis(self):
        state, shared = anticipated(flight_frames=8)
        state.engine.puck_owner, state.engine.pass_target = -1, 1
        state.team1.pass_attempts = 1
        shared.observe(state, [0, 1], 2)
        with patch('nhl94_ai.agents.teammate_receiving.one_timer_reception',
                   return_value={'gap': -3, 'frames': 8}), \
             patch('nhl94_ai.agents.teammate_receiving.shot_value', return_value=30):
            action, _, role = shared.receiver_action(state, 2, 2, np.zeros(12))
        self.assertEqual(role, 'one-timer-receiver')
        self.assertEqual(action[Buttons.INPUT_C], 1)
        shared.accept_applied_action(2, 1, 2, action, np.zeros(12))
        self.assertEqual(shared.events[-1]['basis'], 'live')

    def test_flag_plumbing_requires_coordinated_opt_in(self):
        from nhl94_ai.agents.teammates import TeammateCoordinator
        from nhl94_ai.evaluation.teammates import build_parser
        self.assertFalse(TeammateCoordinator(team_args()).receiving.anticipate_cue)
        self.assertTrue(TeammateCoordinator(team_args(teammate_cue_anticipation=True)).receiving.anticipate_cue)
        with self.assertRaises(ValueError):
            TeammateCoordinator(team_args(teammate_cue_anticipation=True), strategy='independent')
        self.assertFalse(build_parser().parse_args(['--output', '/tmp/unused']).teammate_cue_anticipation)
        self.assertTrue(build_parser().parse_args(
            ['--output', '/tmp/unused', '--teammate-cue-anticipation']).teammate_cue_anticipation)
