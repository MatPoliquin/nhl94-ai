"""Receiving requests, applied edges and native arming are distinct events."""
import unittest
from unittest.mock import patch

import numpy as np

from nhl94_ai.agents.teammates import TeammateCoordinator
from tests.support.teammates import three_args, three_info
from tests.unit.test_teammate_reception_followup import flight


class ReceivingLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.state, self.receiving = flight()
        self.receiving.pending.purpose = 'one-timer'
        self.state.team1.players[1].x, self.state.team1.players[1].y = 35, 210
        self.state.engine.pass_target = 1
        self.contact = patch('nhl94_ai.agents.teammate_receiving.one_timer_reception',
                             return_value={'gap': -4, 'frames': 8})
        self.value = patch('nhl94_ai.agents.teammate_receiving.shot_value', return_value=30)
        self.contact.start()
        self.value.start()
        self.addCleanup(self.contact.stop)
        self.addCleanup(self.value.stop)

    def test_suppressed_cue_remains_available_until_the_correct_controller_applies_it(self):
        receiving, state = self.receiving, self.state
        request = receiving.receiver_action(state, 2, 13, np.zeros(12))[0]
        self.assertTrue(request[8])
        self.assertIsNone(receiving.pending.cue)
        for slot, applied in ((-1, np.zeros(12)), (0, request)):
            receiving.accept_applied_action(2, slot, 13, applied, np.zeros(12))
            self.assertIsNone(receiving.pending.cue)
        retry = receiving.receiver_action(state, 2, 14, np.zeros(12))[0]
        receiving.accept_applied_action(2, 1, 14, retry, np.zeros(12))
        self.assertEqual(receiving.pending.cue, 14)
        self.assertFalse(receiving.receiver_action(state, 2, 15, np.zeros(12))[0][8])
        self.assertEqual([e['phase'] for e in receiving.events].count('one-timer-cue'), 1)

    def test_native_arming_does_not_follow_from_submitting_c(self):
        receiving, state = self.receiving, self.state
        action = receiving.receiver_action(state, 2, 13, np.zeros(12))[0]
        receiving.accept_applied_action(2, 1, 13, action, np.zeros(12))
        receiving.observe(state, [0, 1, 2], 14)
        self.assertIsNone(receiving.pending.armed)
        state.team1.players[1].is_one_timer = True
        receiving.observe(state, [0, 1, 2], 15)
        self.assertEqual(receiving.pending.armed, 15)
        receiving.observe(state, [0, 1, 2], 16)
        self.assertEqual([e['phase'] for e in receiving.events].count('one-timer-armed'), 1)

    def test_same_receiver_on_a_different_controller_routes_c_to_its_actual_owner(self):
        receiving, state = self.receiving, self.state
        receiving.observe(state, [1, 0, 2], 13)
        self.assertEqual(receiving.pending.receiver_controller, 1)
        self.assertIsNone(receiving.receiver_action(state, 2, 13, np.zeros(12)))
        self.assertTrue(receiving.receiver_action(state, 1, 13, np.zeros(12))[0][8])

    def test_cpu_receiver_native_control_transfer_does_not_request_another_c(self):
        receiving, state = self.receiving, self.state
        receiving.pending.receiver_controller = None
        state.team1.players[1].is_one_timer = True
        receiving.observe(state, [1, 0, 2], 13)
        self.assertEqual(receiving.pending.receiver_controller, 1)
        self.assertEqual(receiving.pending.armed, 13)
        self.assertIsNone(receiving.pending.cue)
        self.assertFalse(receiving.receiver_action(state, 1, 13, np.zeros(12))[0].any())
        state.team1.players[1].is_one_timer = False
        receiving.observe(state, [1, 0, 2], 14)
        self.assertFalse(receiving.receiver_action(state, 1, 14, np.zeros(12))[0][8])

    def test_coordinator_unknown_table_cannot_consume_the_receivers_cue(self):
        ai = TeammateCoordinator(three_args())
        ai.act(three_info())
        ai.receiving = self.receiving
        ai.receiving.pending.expiry = 100
        info = three_info(puck_owner=-1, p1_pass_attempts=0, pass_target=1,
                          defense_team3=0, p1_2_x=35, p1_2_y=210)
        result = ai.act(info)
        self.assertTrue(result.diagnostics['teammates']['transient'])
        self.assertFalse(result.action.any())
        self.assertIsNone(ai.receiving.pending.cue)

    def test_accounting_and_routing_do_not_depend_on_transient_preservation(self):
        self.receiving.preserve_transient = False
        self.receiving.observe(self.state, [1, 0, 2], 13)
        self.assertEqual(self.receiving.pending.receiver_controller, 1)
        action = self.receiving.receiver_action(self.state, 1, 13, np.zeros(12))[0]
        self.assertIsNone(self.receiving.pending.cue)
        self.receiving.accept_applied_action(1, 1, 13, action, np.zeros(12))
        self.assertEqual(self.receiving.pending.cue, 13)
        self.state.team1.players[1].is_one_timer = True
        self.receiving.observe(self.state, [1, 0, 2], 14)
        self.assertEqual(self.receiving.pending.armed, 14)
