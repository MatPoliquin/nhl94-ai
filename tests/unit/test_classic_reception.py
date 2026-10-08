"""Receiver correction must preserve possession, selection and button ownership."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.agents.motion import puck_path
from nhl94_ai.agents.receiving import passive_reception
from nhl94_ai.env.intents import HOCKEY_INTENT_CHANGE_PLAYER, HOCKEY_INTENT_NOOP
from nhl94_ai.game.constants import GameConsts as Buttons
from tests.unit.test_classic_offense import offense_state


def reception_state():
    fixture = json.loads((Path(__file__).parents[1] / 'fixtures/classic-missed-reception.json').read_text())
    state = offense_state()
    for name, value in fixture['receiver_feedback'].items():
        setattr(state.team1.players[0], name, value)
    for name, value in fixture['puck_feedback'].items():
        setattr(state.puck, name, value)
    state.team1.defense_control = 3
    state.engine.puck_owner = -247
    return state


def waiting_model(schema='FILTERED'):
    model = ClassicAIV1Model(SimpleNamespace(action_type=schema))
    model.defense.frames = 5
    model.offense.pending = {'passer': 3, 'receiver': 0, 'actual_receiver': 0,
                             'launched': True, 'flight_observed': True, 'frame': 0,
                             'deadline': 50, 'purpose': 'advance-pass', 'point': (-94, -136),
                             'passes_before': 0}
    return model


class ReceptionTests(unittest.TestCase):
    def test_live_native_feedback_exposes_marginal_contact_after_receive_assignment(self):
        state = reception_state()
        before = deepcopy(vars(state.team1.players[0]))
        forecast = passive_reception(state, state.team1.players[0])
        self.assertGreater(forecast['gap'], -2)
        self.assertGreater(forecast['frames'], state.team1.players[0].decision_timer)
        self.assertEqual(before, vars(state.team1.players[0]))
        dense = puck_path(state.puck, 40, sample_interval=1)
        self.assertEqual([item for item in dense if item[0] % 4 == 0], puck_path(state.puck, 40))

    def test_good_body_contact_and_missing_feedback_do_not_request_a_switch(self):
        state = reception_state()
        receiver = state.team1.players[0]
        receiver.x, receiver.y = state.puck.x, state.puck.y - 20
        receiver.precise_x = receiver.precise_y = None
        receiver.motion_x = receiver.motion_y = 0
        receiver.decision_timer = 0
        self.assertIsNone(waiting_model()._receive_pass(state))
        state = reception_state()
        state.team1.players[0].stick_x = None
        self.assertIsNone(waiting_model()._receive_pass(state))

    def test_switch_has_fresh_b_edge_and_steering_requires_confirmed_receiver(self):
        for schema in ('FILTERED', 'HOCKEY_INTENT_DPAD'):
            state, model = reception_state(), waiting_model(schema)
            with patch('nhl94_ai.agents.classic_v1.DefenseController._likely_switch', return_value=0):
                action, intent = model._receive_pass(state)
                self.assertEqual(intent, HOCKEY_INTENT_CHANGE_PLAYER)
                self.assertEqual(action[Buttons.INPUT_B], 1)
                self.assertFalse(action[4:9].any())
                model._encode(action, intent)
                model.defense.frames += 1
                action, intent = model._receive_pass(state)
                self.assertEqual(intent, HOCKEY_INTENT_NOOP)
                self.assertFalse(action.any())
                state.team1.defense_control = 0
                action, intent = model._receive_pass(state)
                self.assertTrue(action[4:8].any())
                self.assertFalse(action[Buttons.INPUT_B] or action[Buttons.INPUT_C])
                self.assertEqual(model._last_decision, 'receive-pass')

    def test_wrong_native_switch_target_is_not_requested_and_wait_is_bounded(self):
        state, model = reception_state(), waiting_model()
        with patch('nhl94_ai.agents.classic_v1.DefenseController._likely_switch', return_value=2):
            self.assertFalse(model._receive_pass(state)[0].any())
            model.defense.frames += 16
            self.assertIsNone(model._receive_pass(state))
        self.assertIsNone(model.offense.pending)
        self.assertEqual(model.offense.last_pass['outcome'], 'reception-unreachable')

    def test_unexpected_switch_and_unavailable_receiver_release_to_defense(self):
        for unavailable in (False, True):
            state, model = reception_state(), waiting_model()
            model._receive_pass(state)
            if unavailable:
                state.team1.players[0].unavailable = 4
            else:
                state.team1.defense_control = 2
            self.assertIsNone(model._receive_pass(state))
            self.assertIsNone(model.offense.pending)
            self.assertTrue(model._defending(state))

    def test_no_correction_before_launch_during_stoppage_or_after_possession(self):
        for condition in ('windup', 'stoppage', 'possession', 'unknown-recipient', 'unknown-control'):
            state, model = reception_state(), waiting_model()
            if condition == 'windup':
                model.offense.pending['launched'] = False
            elif condition == 'stoppage':
                state.engine.clock_stopped = True
            elif condition == 'unknown-recipient':
                model.offense.pending['actual_receiver'] = None
            elif condition == 'unknown-control':
                state.team1.defense_control = -1
            else:
                state.engine.puck_owner = 0
            self.assertIsNone(model._receive_pass(state))

    def test_observed_reception_and_turnover_end_correction_on_the_next_frame(self):
        for owner, outcome in ((0, 'received'), (6, 'intercepted')):
            state, model = reception_state(), waiting_model()
            model._receive_pass(state)
            state.engine.puck_owner = owner
            model._observe_ordinary_pass(state)
            self.assertIsNone(model.offense.pending)
            self.assertEqual(model.offense.last_pass['outcome'], outcome)
            self.assertIsNone(model._receive_pass(state))

    def test_one_timer_has_no_ordinary_reception_controller(self):
        model, state = waiting_model(), reception_state()
        model.offense.pending = None
        model._one_timer = (3, 0, 60)
        self.assertIsNone(model._receive_pass(state))


if __name__ == '__main__':
    unittest.main()
