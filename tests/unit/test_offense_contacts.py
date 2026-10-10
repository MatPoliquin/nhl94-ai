"""Contact context is read-only and refers to the input before the observation."""
from copy import deepcopy
import unittest

from nhl94_ai.evaluation.offense_metrics import OffenseMetrics
from tests.unit.test_classic_offense import offense_state


class OffenseContactTests(unittest.TestCase):
    def test_contacts_preserve_preceding_phase_ownership_and_input(self):
        state = offense_state()
        metrics = OffenseMetrics()
        goalie = state.team2.goalie
        goalie.contact_impact, goalie.contact_player = 0, 0
        metrics.observe(0, state)
        diagnostics = {'phase': 'offense', 'carry_safe': False, 'carry_status': 'uncertified'}
        buttons = [0] * 12
        buttons[4] = 1
        before = deepcopy(vars(state.team1.players[0]))
        metrics.record_action(0, state, 'carry-opportunity', diagnostics, buttons, target=(35, 235))
        diagnostics['phase'], buttons[4] = 'defense', 0
        state.engine.puck_owner = -1
        goalie.contact_impact = 12
        metrics.observe(1, state)
        event = metrics.goalie_contacts[0]
        self.assertEqual(metrics.summary()['goalie_contact_impulses'], 1)
        self.assertEqual(event['preceding_action']['frame'], 0)
        self.assertEqual(event['preceding_action']['phase'], 'offense')
        self.assertTrue(event['preceding_action']['carrying'])
        self.assertEqual(event['preceding_action']['buttons'][4], 1)
        self.assertEqual(event['owner'], -1)
        self.assertEqual(vars(state.team1.players[0]), before)
        metrics.observe(2, state)
        self.assertEqual(len(metrics.goalie_contacts), 1)

    def test_non_carrying_contact_is_not_attributed_to_an_old_carry(self):
        state = offense_state()
        metrics = OffenseMetrics()
        goalie = state.team2.goalie
        goalie.contact_impact, goalie.contact_player = 0, 0
        metrics.observe(0, state)
        metrics.record_action(0, state, 'carry', {'phase': 'offense'}, [0] * 12)
        state.engine.puck_owner = 6
        metrics.record_action(1, state, 'cover', {'phase': 'defense'}, [0] * 12)
        goalie.contact_impact = 10
        metrics.observe(2, state)
        action = metrics.goalie_contacts[0]['preceding_action']
        self.assertEqual(action['phase'], 'defense')
        self.assertFalse(action['carrying'])
        self.assertEqual(action['decision'], 'cover')

    def test_missing_action_context_is_explicit_and_history_is_bounded(self):
        state = offense_state()
        metrics = OffenseMetrics()
        state.team2.goalie.contact_impact, state.team2.goalie.contact_player = 0, 0
        metrics.observe(0, state)
        state.team2.goalie.contact_impact = 2
        metrics.observe(1, state)
        self.assertIsNone(metrics.goalie_contacts[0]['preceding_action'])
        for frame in range(12):
            metrics.record_action(frame, state, 'carry', {}, [0] * 12)
        self.assertEqual(len(metrics.action_history), 8)
        self.assertEqual(metrics.action_history[0]['frame'], 4)


if __name__ == '__main__':
    unittest.main()
