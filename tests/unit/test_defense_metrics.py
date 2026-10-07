"""Zone attribution and frame/event units for read-only defensive telemetry."""
import unittest
from types import SimpleNamespace

import numpy as np

from nhl94_ai.evaluation.defense_metrics import CarrierDefenseMetrics, GoalieDefenseMetrics
from nhl94_ai.game.constants import GameConsts as Buttons
from tests.unit.test_classic_defense import defense_state


class CarrierDefenseMetricsTests(unittest.TestCase):
    def setUp(self):
        self.state = defense_state()
        self.state.team2.players[0].y = 0
        self.state.team1.players[0].y = 24
        self.metrics = CarrierDefenseMetrics()
        self.details = {'decision': 'intercept-carrier', 'mode': 'skating'}

    def test_counts_behind_and_regained_position_without_claiming_a_tackle(self):
        self.metrics.observe(self.state, self.details)
        self.metrics.after_step(6)
        self.state.team1.players[0].y = -24
        self.metrics.observe(self.state, self.details)
        self.metrics.after_step(-1)
        counts = self.metrics.summary()['neutral']
        self.assertEqual(counts['frames'], 2)
        self.assertEqual(counts['behind_carrier_frames'], 1)
        self.assertEqual(counts['goal_side_frames'], 1)
        self.assertEqual(counts['goal_side_regained'], 1)
        self.assertEqual(counts['cutoff_frames'], 2)
        self.assertNotIn('direct_controlled_recoveries', counts)

    def test_direct_recovery_is_an_event_not_a_repeated_frame(self):
        self.metrics.observe(self.state, dict(self.details, mode='poke-request'))
        self.metrics.after_step(0)
        self.metrics.after_step(0)
        counts = self.metrics.summary()['neutral']
        self.assertEqual(counts['poke_requests'], 1)
        self.assertEqual(counts['direct_controlled_recoveries'], 1)

    def test_loose_pucks_goalies_and_stoppages_do_not_invent_carrier_frames(self):
        for owner, details in ((-1, self.details), (11, self.details), (6, {})):
            self.state.engine.puck_owner = owner
            self.metrics.observe(self.state, details)
            self.metrics.after_step(0)
        self.assertEqual(self.metrics.summary(), {})

    def test_away_orientation_and_zone_boundaries(self):
        self.state.team1.controller, self.state.team2.controller = 2, 1
        self.state.team1.defense_control, self.state.engine.puck_owner = 6, 0
        self.state.team1.net.y = 264
        for y, zone in ((87, 'neutral'), (88, 'defensive'), (-88, 'attacking')):
            self.state.team2.players[0].y = y
            self.state.team1.players[0].y = y + 24
            self.metrics.observe(self.state, self.details)
            self.metrics.after_step(6)
            self.assertEqual(self.metrics.summary()[zone]['goal_side_frames'], 1)
            self.assertEqual(self.metrics.summary()[zone]['direct_controlled_recoveries'], 1)


class GoalieDefenseMetricsTests(unittest.TestCase):
    def test_goal_counters_are_separate_when_native_updates_are_not_simultaneous(self):
        state = defense_state()
        state.team1.defense_goalie, state.team1.defense_control = 5, 5
        state.team1.goalie.selection_flags, state.team1.goalie.live_state_flags = 8, 0
        state.team1.goalie.motion_x = state.team1.goalie.motion_y = 0
        metrics = GoalieDefenseMetrics(0, 0)
        action = np.zeros(12, dtype=np.int8)
        action[Buttons.INPUT_C] = 1
        metrics.observe(state, SimpleNamespace(phase='save-request'), action, 10)
        metrics.after_step(0, 1)
        metrics.observe(state, SimpleNamespace(phase='save-recovery'), action, 11)
        metrics.after_step(1, 1)
        metrics.after_step(1, 1)
        events = metrics.summary()['goal_events']
        self.assertEqual([event['counter'] for event in events], ['one_timer_goals', 'score'])
        self.assertTrue(all(event['effective_manual'] for event in events))
        self.assertEqual(metrics.summary()['input_frames'], {'C-only': 2})

    def test_CPU_policy_never_claims_manual_control(self):
        state = defense_state()
        state.team1.defense_goalie, state.team1.defense_control = 5, 5
        state.team1.goalie.selection_flags, state.team1.goalie.live_state_flags = 8, 0
        metrics = GoalieDefenseMetrics(0, 0)
        metrics.observe(state, None, np.zeros(12, dtype=np.int8), 1)
        metrics.after_step(1, 0)
        self.assertFalse(metrics.events[0]['effective_manual'])


if __name__ == '__main__':
    unittest.main()
