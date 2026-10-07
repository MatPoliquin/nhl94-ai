"""Native counters and stable possession distinguish attack endings from rebounds."""
import unittest

from nhl94_ai.evaluation.carry_outcomes import CarryOutcomes
from tests.unit.test_classic_offense import offense_state


class CarryOutcomeTests(unittest.TestCase):
    def setUp(self):
        self.state = offense_state()
        self.state.engine.clock_stopped = False
        self.state.engine.puck_owner = 0
        self.state.team1.one_timer_attempts = 0
        self.state.team1.stats.shots = 0
        self.outcomes = CarryOutcomes(self.state, 0, 0)

    def test_loose_puck_is_not_productive_possession_or_turnover(self):
        self.state.engine.puck_owner = -1
        self.assertFalse(self.outcomes.observe(1, self.state, 0, 0))
        self.outcomes.record_options([])
        summary = self.outcomes.summary()
        self.assertEqual(summary['possession_frames'], {'loose': 1})
        self.assertEqual(summary['ordinary_turnovers'], 0)
        self.assertEqual(summary['end_reason'], 'timeout')

    def test_attacking_possession_excludes_loose_puck_time(self):
        self.state.team1.control = 1
        self.state.team1.net.y, self.state.team2.net.y = -240, 240
        self.state.team1.get_player_by_scnum(0).y = 180
        self.outcomes.observe(1, self.state, 0, 0)
        self.state.engine.puck_owner = -1
        self.outcomes.observe(2, self.state, 0, 0)
        summary = self.outcomes.summary()
        self.assertEqual(summary['controlled_offensive_zone_frames'], 1)
        self.assertEqual(summary['team_offensive_zone_frames'], 1)
        self.assertEqual(summary['possession_fractions']['loose'], 0.5)
        self.assertEqual(summary['maximum_controlled_attack_depth'], 180)

    def test_brief_opponent_touch_is_not_a_confirmed_turnover(self):
        self.state.engine.puck_owner = 6
        for frame in range(1, 4):
            self.assertFalse(self.outcomes.observe(frame, self.state, 0, 6))
        self.state.engine.puck_owner = 0
        self.assertFalse(self.outcomes.observe(4, self.state, 0, 0))
        self.assertEqual(self.outcomes.summary()['ordinary_turnovers'], 0)

    def test_four_frames_of_opponent_control_confirm_turnover(self):
        self.state.engine.puck_owner = 6
        for frame in range(1, 4):
            self.assertFalse(self.outcomes.observe(frame, self.state, 0, 6))
        self.assertTrue(self.outcomes.observe(4, self.state, 0, 6))
        self.assertEqual(self.outcomes.summary()['end_reason'], 'ordinary-turnover')
        self.assertEqual(self.outcomes.summary()['ordinary_turnovers'], 1)

    def test_native_attempt_continues_until_save_and_is_not_turnover(self):
        self.state.team1.one_timer_attempts = 1
        self.state.engine.shot_player = 1
        self.state.engine.puck_owner = -1
        self.assertFalse(self.outcomes.observe(1, self.state, 0, 1))
        self.state.team1.stats.shots = 1
        self.state.engine.puck_owner = self.state.team2.goalie_scnum()
        for frame in range(2, 5):
            self.assertFalse(self.outcomes.observe(frame, self.state, 0, self.state.team2.goalie_scnum()))
        self.assertTrue(self.outcomes.observe(5, self.state, 0, self.state.team2.goalie_scnum()))
        summary = self.outcomes.summary()
        self.assertEqual(summary['ordinary_turnovers'], 0)
        self.assertEqual(summary['shot_events'][0]['outcome'], 'save-held')
        self.assertTrue(summary['shot_events'][0]['recorded_shot'])

    def test_goal_after_goalie_touch_wins_over_save_classification(self):
        self.state.team1.one_timer_attempts = 1
        self.state.engine.shot_player = 1
        self.state.engine.puck_owner = -1
        self.outcomes.observe(1, self.state, 0, 1)
        self.assertFalse(self.outcomes.observe(2, self.state, 0, self.state.team2.goalie_scnum()))
        self.assertTrue(self.outcomes.observe(3, self.state, 1, self.state.team2.goalie_scnum()))
        self.assertEqual(self.outcomes.summary()['shot_events'][0]['outcome'], 'goal')

    def test_unresolved_native_attempt_is_not_a_success_at_timeout(self):
        self.state.team1.one_timer_attempts = 1
        self.state.engine.shot_player = 1
        self.state.engine.puck_owner = -1
        self.outcomes.observe(1, self.state, 0, 1)
        summary = self.outcomes.summary()
        self.assertEqual(summary['end_reason'], 'timeout')
        self.assertEqual(summary['shot_events'][0]['outcome'], 'unresolved-at-timeout')

    def test_loose_goalie_rebound_is_followed_until_a_later_goal(self):
        self.state.team1.one_timer_attempts = 1
        self.state.engine.shot_player = 1
        self.state.engine.puck_owner = -1
        self.outcomes.observe(1, self.state, 0, 1)
        goalie = self.state.team2.goalie_scnum()
        for frame in range(2, 31):
            self.assertFalse(self.outcomes.observe(frame, self.state, 0, goalie))
        self.assertTrue(self.outcomes.observe(31, self.state, 1, goalie))
        self.assertEqual(self.outcomes.summary()['shot_events'][0]['outcome'], 'goal')

    def test_normal_release_needs_animation_contact_and_puck_impulse(self):
        player = self.state.team1.get_player_by_scnum(0)
        player.live_anim, player.live_anim_frame = 0x7FC, 0x1C
        self.state.puck.motion_x, self.state.puck.motion_y = 0, 0
        outcomes = CarryOutcomes(self.state, 0, 0)
        self.state.engine.shot_player = 0
        self.state.engine.puck_owner = -1
        self.state.puck.motion_y = 4
        self.assertFalse(outcomes.observe(1, self.state, 0, 0))
        self.assertEqual(outcomes.summary()['shot_events'][0]['release_frame'], 1)

    def test_losing_puck_early_in_windup_does_not_claim_a_release(self):
        player = self.state.team1.get_player_by_scnum(0)
        player.live_anim, player.live_anim_frame = 0x7FC, 8
        self.state.puck.motion_x, self.state.puck.motion_y = 0, 0
        outcomes = CarryOutcomes(self.state, 0, 0)
        self.state.engine.shot_player = 0
        self.state.engine.puck_owner = -1
        self.state.puck.motion_y = 4
        outcomes.observe(1, self.state, 0, 6)
        self.assertEqual(outcomes.summary()['shot_events'], [])

    def test_latched_shooter_and_c_input_do_not_create_shots(self):
        self.state.engine.shot_player = 0
        self.state.c_pressed = True
        self.state.engine.puck_owner = -1
        self.outcomes.observe(1, self.state, 0, 0)
        self.assertEqual(self.outcomes.summary()['shot_events'], [])

    def test_option_windows_count_creation_not_every_frame(self):
        options = [{'value': 42}]
        for sample in ([], options, options, [], options):
            self.outcomes.record_options(sample)
        self.state.engine.puck_owner = -1
        self.outcomes.observe(1, self.state, 0, 0)
        summary = self.outcomes.summary()
        self.assertEqual(summary['one_timer_option_windows'], 2)
        self.assertEqual(summary['frames_with_one_timer_option'], 3)
        self.assertEqual(summary['best_one_timer_value'], 42)


if __name__ == '__main__':
    unittest.main()
