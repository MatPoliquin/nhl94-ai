"""Retention labels distinguish accidental loss, brief touches and releases."""
import unittest

import numpy as np

from nhl94_ai.evaluation.retention import RetentionOutcome, movement_action, movement_id, admissible, summarize
from tests.unit.test_classic_offense import offense_state


class RetentionTests(unittest.TestCase):
    def setUp(self):
        self.state = offense_state()
        self.state.engine.clock_stopped = False
        self.state.engine.puck_owner = 0
        self.state.team1.one_timer_attempts = 0
        self.state.team1.pass_attempts = 0
        self.state.team1.stats.shots = 0
        self.outcome = RetentionOutcome(self.state, 0, 0)

    def advance(self, frame, owner):
        self.state.engine.puck_owner = owner
        self.outcome.observe(frame, self.state, 0, owner)

    def test_brief_separation_is_not_confirmed_loss(self):
        for frame in range(1, 4):
            self.advance(frame, -1)
        self.advance(4, 0)
        self.assertEqual(self.outcome.summary()['loss'], 0)

    def test_loose_puck_is_loss_but_not_confirmed_turnover(self):
        for frame in range(1, 5):
            self.advance(frame, -1)
        result = self.outcome.summary()
        self.assertEqual((result['loss'], result['turnover'], result['separation_frame']), (1, 0, 1))

    def test_confirmed_opponent_control_is_turnover(self):
        for frame in range(1, 5):
            self.advance(frame, 6)
        self.assertEqual(self.outcome.summary()['turnover'], 1)

    def test_native_pass_release_is_not_carry_loss(self):
        self.state.team1.pass_attempts += 1
        for frame in range(1, 9):
            self.advance(frame, -1 if frame < 3 else 6)
        result = self.outcome.summary()
        self.assertEqual((result['loss'], result['turnover'], result['release_kind']), (0, 0, 'pass'))

    def test_pass_button_without_native_release_does_not_hide_loss(self):
        self.state.b_pressed = True
        for frame in range(1, 5):
            self.advance(frame, 6)
        self.assertEqual(self.outcome.summary()['loss'], 1)

    def test_shot_release_is_not_carry_loss(self):
        self.state.engine.shot_player = 0
        self.state.team1.one_timer_attempts += 1
        for frame in range(1, 9):
            self.advance(frame, -1 if frame < 3 else 11)
        result = self.outcome.summary()
        self.assertEqual((result['loss'], result['turnover'], result['release_kind']), (0, 0, 'shot'))

    def test_recovery_and_later_release_do_not_erase_loss(self):
        for frame in range(1, 5):
            self.advance(frame, -1)
        self.advance(5, 0)
        self.state.team1.pass_attempts += 1
        self.advance(6, -1)
        self.assertEqual(self.outcome.summary()['loss'], 1)

    def test_movement_rotation_and_no_action_buttons(self):
        for command in range(9):
            for sign in (-1, 1):
                action = movement_action(command, sign)
                self.assertEqual(movement_id(action, sign), command)
                self.assertLessEqual(np.count_nonzero(action), 2)
        with self.assertRaises(ValueError):
            movement_action(9, 1)

    def test_admission_uses_prediction_not_observed_future(self):
        rows = [dict(features=dict(projected_dy=y, pad_y=pad))
                for y, pad in ((5, 1), (5, -1), (4, 0), (5, 0))]
        self.assertEqual(admissible(rows, 0), [0, 3])

    def test_avoidable_losses_exclude_retreat_and_censored_branches(self):
        base = dict(loss=1, turnover=1, release_frame=None, progress=10, label_valid=True)
        row = dict(source='game', baseline=0, admitted=[0, 1], initial_safe_pass=False,
                   initial_one_timer=False, branches=[base, dict(base, loss=0, progress=-1)])
        self.assertEqual(summarize([row])['progress_preserving_avoidable_losses'], 0)
        row['branches'][1].update(progress=10, label_valid=False)
        self.assertEqual(summarize([row])['progress_preserving_avoidable_losses'], 0)
        row['branches'][1]['label_valid'] = True
        self.assertEqual(summarize([row])['progress_preserving_avoidable_losses'], 1)


if __name__ == '__main__':
    unittest.main()
