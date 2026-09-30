"""Pass outcomes distinguish button requests, reception and actual ROM shots."""
from copy import deepcopy
import unittest

from nhl94_ai.evaluation.pass_outcomes import PassOutcomes


def initial():
    return dict(puck_owner=6, pass_target=9, last_puck_player=0, shot_player=2,
                p2_pass_attempts=10, bench_one_timers2=3)


class PassOutcomeTests(unittest.TestCase):
    def test_stale_target_is_ignored_until_a_new_targeted_pass(self):
        tracker, info = PassOutcomes(2), initial()
        original = deepcopy(info)
        tracker.start(0, info, 6, 7)
        tracker.observe(1, info)
        self.assertIsNone(tracker.pending['actual'])
        self.assertEqual(info, original)
        info.update(puck_owner=-256, p2_pass_attempts=11, pass_target=8, last_puck_player=6)
        tracker.observe(4, info)
        info.update(puck_owner=8)
        tracker.observe(9, info)
        self.assertEqual(tracker.summary(), {'outcomes': {'received_without_shot': 1}, 'wrong_recipient': 1})

    def test_rom_counter_and_shooter_confirm_a_one_timer_even_after_redirection(self):
        tracker, info = PassOutcomes(2), initial()
        tracker.start(0, info, 6, 7)
        info.update(puck_owner=-1, p2_pass_attempts=11, pass_target=8)
        tracker.observe(4, info)
        info.update(bench_one_timers2=4, shot_player=8)
        tracker.observe(15, info)
        tracker.observe(16, info)
        self.assertEqual(len(tracker.events), 1)
        self.assertEqual(tracker.events[0]['outcome'], 'one_timer_shot')

    def test_interception_and_loss_before_release_are_different(self):
        for launched in (False, True):
            tracker, info = PassOutcomes(2), initial()
            tracker.start(0, info, 6, 7)
            if launched:
                info.update(puck_owner=-256, p2_pass_attempts=11, pass_target=7)
                tracker.observe(4, info)
            info['puck_owner'] = 0
            tracker.observe(8, info)
            expected = 'intercepted' if launched else 'lost_before_release'
            self.assertEqual(tracker.events[0]['outcome'], expected)

    def test_timeout_and_period_end_finish_each_attempt_once(self):
        tracker, info = PassOutcomes(2), initial()
        tracker.start(10, info, 6, 7)
        tracker.observe(130, info)
        self.assertEqual(tracker.events[0]['outcome'], 'not_released')
        tracker.start(150, info, 6, 7)
        tracker.finish(160, info, 'period_ended')
        tracker.finish(161, info, 'period_ended')
        self.assertEqual(len(tracker.events), 2)


if __name__ == '__main__':
    unittest.main()
