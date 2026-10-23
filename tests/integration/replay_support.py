"""Native replay reset/restore checks; requires the installed Genesis ROM.

Run with python -m tests.integration.replay_support.
"""
from copy import deepcopy
import hashlib
import unittest

import numpy as np

from nhl94_ai.env.factory import make_retro
from nhl94_ai.evaluation.replay_support import advance, prepare_period, read_view, restore
from nhl94_ai.game.state import NHL94GameState


class NativeReplaySupportTests(unittest.TestCase):
    def trajectory(self, env, decoded, action, side):
        current = deepcopy(decoded)
        result = []
        for _ in range(24):
            advance(env, action, side)
            _, frame_info = read_view(env, current, side)
            self.assertEqual((frame_info['bench_team1'], frame_info['bench_team2']), (side, 0))
            result.append(hashlib.sha256(env.get_ram().tobytes()).hexdigest())
        return result

    def test_restored_input_sequence_reproduces_ram_on_both_sides(self):
        for side in (1, 2):
            with self.subTest(side=side):
                env = make_retro(game='NHL94-Genesis-v0',
                                 state='SabresVsMightyDucks.ManualGoalie.Start',
                                 num_players=1, goalie_policy='selective')
                try:
                    info = prepare_period(env, 7, side)
                    self.assertEqual((info['bench_team1'], info['bench_team2']), (side, 0))
                    state = NHL94GameState(5)
                    read_view(env, state, side)
                    snapshot = env.em.get_state()
                    digest = hashlib.sha256(env.get_ram().tobytes()).hexdigest()
                    decoded = deepcopy(state)
                    action = np.zeros(12, dtype=np.int8)
                    action[7] = 1

                    first = self.trajectory(env, decoded, action, side)
                    restore(env, snapshot, digest)
                    self.assertEqual(first, self.trajectory(env, decoded, action, side))
                    with self.assertRaises(AssertionError):
                        restore(env, snapshot, '0' * 64)
                finally:
                    env.close()


if __name__ == '__main__':
    unittest.main()
