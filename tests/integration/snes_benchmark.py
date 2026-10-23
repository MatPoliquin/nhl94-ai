"""Seeded paired SNES benchmark checks on the real native integration.

Run explicitly with the local stable-retro NHL94-Snes-v0 integration and the
imported SNES ROM installed:
    python -m tests.integration.snes_benchmark
"""
import unittest

from nhl94_ai.evaluation.snes_benchmark import compare, summarize
from nhl94_ai.evaluation.snes_session import classic_options, play_snes_period, seed_words

from types import SimpleNamespace


def _options():
    return classic_options(SimpleNamespace(classic_config=None))


def _job(seed, frames=360):
    return {'side': 'home', 'human': False, 'goalie_policy': _options()['goalie_policy'],
            'frame_skip': 4, 'classic_options': _options(), 'seed': seed,
            'max_frames': frames}


class SnesSeeding(unittest.TestCase):
    def test_seed_words_avoid_the_zero_remap(self):
        for seed in range(50):
            word1, word2 = seed_words(seed)
            self.assertTrue(word1 and word2)
        self.assertEqual(len({seed_words(seed) for seed in range(50)}), 50)

    def test_same_seed_reproduces_the_period_exactly(self):
        first = play_snes_period(_job(7))
        second = play_snes_period(_job(7))
        self.assertEqual(first['final_ram_sha256'], second['final_ram_sha256'])
        self.assertEqual(first['goals'], second['goals'])

    def test_different_seeds_diverge(self):
        base = play_snes_period(_job(7))
        others = [play_snes_period(_job(seed)) for seed in (8, 9)]
        self.assertNotEqual(base['final_ram_sha256'], others[0]['final_ram_sha256'])
        self.assertNotEqual(base['final_ram_sha256'], others[1]['final_ram_sha256'])
        self.assertNotEqual(others[0]['final_ram_sha256'], others[1]['final_ram_sha256'])

    def test_paired_compare_shares_seeds_and_summarizes(self):
        options = {**_options(), 'model_args': {}}
        rows = compare(options, options, [3, 4], max_frames=240)
        self.assertEqual(len(rows), 4)
        self.assertEqual({row['seed'] for row in rows}, {3, 4})
        self.assertEqual({(row['home_policy'], row['away_policy']) for row in rows},
                         {('a', 'b'), ('b', 'a')})
        summary = summarize(rows)
        self.assertEqual(summary['a']['periods'], 4)
        self.assertEqual(summary['a']['goal_diff'], 0)  # identical policies draw


if __name__ == '__main__':
    unittest.main()
