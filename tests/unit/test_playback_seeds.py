"""Opt-in ROM seed sequences survive explicit and vector episode resets."""
import unittest
from unittest.mock import patch

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

from nhl94_ai.cli import main
from nhl94_ai.env.factory import build_single_nhl94_env, init_play_env
from nhl94_ai.env.observation import NHL94Observation2PEnv
from nhl94_ai.env.wrappers import EpisodeROMSeed
from nhl94_ai.evaluation.play import NHL94Player, parse_cmdline
from nhl94_ai.game.ram import ROM_RNG_ADDRESS
from tests.unit.test_away_play import AwayFixture, away_args
from tests.unit.test_environment_contracts import FixtureEnv


class SeedFixture(FixtureEnv):
    def __init__(self):
        super().__init__()
        self.step_seeds = []

    def step(self, action):
        self.step_seeds.append(self.data.memory.extract(ROM_RNG_ADDRESS, '>u4'))
        return super().step(action)


class PlaybackSeedContracts(unittest.TestCase):
    def test_installed_and_legacy_parsers_keep_saved_rng_by_default(self):
        self.assertIsNone(parse_cmdline([]).seed)
        for flags, expected in (([], None), (['--seed', '12000'], 12000), (['--seed', '0'], 0)):
            with self.subTest(flags=flags), patch('nhl94_ai.evaluation.play.run') as run:
                main(['play', '--agent', 'classic-v1', *flags])
            self.assertEqual(run.call_args.args[0].seed, expected)
        self.assertEqual(parse_cmdline(['--seed', '4294967295']).seed, 2**32 - 1)

    def test_invalid_seeds_and_games_fail_before_emulator_creation(self):
        for seed in (-1, 2**32, True, 1.5, '42'):
            args = away_args()
            args.seed = seed
            with self.subTest(seed=seed), patch('nhl94_ai.evaluation.play.init_env') as init:
                with self.assertRaisesRegex(ValueError, 'uint32'):
                    NHL94Player(args, None, need_display=False)
                init.assert_not_called()
            with self.subTest(seed=seed), self.assertRaisesRegex(ValueError, 'uint32'):
                EpisodeROMSeed(FixtureEnv(), seed)
        args = parse_cmdline(['--env', 'OtherGame', '--seed', '42'])
        with patch('nhl94_ai.evaluation.play.init_env') as init:
            with self.assertRaisesRegex(ValueError, 'Unsupported environment'):
                NHL94Player(args, None, need_display=False)
            init.assert_not_called()

    def test_explicit_resets_write_successive_uint32_seeds_and_preserve_kwargs(self):
        base = SeedFixture()
        env = EpisodeROMSeed(base, 2**32 - 1)
        self.addCleanup(env.close)
        for seed in (2**32 - 1, 0, 1):
            with patch.object(base, 'reset', wraps=base.reset) as reset:
                _, info = env.reset(seed=99, options={'example': True})
            reset.assert_called_once_with(seed=99, options={'example': True})
            self.assertEqual(base.data.memory.extract(ROM_RNG_ADDRESS, '>u4'), seed)
            self.assertEqual(info['episode_seed'], seed)
            _, _, _, _, info = env.step(np.zeros(12, dtype=np.int8))
            self.assertEqual(info['episode_seed'], seed)
            self.assertEqual(base.step_seeds[-1], seed)
            base.data.memory.assign(ROM_RNG_ADDRESS, '>u4', 123)
            self.assertEqual(env.step(np.zeros(12, dtype=np.int8))[-1]['episode_seed'], seed)

    def test_new_environment_reproduces_the_sequence(self):
        sequences = []
        for _ in range(2):
            env = EpisodeROMSeed(FixtureEnv(), 12000)
            self.addCleanup(env.close)
            sequences.append([env.reset()[1]['episode_seed'] for _ in range(3)])
        self.assertEqual(sequences, [[12000, 12001, 12002]] * 2)

    def test_failed_ram_write_does_not_advance_the_seed(self):
        base = FixtureEnv()
        env = EpisodeROMSeed(base, 42)
        self.addCleanup(env.close)
        base.reset()
        with patch.object(base, 'reset', return_value=(None, {})), \
                patch.object(base.data.memory, 'assign', side_effect=RuntimeError('RAM write failed')):
            with self.assertRaisesRegex(RuntimeError, 'RAM write failed'):
                env.reset()
        self.assertEqual(env.next_seed, 42)
        self.assertIsNone(env.episode_seed)

    def test_seed_precedes_first_observation_warmup_frame(self):
        base = SeedFixture()
        args = parse_cmdline(['--mode', 'model_vs_game', '--nn', 'ClassicAIV1',
                             '--env', 'NHL94-Genesis-v0', '--rf', 'PostPlay'])
        env = NHL94Observation2PEnv(EpisodeROMSeed(base, 42), args, 1, 'PostPlay')
        self.addCleanup(env.close)
        for seed in (42, 43):
            observation, info = env.reset()
            self.assertEqual(base.step_seeds[-1], seed)
            self.assertEqual(info['episode_seed'], seed)
            self.assertEqual(np.asarray(observation).shape, (env.NUM_PARAMS,))

    def test_vector_auto_reset_advances_seed_and_preserves_away_assignment(self):
        base = AwayFixture()
        inner = NHL94Observation2PEnv(EpisodeROMSeed(base, 42), away_args(), 1, 'PostPlay')
        env = DummyVecEnv([lambda: inner])
        self.addCleanup(env.close)
        env.reset()
        self.assertEqual(env.reset_infos[0]['episode_seed'], 42)
        for seed in (42, 43):
            base.info['time'] = 9
            _, _, done, info = env.step([np.zeros(12, dtype=np.int8)])
            self.assertTrue(done[0])
            self.assertEqual(info[0]['episode_seed'], seed)
            self.assertEqual(env.reset_infos[0]['episode_seed'], seed + 1)
            self.assertEqual(base.data.memory.extract(ROM_RNG_ADDRESS, '>u4'), seed + 1)
            self.assertEqual((env.reset_infos[0]['defense_team1'], env.reset_infos[0]['defense_team2']),
                             (2, 0))
            self.assertEqual(inner.game_state.team2.defense_control, 8)

    def test_factory_only_changes_rom_seeds_when_explicitly_requested(self):
        for seed in (None, 0, 42):
            base = SeedFixture()
            args = away_args('--side=home')
            args.seed = 99
            with self.subTest(seed=seed), \
                    patch('nhl94_ai.env.factory.make_retro', return_value=base), \
                    patch('nhl94_ai.env.factory.EpisodeROMSeed', wraps=EpisodeROMSeed) as wrapper:
                env = build_single_nhl94_env(args, {}, use_frame_skip=False, episode_rom_seed=seed)
                self.addCleanup(env.close)
                _, info = env.reset()
                if seed is None:
                    wrapper.assert_not_called()
                    self.assertNotIn('episode_seed', info)
                else:
                    wrapper.assert_called_once_with(base, seed)
                    self.assertEqual(info['episode_seed'], seed)
                    self.assertEqual(base.step_seeds, [seed])

    def test_watched_factory_passes_the_seed_to_the_episode_environment(self):
        args = away_args('--seed=42')
        with patch('nhl94_ai.env.factory.get_button_names', return_value=[]), \
                patch('nhl94_ai.env.factory.init_env') as init:
            env = init_play_env(args, 1, {}, need_display=False)
        self.assertIs(env, init.return_value)
        self.assertEqual(init.call_args.kwargs['episode_rom_seed'], 42)

    def test_target_playback_uses_the_same_episode_seed_hook(self):
        player = NHL94Player.__new__(NHL94Player)
        player.args = parse_cmdline(['--seed', '42', '--model_1', 'policy.zip',
                                    '--action_type', 'TARGET_POSITION'])
        player.args.hyperparams_dict = {}
        player.need_display = False
        player.logger = None
        with patch('nhl94_ai.evaluation.play.EnvironmentConfig.from_args'), \
                patch('nhl94_ai.evaluation.play.init_env') as init, \
                patch('nhl94_ai.evaluation.play.init_model'):
            player.init_target_mode()
        self.assertEqual(init.call_args.kwargs['episode_rom_seed'], 42)


if __name__ == '__main__':
    unittest.main()
