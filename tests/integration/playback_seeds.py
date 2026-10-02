"""Verify ROM seed writes and reproducible variation in the exact watched command."""
import hashlib
import os
from unittest.mock import patch

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

from nhl94_ai.cli import main
from nhl94_ai.config import load_hyperparams, resolve_hyperparams_for_model
from nhl94_ai.env.factory import build_single_nhl94_env, make_retro
from nhl94_ai.env.wrappers import EpisodeROMSeed
from nhl94_ai.evaluation.play import NHL94Player
from nhl94_ai.game.ram import ROM_RNG_ADDRESS


class Finished(Exception):
    """Stop a short replay without waiting for a full period."""


def replay(player, expected_seed):
    for model in player.ai_sys.models:
        if model is not None:
            model.reset()
    digest = hashlib.sha256()
    frames = 0
    shown_step = player.display_env.step

    def capture(actions):
        nonlocal frames
        result = shown_step(actions)
        info = result[3][0]
        assert not result[2][0], 'Seed fixture ended before the bounded replay finished'
        assert (info['defense_team1'], info['defense_team2']) == (2, 0)
        if expected_seed is None:
            assert 'episode_seed' not in info
        else:
            assert info['episode_seed'] == expected_seed
        digest.update(np.asarray(actions, dtype=np.int8).tobytes())
        digest.update(str(tuple(info[field] for field in ('puck_x', 'puck_y', 'puck_owner'))).encode('ascii'))
        frames += 1
        if frames == 600:
            raise Finished
        return result

    with patch.object(player.display_env, 'step', side_effect=capture):
        try:
            player.play(continuous=False, need_reset=False)
        except Finished:
            pass
    assert frames == 600
    return digest.hexdigest()


def check(args):
    args.hyperparams_dict = resolve_hyperparams_for_model(load_hyperparams(args.hyperparams), args.nn)
    inner = build_single_nhl94_env(
        args, args.hyperparams_dict, use_frame_skip=False, episode_rom_seed=args.seed,
    )
    vector = DummyVecEnv([lambda: inner])
    try:
        vector.reset()
        assert vector.reset_infos[0]['episode_seed'] == 12000
        for seed in (12000, 12001):
            inner.unwrapped.data.memory.assign(0xFFC468, '>u2', 9)
            _, _, done, info = vector.step([np.zeros(12, dtype=np.int8)])
            assert done[0], 'Clock cutoff did not trigger the real vector auto-reset'
            assert info[0]['episode_seed'] == seed
            assert vector.reset_infos[0]['episode_seed'] == seed + 1
            assert (vector.reset_infos[0]['defense_team1'], vector.reset_infos[0]['defense_team2']) == (2, 0)
    finally:
        vector.close()
    sequences = []
    for _ in range(2):
        player = NHL94Player(args, None, need_display=True)
        try:
            player.display_frame_interval = 0
            sequences.append([replay(player, seed) for seed in (12000, 12001)])
        finally:
            player.close()
    assert sequences[0] == sequences[1], 'Equal seed sequences did not reproduce watched play'
    assert sequences[0][0] != sequences[0][1], 'Different episode seeds did not vary watched play'
    args.seed = None
    player = NHL94Player(args, None, need_display=True)
    try:
        player.display_frame_interval = 0
        assert replay(player, None) == replay(player, None), 'Default fixed-save playback changed'
    finally:
        player.close()
    print('PASS: watched away Classic/manual-goalie play varies per episode, reproduces by seed, '
          'and retains deterministic fixed-save defaults')


def run():
    env = EpisodeROMSeed(make_retro(
        game='NHL94-Genesis-v0', state='SabresVsMightyDucks.ManualGoalie.Start',
        num_players=1, goalie_policy='selective',
    ), 12000)
    try:
        for seed in (12000, 12001):
            _, info = env.reset()
            assert info['episode_seed'] == seed
            assert env.data.memory.extract(ROM_RNG_ADDRESS, '>u4') == seed
    finally:
        env.close()
    with patch.dict(os.environ, {'SDL_VIDEODRIVER': 'dummy', 'SDL_AUDIODRIVER': 'dummy'}), \
            patch('nhl94_ai.evaluation.play.run', side_effect=check):
        main(['play', '--agent', 'classic-v1', '--env', 'NHL94-Genesis-v0',
              '--mode', 'model_vs_game', '--state', 'SabresVsMightyDucks.ManualGoalie.Start',
              '--side', 'away', '--goalie-policy', 'selective', '--max_playback_speed', '1.0',
              '--seed', '12000'])


if __name__ == '__main__':
    run()
