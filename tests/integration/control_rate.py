"""Verify custom Classic input intervals through real watched-play paths.

Requires the installed ROM. Display rendering is disabled, but the regular
play loop dispatches every native input. No game outcome is asserted.
"""
import argparse
from unittest.mock import patch

import numpy as np

from nhl94_ai.config import load_hyperparams, resolve_hyperparams_for_model
from nhl94_ai.evaluation.play import NHL94Player, parse_cmdline


class Finished(Exception):
    """Stop after the requested native input count."""


def replay(interval, side, frames=241):
    args = parse_cmdline(['--mode', 'model_vs_game', '--nn', 'ClassicAIV1',
                         '--env', 'NHL94-Genesis-v0', '--rf', 'PostPlay',
                         '--state', 'SabresVsMightyDucks.ManualGoalie.Start',
                         '--goalie-policy', 'selective', '--side', side,
                         '--classic-control-interval', str(interval), '--seed', '26501'])
    args.hyperparams_dict = resolve_hyperparams_for_model(load_hyperparams(args.hyperparams), args.nn)
    player = NHL94Player(args, None, need_display=False)
    actions = []
    step = player.display_env.step

    def capture(action):
        actions.append(np.asarray(action[0]).copy())
        if interval:
            np.testing.assert_array_equal(actions[-1], actions[(len(actions) - 1) // interval * interval])
        result = step(action)
        assert not result[2][0], 'Playback ended before the cadence verification completed'
        if len(actions) >= frames:
            raise Finished
        return result

    try:
        with patch.object(player.display_env, 'step', side_effect=capture):
            try:
                player.play(continuous=False, need_reset=False)
            except Finished:
                pass
        agent = player.ai_sys.models[1]
        assert agent.scheduler.frames == frames
        if interval:
            assert agent.control_rate.observations == (frames + interval - 1) // interval
        else:
            assert agent.control_rate is None
        agent.reset()
        assert agent.scheduler.frames == 0
        if interval:
            assert agent.control_rate.remaining == agent.control_rate.observations == 0
        print(f'PASS: play {side}, interval={interval}, {frames} native inputs, reset clean')
    finally:
        player.display_env.close()
        player.p1_env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--intervals', nargs='+', type=int, default=[0, 4, 10, 20])
    args = parser.parse_args()
    for interval in args.intervals:
        for side in ('home', 'away'):
            replay(interval, side)


if __name__ == '__main__':
    main()
