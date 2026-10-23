"""Replay the default V1 play command through its actual watched-play path.

Requires the ROM. Rendering is disabled; defense runs every frame and offense
every four frames. Capture the first PostPlay ending before the vector reset.
"""
import hashlib
from unittest.mock import patch

from nhl94_ai.config import load_hyperparams, resolve_hyperparams_for_model
from nhl94_ai.evaluation.play import NHL94Player, parse_cmdline


class SessionFinished(Exception):
    """Stop on the first terminal transition instead of continuing after reset."""


def replay():
    args = parse_cmdline(['--mode=model_vs_game', '--nn=ClassicAIV1',
                         '--env=NHL94-Genesis-v0', '--rf=PostPlay'])
    args.hyperparams_dict = resolve_hyperparams_for_model(load_hyperparams(args.hyperparams), args.nn)
    player = NHL94Player(args, None, need_display=False)
    digest, terminal = hashlib.sha256(), {}
    frames = 0
    defense_frames = 0
    step = player.display_env.step

    def capture(actions):
        nonlocal frames, defense_frames
        frames += 1
        assert frames < 45000, 'Default play did not finish within the frame budget'
        digest.update(bytes(int(value) for value in actions[0]))
        defense_frames += bool(player.ai_sys.last_diagnostics.get('classic_defense'))
        result = step(actions)
        if result[2][0]:
            terminal.update(result[3][0])
            raise SessionFinished
        return result

    try:
        with patch.object(player.display_env, 'step', side_effect=capture):
            try:
                player.play(continuous=False, need_reset=False)
            except SessionFinished:
                pass
    finally:
        player.display_env.close()
        player.p1_env.close()

    assert terminal and defense_frames > 100
    return digest.hexdigest(), frames, defense_frames, tuple(terminal[f'p{i}_score'] for i in (1, 2))


def main():
    first = replay()
    assert replay() == first, 'Default defensive playback is not deterministic'
    print(f'PASS: default V1 defensive playback reproduces actions, frames and scores: {first}')


if __name__ == '__main__':
    main()
