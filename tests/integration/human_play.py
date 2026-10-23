"""ROM-backed keyboard-vs-Classic routing through the real playback and display."""
from collections import defaultdict
import os
from unittest.mock import patch

import numpy as np

from nhl94_ai.config import load_hyperparams, resolve_hyperparams_for_model
from nhl94_ai.evaluation.play import NHL94Player, parse_cmdline
from nhl94_ai.game.constants import GameConsts as Buttons
from nhl94_ai.game.specs import GAMES


class Finished(Exception):
    """End the bounded playback after exercising both keyboard directions."""


def check_game(game):
    import pygame
    from nhl94_ai.ui.targets import TARGET_GREEN
    args = parse_cmdline(['--mode=player_vs_model', '--nn=ClassicAIV1',
                         f'--env={game}', '--rf=PostPlay', '--max_playback_speed=0.5'])
    args.hyperparams_dict = resolve_hyperparams_for_model(load_hyperparams(args.hyperparams), args.nn)
    player = NHL94Player(args, None, need_display=True)
    try:
        display = player.display_env
        assert player.p1_env.action_space.shape == (12,), 'Policy input/action contract changed'
        assert display.env.action_space.shape == (24,)
        assert player.display_frame_interval == 1 / 30
        player.display_frame_interval = 0  # No wall-clock throttling in this bounded ROM check.
        frame = ai_frames = human_frames = marker_frames = 0
        started_at = None
        signs = set()
        sent_step, shown_step = display.env.step, display.step

        def pressed():
            keys = defaultdict(bool)
            if started_at is None:
                keys[pygame.K_x] = frame % 32 == 0
            elif frame - started_at < 360:
                phase = frame - started_at
                keys[pygame.K_RIGHT if phase < 180 else pygame.K_LEFT] = True
                keys[pygame.K_UP if phase < 180 else pygame.K_DOWN] = True
                keys[pygame.K_c] = phase % 80 == 40
            return keys

        def capture_input(actions):
            nonlocal ai_frames, human_frames, started_at
            action = np.asarray(actions)
            assert action.shape == (1, 24), action.shape
            expected = np.zeros(12, dtype=np.int8)
            for key, index in display.key_action_map.items():
                expected[index] = pressed()[key]
            np.testing.assert_array_equal(action[0, 12:], expected)
            model = player.ai_sys.models[player.ai_sys.model_in_use]
            np.testing.assert_array_equal(action[0, :12], model.get_action_preferences()[0])
            ai_frames += bool(action[0, :12].any())
            human_frames += bool(action[0, 12:].any())
            result = sent_step(actions)
            assert not result[2][0], 'Fixture ended before input coverage completed'
            info = result[3][0]
            if started_at is None and info['time'] & 65535 < 300:
                started_at = frame + 1
            assert (info['defense_team1'], info['defense_team2']) == (1, 2)
            opponent = display.env.get_attr('opponent_action_state')[0]['last_env_action']
            np.testing.assert_array_equal(opponent, expected)
            slot = info['defense_control2']
            if 6 <= slot < 6 + GAMES[game].skaters_per_team:
                vx = info[f'defense_{slot}_vx']
                if abs(vx) > 256:
                    signs.add(1 if vx > 0 else -1)
            return result

        def capture_display(actions):
            nonlocal frame, marker_frames
            result = shown_step(actions)
            pixels = pygame.surfarray.array3d(display.game_surf)
            marker_frames += bool(display.classic_defense and np.any(np.all(pixels == TARGET_GREEN, axis=2)))
            frame += 1
            assert frame < 1400, 'Faceoff/play did not start within the frame budget'
            if started_at is not None and frame - started_at == 376:
                raise Finished
            return result

        with patch('pygame.key.get_pressed', side_effect=pressed), \
                patch.object(display.env, 'step', side_effect=capture_input), \
                patch.object(display, 'step', side_effect=capture_display):
            try:
                player.play(continuous=False, need_reset=False)
            except Finished:
                pass
        assert ai_frames > 20 and human_frames >= 360
        assert signs == {-1, 1}, 'Keyboard did not move the actual away skater in both directions'
        assert marker_frames > 0, 'Classic defensive target was hidden during human play'
        print(f'PASS: {game}: keyboard controls P2, AI controls P1, releases arrive, green target stays visible')
    finally:
        player.close()


if __name__ == '__main__':
    with patch.dict(os.environ, {'SDL_VIDEODRIVER': 'dummy', 'SDL_AUDIODRIVER': 'dummy'}):
        for variant in GAMES:
            check_game(variant)
