"""Bounded Montreal CPU playback through the installed command and real display."""
import argparse
from functools import partial
import os
from unittest.mock import patch

import numpy as np

from nhl94_ai.cli import main
from nhl94_ai.config import load_hyperparams, resolve_hyperparams_for_model
from nhl94_ai.evaluation.play import NHL94Player


class Finished(Exception):
    """End after verifying actual AI inputs, CPU movement and visible targets."""


def check(args, *, one_timer_period=False):
    import pygame
    from nhl94_ai.ui.targets import TARGET_GREEN
    assert (args.nn, args.mode, args.side, args.num_players) == (
        'ClassicAIV1', 'model_vs_game', 'away', 1)
    args.hyperparams_dict = resolve_hyperparams_for_model(load_hyperparams(args.hyperparams), args.nn)
    player = NHL94Player(args, None, need_display=True)
    try:
        display = player.display_env
        assert player.display_frame_interval == 1 / 60
        player.display_frame_interval = 0
        frames = active = defense = markers = cpu_motion = reassignments = 0
        restoring = 0
        slots = set()
        terminal = {}
        shown_step = display.step

        def capture(actions):
            nonlocal frames, active, defense, markers, cpu_motion, restoring, reassignments
            result = shown_step(actions)
            info = result[3][0]
            if result[2][0]:
                assert one_timer_period, 'Fixture ended before controller checks completed'
                terminal.update(info)
                raise Finished
            assert (info['defense_team1'], info['defense_team2']) == (2, 0)
            reassignments += 'away_control_restored_from' in info
            actual = info['defense_control1']
            if actual >= 0:
                assert 6 <= actual <= 11, 'Classic controls a home player'
                if actual < 11:
                    skater = display.game_state.team2.get_player_by_scnum(actual)
                    flags = info[f'defense_{actual}_flags']
                    if flags & 8 or skater.is_one_timer:
                        restoring = 0
                    else:
                        restoring += 1
                        animation_restore = info[f'defense_{actual}_unavailable'] & 2 and restoring == 1
                        assert restoring <= 4 and (flags & 2 or animation_restore), (frames, actual, flags, restoring)
                slots.add(actual)
            assert all(not info[f'defense_{slot}_flags'] & 8 for slot in range(5)), \
                'Montreal still has a joystick-controlled skater'
            for controller, slot in ((1, -1), (2, actual)):
                team = getattr(display.game_state, f'team{controller}')
                assert team.controlled_scnum() == slot
                owner = info['puck_owner']
                assert team.player_haspuck == (team.owns_scnum(owner) and owner != team.goalie_scnum())
            diagnostics = player.ai_sys.last_diagnostics.get('classic_defense')
            if diagnostics:
                assert diagnostics['actual_slot'] < 0 or 6 <= diagnostics['actual_slot'] < 11
                assert all(6 <= slot < 11 for slot in diagnostics.get('arrival_frames', {}))
                defense += 1
            pixels = pygame.surfarray.array3d(display.game_surf)
            markers += bool(np.any(np.all(pixels == TARGET_GREEN, axis=2)))
            active += bool(np.asarray(actions).any())
            cpu_motion += any(abs(info[f'defense_{slot}_vx']) > 256
                              or abs(info[f'defense_{slot}_vy']) > 256 for slot in range(5))
            frames += 1
            assert frames < 45000, 'Montreal period did not end within the frame budget'
            if frames == 1200 and not one_timer_period:
                raise Finished
            return result

        with patch.object(display, 'step', side_effect=capture):
            try:
                player.play(continuous=False, need_reset=False)
            except Finished:
                pass
        assert active > 50 and defense > 50 and markers > 0 and cpu_motion > 50
        assert slots
        if one_timer_period:
            assert terminal, 'Full-period verification stopped before the clock cutoff'
            # Legacy aliases span the preceding stat; the counter is the low word.
            one_timers = terminal['p2_onetimer'] & 65535
            assert one_timers > 0, 'Exact watched Montreal command produced no one-timer'
            print(f'PASS: exact watched Montreal command completes {one_timers} ROM-counted '
                  f'one-timers; Quebec {terminal["p2_score"]}, Montreal {terminal["p1_score"]}')
        display.reset()
        info = display.env.reset_infos[0]
        assert (info['defense_team1'], info['defense_team2']) == (2, 0)
        assert 6 <= info['defense_control1'] < 11
        assert display.classic_defense == {}
        print(f'PASS: Montreal CPU vs away Classic: {frames} frames; '
              f'{active} active AI frames; {cpu_motion} CPU-motion frames; '
              f'{markers} visible-target frames; {reassignments} cross-team reassignments restored; '
              'away control survives reset')
    finally:
        player.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--one-timer-period', action='store_true')
    options = parser.parse_args()
    with patch.dict(os.environ, {'SDL_VIDEODRIVER': 'dummy', 'SDL_AUDIODRIVER': 'dummy'}), \
            patch('nhl94_ai.evaluation.play.run',
                  side_effect=partial(check, one_timer_period=options.one_timer_period)):
        main(['play', '--agent', 'classic-v1', '--env', 'NHL94-Genesis-v0',
              '--state', 'CanadiensVsNordiques.start', '--side', 'away',
              '--max_playback_speed', '1.0'])
