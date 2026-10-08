"""Explicit away/selected-goalie debug replay with a RAM-frozen pause."""
from copy import deepcopy
import argparse
import hashlib
from unittest.mock import patch

import numpy as np

from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.agents.decisions import DecisionSnapshot
from nhl94_ai.config import load_hyperparams, resolve_hyperparams_for_model
from nhl94_ai.evaluation.play import NHL94Player, parse_cmdline


def native_debug_pause(screenshot=None):
    import pygame
    args = parse_cmdline([
        '--nn=ClassicAIV1', '--env=NHL94-Genesis-v0', '--mode=model_vs_game',
        '--state=SabresVsMightyDucks.ManualGoalie.Start', '--side=away',
        '--goalie-policy=selective', '--max_playback_speed=1.0', '--seed=12000', '--rf=PostPlay',
    ])
    args.hyperparams_dict = resolve_hyperparams_for_model(
        load_hyperparams(args.hyperparams, required=True), args.nn)
    player = NHL94Player(args, None, need_display=True)
    try:
        display = player.display_env
        state = display.reset()
        info = player._scripted_reset_info()
        agent = player.ai_sys.models[player.ai_sys.model_in_use]
        controller = agent.controller
        original_act = agent.act
        comparisons = []
        def checked_act(inputs, deterministic=True):
            reference = ClassicAIV1Model(args=controller.args)
            reference.__dict__.update(deepcopy({
                name: value for name, value in controller.__dict__.items() if name not in ('args', 'env')}))
            expected = reference.predict_frame(deepcopy(inputs.game_state), agent.frame_skip, deterministic)[0]
            result = original_act(inputs, deterministic)
            np.testing.assert_array_equal(result.action, expected)
            assert isinstance(result.decision, DecisionSnapshot)
            assert result.decision.action == tuple(result.action)
            assert (controller._tick, controller.defense.frames, controller._frame_remaining) == (
                reference._tick, reference.defense.frames, reference._frame_remaining)
            comparisons.append(True)
            return result
        agent.act = checked_act
        for elapsed in range(900):
            player._wait_for_display()
            actions = player.ai_sys.predict(state, info=info, deterministic=True)
            display.set_ai_sys_info(player.ai_sys)
            state, _, done, info = display.step([actions[0]])
            assert not any(done), 'Probe ended before its inspection point'
            scores = display.score_evaluation
            if scores and any(row['pass'].get('value') is not None for row in scores['teammate_scores']):
                break
        else:
            raise AssertionError('No numeric teammate evaluation during the bounded away replay')
        def ram_digest():
            return hashlib.sha256(display.env.env_method('get_ram')[0].tobytes()).hexdigest()
        before_ram = ram_digest()
        before_clocks = controller._tick, controller.defense.frames, display.playback_frames
        before_scores = deepcopy(display.score_evaluation)
        waits = []
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))
        def inspect(_milliseconds):
            assert ram_digest() == before_ram
            assert (controller._tick, controller.defense.frames, display.playback_frames) == before_clocks
            assert display.score_evaluation == before_scores
            waits.append(True)
            if len(waits) == 1:
                logical = display.inspector.tab_rects['defense'].center
                rect = display.presentation_rect
                position = (round(rect.x + logical[0] * rect.width / display.CANVAS_WIDTH),
                            round(rect.y + logical[1] * rect.height / display.CANVAS_HEIGHT))
                pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=position))
            else:
                assert display.inspector.selected_mode == 'defense'
                assert display.inspector.historical
                assert not display.inspector.highlighted_ids
                assert display.inspector.max_scroll == 0
            keys = (pygame.K_8,) if len(waits) == 1 else (pygame.K_8, pygame.K_SPACE)
            for key in keys:
                pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=key))
        with patch('pygame.time.wait', side_effect=inspect):
            player._wait_for_display()
        assert len(waits) == 2 and not display.paused
        assert display.inspector.selected_mode == display.inspector.current.active_mode
        assert not display.show_planner_overlay
        assert ram_digest() == before_ram
        actions = player.ai_sys.predict(state, info=info, deterministic=True)
        display.set_ai_sys_info(player.ai_sys)
        display.step([actions[0]])
        assert display.playback_frames == before_clocks[2] + 1
        assert len(comparisons) == elapsed + 2
        if screenshot:
            pygame.image.save(display.screen, screenshot)
        print(f'PASS: seeded away Classic/selective-goalie replay captures scores at frame {elapsed + 1}; '
              'inspector outputs match the bare native-cadence controller; '
              'pause freezes ROM RAM, policy clocks and evaluation, serves tabs and overlay keys, then resumes')
    finally:
        player.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--screenshot', help='Save the verified logical 1920x1080 inspector canvas.')
    native_debug_pause(parser.parse_args().screenshot)
