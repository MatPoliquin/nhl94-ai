"""Quiet debug defaults, persistent evaluations and true playback pause."""
from copy import deepcopy
import importlib.util
import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from nhl94_ai.evaluation.play import NHL94Player, parse_cmdline
from nhl94_ai.game.constants import GameConsts as Buttons
from tests.unit.test_environment_contracts import wrapped
from tests.unit.test_target_interfaces import target_args


def evaluation(frame=4, value=42.5):
    return {
        'evaluation_frame': frame, 'evaluation_carrier': 6,
        'actual_slot': 6, 'desired_slot': 7, 'mode': 'advance-pass',
        'target': (0, -200), 'retained_value': 12,
        'teammate_scores': [{
            'slot': 7, 'selected': True,
            'pass': {'status': 'safe', 'value': value, 'shot_value': 10,
                     'continuation_value': 20, 'continuation_status': 'safe'},
            'one_timer': None,
        }],
    }


@unittest.skipUnless(importlib.util.find_spec('pygame'), 'optional display extra is not installed')
class DebugPlaybackTests(unittest.TestCase):
    def setUp(self):
        variables = patch.dict(os.environ, {'SDL_VIDEODRIVER': 'dummy', 'SDL_AUDIODRIVER': 'dummy'})
        variables.start()
        self.addCleanup(variables.stop)
        import pygame
        pygame.init()
        pygame.event.clear()
        self.addCleanup(pygame.quit)

    def display(self, *, terminal=False):
        from stable_baselines3.common.vec_env import DummyVecEnv
        from nhl94_ai.ui.debug import NHL94DebugDisplay
        inner = wrapped()
        inner.unwrapped.render_mode = 'rgb_array'
        inner.env.truncate = terminal
        vector = DummyVecEnv([lambda: inner])
        self.addCleanup(vector.close)
        vector.reset()
        args = target_args(nn='ClassicAIV1', action_type='FILTERED', mode='model_vs_game')
        return NHL94DebugDisplay(vector, args, 0, 'ClassicAIV1', []), inner

    def test_controls_are_printed_in_console_and_compact_stats_stay_in_bottom_card(self):
        import pygame
        with patch('builtins.print') as console:
            display, _ = self.display()
        output = '\n'.join(str(call.args[0]) for call in console.call_args_list)
        self.assertIn('NHL94 debug controls', output)
        self.assertIn('SPACE / P', output)
        self.assertIn('Mouse wheel / PgUp / PgDn', output)
        display.set_ai_sys_info(SimpleNamespace(last_diagnostics={'classic_offense': evaluation()}))
        screen = Mock(wraps=display.screen)
        with patch.object(display, 'screen', screen), patch.object(display, 'font', wraps=display.font) as font, \
                patch('pygame.draw.rect'):
            display._draw_stats(display.game_state.team1, display.game_state.team2)
        texts = [call.args[0] for call in font.render.call_args_list]
        self.assertIn('Score', texts)
        self.assertIn('One-timers', texts)
        self.assertIn('Puck', texts)
        self.assertFalse(any('Arrows' in text or 'SPACE' in text or 'OVERLAYS' in text or '[1]' in text for text in texts))
        for call in screen.blit.call_args_list:
            surface, position = call.args[:2]
            self.assertTrue(display.stats_rect.contains(pygame.Rect(position, surface.get_size())))
        with patch('builtins.print') as console:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_3))
            display.process_events()
        console.assert_called_once_with('velocities: ON')

    def test_stats_share_inspector_background_fonts_and_text_palette(self):
        import pygame
        from nhl94_ai.ui.decision_panel import BACKGROUND, CYAN, WHITE
        display, _ = self.display()
        display.screen.fill((0, 0, 0))
        with patch.object(display, 'font', wraps=display.font) as font, \
                patch.object(display, 'big_font', wraps=display.big_font) as heading:
            display._draw_stats(display.game_state.team1, display.game_state.team2)
        self.assertEqual(display.screen.get_at((display.stats_rect.right - 1, display.stats_rect.bottom - 1))[:3],
                         BACKGROUND)
        self.assertEqual(heading.render.call_args.args[2], WHITE)
        for call in font.render.call_args_list:
            self.assertEqual(call.args[2], CYAN if call.args[0] in ('Statistic', 'TEAM 1', 'TEAM 2') else WHITE)
        self.assertEqual(display.screen.get_at((display.stats_rect.left - 1, display.stats_rect.bottom - 1))[:3],
                         (0, 0, 0))
        self.assertEqual(display.screen.get_at((display.stats_rect.x + display.HUD_PADDING + 180 - 10,
                                               display.stats_rect.y + display.HUD_PADDING + display.HUD_HEADER_HEIGHT + 8))[:3],
                         display.COLOR_RED)

    def test_only_teammate_scores_are_enabled_by_default_and_all_layers_can_be_toggled(self):
        import pygame
        display, _ = self.display()
        attributes = (
            'show_passing_lanes', 'show_one_timer_lanes', 'show_velocities', 'show_orientations',
            'show_distances', 'show_clear_shot_lanes', 'show_open_net_shots', 'show_planner_overlay',
        )
        self.assertTrue(display.show_teammate_scores)
        self.assertTrue(all(not getattr(display, name) for name in attributes))
        for key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4,
                    pygame.K_5, pygame.K_6, pygame.K_7, pygame.K_8, pygame.K_9):
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=key))
        display.process_events()
        self.assertTrue(all(getattr(display, name) for name in attributes))
        self.assertFalse(display.show_teammate_scores)

    def test_score_snapshot_survives_defense_goalie_and_unevaluated_frames(self):
        display, _ = self.display()
        original = evaluation()
        display.set_ai_sys_info(SimpleNamespace(last_diagnostics={'classic_offense': original}))
        cached = deepcopy(display.score_evaluation)
        original['teammate_scores'][0]['pass']['value'] = -100
        original_frame = display.score_evaluation_frame
        for packet in (
                {'classic_defense': {'target': (0, 0)}},
                {'classic_goalie': {'target': (0, 250)}},
                {'classic_offense': {'teammate_scores': [{
                    'slot': 7, 'selected': False, 'pass': {'status': 'not-evaluated'},
                }]}}, {}):
            display.playback_frames += 60
            display.set_ai_sys_info(SimpleNamespace(last_diagnostics=packet))
            self.assertEqual(display.score_evaluation, cached)
            self.assertEqual(display.score_evaluation_frame, original_frame)
            self.assertTrue(display._scores_historical())

    def test_pending_pass_does_not_refresh_the_evaluation_age(self):
        display, _ = self.display()
        details = evaluation()
        display.set_ai_sys_info(SimpleNamespace(last_diagnostics={'classic_offense': details}))
        captured = display.score_evaluation_frame
        display.playback_frames += 120
        details['mode'] = 'pass-flight'
        display.set_ai_sys_info(SimpleNamespace(last_diagnostics={'classic_offense': details}))
        self.assertEqual(display.score_evaluation_frame, captured)
        fresh = evaluation(frame=124, value=30)
        display.set_ai_sys_info(SimpleNamespace(last_diagnostics={'classic_offense': fresh}))
        self.assertEqual(display.score_evaluation_frame, display.playback_frames + 1)
        self.assertEqual(display.score_evaluation['teammate_scores'][0]['pass']['value'], 30)

    def test_fresh_rejected_options_replace_old_numbers_instead_of_hiding_new_risk(self):
        display, _ = self.display()
        display.set_ai_sys_info(SimpleNamespace(last_diagnostics={'classic_offense': evaluation()}))
        blocked = evaluation(frame=8)
        blocked['teammate_scores'][0]['pass'] = {'status': 'moving-interception'}
        display.set_ai_sys_info(SimpleNamespace(last_diagnostics={'classic_offense': blocked}))
        self.assertEqual(display.score_evaluation['evaluation_frame'], 8)
        self.assertNotIn('value', display.score_evaluation['teammate_scores'][0]['pass'])

    def test_away_scores_are_drawn_even_when_selective_goalie_diagnostics_take_priority(self):
        import pygame
        from nhl94_ai.ui.targets import draw_teammate_scores
        display, _ = self.display()
        display.set_ai_sys_info(SimpleNamespace(last_diagnostics={'classic_offense': evaluation()}))
        display.set_ai_sys_info(SimpleNamespace(last_diagnostics={'classic_goalie': {'target': (0, 250)}}))
        with patch('nhl94_ai.ui.targets.draw_teammate_scores', wraps=draw_teammate_scores) as draw:
            display.draw_frame(np.zeros((224, 256, 3), dtype=np.uint8), [{}])
        details = draw.call_args.args[3]
        self.assertTrue(draw.call_args.kwargs['compact'])
        self.assertTrue(details['historical'])
        self.assertEqual(details['actual_slot'], 6)
        self.assertEqual(details['teammate_scores'][0]['slot'], 7)
        self.assertTrue(np.any(pygame.surfarray.array3d(display.debug_surf)))

    def test_episode_reset_and_vector_auto_reset_clear_old_scores(self):
        for terminal in (False, True):
            with self.subTest(terminal=terminal):
                display, _ = self.display(terminal=terminal)
                display.set_ai_sys_info(SimpleNamespace(last_diagnostics={'classic_offense': evaluation()}))
                if terminal:
                    display.step([np.zeros(12, dtype=np.int8)])
                else:
                    display.reset()
                self.assertIsNone(display.score_evaluation)
                self.assertEqual(display.playback_frames, 0)

    def test_pause_blocks_real_steps_but_keeps_controls_and_screenshots_responsive(self):
        import pygame
        display, _ = self.display()
        display.paused = True
        waits = []
        with patch.object(display.env, 'step', wraps=display.env.step) as step, \
                patch('pygame.image.save') as screenshot:
            def waiting(_milliseconds):
                self.assertEqual(step.call_count, 0)
                self.assertEqual(display.playback_frames, 0)
                waits.append(True)
                keys = (pygame.K_3, pygame.K_F2) if len(waits) == 1 else (pygame.K_p,)
                for key in keys:
                    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=key))
            with patch('pygame.time.wait', side_effect=waiting):
                display.step([np.zeros(12, dtype=np.int8)])
            self.assertEqual(step.call_count, 1)
            self.assertEqual(len(waits), 2)
            self.assertTrue(display.show_velocities)
            self.assertFalse(display.paused)
            self.assertEqual(display.playback_frames, 1)
            self.assertFalse(np.asarray(step.call_args.args[0])[0, Buttons.INPUT_START])
            screenshot.assert_called_once()

    def test_repeat_pause_key_does_not_accidentally_resume(self):
        import pygame
        display, _ = self.display()
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))
        display.process_events()
        self.assertTrue(display.paused)
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE, repeat=True))
        display.process_events()
        self.assertTrue(display.paused)
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))
        display.process_events()
        self.assertFalse(display.paused)

    def test_quit_while_paused_never_steps_the_emulator(self):
        import pygame
        display, _ = self.display()
        display.paused = True
        def waiting(_milliseconds):
            pygame.event.post(pygame.event.Event(pygame.QUIT))
        with patch.object(display.env, 'step') as step, \
                patch.object(display.env, 'close', wraps=display.env.close) as close, \
                patch('pygame.time.wait', side_effect=waiting):
            with self.assertRaisesRegex(SystemExit, 'User requested exit'):
                display.wait_until_running()
        step.assert_not_called()
        close.assert_called_once()

    def test_player_pauses_before_policy_prediction_and_resets_playback_pacing_on_resume(self):
        import pygame
        display, _ = self.display(terminal=True)
        player = NHL94Player.__new__(NHL94Player)
        player.args = parse_cmdline(['--mode=model_vs_game', '--nn=ClassicAIV1', '--env=NHL94-Genesis-v0'])
        player.need_display, player.display_env = True, display
        player.ai_sys = Mock(models=[Mock()], last_diagnostics={})
        player.ai_sys.predict.return_value = np.zeros((1, 12), dtype=np.int8)
        player._reset_playback_timer = Mock()
        player._throttle_display_frame = Mock()
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))
        with patch.object(display.env, 'step', wraps=display.env.step) as step:
            def waiting(_milliseconds):
                player.ai_sys.predict.assert_not_called()
                step.assert_not_called()
                pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))
            with patch('pygame.time.wait', side_effect=waiting):
                player.play_player_or_game_mode(continuous=False, need_reset=False)
        player.ai_sys.predict.assert_called_once()
        self.assertEqual(step.call_count, 1)
        self.assertEqual(player._reset_playback_timer.call_count, 2)

    def test_pause_between_repeated_frames_precedes_the_next_scripted_prediction(self):
        import pygame
        display, inner = self.display()
        player = NHL94Player.__new__(NHL94Player)
        player.args = parse_cmdline(['--mode=model_vs_game', '--nn=ClassicAIV1', '--env=NHL94-Genesis-v0'])
        player.need_display, player.display_env = True, display
        player.ai_sys = Mock(models=[Mock()], last_diagnostics={})
        player.ai_sys.predict.return_value = np.zeros((1, 12), dtype=np.int8)
        player._reset_playback_timer = Mock()
        def throttle():
            if player._throttle_display_frame.call_count == 1:
                pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_p))
        player._throttle_display_frame = Mock(side_effect=throttle)
        with patch.object(display.env, 'step', wraps=display.env.step) as step:
            def waiting(_milliseconds):
                self.assertEqual(player.ai_sys.predict.call_count, 1)
                self.assertEqual(step.call_count, 1)
                inner.env.truncate = True
                pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_p))
            with patch('pygame.time.wait', side_effect=waiting):
                player.play_player_or_game_mode(continuous=False, need_reset=False)
        self.assertEqual(player.ai_sys.predict.call_count, 2)
        self.assertEqual(step.call_count, 2)

    def test_target_policy_is_not_evaluated_while_playback_is_paused(self):
        import pygame
        from stable_baselines3.common.vec_env import DummyVecEnv
        from nhl94_ai.ui.debug import NHL94DebugDisplay
        from tests.unit.test_target_control import target_env
        inner = target_env()
        inner.unwrapped.render_mode = 'rgb_array'
        inner.unwrapped.truncate = True
        vector = DummyVecEnv([lambda: inner])
        self.addCleanup(vector.close)
        vector.reset()
        args = target_args(mode='model_vs_game')
        display = NHL94DebugDisplay(vector, args, 0, 'MlpPolicy', [])
        player = NHL94Player.__new__(NHL94Player)
        player.args, player.need_display, player.display_env = args, True, display
        player.args.deterministic = True
        player.target_agent = Mock()
        player.target_agent.act.return_value = SimpleNamespace(action=np.zeros(2, dtype=np.float32))
        player._reset_playback_timer = Mock()
        player._throttle_display_frame = Mock()
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))
        def waiting(_milliseconds):
            player.target_agent.act.assert_not_called()
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))
        with patch('pygame.time.wait', side_effect=waiting):
            player.play_target_mode(continuous=False)
        player.target_agent.act.assert_called_once()


if __name__ == '__main__':
    unittest.main()
