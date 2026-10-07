"""Green target markers, camera projection and both display entry points."""
from collections import deque
import importlib.util
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from tests.unit.test_target_control import target_env
from tests.unit.test_target_interfaces import target_args


class TargetCameraTests(unittest.TestCase):
    def test_camera_aliases_do_not_change_policy_observations_or_buttons(self):
        env = target_env()
        self.addCleanup(env.close)
        frames = []
        for camera in ((-60, -200), (60, 256)):
            env.reset(seed=0)
            fields = env.env.data.memory.fields
            self.assertEqual(fields['target_scroll_x'], (0xFFBD1E, '>i2'))
            self.assertEqual(fields['target_scroll_y'], (0xFFBD20, '>i2'))
            env.env.info.update(target_scroll_x=-64-camera[0], target_scroll_y=-camera[1])
            observation, _, _, _, info = env.step([0, 0])
            self.assertEqual(observation.shape, (322,))
            frames.append((observation, info['target_control']))
        np.testing.assert_array_equal(frames[0][0], frames[1][0])
        self.assertEqual(frames[0][1], frames[1][1])

    def test_render_camera_uses_scroll_queued_before_the_emulator_step(self):
        env = target_env()
        self.addCleanup(env.close)
        env.reset(seed=0)
        env.target_info.update(target_scroll_x=-39, target_scroll_y=194)
        env.env.info.update(target_scroll_x=-42, target_scroll_y=184)
        _, _, _, _, info = env.step([0, 0])
        self.assertEqual((info['target_camera_x'], info['target_camera_y']), (-25, -194))
        _, _, _, _, info = env.step([0, 0])
        self.assertEqual((info['target_camera_x'], info['target_camera_y']), (-22, -184))


@unittest.skipUnless(importlib.util.find_spec('pygame'), 'optional display extra is not installed')
class TargetOverlayTests(unittest.TestCase):
    def setUp(self):
        variables = patch.dict(os.environ, {'SDL_VIDEODRIVER': 'dummy', 'SDL_AUDIODRIVER': 'dummy'})
        variables.start()
        self.addCleanup(variables.stop)
        import pygame
        pygame.init()
        self.addCleanup(pygame.quit)

    def test_goalie_target_is_cyan_and_inactive_diagnostics_do_not_hide_defense(self):
        import pygame
        from nhl94_ai.agents.goalie import GoalieController
        from nhl94_ai.ui.targets import CYAN, TARGET_GREEN, draw_game_target, select_target
        from tests.unit.test_manual_goalie import goalie_state
        manager = GoalieController('selective')
        manager.step(goalie_state(True))
        surface = pygame.Surface((256, 224))
        info = {'classic_goalie': manager.diagnostics, 'target_camera_x': 0, 'target_camera_y': -200}
        draw_game_target(surface, surface.get_rect(), (256, 224), info)
        self.assertTrue(np.any(np.all(pygame.surfarray.array3d(surface) == CYAN, axis=2)))
        defense = {'phase': 'defense', 'target': (0, -200)}
        info.update(classic_goalie={'target': None}, classic_defense=defense)
        self.assertIs(select_target(info), defense)
        surface.fill((0, 0, 0))
        draw_game_target(surface, surface.get_rect(), (256, 224), info)
        self.assertTrue(np.any(np.all(pygame.surfarray.array3d(surface) == TARGET_GREEN, axis=2)))

    def test_world_view_overlays_mark_the_away_skater_not_home(self):
        import pygame
        from nhl94_ai.ui.targets import CYAN, GREEN, draw_offense_overlay, draw_target_overlay
        from tests.unit.test_classic_defense import defense_state
        state = defense_state()
        details = dict(target=(0, 200), destination=(0, 200), waypoint=(0, 200),
                       actual_slot=6, desired_slot=7, mode='tracking')
        surface = pygame.Surface((600, 700))
        font = pygame.font.Font(None, 16)
        def transform(x, y):
            return round(300 + x), round(350 - y)
        with patch('nhl94_ai.ui.targets.pygame.draw.circle', wraps=pygame.draw.circle) as draw:
            draw_target_overlay(surface, transform, state, details, font)
        player = state.team2.players[0]
        desired = state.team2.players[1]
        draw.assert_any_call(surface, GREEN, transform(player.x, player.y), 12, 3)
        draw.assert_any_call(surface, CYAN, transform(desired.x, desired.y), 16, 2)
        details.update(phase='offense', reason='advance')
        with patch('nhl94_ai.ui.targets.pygame.draw.circle', wraps=pygame.draw.circle) as draw:
            draw_offense_overlay(surface, transform, state, details, font, (12, 12))
        draw.assert_any_call(surface, GREEN, transform(player.x, player.y), 12, 3)

    def test_teammate_labels_show_scores_continuations_rejections_and_away_slots(self):
        import pygame
        from nhl94_ai.ui.targets import draw_teammate_scores
        from tests.unit.test_classic_defense import defense_state
        state = defense_state()
        details = dict(actual_slot=6, desired_slot=7, teammate_scores=[
            {'slot': 7, 'selected': True, 'pass': {
                'status': 'safe', 'value': 42.5, 'shot_value': 0, 'continuation_value': 21,
                'continuation_position_value': 35, 'continuation_finish_value': 10,
                'worthwhile': True}, 'one_timer': {'status': 'safe', 'shot_value': 36}},
            {'slot': 8, 'selected': False, 'pass': {'status': 'moving-interception'}, 'one_timer': None},
        ])
        font = pygame.font.Font(None, 16)
        rendered = []
        wrapped_font = SimpleNamespace(
            get_linesize=font.get_linesize,
            render=lambda text, *args: (rendered.append(text), font.render(text, *args))[1])
        surface = pygame.Surface((600, 700))
        drawn = []
        def transform(x, y):
            drawn.append((x, y))
            return round(300 + x), round(350 - y)
        draw_teammate_scores(surface, transform, state, details, wrapped_font)
        self.assertEqual(drawn, [(state.team2.players[i].x, state.team2.players[i].y) for i in (1, 2)])
        self.assertEqual(rendered, ['7 *P: 42.5', 'Pos: 0.0 / carry: 35.0', 'Finish: 10.0', 'OT: 36.0',
                                    '8 P: --', 'moving-interception'])
        self.assertNotIn('8 P: 0.0', rendered)

    def test_selected_one_timer_and_unknown_carry_do_not_show_a_fake_pass_choice(self):
        import pygame
        from nhl94_ai.ui.targets import draw_teammate_scores
        from tests.unit.test_classic_defense import defense_state
        state = defense_state()
        details = dict(actual_slot=6, desired_slot=7, mode='one-timer-pass', teammate_scores=[
            {'slot': 7, 'selected': True, 'pass': {
                'status': 'safe', 'value': 42.5, 'shot_value': 0,
                'continuation_status': 'no-modeled-reception-contact'},
             'one_timer': {'status': 'safe', 'shot_value': 36}},
        ])
        font = pygame.font.Font(None, 16)
        rendered = []
        wrapped_font = SimpleNamespace(
            get_linesize=font.get_linesize,
            render=lambda text, *args: (rendered.append(text), font.render(text, *args))[1])
        surface = pygame.Surface((600, 700))
        surface.set_clip(pygame.Rect(100, 100, 400, 500))
        draw_teammate_scores(surface, lambda x, y: (300 + x, 350 - y), state, details, wrapped_font)
        self.assertEqual(rendered, ['7 P: 42.5', 'Pos: 0.0 / carry: --',
                                    'no-modeled-reception-contact', '*OT: 36.0'])
        self.assertEqual(surface.get_clip(), pygame.Rect(100, 100, 400, 500))

    def test_marker_is_a_hollow_green_square_not_a_cross(self):
        import pygame
        from nhl94_ai.ui.targets import draw_target_box, TARGET_GREEN
        surface = pygame.Surface((50, 50))
        surface.fill((30, 40, 50))
        draw_target_box(surface, (25, 25))
        pixels = pygame.surfarray.array3d(surface)
        self.assertTrue(np.all(pixels[17:33, 17] == TARGET_GREEN))
        self.assertTrue(np.all(pixels[17, 17:33] == TARGET_GREEN))
        np.testing.assert_array_equal(pixels[25, 25], (30, 40, 50))
        np.testing.assert_array_equal(pixels[16, 17], (30, 40, 50))

    def test_game_projection_tracks_camera_scaling_and_viewport_offset(self):
        import pygame
        from nhl94_ai.ui.targets import draw_game_target, draw_target_box
        surface = pygame.Surface((900, 900))
        for camera in ((-60, -200), (0, -179), (60, -100)):
            for size in ((256, 224), (512, 448), (640, 480)):
                with self.subTest(camera=camera, size=size):
                    rect = pygame.Rect((30, 40), size)
                    info = {'target_control': {'target': (0, -179), 'destination': (10, -190)},
                            'target_camera_x': camera[0], 'target_camera_y': camera[1]}
                    with patch('nhl94_ai.ui.targets.draw_target_box', wraps=draw_target_box) as draw:
                        draw_game_target(surface, rect, (256, 224), info)
                    position = draw.call_args.args[1]
                    self.assertEqual(position, (30 + (128 - camera[0]) * size[0] / 256,
                                                40 + (291 + camera[1]) * size[1] / 224))

    def test_offscreen_targets_are_clipped_not_clamped_and_clip_is_restored(self):
        import pygame
        from nhl94_ai.ui.targets import draw_game_target, TARGET_GREEN
        surface = pygame.Surface((500, 500))
        rect = pygame.Rect(100, 100, 256, 224)
        previous_clip = pygame.Rect(50, 110, 320, 210)
        surface.set_clip(previous_clip)
        info = {'target_control': {'target': (-120, -179)},
                'target_camera_x': 60, 'target_camera_y': -200}
        draw_game_target(surface, rect, (256, 224), info)
        self.assertFalse(np.any(pygame.surfarray.array3d(surface)))
        info['target_control']['target'] = (-68, -179)  # Exactly on the image's left edge.
        draw_game_target(surface, rect, (256, 224), info)
        self.assertEqual(surface.get_clip(), previous_clip)
        x, y = np.where(np.all(pygame.surfarray.array3d(surface) == TARGET_GREEN, axis=2))
        self.assertGreater(len(x), 0)
        self.assertTrue(all(rect.clip(previous_clip).collidepoint(int(px), int(py)) for px, py in zip(x, y)))

    def test_no_target_does_not_draw_or_require_camera_and_missing_camera_fails(self):
        import pygame
        from nhl94_ai.ui.targets import draw_game_target
        surface = pygame.Surface((256, 224))
        for info in ({}, {'target_control': {'target': None}}):
            draw_game_target(surface, surface.get_rect(), surface.get_size(), info)
            self.assertFalse(np.any(pygame.surfarray.array3d(surface)))
        with self.assertRaises(KeyError):
            draw_game_target(surface, surface.get_rect(), surface.get_size(),
                             {'target_control': {'target': (0, -179)}})

    def _vector_env(self, terminal):
        from stable_baselines3.common.vec_env import DummyVecEnv
        env = target_env()
        env.unwrapped.render_mode = 'rgb_array'
        env.unwrapped.truncate = terminal
        vector = DummyVecEnv([lambda: env])
        self.addCleanup(vector.close)
        vector.reset()
        env.env.info.update(target_scroll_x=-4, target_scroll_y=200)
        env.target_info.update(target_scroll_x=-4, target_scroll_y=200)
        return vector

    def test_debug_game_image_has_marker_and_auto_reset_clears_it(self):
        import pygame
        from nhl94_ai.ui.debug import NHL94DebugDisplay
        from nhl94_ai.ui.targets import TARGET_GREEN
        for terminal in (False, True):
            with self.subTest(terminal=terminal):
                display = NHL94DebugDisplay(self._vector_env(terminal),
                                            target_args(mode='model_vs_game'), 0, 'MlpPolicy', [])
                display.step([[0, 0]])
                pixels = pygame.surfarray.array3d(display.game_surf)
                self.assertEqual(bool(np.any(np.all(pixels == TARGET_GREEN, axis=2))), not terminal)

    def test_classic_buttons_show_defensive_target_and_clear_it_after_reset(self):
        import pygame
        from stable_baselines3.common.vec_env import DummyVecEnv
        from nhl94_ai.agents.defense import DefenseController
        from nhl94_ai.ui.debug import NHL94DebugDisplay
        from nhl94_ai.ui.targets import TARGET_GREEN
        from tests.unit.test_environment_contracts import wrapped
        from tests.unit.test_classic_defense import defense_state
        for terminal in (False, True):
            with self.subTest(terminal=terminal):
                inner = wrapped()
                inner.unwrapped.render_mode = 'rgb_array'
                inner.env.truncate = terminal
                inner.env.info.update(defense_scroll_x=-64, defense_scroll_y=179)
                vector = DummyVecEnv([lambda inner=inner: inner])
                self.addCleanup(vector.close)
                vector.reset()
                planner = DefenseController()
                planner.step(defense_state())
                args = target_args(nn='ClassicAIV1', action_type='FILTERED', mode='model_vs_game')
                display = NHL94DebugDisplay(vector, args, 0, 'ClassicAIV1', [])
                display.show_planner_overlay = True
                display.set_ai_sys_info(SimpleNamespace(last_diagnostics={'classic_defense': planner.diagnostics}))
                display.step([np.zeros(12, dtype=np.int8)])
                pixels = pygame.surfarray.array3d(display.game_surf)
                self.assertEqual(bool(np.any(np.all(pixels == TARGET_GREEN, axis=2))), not terminal)

    def test_classic_overlay_draws_covered_open_and_selected_shooting_lanes(self):
        import pygame
        from nhl94_ai.agents.defense import DefenseController
        from nhl94_ai.ui.targets import CYAN, GREEN, PURPLE, draw_target_overlay
        from tests.unit.test_classic_defense import defense_state
        state = defense_state()
        planner = DefenseController()
        planner.step(state)
        details = planner.diagnostics
        surface = pygame.Surface((600, 700))
        def transform(x, y):
            return round(300 + x), round(350 - y)
        font = pygame.font.Font(None, 16)
        with patch('nhl94_ai.ui.targets.pygame.draw.line', wraps=pygame.draw.line) as draw:
            draw_target_overlay(surface, transform, state, details, font)
        for lane in details['lanes']:
            draw.assert_any_call(surface, GREEN if lane['blockers'] else PURPLE,
                                 transform(*lane['origin']), transform(*lane['goal']), 1)
        draw.assert_any_call(surface, CYAN, transform(*details['lanes'][-1]['origin']),
                             transform(*details['shot_goal']), 2)

    def test_body_check_gate_and_estimate_are_visible(self):
        import pygame
        from nhl94_ai.agents.defense import DefenseController
        from nhl94_ai.ui.targets import draw_target_overlay
        from tests.unit.test_classic_defense import defense_state
        planner = DefenseController()
        state = defense_state()
        planner.step(state)
        planner.diagnostics['check'] = {'status': 'ready', 'impact_estimate': 21, 'weight_threshold': 8}
        font = pygame.font.Font(None, 16)
        rendered = []
        wrapped_font = SimpleNamespace(
            get_linesize=font.get_linesize,
            render=lambda text, *args: (rendered.append(text), font.render(text, *args))[1])
        draw_target_overlay(pygame.Surface((600, 700)), lambda x, y: (300 + x, 350 - y),
                            state, planner.diagnostics, wrapped_font)
        self.assertIn('Body check: ready; impact/threshold: 21/8', rendered)

    def test_offensive_target_is_amber_and_clears_on_vector_reset(self):
        import pygame
        from stable_baselines3.common.vec_env import DummyVecEnv
        from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
        from nhl94_ai.ui.debug import NHL94DebugDisplay
        from nhl94_ai.ui.targets import OFFENSE_AMBER, TARGET_GREEN
        from tests.unit.test_environment_contracts import wrapped
        from tests.unit.test_classic_offense import offense_state
        for terminal in (False, True):
            with self.subTest(terminal=terminal):
                inner = wrapped()
                inner.unwrapped.render_mode = 'rgb_array'
                inner.env.truncate = terminal
                inner.env.info.update(defense_scroll_x=-64, defense_scroll_y=30)
                vector = DummyVecEnv([lambda inner=inner: inner])
                self.addCleanup(vector.close)
                vector.reset()
                model = ClassicAIV1Model()
                model.predict_frame(offense_state())
                args = target_args(nn='ClassicAIV1', action_type='FILTERED', mode='model_vs_game')
                display = NHL94DebugDisplay(vector, args, 0, 'ClassicAIV1', [])
                display.show_planner_overlay = True
                display.set_ai_sys_info(SimpleNamespace(last_diagnostics={'classic_offense': model.offense_diagnostics}))
                display.step([np.zeros(12, dtype=np.int8)])
                pixels = pygame.surfarray.array3d(display.game_surf)
                self.assertEqual(bool(np.any(np.all(pixels == OFFENSE_AMBER, axis=2))), not terminal)
                self.assertFalse(np.any(np.all(pixels == TARGET_GREEN, axis=2)))

    def test_live_game_image_has_marker_and_auto_reset_clears_it(self):
        import pygame
        from nhl94_ai.agents.base import FrameRepeatAgent, LearnedAgent
        from nhl94_ai.training.events import LiveTrainingState
        from nhl94_ai.ui.live import LiveTrainingDisplay
        from nhl94_ai.ui.targets import draw_game_target, TARGET_GREEN
        captured = []

        def capture(surface, rect, frame_size, info):
            draw_game_target(surface, rect, frame_size, info)
            captured.append(pygame.surfarray.array3d(surface.subsurface(rect)))

        for terminal in (False, True):
            with self.subTest(terminal=terminal):
                args = target_args(live_window_width=1000, live_window_height=800, live_sleep_per_step=0)
                display = LiveTrainingDisplay(args, LiveTrainingState('', deque()), fps=60, graph_width=260)
                display.env = self._vector_env(terminal)
                display.obs = np.zeros((1, 322), dtype=np.float32)
                display.display_model = SimpleNamespace(
                    predict=lambda *args, **kwargs: (np.zeros((1, 2), dtype=np.float32), None))
                display.target_agent = FrameRepeatAgent(LearnedAgent(display.display_model), 4)
                captured.clear()

                with patch.object(display, '_ensure_model_loaded'), \
                        patch('nhl94_ai.ui.targets.draw_game_target', side_effect=capture), \
                        patch('pygame.display.flip', side_effect=display.stop):
                    display.run()
                self.assertEqual(len(captured), 1)
                self.assertEqual(bool(np.any(np.all(captured[0] == TARGET_GREEN, axis=2))), not terminal)
                if terminal:
                    self.assertIsNone(display.target_diagnostics)
                    self.assertEqual(display.target_frame_info, {})


if __name__ == '__main__':
    unittest.main()
