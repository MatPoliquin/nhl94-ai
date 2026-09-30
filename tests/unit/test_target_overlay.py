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
                display.set_ai_sys_info(SimpleNamespace(last_diagnostics={'classic_defense': planner.diagnostics}))
                display.step([np.zeros(12, dtype=np.int8)])
                pixels = pygame.surfarray.array3d(display.game_surf)
                self.assertEqual(bool(np.any(np.all(pixels == TARGET_GREEN, axis=2))), not terminal)

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
