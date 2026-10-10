"""Full-canvas MP4 capture, wall-clock pacing, and playback cleanup."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from threading import Event, Thread, get_ident
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from nhl94_ai.cli import main, parse_configured
from nhl94_ai.evaluation.play import NHL94Player, build_parser, parse_cmdline
from tests.unit import test_debug_playback


class RecordingArgumentsTests(unittest.TestCase):
    def test_installed_command_and_config_resolve_recording_path(self):
        with patch('nhl94_ai.evaluation.play.run') as run:
            main(['play', '--agent', 'classic-v1', '--record-mp4', 'match.mp4'])
        self.assertEqual(run.call_args.args[0].record_mp4, 'match.mp4')
        self.assertIsNone(parse_cmdline([]).record_mp4)
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'play.json'
            config.write_text(json.dumps({'record_mp4': 'videos/match.mp4'}), encoding='utf-8')
            args = parse_configured(build_parser(), ['--config', str(config)])
            self.assertEqual(args.record_mp4, str(Path(directory) / 'videos/match.mp4'))

    def test_unsupported_displays_fail_before_emulator_creation(self):
        for mode, need_display in (('model_vs_model', True), ('model_vs_game', False)):
            args = parse_cmdline(['--mode', mode, '--record-mp4', 'match.mp4'])
            with patch('nhl94_ai.evaluation.play.init_env') as initialize:
                with self.assertRaisesRegex(ValueError, 'requires the debug display'):
                    NHL94Player(args, None, need_display=need_display)
                initialize.assert_not_called()

    def test_player_starts_recording_and_cleans_up_if_encoder_fails(self):
        args = parse_cmdline(['--mode=model_vs_game', '--record-mp4', 'match.mp4'])
        display = Mock()
        def initialize(player):
            player.display_env = display
            player.p1_env = Mock()
        with patch.object(NHL94Player, 'init_player_or_game_mode', initialize):
            player = NHL94Player(args, None)
            display.start_recording.assert_called_once_with('match.mp4')
            player.close()
            display.reset_mock()
            display.start_recording.side_effect = RuntimeError('encoder unavailable')
            with self.assertRaisesRegex(RuntimeError, 'encoder unavailable'):
                NHL94Player(args, None)
            display.close.assert_called_once()


@unittest.skipUnless(importlib.util.find_spec('pygame'), 'optional display extra is not installed')
class RecordingTests(unittest.TestCase):
    def setUp(self):
        variables = patch.dict(os.environ, {'SDL_VIDEODRIVER': 'dummy', 'SDL_AUDIODRIVER': 'dummy'})
        variables.start()
        self.addCleanup(variables.stop)
        import pygame
        pygame.init()
        pygame.event.clear()
        self.addCleanup(pygame.quit)
        self.directory = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.addCleanup(self.directory.cleanup)

    def test_wall_clock_pacing_keeps_color_orientation_and_latest_fast_frame(self):
        import pygame
        from nhl94_ai.ui.recording import MP4Recorder
        surface = pygame.Surface((4, 2))
        frames = []
        writer = Mock()
        writer.write.side_effect = lambda frame: frames.append(frame.copy())
        now = [0.0]
        with patch('nhl94_ai.ui.recording.cv2.VideoWriter', return_value=writer), \
                patch('nhl94_ai.ui.recording.time.monotonic', side_effect=lambda: now[0]):
            recorder = MP4Recorder(Path(self.directory.name) / 'match.mp4', surface.get_size())
            surface.fill((255, 0, 0))
            surface.set_at((3, 1), (100, 150, 200))
            recorder.capture(surface)
            now[0] = 0.004
            surface.fill((0, 255, 0))
            recorder.capture(surface)
            now[0] = 0.034
            surface.fill((0, 0, 255))
            recorder.capture(surface)
            now[0] = 0.1
            recorder.close()
            recorder.close()
        self.assertEqual(len(frames), 6)
        self.assertEqual(frames[0].shape, (2, 4, 3))
        np.testing.assert_array_equal(frames[0][0, 0], [0, 0, 255])
        np.testing.assert_array_equal(frames[0][1, 3], [200, 150, 100])
        np.testing.assert_array_equal(frames[1][0, 0], [0, 255, 0])
        for frame in frames[2:]:
            np.testing.assert_array_equal(frame[0, 0], [255, 0, 0])
        writer.release.assert_called_once()

    def test_encoder_open_failure_and_invalid_extension_are_reported(self):
        from nhl94_ai.ui.recording import MP4Recorder
        writer = Mock()
        writer.isOpened.return_value = False
        with patch('nhl94_ai.ui.recording.cv2.VideoWriter', return_value=writer):
            with self.assertRaisesRegex(RuntimeError, 'Cannot open MP4 recording'):
                MP4Recorder(Path(self.directory.name) / 'match.mp4', (1920, 1080))
            writer.release.assert_called_once()
            with self.assertRaisesRegex(ValueError, 'ending in .mp4'):
                MP4Recorder(Path(self.directory.name) / 'match.avi', (1920, 1080))

    def test_full_canvas_mp4_survives_resize_pause_reset_and_escape(self):
        import pygame
        helper = test_debug_playback.DebugPlaybackTests()
        self.addCleanup(helper.doCleanups)
        display, _ = helper.display()
        self.addCleanup(display.close)
        path = Path(self.directory.name) / 'new-directory/full-window.mp4'
        now = [0.0]
        with patch('nhl94_ai.ui.recording.time.monotonic', side_effect=lambda: now[0]):
            display.start_recording(path)
            display.reset()
            picture = np.full((224, 256, 3), (30, 120, 210), dtype=np.uint8)
            with patch.object(display.env, 'render', return_value=picture):
                display.step([0] * 12)
            expected = cv2.cvtColor(np.transpose(pygame.surfarray.array3d(display.screen), (1, 0, 2)),
                                    cv2.COLOR_RGB2BGR)
            pygame.event.post(pygame.event.Event(pygame.VIDEORESIZE, w=640, h=480))
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_p))
            now[0] = 0.02
            def resume(_milliseconds):
                self.assertTrue(display.paused)
                now[0] = 0.08
                pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_p))
            with patch.object(display.env, 'step', wraps=display.env.step) as step, \
                    patch('pygame.time.wait', side_effect=resume):
                self.assertTrue(display.wait_until_running())
                step.assert_not_called()
            self.assertEqual(display.window.get_size(), (640, 480))
            now[0] = 0.12
            display.reset()
            display.step([0] * 12)
            now[0] = 0.15
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))
            with patch.object(display.env, 'close', wraps=display.env.close) as close:
                with self.assertRaises(SystemExit):
                    display.process_events()
                display.close()
                close.assert_called_once()
        capture = cv2.VideoCapture(str(path))
        try:
            self.assertTrue(capture.isOpened())
            self.assertEqual(capture.get(cv2.CAP_PROP_FPS), 60)
            self.assertEqual(capture.get(cv2.CAP_PROP_FRAME_COUNT), 9)
            ok, frame = capture.read()
            self.assertTrue(ok)
            self.assertEqual(frame.shape, (1080, 1920, 3))
            # Compare each panel, tolerating lossy MP4 encoding.
            for rect in (pygame.Rect(0, 0, 740, 1080), display.game_rect,
                         display.mini_rink_rect, display.stats_rect):
                area = np.s_[rect.top:rect.bottom, rect.left:rect.right]
                error = np.abs(frame[area].astype(float) - expected[area]).mean()
                self.assertLess(error, 8, (rect, error))
            # Paused frames include the PAUSED label, without advancing the ROM.
            capture.read()
            ok, paused = capture.read()
            self.assertTrue(ok)
            label = paused[12:40, display.game_rect.x + 12:display.game_rect.x + 110]
            self.assertTrue(np.any((label[:, :, 1] > 180) & (label[:, :, 2] > 180) & (label[:, :, 0] < 80)))
        finally:
            capture.release()

    def test_close_releases_encoder_even_when_final_write_fails(self):
        import pygame
        from nhl94_ai.ui.recording import MP4Recorder
        writer = Mock()
        written = Event()
        writer.write.side_effect = lambda _frame: written.set()
        now = [0.0]
        with patch('nhl94_ai.ui.recording.cv2.VideoWriter', return_value=writer), \
                patch('nhl94_ai.ui.recording.time.monotonic', side_effect=lambda: now[0]):
            recorder = MP4Recorder(Path(self.directory.name) / 'match.mp4', (4, 2))
            recorder.capture(pygame.Surface((4, 2)))
            self.assertTrue(written.wait(2))
            now[0] = 0.1
            writer.write.side_effect = RuntimeError('encoder failed')
            with self.assertRaisesRegex(RuntimeError, 'encoder failed'):
                recorder.close()
            writer.release.assert_called_once()

    def test_slow_encoder_does_not_block_capture_and_queue_stays_bounded(self):
        import pygame
        from nhl94_ai.ui.recording import MP4Recorder
        entered, release = Event(), Event()
        frames, threads, failures = [], [], []
        writer = Mock()
        def encode(frame):
            threads.append(get_ident())
            entered.set()
            if not release.wait(5):
                raise RuntimeError('Test encoder was not released')
            frames.append(frame.copy())
        writer.write.side_effect = encode
        now = [0.0]
        with patch('nhl94_ai.ui.recording.cv2.VideoWriter', return_value=writer), \
                patch('nhl94_ai.ui.recording.time.monotonic', side_effect=lambda: now[0]):
            recorder = MP4Recorder(Path(self.directory.name) / 'match.mp4', (4, 2))
            recorder.capture(pygame.Surface((4, 2)))
            try:
                self.assertTrue(entered.wait(2))
                def capture_many():
                    try:
                        surface = pygame.Surface((4, 2))
                        for index in range(1, 11):
                            now[0] = index / 60 + 0.000001
                            surface.fill((index, 100, 200))
                            recorder.capture(surface)
                        # Queued pixels must survive subsequent surface changes.
                        surface.fill((0, 0, 0))
                    except Exception as error:
                        failures.append(error)
                producer = Thread(target=capture_many, daemon=True)
                producer.start()
                producer.join(timeout=2)
                self.assertFalse(producer.is_alive(), 'Capture waited for the encoder')
                self.assertFalse(failures)
                self.assertEqual(len(recorder._pending), recorder.MAX_PENDING_FRAMES)
                self.assertEqual(recorder.dropped_captures, 6)
            finally:
                now[0] = 0.2
                release.set()
                recorder.close()
            self.assertFalse(recorder._worker.is_alive())
        self.assertEqual(len(frames), 12)
        self.assertTrue(all(ident == recorder._worker.ident for ident in threads))
        self.assertNotEqual(recorder._worker.ident, get_ident())
        np.testing.assert_array_equal(frames[-1][0, 0], [200, 100, 10])
        writer.release.assert_called_once()

    def test_worker_errors_reach_capture_and_close_without_hanging(self):
        import pygame
        from nhl94_ai.ui.recording import MP4Recorder
        writer = Mock()
        writer.write.side_effect = OSError('disk write failed')
        with patch('nhl94_ai.ui.recording.cv2.VideoWriter', return_value=writer):
            recorder = MP4Recorder(Path(self.directory.name) / 'match.mp4', (4, 2))
            try:
                recorder.capture(pygame.Surface((4, 2)))
                recorder._worker.join(timeout=2)
                self.assertFalse(recorder._worker.is_alive())
                with self.assertRaisesRegex(RuntimeError, 'disk write failed'):
                    recorder.capture(pygame.Surface((4, 2)))
            finally:
                with self.assertRaisesRegex(RuntimeError, 'disk write failed'):
                    recorder.close()
            recorder.close()
        writer.release.assert_called_once()

    def test_empty_recording_closes_the_waiting_worker(self):
        from nhl94_ai.ui.recording import MP4Recorder
        writer = Mock()
        with patch('nhl94_ai.ui.recording.cv2.VideoWriter', return_value=writer):
            recorder = MP4Recorder(Path(self.directory.name) / 'match.mp4', (4, 2))
            recorder.close()
        self.assertFalse(recorder._worker.is_alive())
        writer.write.assert_not_called()
        writer.release.assert_called_once()


if __name__ == '__main__':
    unittest.main()
