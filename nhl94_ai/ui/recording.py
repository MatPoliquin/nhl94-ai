"""Encode copied Pygame canvases in a worker on a wall-clock MP4 timeline."""
from collections import deque
import math
from pathlib import Path
from threading import Condition, Thread
import time

import cv2
import numpy as np
import pygame


class MP4Recorder:
    FPS = 60
    MAX_PENDING_FRAMES = 4

    def __init__(self, path, size):
        self.path = Path(path).expanduser()
        if self.path.suffix.lower() != '.mp4':
            raise ValueError('--record-mp4 requires a filename ending in .mp4')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.size = size
        self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*'mp4v'), self.FPS, size)
        if not self.writer.isOpened():
            self.writer.release()
            raise RuntimeError(f'Cannot open MP4 recording: {self.path}. Check the output path and OpenCV encoder.')
        self.started_at = None
        self.last_frame = None
        self.frames_written = 0
        self.closed = False
        self.dropped_captures = 0
        self._pending = deque()
        self._condition = Condition()
        self._error = None
        self._end_frame = 0
        self._worker = Thread(target=self._encode, name='nhl94-mp4-encoder', daemon=True)
        self._worker.start()

    def _repeat_until(self, count):
        while self.frames_written < count:
            self.writer.write(self.last_frame)
            self.frames_written += 1

    def capture(self, surface):
        if self.closed:
            return
        self._raise_worker_error()
        if surface.get_size() != self.size:
            raise ValueError(f'Recording canvas must remain {self.size}')
        now = time.monotonic()
        if self.started_at is None:
            self.started_at = now
        frame_index = int((now - self.started_at) * self.FPS)
        # Copy while the canvas belongs to the display thread. No mutable
        # Pygame surface or view crosses into the encoder thread.
        pixels = pygame.image.tobytes(surface, 'RGB')
        with self._condition:
            if len(self._pending) == self.MAX_PENDING_FRAMES:
                self._pending.popleft()
                self.dropped_captures += 1
            self._pending.append((frame_index, pixels))
            self._condition.notify()

    def _encode(self):
        try:
            try:
                while True:
                    with self._condition:
                        self._condition.wait_for(lambda: self._pending or self.closed)
                        if not self._pending:
                            break
                        frame_index, pixels = self._pending.popleft()
                    # Encoding and gap filling never hold the queue lock.
                    # Dropped captures retain their original timestamps, so
                    # overload reduces detail rather than shortening the video.
                    if self.last_frame is not None:
                        self._repeat_until(frame_index)
                    rgb = np.frombuffer(pixels, dtype=np.uint8).reshape(self.size[1], self.size[0], 3)
                    self.last_frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    self._repeat_until(frame_index + 1)
                if self.last_frame is not None:
                    self._repeat_until(self._end_frame)
            finally:
                self.writer.release()
        except Exception as error:
            with self._condition:
                self._error = error
                self._pending.clear()

    def _raise_worker_error(self):
        if self._error is not None:
            raise RuntimeError(f'MP4 recording failed: {self.path}: {self._error}') from self._error

    def close(self):
        if self.closed:
            return
        with self._condition:
            if self.started_at is not None:
                self._end_frame = math.ceil((time.monotonic() - self.started_at) * self.FPS)
            self.closed = True
            self._condition.notify()
        # Freeze the end timestamp before draining so shutdown time is not
        # appended to the recording. This is the only wait for the encoder.
        self._worker.join()
        self._raise_worker_error()
