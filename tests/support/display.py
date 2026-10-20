"""Headless display fixtures with cleanup registered on the calling test case."""
import os
from unittest.mock import patch

from tests.support.environments import target_args, wrapped


def initialize_pygame(test_case):
    variables = patch.dict(os.environ, {'SDL_VIDEODRIVER': 'dummy', 'SDL_AUDIODRIVER': 'dummy'})
    variables.start()
    test_case.addCleanup(variables.stop)
    import pygame
    pygame.init()
    pygame.event.clear()
    test_case.addCleanup(pygame.quit)


def debug_display(test_case, *, terminal=False):
    from stable_baselines3.common.vec_env import DummyVecEnv
    from nhl94_ai.ui.debug import NHL94DebugDisplay
    inner = wrapped()
    inner.unwrapped.render_mode = 'rgb_array'
    inner.env.truncate = terminal
    vector = DummyVecEnv([lambda: inner])
    test_case.addCleanup(vector.close)
    vector.reset()
    args = target_args(nn='ClassicAIV1', action_type='FILTERED', mode='model_vs_game')
    return NHL94DebugDisplay(vector, args, 0, 'ClassicAIV1', []), inner
