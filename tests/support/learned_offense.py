"""Deterministic fake transitions for learned-offense contracts."""
from types import SimpleNamespace

import gymnasium as gym
import numpy as np

from nhl94_ai.agents.learned_offense import N_ACTIONS, OBSERVATION_FIELDS
from nhl94_ai.env.offense import OffenseEnv


class TickEnv(gym.Env):
    """Decisions at ticks zero and three; a delayed goal at tick five."""
    def __init__(self, cutoff=5, truncated=False):
        super().__init__()
        self.observation_space = gym.spaces.Box(-1, 1, (16, len(OBSERVATION_FIELDS)), np.float32)
        self.action_space = gym.spaces.Discrete(N_ACTIONS)
        self.cutoff, self.truncated, self.ticks = cutoff, truncated, 0
        self.actions = []

    def observation(self):
        obs = np.zeros(self.observation_space.shape, np.float32)
        obs[-1, -N_ACTIONS:] = int(self.ticks in (0, 3))
        obs[-1, -N_ACTIONS] = 1
        return obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.ticks = 0
        return self.observation(), {}

    def step(self, action):
        active = self.ticks in (0, 3)
        self.actions.append(int(action))
        self.ticks += 1
        done = self.ticks == self.cutoff
        return (self.observation(), float(done and not self.truncated),
                done and not self.truncated, done and self.truncated,
                dict(native_frames=4, actor_active=active, command_used=active))

    def render(self):
        return None


class DelayedOutcomeEnv(gym.Env):
    """A decision followed by delegated execution; no emulator needed."""
    def __init__(self, cutoff=3, truncation=False):
        super().__init__()
        self.cutoff, self.truncation, self.ticks = cutoff, truncation, 0
        self.observation_space = gym.spaces.Box(-1, 1, (16, len(OBSERVATION_FIELDS)), np.float32)
        self.action_space = gym.spaces.Discrete(N_ACTIONS)

    def observation(self):
        obs = np.zeros(self.observation_space.shape, np.float32)
        obs[-1, -N_ACTIONS:] = 1 if self.ticks == 0 else 0
        obs[-1, -N_ACTIONS] = 1
        return obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.ticks = 0
        return self.observation(), {}

    def step(self, action):
        self.ticks += 1
        end = self.ticks >= self.cutoff
        return self.observation(), float(end and self.cutoff == 3), end and not self.truncation, end and self.truncation, {}

    def render(self):
        return None


def make_offense_env(scores, *, budget=None):
    env = OffenseEnv(max_frames=budget)
    env.side, env.opponent, env.state_name, env.episode_seed = 1, 'cpu', 'fake', 0
    env.frames, env.finished = 1, False
    env.start_score = env.previous_score = (0, 0)
    env.info = dict(p1_score=0, p2_score=0, period=0, bench_clock=300,
                    bench_shots1=0, bench_shots2=0, bench_one_timers1=0, bench_one_timers2=0)
    controller = SimpleNamespace(begin_step=lambda *_: False, frame=lambda *_: np.zeros(12, np.int8),
                                 teacher_command=None, command_used=False)
    env.controllers, env.views = [controller, controller], [None, None]
    import hashlib
    env.digest = hashlib.sha256()
    env.choices = np.zeros(N_ACTIONS, dtype=int)
    env._refresh = lambda: None
    env._append_history = lambda **_kwargs: None
    env._observation = lambda: np.zeros(env.observation_space.shape, np.float32)
    def native_step(_):
        a, b = next(scores)
        return None, 0, False, False, dict(env.info, p1_score=a, p2_score=b)
    env.emulator = SimpleNamespace(step=native_step)
    return env
