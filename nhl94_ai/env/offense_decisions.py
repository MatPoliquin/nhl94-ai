"""Expose actual offensive choices while simulating delegated execution."""
import gymnasium as gym

from nhl94_ai.agents.learned_offense import N_ACTIONS, SCHEMA


def can_choose(observation):
    return observation[-1, -N_ACTIONS:].sum() > 1


class DecisionEnv(gym.Wrapper):
    def __init__(self, env, gamma=0.997):
        super().__init__(env)
        self.gamma = gamma
        self.ready = False

    def reset(self, *, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        # Pre-decision play has no preceding learned action to credit. Use the
        # teacher to reach the first decision, while updating native histories.
        if not can_choose(observation):
            controller = self.unwrapped.controllers[self.unwrapped.side - 1]
            mode = controller.mode
            controller.mode = 'teacher'
            try:
                while not can_choose(observation):
                    observation, _, terminated, truncated, info = self.env.step(0)
                    if terminated or truncated:
                        raise RuntimeError('Episode ended before an offensive decision; use possession snapshots')
            finally:
                controller.mode = mode
        self.ready = True
        return observation, info

    def step(self, action):
        if not self.ready:
            raise RuntimeError('Reset before stepping a completed decision episode')
        reward, elapsed, native_frames, raw_reward = 0.0, 0.0, 0, 0.0
        first = True
        command_used = False
        while True:
            observation, part, terminated, truncated, info = self.env.step(action if first else 0)
            if first and not info['actor_active']:
                raise RuntimeError('Decision transition started without policy control')
            command_used = command_used or info['command_used']
            reward += self.gamma ** elapsed * part
            raw_reward += part
            native_frames += info['native_frames']
            elapsed += info['native_frames'] / SCHEMA['interval']
            first = False
            if terminated or truncated or can_choose(observation):
                break
        self.ready = not (terminated or truncated)
        info.update(actor_active=True, command_used=command_used, elapsed_steps=elapsed,
                    native_frames=native_frames, undiscounted_reward=raw_reward)
        return observation, reward, terminated, truncated, info
