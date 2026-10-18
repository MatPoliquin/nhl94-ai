"""PPO over decision boundaries with elapsed-time discounting.

Gamma is per four-frame environment tick; lambda is per decision. Rewards
inside a transition are already discounted by DecisionEnv. The collector is
the discrete-action subset of SB3's on-policy collector, with duration-aware
timeout bootstrapping and returns (SB3 is distributed under the MIT license).
"""
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.utils import obs_as_tensor


class DecisionRolloutBuffer(RolloutBuffer):
    def reset(self):
        super().reset()
        self.durations = np.ones((self.buffer_size, self.n_envs), np.float32)

    def add(self, obs, action, reward, episode_start, value, log_prob, *, durations=None):
        if durations is None:
            raise ValueError('Decision transitions require elapsed durations')
        self.durations[self.pos] = durations
        super().add(obs, action, reward, episode_start, value, log_prob)

    def compute_returns_and_advantage(self, last_values, dones):
        last_values = last_values.detach().cpu().numpy().flatten()
        advantage = np.zeros(self.n_envs, np.float32)
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                continuing, next_values = 1.0 - dones.astype(np.float32), last_values
            else:
                continuing, next_values = 1.0 - self.episode_starts[step + 1], self.values[step + 1]
            discount = self.gamma ** self.durations[step]
            delta = self.rewards[step] + discount * continuing * next_values - self.values[step]
            advantage = delta + discount * self.gae_lambda * continuing * advantage
            self.advantages[step] = advantage
        self.returns = self.advantages + self.values


class DecisionPPO(PPO):
    """Count actual policy choices as timesteps, including across vector workers."""
    def __init__(self, *args, **kwargs):
        kwargs['rollout_buffer_class'] = DecisionRolloutBuffer
        super().__init__(*args, **kwargs)
        if hasattr(self, 'action_space') and (not isinstance(self.action_space, spaces.Discrete) or self.use_sde):
            raise ValueError('DecisionPPO requires discrete actions without SDE')

    def _setup_model(self):
        # Loading an ordinary PPO archive must not restore its fixed-step buffer.
        self.rollout_buffer_class = DecisionRolloutBuffer
        super()._setup_model()

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        assert self._last_obs is not None
        self.policy.set_training_mode(False)
        rollout_buffer.reset()
        callback.on_rollout_start()
        for _ in range(n_rollout_steps):
            with torch.no_grad():
                actions, values, log_probs = self.policy(obs_as_tensor(self._last_obs, self.device))
            actions = actions.cpu().numpy()
            new_obs, rewards, dones, infos = env.step(actions)
            durations = np.asarray([info['elapsed_steps'] for info in infos], np.float32)
            if np.any(durations <= 0) or not all(info['actor_active'] for info in infos):
                raise ValueError('DecisionPPO requires positive-duration active decision transitions')
            self.num_timesteps += env.num_envs
            callback.update_locals(locals())
            if not callback.on_step():
                return False
            self._update_info_buffer(infos, dones)
            for i, done in enumerate(dones):
                if done and infos[i].get('TimeLimit.truncated', False):
                    terminal_obs = self.policy.obs_to_tensor(infos[i]['terminal_observation'])[0]
                    with torch.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs).item()
                    rewards[i] += self.gamma ** durations[i] * terminal_value
            rollout_buffer.add(self._last_obs, actions.reshape(-1, 1), rewards,
                               self._last_episode_starts, values, log_probs, durations=durations)
            self._last_obs, self._last_episode_starts = new_obs, dones
        with torch.no_grad():
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))
        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        callback.update_locals(locals())
        callback.on_rollout_end()
        return True
