import os
import numpy as np
from stable_baselines3.common.atari_wrappers import WarpFrame, ClipRewardEnv
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
import stable_retro as retro
import cv2


class RewardClipper(gym.RewardWrapper):
    def __init__(self, env, low: float = -1.0, high: float = 1.0):
        super().__init__(env)
        self.low = float(low)
        self.high = float(high)

    def reward(self, reward):
        return float(np.clip(float(reward), self.low, self.high))

class WarpFrameDict(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, grayscale=True):
        super().__init__(env)
        self.width = width
        self.height = height
        self.grayscale = grayscale

        img_space = env.observation_space.spaces['image']
        n_channels = 1 if self.grayscale else img_space.shape[0]

        self.observation_space = gym.spaces.Dict({
            'image': gym.spaces.Box(low=0, high=255, shape=(n_channels, self.height, self.width), dtype=np.uint8),
            'scalar': env.observation_space.spaces['scalar']
        })

    def observation(self, obs):
        img = obs['image']  # Expect shape: (C, H, W)

        # Check shape and type for debugging
        #print(f"Original img shape: {img.shape}, dtype: {img.dtype}")

        # Make sure img is a numpy array
        if not isinstance(img, np.ndarray):
            img = np.array(img)

        # Validate shape has 3 dimensions
        assert len(img.shape) == 3, f"Expected image with 3 dims (C,H,W), got {img.shape}"

        # Convert CHW to HWC for OpenCV
        #img = np.transpose(img, (1, 2, 0))  # (H, W, C)

        # Check shape after transpose
        #print(f"After transpose img shape: {img.shape}, dtype: {img.dtype}")
        #assert img.shape[2] in [1, 3, 4], f"Invalid channel number {img.shape[2]} for cv2"

        if self.grayscale:
            # Convert RGB to grayscale (input must have 3 or 4 channels)
            if img.shape[2] == 1:
                # Already grayscale, just resize
                img = img[:, :, 0]
            else:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # Results in (H, W)

            # Resize grayscale
            img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_AREA)

            # Add channel dimension back (1, H, W)
            img = np.expand_dims(img, axis=0)
        else:
            img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_AREA)
            # Convert back to CHW
            #img = np.transpose(img, (2, 0, 1))

        img = img.astype(np.uint8)

        return {'image': img, 'scalar': obs['scalar']}

class StochasticFrameSkip(gym.Wrapper):
    def __init__(self, env, n, stickprob, *, stop_on_truncation=False):
        gym.Wrapper.__init__(self, env)
        self.n = n
        self.stickprob = stickprob
        self.stop_on_truncation = stop_on_truncation
        self.curac = None
        self.rng = np.random.RandomState()
        self.supports_want_render = hasattr(env, "supports_want_render")
        self.frame_agent = None
        self.record_scripted_frames = False
        self.scripted_teacher_only = False
        self.previous_observation = None

    def reset(self, **kwargs):
        self.curac = None
        result = self.env.reset(**kwargs)
        self.previous_observation = result[0]
        return result

    def step(self, ac):
        done = False
        totrew = 0
        traces = []
        for i in range(self.n):
            # First step after reset, use action
            if self.curac is None:
                self.curac = ac
            # First substep, delay with probability=stickprob
            elif i==0:
                if self.rng.rand() > self.stickprob:
                    self.curac = ac
            # Second substep, new action definitely kicks in
            elif i==1:
                self.curac = ac
            if self.frame_agent is not None:
                if i:
                    output = self.frame_agent.act_frame(
                        self.env.get_wrapper_attr('game_state'), self.previous_observation)
                    if not self.scripted_teacher_only:
                        self.curac = output.action
                elif not self.scripted_teacher_only:
                    self.curac = ac
                diagnostics = dict(self.frame_agent.last_output.diagnostics)
                if self.scripted_teacher_only:
                    diagnostics['teacher_action'] = self.frame_agent.last_output.action.copy()
                before = self.previous_observation
            if self.supports_want_render and i<self.n-1:
                ob, rew, terminated, truncated, info = self.env.step(self.curac, want_render=False)
            else:
                ob, rew, terminated, truncated, info = self.env.step(self.curac)
            totrew += rew
            self.previous_observation = ob
            if self.frame_agent is not None:
                info['classic_defense'] = diagnostics.get('classic_defense', {})
                if self.record_scripted_frames:
                    traces.append((np.asarray(before).copy(), np.asarray(self.curac).copy(), rew,
                                   bool(terminated or truncated), diagnostics))
            if terminated or (truncated and (self.stop_on_truncation or self.frame_agent is not None)): break
        if self.record_scripted_frames:
            info['scripted_frames'] = traces
        return ob, totrew, terminated, truncated, info

    def seed(self, s):
        self.rng.seed(s)
