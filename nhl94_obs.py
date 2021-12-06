"""
NHL94 Observation wrapper
"""

import os, datetime
import argparse
import retro
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from os import environ

# WARNING: NON FUNCTIONAL CODE - WIP
class NHL94ObservationEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

        low = np.array([0, 0, 0, 0], dtype=np.float32)
        high = np.array([float(300), float(300), float(300), float(300)], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

    def reset(self, **kwargs):
        self.env.reset(**kwargs)

        self.state = (0, 0, 0, 0)

        return self.state

    def step(self, ac):
        ob, rew, done, info = self.env.step(ac)

        p1_x = info.get('p1_x')
        p1_y = info.get('p1_y')
        p2_x = info.get('p2_x')
        p2_y = info.get('p2_y')

        print(p1_x)


        self.state = (p1_x, p1_y, p2_x, p2_y)
       
        return ob, rew, done, info

    def seed(self, s):
        self.rng.seed(s)
