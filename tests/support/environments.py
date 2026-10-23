"""ROM-free Gym and vector-environment fixtures for wrapper and UI contracts."""
import copy
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import gymnasium as gym
import numpy as np

from nhl94_ai.env.observation import NHL94Observation2PEnv
from nhl94_ai.env.target_control import TargetPositionController
from nhl94_ai.evaluation.play import parse_cmdline
from nhl94_ai.game.ram import register_target_state
from tests.support.memory import FixtureMemory

ROOT = Path(__file__).resolve().parents[2]


class FixtureEnv(gym.Env):
    def __init__(self, game='NHL94-Genesis-v0', truncate=False):
        self.info = json.loads((ROOT / f'tests/fixtures/{game}.json').read_text(encoding='utf-8'))
        self.observation_space = gym.spaces.Box(0, 255, (224, 256, 3), dtype=np.uint8)
        self.action_space = gym.spaces.MultiBinary(12)
        self.steps = 0
        self.truncate = truncate

    def reset(self, **kwargs):
        super().reset(seed=kwargs.get('seed'))
        self.steps = 0
        self.data = SimpleNamespace(memory=FixtureMemory(self.info))
        return np.zeros(self.observation_space.shape, dtype=np.uint8), copy.deepcopy(self.info)

    def render(self):
        return np.zeros(self.observation_space.shape, dtype=np.uint8)

    def set_value(self, name, value):
        self.info[name] = value

    def step(self, action):
        self.steps += 1
        return np.zeros(self.observation_space.shape, dtype=np.uint8), 0.0, False, self.truncate, copy.deepcopy(self.info)


def wrapped(nn='MlpPolicy', action_type='FILTERED', rf='PostPlay'):
    args = SimpleNamespace(env='NHL94-Genesis-v0', nn=nn, action_type=action_type,
                           seq_len=4, alg='ppo2', hyperparams_dict={})
    base = FixtureEnv()
    if rf in ('DefenseZone', 'SelfPlayDefenseFinetune'):
        base.info.update(puck_owner=6, period=0, time=274)
    return NHL94Observation2PEnv(base, args, 1, rf)


class TargetFixtureEnv(FixtureEnv):
    def __init__(self):
        super().__init__()
        self.info.update(puck_owner=6, period=0, time=274)
        self.actions = []

    def _register(self, name, definition):
        memory = self.data.memory
        memory.fields[name] = definition['address'], definition['type']
        self.info[name] = memory.extract(definition['address'], definition['type'])

    def reset(self, **kwargs):
        result = super().reset(**kwargs)
        self.data.set_variable = self._register
        self.data.lookup_all = self.info.copy
        register_target_state(self)
        self.actions = []
        return result

    def step(self, action):
        self.actions.append(np.asarray(action).copy())
        return super().step(action)


def target_env(task_name='DefenseZone'):
    args = SimpleNamespace(env='NHL94-Genesis-v0', nn='MlpPolicy', rf=task_name,
                           action_type='TARGET_POSITION', seq_len=16, alg='ppo2', num_players=1,
                           hyperparams_dict={})
    return NHL94Observation2PEnv(TargetFixtureEnv(), args, 1, task_name)


def control_frame():
    env = target_env()
    env.reset(seed=0)
    state, info = env.game_state, env.target_info
    for index, player in enumerate(state.team1.players):
        player.x, player.y = -100 + index * 40, 180
        player.motion_x = player.motion_y = 0.0
        player.is_falling = 0.0
        info[f'target_flags_{index}'] = 0
        info[f'target_unavailable_{index}'] = 0
        info[f'target_facing_{index}'] = 0
    state.team1.players[1].x = state.team1.players[1].y = 0
    state.puck.x, state.puck.y = 100, -100
    state.action = [0] * 6
    # Mechanics tests use unrestricted ice; wrapper tests retain DefenseZone bounds.
    env.target_controller = TargetPositionController()
    env.target_controller.observe(state, info)
    return env, state, info


def target_args(**overrides):
    return SimpleNamespace(**{
        'env': 'NHL94-Genesis-v0', 'rf': 'DefenseZone', 'nn': 'MlpPolicy',
        'action_type': 'TARGET_POSITION', 'num_players': 1, 'alg': 'ppo2',
        **overrides,
    })


def away_args(*flags):
    return parse_cmdline(['--mode=model_vs_game', '--nn=ClassicAIV1',
                          '--env=NHL94-Genesis-v0', '--rf=PostPlay', '--side=away', *flags])


class AwayFixture(FixtureEnv):
    def __init__(self):
        super().__init__()
        self.info.update(defense_team1=1, defense_team2=0, defense_control1=2,
                         defense_control2=-1, defense_2_flags=0x88, defense_8_flags=0x40)
        self.saved_info = deepcopy(self.info)

    def reset(self, **kwargs):
        self.info = deepcopy(self.saved_info)
        result = super().reset(**kwargs)
        self.data.lookup_all = lambda: dict(self.info)
        self.data.set_value = self.set_value
        self.data.update_ram = Mock()
        return result


def saved_info():
    path = Path(__file__).resolve().parents[1] / 'fixtures/NHL94-Genesis-v0.json'
    info = json.loads(path.read_text(encoding='utf-8'))
    for team in (1, 2):
        info.update({f'p{team}_score': 3, f'p{team}_passing': 19,
                     f'p{team}_bodychecks': 7, f'p{team}_shots': (123 << 16) | 12,
                     f'p{team}_onetimer': (321 << 16) | 4})
    return info


class StatisticsEnv:
    def __init__(self, events=None, terminal=True):
        self.events = events or {}
        self.terminal = terminal
        self.reset_infos = []

    def reset(self):
        self.reset_infos = [saved_info()]
        return np.zeros((1, 310), dtype=np.float32)

    def step(self, action):
        info = saved_info()
        for key, value in self.events.items():
            info[key] += value
        # A VecEnv replaces reset_infos before returning a terminal transition.
        self.reset_infos = [{key: 0 for key in info}]
        return np.zeros((1, 310)), np.array([1.0]), np.array([self.terminal]), [info]
