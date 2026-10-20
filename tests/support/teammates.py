"""Controller arguments, RAM data, and multi-controller environment fixtures."""
from copy import deepcopy
from types import SimpleNamespace

import gymnasium as gym
import numpy as np

from tests.support.memory import FixtureMemory
from tests.support.environments import FixtureEnv


def team_args(**values):
    return SimpleNamespace(**dict({'mode': 'team_vs_game', 'env': 'NHL94-Genesis-v0',
                                   'nn': 'ClassicAIV1', 'num_players': 2, 'rf': 'PostPlay',
                                   'action_type': 'FILTERED', 'goalie_policy': 'off',
                                   'hyperparams_dict': {'frame_skip': 4}}, **values))


def team_info(**values):
    info = FixtureEnv().info
    info.update(defense_team1=1, defense_team2=1, defense_control1=0, defense_control2=1,
                teammate_clock=274, goalie_mode_1=1, goalie_mode_2=1,
                goalie_clock_flags=0, goalie_input_block=0, puck_owner=0,
                p1_pass_attempts=0, pass_target=1)
    info.update(values)
    return info


class FixtureData:
    def __init__(self, info):
        self.memory = FixtureMemory(info)
        for name, address, kind in (
                ('defense_team1', 0xFFC328, '>u2'), ('defense_team2', 0xFFC32A, '>u2'),
                ('defense_control1', 0xFFC320, '>i2'), ('defense_control2', 0xFFC322, '>i2'),
                ('teammate_clock', 0xFFC468, '>u2'), ('goalie_mode_1', 0xFFD05A, '>u2'),
                ('goalie_mode_2', 0xFFD05C, '>u2')):
            self.memory.fields[name] = address, kind
            if name in info:
                self.memory.assign(address, kind, info[name])

    def set_variable(self, name, spec):
        self.memory.fields[name] = spec['address'], spec['type']

    @staticmethod
    def filter_action(buttons):
        return buttons & ~sum(1 << index for index in (1, 2, 3, 9, 10, 11))

    def update_ram(self):
        for name, (address, kind) in self.memory.fields.items():
            self.memory.info[name] = self.memory.extract(address, kind)

    def lookup_all(self):
        self.update_ram()
        return deepcopy(self.memory.info)


class TeammateFixture(FixtureEnv):
    def __init__(self):
        super().__init__()
        self.initial = team_info(defense_team2=0, defense_control1=2, defense_control2=-1)
        self.info = deepcopy(self.initial)
        self.data = FixtureData(self.info)
        self.action_space = gym.spaces.MultiBinary(24)
        self.render_mode = 'rgb_array'

    def reset(self, **kwargs):
        gym.Env.reset(self, seed=kwargs.get('seed'))
        self.info = deepcopy(self.initial)
        self.data = FixtureData(self.info)
        self.steps = 0
        return self.render(), self.data.lookup_all()

    def step(self, action):
        self.steps += 1
        self.last_action = np.asarray(action).copy()
        return self.render(), 0.0, False, False, self.data.lookup_all()


def three_info(human=False, **values):
    info = team_info(defense_team3=1, defense_team4=2 if human else 0,
                     defense_control3=2, defense_control4=6 if human else -1)
    for slot in range(12):
        info[f'defense_{slot}_flags'] = (8 if slot in (0, 1, 2) or human and slot == 6 else 0) | (64 if slot >= 6 else 0)
    info.update(values)
    return info


def three_args(human=False, **values):
    return team_args(team_size=3, num_players=4 if human else 3,
                     mode='team_vs_player' if human else 'team_vs_game', **values)


class MultitapFixture(TeammateFixture):
    def reset(self, **kwargs):
        super().reset(**kwargs)
        for port in (3, 4):
            for kind, base, dtype in (('team', 0xFFC328, '>u2'), ('control', 0xFFC320, '>i2')):
                self.data.set_variable(f'defense_{kind}{port}', {'address': base + (port - 1) * 2, 'type': dtype})
        for slot in range(12):
            self.data.set_variable(f'defense_{slot}_flags',
                                   {'address': 0xFFB04A + slot * 128 + 0x62, 'type': '|u1'})
        return self.render(), self.data.lookup_all()
