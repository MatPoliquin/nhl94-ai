"""Short natural goalie episodes with active-decision time accounting."""
from collections import Counter, OrderedDict, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import gymnasium as gym
import numpy as np

from nhl94_ai.agents.learned_goalie import (
    N_ACTIONS, TOKEN_COUNT, TOKEN_WIDTH, CommandGoalie, SampledGoalie,
    action_mask, decision_available, observation, replace_goalie,
)
from nhl94_ai.evaluation.cpu_benchmark import CPU_RAM, cpu_view
from nhl94_ai.evaluation.goalie import load_start
from nhl94_ai.game.ram import pass_geometry_info, restore_away_control


def bank_rows(banks):
    rows = []
    for path in banks:
        report = json.loads(Path(path).read_text(encoding='utf-8'))
        if report.get('kind') != 'native-selected-manual-goalie-starts':
            raise ValueError('Expected a natural manual-goalie snapshot bank')
        rows.extend(report['starts'])
    if not rows or len({row['path'] for row in rows}) != len(rows):
        raise ValueError('Goalie banks must contain unique, nonempty snapshots')
    return rows


class GoalieEnv(gym.Env):
    """One goalie controls movement, C saves and A dives; other play stays native."""
    metadata = {'render_modes': ['rgb_array']}

    def __init__(self, banks, *, mode='learned', interval=4, horizon=180, gamma=0.997):
        super().__init__()
        if mode not in ('learned', 'classic', 'contact') or interval < 1 or horizon < interval:
            raise ValueError('Invalid goalie mode, interval or episode horizon')
        if not 0 < gamma <= 1:
            raise ValueError('Goalie discount must be in (0, 1]')
        self.rows = bank_rows(banks)
        self.groups = defaultdict(list)
        for index, row in enumerate(self.rows):
            self.groups[row['source']].append(index)
        self.sources = tuple(self.groups)
        self.mode, self.interval, self.horizon, self.gamma = mode, interval, horizon, gamma
        self.action_space = gym.spaces.Discrete(N_ACTIONS)
        self.observation_space = gym.spaces.Box(-1, 1, (TOKEN_COUNT * TOKEN_WIDTH + N_ACTIONS,), np.float32)
        self.emulator = None
        self.cache = OrderedDict()
        self.finished = True
        self.total_native_frames = self.total_decisions = self.total_commands_used = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options and 'index' in options:
            index = int(options['index'])
        else:
            source = self.sources[int(self.np_random.integers(len(self.sources)))]
            index = int(self.np_random.choice(self.groups[source]))
        row = self.rows[index]
        if index not in self.cache:
            self.cache[index] = load_start(row['path'])
            if len(self.cache) > 128:
                self.cache.popitem(last=False)
        self.cache.move_to_end(index)
        source, meta = self.cache[index]
        if meta['sha256'] != row['sha256']:
            raise ValueError('Snapshot differs from its bank manifest')
        sample = deepcopy(source)
        if self.emulator is None:
            from nhl94_ai.env.factory import make_retro
            self.emulator = make_retro(game='NHL94-Genesis-v0', state=meta['state'],
                                      num_players=1, goalie_policy='selective')
            for name, (address, kind) in CPU_RAM.items():
                self.emulator.data.set_variable(name, {'address': address, 'type': kind})
            self.emulator.reset(seed=meta['seed'])
        self.emulator.em.set_state(sample['snapshot'])
        self.emulator.data.update_ram()
        if hashlib.sha256(bytes(self.emulator.get_ram())).hexdigest() != sample['ram_sha256']:
            raise RuntimeError('Restored goalie snapshot RAM differs from collection')
        self.info = self.emulator.data.lookup_all()
        self.side, self.start = meta['side'], meta
        self.state, self.view, self.model = sample['root_state'], sample['view'], sample['model']
        replacement = (CommandGoalie() if self.mode == 'learned' else
                       SampledGoalie(interval=self.interval, save_timing=self.mode))
        self.model.goalie = replace_goalie(self.model.goalie, replacement)
        self.initial_metrics = Counter(self.model.goalie.metrics)
        self.frames = self.decisions = self.commands_used = self.manual_frames = 0
        self.owner_run, self.previous_owner = 0, None
        self.initial_goals = self.view.team2.stats.score
        self.initial_one_timers = self.view.team2.one_timer_attempts
        self.finished, self.outcome = False, None
        obs = observation(self.view, self.model.goalie)
        if obs[-N_ACTIONS:].sum() <= 1:
            raise RuntimeError('Goalie snapshot has no available decision')
        return obs, {'start': meta['path'], 'source': meta['source'], 'ram_sha256': sample['ram_sha256']}

    def _ending(self):
        view = self.view
        owner = view.engine.puck_owner
        self.owner_run = self.owner_run + 1 if owner == self.previous_owner else 1
        self.previous_owner = owner
        if view.team2.stats.score > self.initial_goals:
            return 'goal', -1.0
        if view.team1.owns_scnum(owner) and self.owner_run >= 4:
            if owner == view.team1.defense_goalie:
                effective = bool(view.team1.goalie.selection_flags & 8
                                 and not view.team1.goalie.live_state_flags & 4)
                return ('manual-catch' if effective else 'automatic-catch'), 1.0
            return 'team-recovery', 1.0
        if view.engine.clock_stopped:
            return 'stoppage', 0.0
        if self.frames >= self.horizon:
            return 'timeout', 0.0
        return None, 0.0

    def step(self, action):
        if self.finished:
            raise RuntimeError('Reset before stepping a completed goalie episode')
        current = action_mask(self.view, self.model.goalie)
        if current.sum() <= 1:
            raise RuntimeError('Goalie transitions must start on an active choice')
        if self.mode == 'learned' and not current[int(action)]:
            raise ValueError('Goalie action is masked')
        manager = self.model.goalie
        before_commands = getattr(manager, 'commands_used', 0)
        self.decisions += 1
        self.total_decisions += 1
        reward, elapsed = 0.0, 0
        # Animate committed saves without charging forced ticks as policy choices.
        while True:
            if self.mode == 'learned':
                manager.command = int(action) if elapsed < self.interval else 0
            buttons = self.model.predict_frame(self.view, 4)[0]
            effective = bool(self.view.team1.goalie.selection_flags & 8
                             and not self.view.team1.goalie.live_state_flags & 4
                             and self.view.team1.defense_control == self.view.team1.defense_goalie)
            self.manual_frames += effective
            *_, self.info = self.emulator.step(buttons)
            if self.side == 2:
                self.info = restore_away_control(self.emulator.data, self.info, controller_prefix='bench',
                                                player_prefix='cpu', slots=6)
            self.info = pass_geometry_info(self.emulator, self.info)
            self.view = cpu_view(self.state, self.info, self.side)
            self.frames += 1
            self.total_native_frames += 1
            elapsed += 1
            self.outcome, part = self._ending()
            reward += self.gamma ** ((elapsed - 1) / self.interval) * part
            self.finished = self.outcome is not None
            if self.finished or elapsed % self.interval == 0 and decision_available(self.view, manager):
                break
        used = getattr(manager, 'commands_used', 0) > before_commands
        if self.mode == 'learned' and not used:
            raise RuntimeError('An available goalie choice did not reach positioning control')
        self.commands_used += used
        self.total_commands_used += used
        info = {'actor_active': True, 'command_used': used, 'elapsed_steps': elapsed / self.interval,
                'native_frames': elapsed, 'outcome': self.outcome}
        if self.finished:
            info['result'] = self.result()
        return observation(self.view, manager), float(reward), self.finished, False, info

    def result(self):
        return {'start': self.start['path'], 'source': self.start['source'], 'side': self.side,
                'matchup': self.start['matchup'], 'seed': self.start['seed'], 'outcome': self.outcome,
                'goal': int(self.outcome == 'goal'), 'frames': self.frames,
                'decisions': self.decisions, 'commands_used': self.commands_used,
                'effective_manual_frames': self.manual_frames,
                'opponent_one_timers': self.view.team2.one_timer_attempts - self.initial_one_timers,
                'goalie_metrics': dict(Counter(self.model.goalie.metrics) - self.initial_metrics)}

    def exposure(self):
        return {'native_frames': self.total_native_frames, 'decisions': self.total_decisions,
                'commands_used': self.total_commands_used}

    def render(self):
        return self.emulator.get_screen() if self.emulator is not None else None

    def close(self):
        if self.emulator is not None:
            self.emulator.close()
            self.emulator = None
