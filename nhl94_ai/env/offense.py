"""Full-period hybrid offense environment, including frozen-opponent self-play.

One gym step is exactly four native frames unless the episode ends. Native
execution and Classic defense continue between policy decisions. Both sides
have separate controllers and decision-time histories.
"""
from collections import deque
import copy
import gzip
import hashlib
import json
from pathlib import Path
import pickle

import gymnasium as gym
import numpy as np

from nhl94_ai.agents.learned_offense import (
    N_ACTIONS, OBSERVATION_FIELDS, PossessionController, encode_offense, SCHEMA, attack_sign,
)
from nhl94_ai.evaluation.benchmark import away_view, update_state
from nhl94_ai.evaluation.cpu_benchmark import CPU_RAM, cpu_view, select_side
from nhl94_ai.game.ram import pass_geometry_info, restore_away_control
from nhl94_ai.game.state import NHL94GameState


def score_reward(previous, current, side):
    change = (current[0] - previous[0]) - (current[1] - previous[1])
    return float(change if side == 1 else -change)


def reset_options(args, episode, opponent=None):
    """Pair both team assignments under the same source seed and roster."""
    result = dict(rom_seed=args.seed + episode // 2,
                  side=args.side or 1 + episode % 2,
                  state=args.states[(episode // 2) % len(args.states)])
    if opponent is not None:
        result['opponent'] = opponent
    return result


def two_player_state(name):
    import stable_retro.data as data
    states = data.list_states('NHL94-Genesis-v0')
    base = name.removesuffix('.state')
    candidates = (base, base + '.2P', base.replace('.Start', '.2P.Start'))
    matches = {state.lower(): state for state in states if '.2p' in state.lower()}
    for candidate in candidates:
        if candidate.lower() in matches:
            return matches[candidate.lower()]
    raise ValueError(f'No two-controller save for {name}; choose a .2P state for self-play')


class OffenseEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array']}

    def __init__(self, *, states=('MightyDucksVsAllStarCampbell.ManualGoalie.Start',), opponents=('cpu',),
                 learner='learned', seconds=300, goalie_policy='selective',
                 side=None, max_frames=None, starts=(), start_probability=0.0,
                 task='period', attack_frames=600, control_scope='offense', gamma=0.997):
        super().__init__()
        if learner not in ('learned', 'teacher', 'classic') or side not in (None, 1, 2):
            raise ValueError('Invalid learner mode or side')
        if not states or not opponents or not 1 <= seconds <= 1200:
            raise ValueError('Provide states, opponents, and 1..1200 game-clock seconds')
        if not 0 <= start_probability <= 1 or start_probability and not starts:
            raise ValueError('Start probability needs a nonempty snapshot bank and must be in [0, 1]')
        if max_frames is not None and max_frames < 4:
            raise ValueError('max_frames must be at least four')
        if task not in ('period', 'attack', 'opportunity') or attack_frames < 4:
            raise ValueError('Task must be period, attack or opportunity, with at least four attack frames')
        if task != 'period' and not starts:
            raise ValueError('Possession tasks require a snapshot bank')
        if control_scope not in ('offense', 'carry') or not 0 < gamma < 1:
            raise ValueError('Invalid control scope or discount')
        if task == 'opportunity' and control_scope != 'carry':
            raise ValueError('The opportunity task requires carry control')
        self.control_scope, self.gamma, self.opportunity_reward = control_scope, gamma, None
        self.task, self.attack_frames, self.attack_start = task, attack_frames, None
        self.states, self.pool = tuple(states), tuple(opponents)
        self.learner, self.seconds, self.goalie_policy = learner, seconds, goalie_policy
        self.fixed_side, self.max_frames = side, max_frames
        self.starts, self.start_probability = tuple(starts), start_probability
        self.start_metadata = {path: json.loads(Path(str(path) + '.json').read_text(encoding="utf-8")) for path in self.starts}
        self.action_space = gym.spaces.Discrete(N_ACTIONS)
        self.observation_space = gym.spaces.Box(-1, 1, (SCHEMA['history'], len(OBSERVATION_FIELDS)), np.float32)
        self.emulator = None
        self.state_name = None
        self.opponent_model = None
        self._model_path = None
        self.finished = True

    def set_pool(self, opponents):
        """Only reset selects an opponent: updates never change a live match."""
        if not opponents:
            raise ValueError('Opponent pool cannot be empty')
        self.pool = tuple(opponents)

    def _create_emulator(self, name, cpu):
        from nhl94_ai.env.factory import make_retro
        actual = name if cpu else two_player_state(name)
        if self.emulator is not None and getattr(self, '_emulator_key', (None, None))[1] == cpu:
            if self.state_name != name:
                from stable_retro.data import Integrations
                self.emulator.load_state(actual, Integrations.ALL)
                self.state_name = name
                self._emulator_key = (name, cpu)
            return
        if self.emulator is not None:
            self.emulator.close()
        self.emulator = make_retro(game='NHL94-Genesis-v0', state=actual,
                                   num_players=1 if cpu else 2, goalie_policy=self.goalie_policy)
        for key, (address, dtype) in CPU_RAM.items():
            self.emulator.data.set_variable(key, {'address': address, 'type': dtype})
        self.state_name = name
        self._emulator_key = (name, cpu)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = dict(options or {})
        self.attack_start = None
        if self.task != 'period' and 'snapshot' not in options:
            compatible = [path for path, meta in self.start_metadata.items()
                          if meta['opponent'] in self.pool and meta['state_name'] in self.states
                          and (self.fixed_side is None or meta['side'] == self.fixed_side)
                          and all(meta[key] == options[option] for key, option in
                                  (('side', 'side'), ('state_name', 'state'), ('opponent', 'opponent'))
                                  if option in options)]
            if not compatible:
                raise ValueError('No compatible offensive snapshots')
            # Equal source-game probability, independent of snapshot count.
            sources = sorted({(self.start_metadata[p]['state_name'], self.start_metadata[p]['source_seed'])
                              for p in compatible})
            source = sources[int(self.np_random.integers(len(sources)))]
            candidates = [p for p in compatible if
                          (self.start_metadata[p]['state_name'], self.start_metadata[p]['source_seed']) == source]
            options['snapshot'] = candidates[int(self.np_random.integers(len(candidates)))]
        if self.task != 'period':
            meta = self.start_metadata.get(options['snapshot'])
            if meta is None:
                meta = json.loads(Path(str(options['snapshot']) + '.json').read_text(encoding='utf-8'))
            options.update(state=meta['state_name'], side=meta['side'], opponent=meta['opponent'])
        self.episode_seed = int(options.get('rom_seed', self.np_random.integers(0, 2**32)))
        self.side = int(options.get('side', self.fixed_side or self.np_random.integers(1, 3)))
        self.opponent = str(options.get('opponent', self.np_random.choice(self.pool)))
        cpu = self.opponent == 'cpu'
        name = str(options.get('state', self.np_random.choice(self.states)))
        start_file = options.get('snapshot')
        if start_file is None and self.starts and self.np_random.random() < self.start_probability:
            compatible = [path for path, meta in self.start_metadata.items()
                          if meta['opponent'] == self.opponent and meta['state_name'] == name
                          and meta['side'] == self.side]
            if compatible:
                start_file = str(self.np_random.choice(compatible))
        if start_file:
            meta = json.loads(Path(str(start_file) + '.json').read_text(encoding="utf-8"))
            if (meta['opponent'], meta['state_name'], meta['side']) != (self.opponent, name, self.side):
                raise ValueError('Snapshot opponent, state and side must match the episode')
        self._create_emulator(name, cpu)
        self.emulator.reset(seed=self.episode_seed)
        self.emulator.data.update_ram()
        initial = self.emulator.data.lookup_all()
        if cpu:
            select_side(self.emulator.data, initial, self.side)
        elif (initial['bench_team1'], initial['bench_team2']) != (1, 2):
            raise ValueError('Self-play requires controllers assigned to home and away')
        self.emulator.data.set_value('bench_rng', self.episode_seed)
        self.emulator.data.set_value('bench_clock', self.seconds)
        *_, self.info = self.emulator.step(np.zeros(12 if cpu else 24, dtype=np.int8))
        self.game_state = NHL94GameState(5)
        self.controllers = [PossessionController(mode='classic', goalie_policy=self.goalie_policy) for _ in range(2)]
        self.controllers[self.side - 1].mode = self.learner
        self.controllers[self.side - 1].control_scope = self.control_scope
        if self.opponent not in ('cpu', 'classic'):
            from nhl94_ai.models.offense import load_offense_model
            path = str(Path(self.opponent).expanduser().resolve())
            if self._model_path != path:
                self.opponent_model = load_offense_model(path, expected_task=self.task)
                self._model_path = path
            self.controllers[2 - self.side].mode = 'learned'
            self.controllers[2 - self.side].control_scope = self.opponent_model.offense_control_scope
        self.histories = [deque(maxlen=SCHEMA['history']) for _ in range(2)]
        self.frames = 1
        self.start_score = self._scores()
        self.previous_score = self.start_score
        self.digest = hashlib.sha256()
        self.choices = np.zeros(N_ACTIONS, dtype=np.int64)
        self.finished = False
        self._refresh()
        self._append_history(initial=True)
        if start_file:
            self._restore_start(start_file)
        self.snapshot = str(start_file) if start_file else None
        if self.task != 'period':
            if not self.controllers[self.side - 1].available(self.views[self.side - 1]):
                raise ValueError('Attack snapshot must start at an offensive decision')
            self.attack_start = self.frames
            # Show the finite task horizon in the existing clock feature. Old
            # history remains physical history; the newest sample starts task time.
            for history in self.histories:
                history.pop()
            self._append_history()
        self.opportunity_reward = None
        if self.task == 'opportunity':
            from nhl94_ai.env.opportunity import OpportunityReward
            self.opportunity_reward = OpportunityReward(
                self.views[self.side - 1], self.controllers[self.side - 1], self.info, self.side, self.gamma)
        return self._observation(), self._info()

    def _scores(self):
        return int(self.info['p1_score']), int(self.info['p2_score'])

    def _refresh(self):
        info = pass_geometry_info(self.emulator, self.info)
        if self.opponent == 'cpu':
            if self.side == 2:
                info = restore_away_control(self.emulator.data, info,
                                           controller_prefix='bench', player_prefix='cpu')
            view = cpu_view(self.game_state, info, self.side)
            self.views = [view, away_view(view)] if self.side == 1 else [away_view(view), view]
        else:
            update_state(self.game_state, info)
            self.views = [self.game_state, away_view(self.game_state)]
        self.info = info

    def _append_history(self, *, initial=False):
        scores = self._scores()
        clock = (max(0, self.attack_frames - (self.frames - self.attack_start)) / 60
                 if self.attack_start is not None else self.info['bench_clock'])
        for i in range(2):
            features = encode_offense(self.views[i], self.controllers[i], clock=clock,
                                      score_difference=scores[i] - scores[1 - i])
            if initial:
                self.histories[i].extend(features.copy() for _ in range(SCHEMA['history']))
            else:
                self.histories[i].append(features)

    def _observation(self):
        return np.stack(self.histories[self.side - 1])

    def _info(self):
        scores = self._scores()
        return dict(side=self.side, opponent=self.opponent, state=self.state_name, seed=self.episode_seed,
                    frames=self.frames, clock=int(self.info['bench_clock']),
                    goals_for=scores[self.side - 1] - self.start_score[self.side - 1],
                    goals_against=scores[2 - self.side] - self.start_score[2 - self.side],
                    shots=[int(self.info[f'bench_shots{i}']) for i in (1, 2)],
                    one_timers=[int(self.info[f'bench_one_timers{i}']) for i in (1, 2)])

    def _attack_outcome(self):
        """A pass or loose puck continues; a goal or lost attack ends the task."""
        if self.attack_start is None:
            return None
        scores = self._scores()
        if scores[self.side - 1] > self.start_score[self.side - 1]:
            return 'goal'
        if scores[2 - self.side] > self.start_score[2 - self.side]:
            return 'conceded'
        view = self.views[self.side - 1]
        owner = view.engine.puck_owner
        if view.team2.owns_scnum(owner) or owner == view.team1.goalie_scnum():
            return 'turnover'
        carrier = view.team1.get_player_by_scnum(owner)
        if view.team1.owns_scnum(owner) and carrier is not None and carrier.y * attack_sign(view) < 100:
            return 'retreat'
        if self.frames - self.attack_start >= self.attack_frames:
            return 'timeout'
        return None

    def step(self, action):
        if self.finished:
            raise RuntimeError('Reset before stepping a completed episode')
        learner = self.controllers[self.side - 1]
        active = learner.begin_step(self.views[self.side - 1], int(action))
        if self.opponent != 'cpu':
            opponent = self.controllers[2 - self.side]
            choice = 0
            if opponent.mode == 'learned' and opponent.available(self.views[2 - self.side]):
                choice, _ = self.opponent_model.predict(np.stack(self.histories[2 - self.side]), deterministic=True)
            opponent.begin_step(self.views[2 - self.side], int(choice))
        reward = 0.0
        terminated = truncated = False
        teacher_command = None
        used = False
        outcome = None
        for frame in range(SCHEMA['interval']):
            if self.opponent == 'cpu':
                buttons = learner.frame(self.views[self.side - 1])
            else:
                buttons = np.concatenate([controller.frame(view) for controller, view in zip(self.controllers, self.views)])
            if frame == 0:
                teacher_command = learner.teacher_command if active else None
            used = used or learner.command_used
            setup_reward = self.opportunity_reward.before_frame(learner) if self.task == 'opportunity' else 0.0
            self.digest.update(buttons.tobytes())
            _, _, raw_end, raw_cut, self.info = self.emulator.step(buttons)
            self.frames += 1
            scores = self._scores()
            if self.task != 'opportunity':
                reward += score_reward(self.previous_score, scores, self.side)
            self.previous_score = scores
            self._refresh()
            terminated = bool(raw_end or self.info['bench_clock'] == 0 or self.info['period'] != 0)
            if self.task == 'attack':
                outcome = self._attack_outcome() or ('period_end' if terminated else None)
                terminated = bool(outcome)
                # Binary task success; no movement, shot-quality, or pass shaping.
                reward = float(outcome == 'goal')
            elif self.task == 'opportunity':
                outcome = self._attack_outcome() or ('period_end' if terminated else None)
                view = self.views[self.side - 1]
                owner = view.engine.puck_owner
                own_skater = view.team1.owns_scnum(owner) and owner != view.team1.goalie_scnum()
                part, outcome = self.opportunity_reward.after_frame(self.info, outcome, own_skater=own_skater)
                reward += self.gamma**(frame / SCHEMA['interval']) * (setup_reward + part)
                terminated = bool(outcome)
            budget = self.max_frames or self.seconds * 120 + 6000
            truncated = bool((raw_cut or self.frames >= budget) and not terminated)
            if terminated or truncated:
                break
        self.finished = terminated or truncated
        if self.task == 'opportunity':
            self.opportunity_reward.eligible_decisions += int(active)
            self.opportunity_reward.movement_decisions += int(used)
            reward += self.opportunity_reward.shape(
                self.views[self.side - 1], learner, (frame + 1) / SCHEMA['interval'], terminated)
        self._append_history()
        info = self._info()
        info.update(actor_active=active, command_used=used, teacher_command=teacher_command,
                    native_frames=frame + 1, terminated=terminated, truncated=truncated)
        label = teacher_command if self.learner == 'teacher' else int(action) if used else None
        if label is not None:
            self.choices[label] += 1
        if self.finished:
            info.update(actions_sha256=self.digest.hexdigest(), choices=self.choices.tolist(),
                        completed=bool(self.info['bench_clock'] == 0 or self.info['period'] != 0))
            if self.task != 'period':
                info.update(outcome=outcome, success=outcome == 'goal', completed=terminated,
                            attempt_frames=self.frames - self.attack_start, snapshot=self.snapshot)
            if self.task == 'opportunity':
                info['opportunity'] = self.opportunity_reward.summary(self.info)
        return self._observation(), reward, terminated, truncated, info

    def save_start(self, path):
        """Trusted local curriculum snapshots; keep source games in one data split."""
        payload = dict(schema=SCHEMA, opponent=self.opponent, state_name=self.state_name,
                       side=self.side, source_seed=self.episode_seed,
                       emulator_state=self.emulator.em.get_state(),
                       game_state=self.game_state, controllers=self.controllers,
                       histories=self.histories, info=self.info, frames=self.frames)
        with gzip.open(path, 'wb') as stream:
            pickle.dump(payload, stream, protocol=5)
        Path(str(path) + '.json').write_text(json.dumps(dict(
            schema=SCHEMA, opponent=self.opponent, state_name=self.state_name,
            side=self.side, source_seed=self.episode_seed,
            sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest()), indent=2) + '\n', encoding='utf-8')

    def _restore_start(self, path):
        metadata = json.loads(Path(str(path) + '.json').read_text(encoding="utf-8"))
        if metadata['sha256'] != hashlib.sha256(Path(path).read_bytes()).hexdigest():
            raise ValueError('Snapshot hash mismatch')
        with gzip.open(path, 'rb') as stream:
            data = pickle.load(stream)
        if data['schema'] != SCHEMA:
            raise ValueError('Snapshot schema mismatch')
        if (data['opponent'], data['state_name'], data['side']) != (self.opponent, self.state_name, self.side):
            raise ValueError('Snapshot opponent, state and side must match the episode')
        self.emulator.em.set_state(data['emulator_state'])
        self.emulator.data.update_ram()
        self.game_state = copy.deepcopy(data['game_state'])
        self.controllers = copy.deepcopy(data['controllers'])
        for controller in self.controllers:
            controller.control_scope = 'offense'
        self.controllers[self.side - 1].mode = self.learner
        self.controllers[self.side - 1].control_scope = self.control_scope
        if self.opponent not in ('cpu', 'classic'):
            self.controllers[2 - self.side].control_scope = self.opponent_model.offense_control_scope
        self.histories = copy.deepcopy(data['histories'])
        if self.control_scope == 'carry':
            for features in self.histories[self.side - 1]:
                features[-N_ACTIONS + 9:] = 0
        self.info = self.emulator.data.lookup_all()
        self.frames = data['frames']
        self.episode_seed = data['source_seed']
        self.start_score = self.previous_score = self._scores()
        self._refresh()

    def render(self):
        return self.emulator.render()

    def close(self):
        if self.emulator is not None:
            self.emulator.close()
            self.emulator = None
