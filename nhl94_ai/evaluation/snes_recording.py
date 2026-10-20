"""Native SNES inputs, decoded observations and snapshots for mechanics review.

Frame N records the decoded observations before input N and the RAM hash after
input N, matching the Genesis native recorder's timeline. Snapshots contain
emulator state only, not the Classic controllers' Python state.
"""
from collections import Counter
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from nhl94_ai.agents.classic_tuning import DEFAULT
from nhl94_ai.evaluation import human_recording
from nhl94_ai.evaluation.human_recording import (
    json_value, ram_hash, sha256, source_provenance, write_json,
)
from nhl94_ai.game.platforms import get_backend

GAME = 'NHL94-Snes-v0'
HUMAN_SCHEMA = 'nhl94-snes-human-inputs-v1'
NATIVE_SCHEMA = 'nhl94-snes-native-inputs-v1'
CORE = 'snes9x_libretro.so'
BACKEND = get_backend('snes')
SIDES = ('home', 'away')


def runtime_provenance():
    return human_recording.runtime_provenance(CORE)


def rom_sha256():
    import stable_retro
    from stable_retro.data import get_romfile_path
    return sha256(Path(get_romfile_path(GAME, stable_retro.data.Integrations.ALL)).read_bytes())


def scoreboard(info):
    return {'goals': [int(info[f'p{i}_score']) for i in (1, 2)],
            'shots': [int(info[f'p{i}_shots']) for i in (1, 2)]}


def classic_diagnostics(classic):
    if classic is None:
        return None
    return {'decision': classic._last_decision, 'target': classic._last_target,
            'input_owner': classic.input_owner,
            'scheduler_decisions': classic.scheduler.decisions,
            'offense': classic.offense_diagnostics,
            'defense': classic.defense_diagnostics,
            'goalie': classic.goalie_diagnostics}


class SnesRecording:
    """Write incrementally so an early quit retains a useful incomplete period."""

    def __init__(self, directory, settings, *, snapshot_interval=600):
        if snapshot_interval < 1:
            raise ValueError('snapshot_interval must be positive')
        if settings.get('side') not in SIDES:
            raise ValueError("SNES recordings require side 'home' or 'away'")
        self.path = Path(directory)
        self.path.mkdir(parents=True, exist_ok=False)
        self.settings = settings
        self.human = bool(settings.get('human', False))
        self.human_side = (1 if settings['side'] == 'home' else 2) if self.human else None
        self.controllers = {side: 'classic' for side in SIDES}
        if self.human:
            self.controllers[settings['side']] = 'human'
        self.snapshot_interval = snapshot_interval
        self.frames = 0
        self.stream = None
        self.snapshots = []
        self.events = []
        self.decisions = {side: Counter() for side in SIDES}
        self.actions = hashlib.sha256()
        self.last_info = {}
        self.final_ram = None
        self.last_owner_side = None
        self.loose_frames = 0

    def snapshot(self, env):
        state = env.em.get_state()
        name = f'frame-{self.frames:07d}.state.gz'
        (self.path / name).write_bytes(gzip.compress(state, mtime=0))
        entry = {'frame': self.frames, 'file': name, 'state_sha256': sha256(state),
                 'ram_sha256': ram_hash(env)}
        self.snapshots.append(entry)
        write_json(self.path / 'snapshots.json', self.snapshots)
        return entry

    def start(self, env, info):
        self.last_info = dict(info)
        self.final_ram = ram_hash(env)
        initial = self.snapshot(env)
        self.stream = gzip.open(self.path / 'inputs.jsonl.gz', 'wt', encoding='utf-8', compresslevel=1)
        header = {'type': 'header',
                  'schema': HUMAN_SCHEMA if self.human else NATIVE_SCHEMA, 'game': GAME,
                  'platform': BACKEND.name, 'buttons': list(env.buttons), 'players': 2,
                  'score_order': ['home', 'away'], 'human_side': self.human_side,
                  'controllers': self.controllers,
                  'controller_assignment': {key: info.get(key) for key in (
                      'defense_team1', 'defense_team2', 'defense_control1', 'defense_control2',
                      'p1_control_slot', 'p2_control_slot')},
                  'action_schema': self.settings.get('action_type', 'FILTERED'),
                  'goalie_policy': self.settings.get('goalie_policy', 'selective'),
                  'decision_cadence': {'frame_skip': self.settings.get('frame_skip', 4),
                                       'classic_control_interval':
                                           self.settings.get('classic_control_interval', 0)},
                  'settings': self.settings,
                  'classic_tuning': self.settings.get('params', asdict(DEFAULT)),
                  'runtime': runtime_provenance(), 'sources': source_provenance(),
                  'rom_sha256': rom_sha256(),
                  'initial_state_sha256': sha256(env.initial_state),
                  'initial_snapshot': initial, 'initial_scoreboard': scoreboard(info)}
        self._write(header)
        self.stream.flush()

    def _write(self, row):
        self.stream.write(json.dumps(json_value(row), separators=(',', ':'), allow_nan=False) + '\n')

    def before_step(self, env, info, classics):
        """Snapshot the decoded native frame and each side's Classic diagnostics."""
        self.pending = json_value({
            'observations': info,
            'classic': {side: classic_diagnostics(classics.get(side)) for side in SIDES},
        })

    def after_step(self, env, buttons, info):
        buttons = np.asarray(buttons, dtype=np.int8)
        self.frames += 1
        self.actions.update(buttons.tobytes())
        self.final_ram = ram_hash(env)
        previous, current = scoreboard(self.last_info), scoreboard(info)
        for field, event in (('goals', 'goal'), ('shots', 'shot')):
            for index, (before, after) in enumerate(zip(previous[field], current[field])):
                if after > before:
                    self.events.append({'frame': self.frames, 'type': event,
                                        'side': index + 1, 'count': after - before})
        owner = info.get('puck_owner', -1)
        if 0 <= owner < 5 or 6 <= owner < 11:
            side = 1 if owner < 5 else 2
            if self.last_owner_side not in (None, side) and self.loose_frames <= 90:
                self.events.append({'frame': self.frames, 'type': 'possession_change',
                                    'from_side': self.last_owner_side, 'to_side': side})
            self.last_owner_side, self.loose_frames = side, 0
        elif owner in (5, 11):
            self.last_owner_side, self.loose_frames = None, 0
        else:
            self.loose_frames += 1
        for side in SIDES:
            diagnostics = self.pending['classic'][side]
            if diagnostics is not None:
                self.decisions[side][diagnostics['decision']] += 1
        self._write({'type': 'frame', 'frame': self.frames, 'buttons': buttons.tolist(),
                     **self.pending, 'after': {**current, 'period': info['period'],
                                               'clock': info['time']},
                     'ram_sha256': self.final_ram})
        self.last_info = dict(info)
        if self.frames % self.snapshot_interval == 0:
            self.snapshot(env)
        if self.frames % 60 == 0:
            self.stream.flush()

    def finish(self, result=None, *, error=None):
        result = dict(result or {}, frames=self.frames,
                      actions_sha256=self.actions.hexdigest(), final_ram_sha256=self.final_ram)
        if error is not None:
            result.update(completed=False, error=str(error))
        result.setdefault('completed', False)
        if self.last_info:
            result.update(scoreboard(self.last_info))
        result['human_side'] = self.human_side
        result['controllers'] = self.controllers
        result['decision_frames'] = {side: dict(self.decisions[side]) for side in SIDES}
        result['events'] = self.events
        try:
            if self.stream is not None:
                self._write({'type': 'footer', 'frames': self.frames,
                             'actions_sha256': self.actions.hexdigest(),
                             'final_ram_sha256': self.final_ram, 'completed': result['completed']})
        finally:
            self.close()
        write_json(self.path / 'results.json', result)
        return result

    def close(self):
        if self.stream is not None:
            self.stream.close()
            self.stream = None
