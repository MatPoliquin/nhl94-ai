"""Portable native inputs, observations and snapshots for human/Classic review.

Frame N records the observations before input N and the RAM hash after input N.
Snapshots contain emulator state only, not the Classic controller's Python state.
"""
from collections import Counter
from dataclasses import asdict
import gzip
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np

from nhl94_ai.agents.classic_tuning import DEFAULT

SCHEMA = 'nhl94-human-inputs-v1'
NATIVE_SCHEMA = 'nhl94-native-inputs-v1'
GAME = 'NHL94-Genesis-v0'


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def ram_hash(env):
    return sha256(env.get_ram().tobytes())


def json_value(value):
    """Normalize NumPy diagnostics and unavailable (non-finite) estimates."""
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    """Replace complete reports atomically, keeping the last valid report on error."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(json_value(value), indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def runtime_provenance():
    import stable_retro
    import stable_retro._retro as native
    core = Path(stable_retro.core_path()) / 'genesis_plus_gx_libretro.so'
    return {'stable_retro_version': version('stable-retro'),
            'native_sha256': sha256(Path(native.__file__).read_bytes()),
            'core_sha256': sha256(core.read_bytes())}


def source_provenance():
    root = Path(__file__).resolve().parents[1]
    return {str(path.relative_to(root)): sha256(path.read_bytes())
            for path in sorted(root.rglob('*.py'))}


def scoreboard(info):
    return {'goals': [int(info[f'p{i}_score']) for i in (1, 2)],
            'shots': [int(info[f'bench_shots{i}']) for i in (1, 2)]}


class HumanRecording:
    """Write incrementally so an early quit retains a useful incomplete period."""

    def __init__(self, directory, settings, *, snapshot_interval=600):
        if snapshot_interval < 1:
            raise ValueError('snapshot_interval must be positive')
        self.path = Path(directory)
        self.path.mkdir(parents=True, exist_ok=False)
        self.settings = settings
        self.human = settings.get('human', True)
        self.classic_side = 3 - settings['side'] if self.human else settings['side']
        self.snapshot_interval = snapshot_interval
        self.frames = 0
        self.stream = None
        self.snapshots = []
        self.events = []
        self.decisions = Counter()
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

    def start(self, env, info, *, opponent_metadata=None):
        self.last_info = dict(info)
        self.final_ram = ram_hash(env)
        initial = self.snapshot(env)
        self.stream = gzip.open(self.path / 'inputs.jsonl.gz', 'wt', encoding='utf-8', compresslevel=1)
        header = {'type': 'header', 'schema': SCHEMA if self.human else NATIVE_SCHEMA, 'game': GAME,
                  'buttons': list(env.buttons), 'players': 2, 'score_order': ['home', 'away'],
                  'human_side': self.settings['side'] if self.human else None,
                  'classic_side': self.classic_side, 'opponent_metadata': opponent_metadata,
                  'settings': self.settings, 'classic_tuning': self.settings.get('params', asdict(DEFAULT)),
                  'runtime': runtime_provenance(), 'sources': source_provenance(),
                  'rom_sha256': sha256(Path(env.pass_geometry_rom).read_bytes()),
                  'initial_snapshot': initial, 'initial_scoreboard': scoreboard(info)}
        self._write(header)
        self.stream.flush()

    def _write(self, row):
        self.stream.write(json.dumps(json_value(row), separators=(',', ':'), allow_nan=False) + '\n')

    def before_step(self, env, info, classic):
        from nhl94_ai.game.ram import pass_geometry_info
        self.pending = json_value({
            'observations': pass_geometry_info(env, info),
            'classic': {'decision': classic._last_decision, 'target': classic._last_target,
                        'input_owner': classic.input_owner,
                        'scheduler_decisions': classic.scheduler.decisions,
                        'offense': classic.offense_diagnostics,
                        'defense': classic.defense_diagnostics,
                        'goalie': classic.goalie_diagnostics},
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
        if info.get('goalie_clock_flags', 0) & 0x11 or info.get('sflags', 0) & 0x0100:
            self.last_owner_side, self.loose_frames = None, 0
        elif 0 <= owner < 5 or 6 <= owner < 11:
            side = 1 if owner < 5 else 2
            if self.last_owner_side not in (None, side) and self.loose_frames <= 90:
                self.events.append({'frame': self.frames, 'type': 'possession_change',
                                    'from_side': self.last_owner_side, 'to_side': side})
            self.last_owner_side, self.loose_frames = side, 0
        elif owner in (5, 11):
            self.last_owner_side, self.loose_frames = None, 0
        else:
            self.loose_frames += 1
        self.decisions[self.pending['classic']['decision']] += 1
        self._write({'type': 'frame', 'frame': self.frames, 'buttons': buttons.tolist(),
                     **self.pending, 'after': {**current, 'period': info['period'],
                                              'clock': info['bench_clock']},
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
        result['human_side'] = self.settings['side'] if self.human else None
        result['classic_side'] = self.classic_side
        result['decision_frames'] = dict(self.decisions)
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
