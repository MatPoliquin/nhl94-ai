"""Shared Genesis replay operations; experiment decisions stay in their runners."""
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from nhl94_ai.agents.defense import controlled_slot
from nhl94_ai.game.genesis_match import CPU_RAM, cpu_view, select_side
from nhl94_ai.game.ram import pass_geometry_info, restore_away_control


def prepare_period(env, seed, side, *, seconds=300):
    """Reset a Genesis CPU fixture and advance its neutral setup frame.

    Keep registration, reset, controller selection, ROM seeding, clock setup,
    and the first input in this order. The neutral frame counts as frame one
    in existing replay reports.
    """
    for name, (address, kind) in CPU_RAM.items():
        env.data.set_variable(name, {'address': address, 'type': kind})
    env.reset(seed=seed)
    env.data.update_ram()
    select_side(env.data, env.data.lookup_all(), side)
    env.data.set_value('bench_rng', seed)
    env.data.set_value('bench_clock', seconds)
    *_, info = env.step(np.zeros(12, dtype=np.int8))
    return info


def replay_source_hashes(*extra_files):
    """Fingerprint shared replay dependencies and extra package-relative files."""
    root = Path(__file__).resolve().parents[2]
    package = root / 'nhl94_ai'
    paths = {path for directory in ('agents', 'env', 'game', 'tasks')
             for path in (package / directory).rglob('*.py')}
    paths.update(package / 'evaluation' / name for name in (
        'possession_ablation_replay.py', 'refinement_replay.py', 'cpu_benchmark.py',
        'benchmark.py', 'carry_outcomes.py', 'carry_replay.py', 'replay_support.py'))
    paths.update(package / name for name in extra_files)
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)}


def write_replay_report(output, report):
    """Write compact JSON, with deterministic gzip for a .gz destination."""
    data = (json.dumps(report, separators=(',', ':'), allow_nan=False) + '\n').encode()
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(data, mtime=0) if path.suffix == '.gz' else data)


def read_view(env, state, side):
    env.data.update_ram()
    info = pass_geometry_info(env, env.data.lookup_all())
    return cpu_view(state, info, side), info


def advance(env, action, side):
    *_, info = env.step(action)
    if side == 2:
        restore_away_control(env.data, info, controller_prefix='bench', player_prefix='cpu', slots=6)


def restore(env, snapshot, digest):
    env.em.set_state(snapshot)
    env.data.update_ram()
    assert hashlib.sha256(env.get_ram().tobytes()).hexdigest() == digest


def observation(view):
    fields = ('x', 'y', 'precise_x', 'precise_y', 'motion_x', 'motion_y', 'facing',
              'role', 'speed', 'agility', 'weight', 'energy', 'passing', 'stick',
              'shot_power', 'shot_accuracy', 'stick_x', 'stick_y', 'live_anim')
    return {'owner': view.engine.puck_owner, 'controlled': controlled_slot(view.team1),
            'net_y': view.team2.net.y,
            'puck': {name: getattr(view.puck, name) for name in ('x', 'y', 'motion_x', 'motion_y', 'height')},
            'teams': [[{name: getattr(player, name) for name in fields}
                       for player in (*team.players, team.goalie)] for team in (view.team1, view.team2)]}
