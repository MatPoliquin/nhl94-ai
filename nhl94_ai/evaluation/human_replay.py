"""Verify recorded native inputs without running either player's decision code."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from nhl94_ai.env.factory import make_retro
from nhl94_ai.evaluation.cpu_benchmark import CPU_RAM
from nhl94_ai.evaluation.external import BUTTONS, MatchViewer
from nhl94_ai.evaluation.human_recording import (
    GAME, SCHEMA, NATIVE_SCHEMA, ram_hash, runtime_provenance, scoreboard, sha256,
)


def artifact_path(directory, name):
    path = (directory / name).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError('Replay artifact must be inside its recording directory')
    return path


def replay(path, *, start_frame=0, watch=False):
    if isinstance(start_frame, bool) or not isinstance(start_frame, int) or start_frame < 0:
        raise ValueError('start_frame must be a nonnegative integer')
    path = Path(path).expanduser().resolve()
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        header = json.loads(next(stream))
        if (header.get('schema') not in (SCHEMA, NATIVE_SCHEMA) or header.get('type') != 'header'
                or header.get('game') != GAME or header.get('players') != 2
                or header.get('buttons') != list(BUTTONS)):
            raise ValueError('Unsupported human replay schema or controller layout')
        if header['runtime'] != runtime_provenance():
            raise ValueError('Replay requires the recorded stable-retro native extension and emulator core')
        snapshot = header['initial_snapshot']
        if start_frame:
            snapshots = json.loads((path.parent / 'snapshots.json').read_text(encoding='utf-8'))
            snapshot = max((row for row in snapshots if row['frame'] <= start_frame),
                           key=lambda row: row['frame'])
        native_state = gzip.decompress(artifact_path(path.parent, snapshot['file']).read_bytes())
        if sha256(native_state) != snapshot['state_sha256']:
            raise ValueError('Snapshot checksum mismatch')
        env = make_retro(game=GAME, num_players=2, goalie_policy='selective')
        viewer = None
        try:
            if sha256(Path(env.pass_geometry_rom).read_bytes()) != header['rom_sha256']:
                raise ValueError('Replay ROM does not match the recording')
            if tuple(env.buttons) != BUTTONS:
                raise ValueError('Replay native button order does not match')
            for name, (address, kind) in CPU_RAM.items():
                env.data.set_variable(name, {'address': address, 'type': kind})
            env.reset()
            # reset() advances a native frame. Restore AFTER reset to avoid
            # introducing that extra frame into the recorded timeline.
            if env.em.set_state(native_state) is False:
                raise RuntimeError('The emulator rejected the recorded snapshot')
            env.data.reset()
            env.data.update_ram()
            if ram_hash(env) != snapshot['ram_sha256']:
                raise RuntimeError('Snapshot restore RAM does not match the recording')
            frames, digest, footer = 0, hashlib.sha256(), None
            info = env.data.lookup_all()
            for line in stream:
                row = json.loads(line)
                if footer is not None:
                    raise ValueError('Unexpected data after replay footer')
                if row.get('type') == 'footer':
                    footer = row
                    continue
                buttons = np.asarray(row.get('buttons'))
                number = row.get('frame')
                valid_number = isinstance(number, int) and not isinstance(number, bool) and number == frames + 1
                if (row.get('type') != 'frame' or not valid_number or buttons.shape != (24,)
                        or not np.isin(buttons, (0, 1)).all()):
                    raise ValueError(f'Invalid input sequence at frame {frames + 1}')
                buttons = buttons.astype(np.int8)
                digest.update(buttons.tobytes())
                frames += 1
                if frames <= snapshot['frame']:
                    if frames == snapshot['frame'] and row['ram_sha256'] != ram_hash(env):
                        raise RuntimeError('Seek snapshot does not match its recorded frame')
                    continue
                if watch and frames > start_frame and viewer is None:
                    env._update_obs()
                    human_side = header.get('human_side')
                    viewer = MatchViewer(env, human_side or header['classic_side'], header['settings']['seed'], info,
                                         opponent_name='Recorded Classic' if human_side else
                                         (header.get('opponent_metadata') or {}).get('name', 'Recorded opponent'),
                                         local_name='Recorded human' if human_side else 'Recorded Classic')
                if viewer is not None and not viewer.before_frame():
                    return {'verified': False, 'user_quit': True, 'frames': frames - 1}
                _, _, _, _, info = env.step(buttons)
                if ram_hash(env) != row['ram_sha256']:
                    raise RuntimeError(f'Replay diverged at native frame {frames}')
                actual = {**scoreboard(info), 'period': info['period'], 'clock': info['bench_clock']}
                if actual != row['after']:
                    raise RuntimeError(f'Replay scoreboard diverged at native frame {frames}')
                if viewer is not None:
                    viewer.show(env, info)
            if footer is None:
                raise ValueError('Missing replay footer: the recording is incomplete or damaged')
            if start_frame > frames or snapshot['frame'] > frames:
                raise ValueError('start_frame is beyond the recording')
            if (footer['frames'] != frames or footer['actions_sha256'] != digest.hexdigest()
                    or footer['final_ram_sha256'] != ram_hash(env)):
                raise RuntimeError('Replay footer/input checksum does not match')
            return {'verified': True, 'frames': frames, 'verified_from_frame': snapshot['frame'],
                    'completed': footer['completed'], 'actions_sha256': digest.hexdigest(),
                    'final_ram_sha256': ram_hash(env), **scoreboard(info)}
        finally:
            try:
                if viewer is not None:
                    viewer.close()
            finally:
                env.close()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('replay', help='Recorded period inputs.jsonl.gz path')
    parser.add_argument('--start-frame', type=int, default=0, help='Seek via nearest earlier snapshot')
    parser.add_argument('--watch', action='store_true', help='Display the recorded match at normal speed')
    return parser


def run(args):
    result = replay(args.replay, start_frame=args.start_frame, watch=args.watch)
    print(json.dumps(result, indent=2), flush=True)
    return result
