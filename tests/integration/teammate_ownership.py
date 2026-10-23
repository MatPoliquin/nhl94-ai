"""Retained native goalie snapshot capture/probes and faceoff input ownership.

Capture using the preserved original source on sys.path, then probe the same
snapshot with the candidate. No gameplay RAM is rewritten. Snapshot fixtures
contain private native state and belong in an ignored caller-selected directory.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from nhl94_ai.agents.teammates import TeammateCoordinator
from nhl94_ai.evaluation.teammates import DEFAULT_STATE, make_match_env, prepare_match, source_hash
from nhl94_ai.evaluation.teammate_diagnostics import TeammateDiagnostics
from nhl94_ai.game.ram import teammate_controller_info
from tests.support.teammates import team_args


def capture(output, side='home', size=2, seed=94):
    output.mkdir(parents=True, exist_ok=False)
    with make_match_env(DEFAULT_STATE, team_size=size) as env:
        info = prepare_match(env, seed=seed, seconds=300, side=1 if side == 'home' else 2, team_size=size)
        agent = TeammateCoordinator(team_args(side=side, team_size=size))
        for frame in range(18000):
            info = agent.controllers[0].platform.frame_info(env, info)
            view = teammate_controller_info(info, side=1 if side == 'home' else 2, team_size=size)
            goalie = 5 if side == 'home' else 11
            slots = [view[f'defense_control{i+1}'] for i in range(size)]
            if info['puck_owner'] == goalie and goalie in slots and not info['goalie_clock_flags'] & 0x11:
                snapshot = env.em.get_state()
                (output / 'snapshot.state.gz').write_bytes(gzip.compress(snapshot))
                meta = {'side': side, 'size': size, 'seed': seed, 'frame': frame,
                        'slots': slots, 'source_sha256': source_hash(),
                        'ram_sha256': hashlib.sha256(env.get_ram().tobytes()).hexdigest(),
                        'rom_sha256': hashlib.sha256(Path(env.pass_geometry_rom).read_bytes()).hexdigest(),
                        'snapshot_sha256': hashlib.sha256(snapshot).hexdigest()}
                (output / 'capture.json').write_text(json.dumps(meta, indent=2) + '\n')
                return meta
            *_, info = env.step(agent.act(info).action)
    raise AssertionError('No selected goalie within 18000 native frames')


def probe(directory):
    meta = json.loads((directory / 'capture.json').read_text())
    snapshot = gzip.decompress((directory / 'snapshot.state.gz').read_bytes())
    assert hashlib.sha256(snapshot).hexdigest() == meta['snapshot_sha256']
    side, size = meta['side'], meta['size']
    goalie = 5 if side == 'home' else 11
    owner_index = meta['slots'].index(goalie)
    outcomes = {}
    with make_match_env(DEFAULT_STATE, team_size=size) as env:
        prepare_match(env, seed=meta['seed'], seconds=300, side=1 if side == 'home' else 2, team_size=size)
        assert hashlib.sha256(Path(env.pass_geometry_rom).read_bytes()).hexdigest() == meta['rom_sha256']
        for variant in ('neutral', 'owner-b', 'other-b', 'owner-pulses', 'candidate'):
            assert env.em.set_state(snapshot)
            env.data.update_ram()
            assert hashlib.sha256(env.get_ram().tobytes()).hexdigest() == meta['ram_sha256']
            agent = TeammateCoordinator(team_args(side=side, team_size=size))
            metrics, rows = TeammateDiagnostics(), []
            for frame in range(1, 181):
                info = agent.controllers[0].platform.frame_info(env, env.data.lookup_all())
                result = agent.act(info)
                action = np.zeros(12*size, dtype=np.int8)
                if variant == 'candidate':
                    action = result.action
                elif frame <= 4 and variant in ('owner-b', 'other-b'):
                    index = owner_index if variant == 'owner-b' else (owner_index + 1) % size
                    action[index*12] = 1
                elif variant == 'owner-pulses' and (frame - 1) % 8 < 4:
                    action[owner_index*12] = 1
                diagnostics = result.diagnostics['teammates']
                metrics.observe(frame, agent.states[0], diagnostics, action)
                rows.append({'frame': frame, 'owner': info['puck_owner'], 'buttons': action.tolist(),
                             'roles': [row['role'] for row in diagnostics['controllers']],
                             'requests': [c.pass_intention() for c in agent.controllers],
                             'passes': agent.states[0].team1.pass_attempts})
                *_, info = env.step(action)
            outcomes[variant] = {'first_owner_exit': next((r['frame'] for r in rows if r['owner'] != goalie), None),
                                  'measurements': metrics.summary(), 'rows': rows}
    report = {'capture': meta, 'candidate_source_sha256': source_hash(), 'outcomes': outcomes}
    (directory / 'probe.json').write_text(json.dumps(report, indent=2) + '\n')
    # Some captures are still in the catch animation: retain the initial pulse
    # result separately and test readiness with bounded release/press pulses.
    owner_exit = outcomes['owner-pulses']['first_owner_exit']
    assert owner_exit is not None, 'Goalie did not accept repeated owner input within 180 frames'
    for variant in ('neutral', 'other-b'):
        assert outcomes[variant]['first_owner_exit'] is None or outcomes[variant]['first_owner_exit'] > owner_exit
    for row in outcomes['candidate']['rows']:
        for index, request in enumerate(row['requests']):
            if request is not None and request['passer'] == goalie and row['owner'] == goalie:
                assert meta['slots'][index] == goalie
    return {key: value['first_owner_exit'] for key, value in outcomes.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('capture', 'probe'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--side', choices=('home', 'away'), default='home')
    parser.add_argument('--size', type=int, choices=(2, 3), default=2)
    parser.add_argument('--seed', type=int, default=94)
    args = parser.parse_args()
    print(capture(args.output, args.side, args.size, args.seed) if args.command == 'capture' else probe(args.output))


if __name__ == '__main__':
    main()
