"""Replay frozen input traces without depending on a retired scripted agent.

Run explicitly; ROMs are never bundled.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
from nhl94_ai.training.datasets import build_single_nhl94_env
from nhl94_ai.evaluation.play import parse_cmdline
from nhl94_ai.config import load_hyperparams, resolve_hyperparams_for_model

FIXTURE = ROOT / 'tests/fixtures/traces.json'
CASES = [
    ('NHL941on1-Genesis-v0', 'FILTERED'),
    ('NHL942on2-Genesis-v0', 'FILTERED'),
    ('NHL94-Genesis-v0', 'FILTERED'),
    ('NHL94-Genesis-v0', 'MULTI_DISCRETE'),
    ('NHL94-Genesis-v0', 'HOCKEY_INTENT_DPAD'),
]


def capture(game, action_type, actions):
    args = parse_cmdline(['--mode=model_vs_game', f'--env={game}',
                         '--nn=ClassicAIV1', '--rf=PostPlay', f'--action_type={action_type}'])
    params = resolve_hyperparams_for_model(load_hyperparams(str(ROOT / 'configs/training/nhl94.json')), args.nn)
    params['frame_skip'] = 4
    env = build_single_nhl94_env(args, params, use_sticky_action=False)
    try:
        obs, info = env.reset(seed=7)
        raw_info = {key: int(value) if isinstance(value, np.integer) else value for key, value in info.items()}
        rows = []
        for recorded_action in actions:
            action = np.asarray(recorded_action, dtype=np.int64 if action_type == 'MULTI_DISCRETE' else np.int8)
            obs, reward, terminated, truncated, _ = env.step(action)
            rows.append({'observation': hashlib.sha256(np.asarray(obs, dtype='<f8').tobytes()).hexdigest(),
                         'action': np.asarray(action).tolist(), 'reward': float(reward),
                         'terminated': bool(terminated), 'truncated': bool(truncated)})
            if terminated or truncated:
                env.reset(seed=7)
        return {'game': game, 'action_type': action_type, 'seed': 7, 'frame_skip': 4,
                'state': args.state, 'rows': rows}, raw_info
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', action='store_true', help='Explicitly replace regression fixtures')
    args = parser.parse_args()
    expected = json.loads(FIXTURE.read_text(encoding='utf-8'))
    inputs = {(trace['game'], trace['action_type']): [row['action'] for row in trace['rows']]
              for trace in expected}
    traces = []
    for game, action_type in CASES:
        trace, raw_info = capture(game, action_type, inputs[game, action_type])
        traces.append(trace)
        if args.record and action_type == 'FILTERED':
            destination = FIXTURE.parent / f'{game}.json'
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(raw_info, sort_keys=True, indent=2) + '\n', encoding='utf-8')
    if args.record:
        FIXTURE.write_text(json.dumps(traces, indent=2) + '\n', encoding='utf-8')
        print('Recorded five baseline traces and three state fixtures')
    else:
        assert traces == expected, 'Gameplay behavior differs from the recorded baseline'
        print('PASS: all five frozen-input gameplay traces match the original implementation')


if __name__ == '__main__':
    main()
