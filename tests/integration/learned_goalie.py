"""Native snapshot reset and exact training/deployment goalie-input parity."""
import argparse
import hashlib
from unittest.mock import patch

import numpy as np
import torch

from nhl94_ai.agents.learned_goalie import replace_goalie
from nhl94_ai.agents.neural_goalie import NeuralGoalie
from nhl94_ai.env.goalie import GoalieEnv, bank_rows
from nhl94_ai.models.goalie import load_model


def episode(args, index, policy, deployed):
    env = GoalieEnv(args.banks, mode='classic' if deployed else 'learned', interval=args.interval)
    try:
        obs, info = env.reset(seed=7, options={'index': index})
        if deployed:
            env.model.goalie = replace_goalie(env.model.goalie,
                                              NeuralGoalie('selective', args.model, interval=args.interval))
        original = env.emulator.step
        actions = []

        def record(action):
            actions.append(np.asarray(action, np.int8).tobytes())
            return original(action)

        with patch.object(env.emulator, 'step', side_effect=record):
            done = False
            while not done:
                command = 0 if deployed else int(policy.predict(obs, deterministic=True)[0])
                obs, _, done, _, _ = env.step(command)
        return {'ram_sha256': info['ram_sha256'], 'actions': actions, 'result': env.result()}
    finally:
        env.close()


def run(args):
    torch.set_num_threads(1)
    policy = load_model(args.model, interval=args.interval)
    rows, chosen, counts = bank_rows(args.banks), [], {}
    for index, row in enumerate(rows):
        matchup = row['matchup']
        if counts.get(matchup, 0) < args.per_matchup:
            chosen.append(index)
            counts[matchup] = counts.get(matchup, 0) + 1
    assert len(counts) == 4, 'Parity check requires both sides of both rosters'
    for index in chosen:
        training = episode(args, index, policy, False)
        deployment = episode(args, index, policy, True)
        assert training['ram_sha256'] == deployment['ram_sha256']
        assert training['actions'] == deployment['actions'], (
            rows[index]['path'], len(training['actions']), len(deployment['actions']),
            next((i for i, (a, b) in enumerate(zip(training['actions'], deployment['actions'])) if a != b), None))
        assert training['result']['outcome'] == deployment['result']['outcome']
        digest = hashlib.sha256(b''.join(training['actions'])).hexdigest()
        print('PASS', rows[index]['matchup'], rows[index]['frame'], len(training['actions']), digest, flush=True)
    print('PASS: native training/deployment actions match on', len(chosen), 'goalie starts', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--banks', nargs='+', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--interval', type=int, default=4)
    parser.add_argument('--per-matchup', type=int, default=3)
    run(parser.parse_args())
