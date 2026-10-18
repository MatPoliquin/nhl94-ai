"""Exercise active goalie choices and forced native states with random legal commands."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

from nhl94_ai.agents.learned_goalie import N_ACTIONS
from nhl94_ai.env.goalie import GoalieEnv
from nhl94_ai.evaluation.goalie import source_hashes


def audit(job):
    args, seed = job
    rng = np.random.default_rng(seed)
    env = GoalieEnv(args.banks, interval=args.interval)
    episodes, choices = 0, []
    try:
        obs, _ = env.reset(seed=seed)
        while env.total_decisions < args.decisions:
            command = int(rng.choice(np.flatnonzero(obs[-N_ACTIONS:])))
            choices.append(command)
            obs, _, done, _, info = env.step(command)
            assert info['command_used'], 'Active choice was ignored'
            if done:
                episodes += 1
                obs, _ = env.reset()
                choices = []
        exposure = env.exposure()
        assert exposure['decisions'] == exposure['commands_used'] == args.decisions
        return {'seed': seed, 'episodes': episodes, **exposure}
    except Exception as error:
        raise RuntimeError({'seed': seed, 'start': env.start['path'], 'commands': choices}) from error
    finally:
        env.close()


def run(args):
    hashes = source_hashes()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(audit, ((args, args.seed + index) for index in range(args.workers))))
    assert hashes == source_hashes(), 'Sources changed during native choice audit'
    report = {'options': vars(args), 'source_hashes': hashes, 'results': results}
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('PASS:', sum(row['decisions'] for row in results), 'legal goalie choices applied;',
          sum(row['native_frames'] for row in results), 'native frames', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--banks', nargs='+', required=True)
    parser.add_argument('--interval', type=int, default=4)
    parser.add_argument('--seed', type=int, default=109000)
    parser.add_argument('--decisions', type=int, default=25000, help='Choices per worker')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--output')
    run(parser.parse_args())
