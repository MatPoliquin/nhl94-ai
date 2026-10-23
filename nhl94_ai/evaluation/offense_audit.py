"""Branch native teacher possessions through the external command executor."""
import copy
from concurrent.futures import ProcessPoolExecutor
import hashlib
import multiprocessing
from pathlib import Path

import numpy as np

from nhl94_ai.agents.learned_offense import ACTION_NAMES, N_ACTIONS, SCHEMA
from nhl94_ai.env.offense import OffenseEnv, reset_options
from nhl94_ai.models.offense import write_json


def capture(env):
    """In-memory rollback, including counters and the running action digest."""
    names = ('game_state', 'controllers', 'histories', 'info', 'frames', 'finished',
             'previous_score', 'choices')
    return dict(state=env.emulator.em.get_state(), ram=env.emulator.get_ram().tobytes(), digest=env.digest.copy(),
                values=copy.deepcopy({name: getattr(env, name) for name in names}))


def restore(env, saved):
    env.emulator.em.set_state(saved['state'])
    env.emulator.data.update_ram()
    if env.emulator.get_ram().tobytes() != saved['ram']:
        raise RuntimeError('Native branch failed to restore identical gameplay RAM')
    for name, value in copy.deepcopy(saved['values']).items():
        setattr(env, name, value)
    env.digest = saved['digest'].copy()
    env._refresh()


def audit(args):
    """Compare a complete commitment, stopping before the next external choice.

    Each branch begins with identical teacher history. This tests execution,
    not a network's accuracy or long-run tactical memory.
    """
    if args.num_envs > 1:
        jobs = []
        for worker in range(min(args.num_envs, args.episodes)):
            options = copy.copy(args)
            options.num_envs = 1
            options.episode_indices = list(range(worker, args.episodes, args.num_envs))
            options.output = str(Path(args.output).with_suffix('')) + f'-worker-{worker}.json'
            jobs.append(options)
        with ProcessPoolExecutor(max_workers=len(jobs), mp_context=multiprocessing.get_context('spawn')) as pool:
            reports = list(pool.map(audit, jobs))
        report = dict(schema=SCHEMA, options=vars(args),
                      branches=[row for result in reports for row in result['branches']],
                      games=[row for result in reports for row in result['games']],
                      counts=np.sum([result['counts'] for result in reports], axis=0).tolist(),
                      mismatches=sum(result['mismatches'] for result in reports),
                      controller_sha256=reports[0]['controller_sha256'], scope=reports[0]['scope'])
        write_json(args.output, report)
        return report
    env = OffenseEnv(states=args.states, opponents=args.opponents, learner='teacher',
                     seconds=args.seconds, goalie_policy=args.goalie_policy, side=args.side,
                     max_frames=args.max_frames, starts=args.starts, start_probability=args.start_probability)
    rows, games = [], []
    counts = np.zeros(N_ACTIONS, dtype=int)
    try:
        for episode in getattr(args, 'episode_indices', range(args.episodes)):
            obs, info = env.reset(seed=args.seed + episode, options=reset_options(args, episode))
            while not env.finished:
                label = None
                if obs[-1, -N_ACTIONS:].sum() > 1:
                    oracle = copy.deepcopy(env.controllers[env.side - 1])
                    view = env.views[env.side - 1]
                    oracle.begin_step(view)
                    oracle.frame(view)
                    label = oracle.teacher_command
                if label is None or counts[label] >= args.audit_per_action:
                    obs, _, _, _, info = env.step(0)
                    continue
                before = capture(env)
                steps = 0
                while True:
                    obs, _, _, _, info = env.step(0)
                    steps += 1
                    if env.finished or obs[-1, -N_ACTIONS:].sum() > 1 or steps >= 256:
                        break
                after = capture(env)
                teacher_hash = env.digest.hexdigest()
                restore(env, before)
                env.learner = env.controllers[env.side - 1].mode = 'learned'
                early_end = False
                for step in range(steps):
                    if env.finished:
                        early_end = True
                        break
                    env.step(label if step == 0 else 0)
                row = dict(action=ACTION_NAMES[label], label=int(label), seed=env.episode_seed,
                           state=env.state_name, side=env.side, start_frame=before['values']['frames'],
                           frames=after['values']['frames'] - before['values']['frames'],
                           steps=steps, horizon_capped=steps == 256,
                           buttons_equal=teacher_hash == env.digest.hexdigest(),
                           native_state_equal=after['ram'] == env.emulator.get_ram().tobytes(),
                           early_end=early_end)
                rows.append(row)
                counts[label] += 1
                restore(env, after)
                env.learner = env.controllers[env.side - 1].mode = 'teacher'
                obs = env._observation()
            games.append(info)
            print(f'audit period {episode + 1}: {len(rows)} branches, '
                  f'{sum(not r["buttons_equal"] or not r["native_state_equal"] for r in rows)} mismatches', flush=True)
    finally:
        env.close()
    report = dict(schema=SCHEMA, options=vars(args), branches=rows, games=games,
                  counts=counts.tolist(), mismatches=sum(
                      not row['buttons_equal'] or not row['native_state_equal'] for row in rows),
                  scope='Teacher-history branches through next decision; exact native input and state comparison')
    # Keep the evidence tied to the implementation that actually ran.
    report['controller_sha256'] = hashlib.sha256(Path('nhl94_ai/agents/learned_offense.py').read_bytes()).hexdigest()
    write_json(args.output, report)
    return report
