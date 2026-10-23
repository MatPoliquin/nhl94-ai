"""Evaluate, audit, and play existing offense policies."""
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import time
import numpy as np
import torch
from nhl94_ai.agents.learned_offense import N_ACTIONS
from nhl94_ai.env.offense import OffenseEnv, SCHEMA, reset_options
from nhl94_ai.models.offense import load_offense_model, write_json
from nhl94_ai.training.unavailable import unavailable


def env_options(args, *, learner='learned', opponents=None):
    starts = args.starts
    return dict(states=args.states, opponents=opponents or args.opponents, learner=learner,
                seconds=args.seconds, goalie_policy=args.goalie_policy, side=args.side,
                max_frames=args.max_frames, starts=starts, start_probability=args.start_probability,
                task=args.task, attack_frames=args.attack_frames, control_scope=args.control_scope, gamma=args.gamma)


def evaluate(args):
    if args.num_envs > 1 and args.phase == 'evaluate':
        jobs = []
        for worker in range(min(args.num_envs, args.episodes)):
            options = copy.copy(args)
            options.num_envs = 1
            options.episode_indices = list(range(worker, args.episodes, args.num_envs))
            options.output = str(Path(args.output).with_suffix('')) + f'-worker-{worker}.json'
            jobs.append(options)
        with ProcessPoolExecutor(max_workers=len(jobs), mp_context=multiprocessing.get_context('spawn')) as pool:
            reports = list(pool.map(evaluate_worker, jobs))
        results = [row for report in reports for row in report['results']]
        return evaluation_report(args, results, tuple(dict.fromkeys(row['agent'] for row in results)))
    model = load_offense_model(args.model, expected_task=args.task,
                               allow_task_transfer=args.allow_task_transfer,
                               expected_control_scope=args.control_scope,
                               allow_control_transfer=args.allow_control_transfer) if args.model else None
    if model is not None and args.task == 'opportunity' and abs(model.gamma - args.gamma) > 1e-12:
        raise ValueError('Opportunity evaluation discount must match the checkpoint')
    agents = args.agents or (('model', 'teacher', 'classic') if model else ('teacher', 'classic'))
    if 'model' in agents and model is None:
        raise ValueError('The model evaluation arm requires --model')
    if args.phase == 'play':
        agents = ('model',) if model else ('classic',)
    results = []
    for opponent in args.opponents:
        for agent in agents:
            env = OffenseEnv(**env_options(args, learner='learned' if agent in ('model', 'shoot', 'coast', 'forward') else agent,
                                          opponents=(opponent,)))
            try:
                for episode in getattr(args, 'episode_indices', range(args.episodes)):
                    options = reset_options(args, episode, opponent)
                    if args.task != 'period':
                        compatible = sorted(p for p in args.starts if env.start_metadata[p]['opponent'] == opponent)
                        options = dict(snapshot=compatible[episode % len(compatible)])
                    obs, _ = env.reset(seed=args.seed + episode, options=options)
                    total = 0.0
                    while True:
                        action = (model.predict(obs, deterministic=True)[0]
                                  if agent == 'model' and obs[-1, -N_ACTIONS:].sum() > 1 else 0)
                        if agent == 'shoot' and obs[-1, -N_ACTIONS + 17]:
                            action = 17
                        if agent == 'forward' and obs[-1, -N_ACTIONS + 1]:
                            action = 1
                        obs, reward, terminated, truncated, info = env.step(action)
                        total += reward
                        if args.phase == 'play':
                            import cv2
                            cv2.imshow('Learned offense', cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR))
                            if cv2.waitKey(1) == 27:
                                return results
                            time.sleep(4 / 60)
                        if terminated or truncated:
                            break
                    info.update(agent=agent, reward=total)
                    results.append(info)
                    print(json.dumps(info), flush=True)
            finally:
                env.close()
    return evaluation_report(args, results, agents)


def evaluate_worker(args):
    torch.set_num_threads(args.threads)
    return evaluate(args)


def task_contract(args):
    from nhl94_ai.env.opportunity import REWARD_VERSION, REWARD_SPEC
    reward = (REWARD_VERSION if args.task == 'opportunity' else
              'goal-before-loss-v1' if args.task == 'attack' else SCHEMA['reward'])
    contract = dict(task=args.task, attack_frames=args.attack_frames, control_scope=args.control_scope,
                    decision_only=args.decision_only, gamma_unit='four-native-frames', reward=reward,
                    observation_clock='attack-seconds-remaining' if args.task != 'period' else 'game-clock',
                    lambda_unit='decision' if args.decision_only else 'four-native-frames')
    if args.task == 'opportunity':
        contract['reward_weights'] = REWARD_SPEC
    return contract


def evaluation_report(args, results, agents):
    if args.task != 'period':
        from nhl94_ai.evaluation.offense_attempts import attempt_report
        report = attempt_report(results, seed=args.seed)
        report['protocol'] = dict(attack_frames=args.attack_frames, policy_interval_frames=SCHEMA['interval'],
                                  model=args.model, task_transfer=args.allow_task_transfer,
                                  outcome_reward='goal=1; turnover/retreat/timeout/period-end=0',
                                  clock='remaining attack seconds', evaluation='deterministic')
        if args.task == 'opportunity':
            from nhl94_ai.evaluation.opportunity import opportunity_report
            report.update(opportunity_report(results, args.seed))
            report['protocol'].update(control_scope=args.control_scope, task_contract=task_contract(args),
                                      control_transfer=args.allow_control_transfer,
                                      outcome_reward='see task_contract.reward_weights')
        write_json(args.output, report)
        return report
    summary = {}
    for opponent in args.opponents:
        for agent in agents:
            rows = [row for row in results if row['agent'] == agent and row['opponent'] == opponent]
            complete = [row for row in rows if row['completed']]
            summary[f'{agent}:{opponent}'] = dict(
                periods=len(complete), incomplete=len(rows) - len(complete),
                goals_for=sum(row['goals_for'] for row in complete),
                goals_against=sum(row['goals_against'] for row in complete),
                wins=sum(row['goals_for'] > row['goals_against'] for row in complete),
                draws=sum(row['goals_for'] == row['goals_against'] for row in complete),
                losses=sum(row['goals_for'] < row['goals_against'] for row in complete))
    # Bootstrap ROM-seed clusters, keeping both team assignments together.
    paired = {}
    for agent in agents:
        if agent == 'classic':
            continue
        for opponent in args.opponents:
            base = {(r['state'], r['seed'], r['side']): r for r in results
                    if r['agent'] == 'classic' and r['opponent'] == opponent and r['completed']}
            clusters = {}
            for row in results:
                key = row['state'], row['seed'], row['side']
                if row['agent'] != agent or row['opponent'] != opponent or not row['completed'] or key not in base:
                    continue
                ref = base[key]
                delta = row['goals_for'] - row['goals_against'] - ref['goals_for'] + ref['goals_against']
                clusters.setdefault(row['seed'], []).append(delta)
            if clusters:
                means = np.asarray([np.mean(values) for values in clusters.values()])
                draws = np.random.default_rng(args.seed).choice(means, (10000, len(means))).mean(axis=1)
                paired[f'{agent}:{opponent}'] = dict(mean_delta=float(means.mean()),
                                                    seed_clusters=len(means),
                                                    interval95=np.quantile(draws, [0.025, 0.975]).tolist())
    report = dict(schema=SCHEMA, results=results, summary=summary, paired_vs_classic=paired)
    write_json(args.output, report)
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('evaluate', 'play', 'audit'))
    parser.add_argument('--output', required=True, help='Evaluation or audit JSON output')
    parser.add_argument('--model', help='Existing policy checkpoint')
    parser.add_argument('--states', nargs='+', default=['MightyDucksVsAllStarCampbell.ManualGoalie.Start'])
    parser.add_argument('--opponents', nargs='+', default=['cpu'], help='cpu, classic, or frozen checkpoint paths')
    parser.add_argument('--goalie-policy', choices=('off', 'selective', 'always'), default='selective')
    parser.add_argument('--side', type=int, choices=(1, 2))
    parser.add_argument('--seconds', type=int, default=300)
    parser.add_argument('--seed', type=int, default=41000)
    parser.add_argument('--episodes', type=int, default=40)
    parser.add_argument('--agents', nargs='+', choices=('model', 'teacher', 'classic', 'shoot', 'coast', 'forward'),
                        help='Evaluation arms; shoot is an immediate wrist-shot baseline')
    parser.add_argument('--max-frames', type=int, help='Diagnostic truncation; use complete periods for comparisons')
    parser.add_argument('--gamma', type=float, default=0.997)
    parser.add_argument('--task', choices=('period', 'attack', 'opportunity'), default='period')
    parser.add_argument('--control-scope', choices=('offense', 'carry'), default='offense',
                        help='Carry masks pass/shot actions and delegates all finishing decisions to Classic')
    parser.add_argument('--allow-control-transfer', action='store_true',
                        help='Explicitly adapt/evaluate weights under a different control scope')
    parser.add_argument('--attack-frames', type=int, default=600, help='Attack task horizon at 60 FPS; goals score 1, other endings 0')
    parser.add_argument('--allow-task-transfer', action='store_true',
                        help='Explicitly evaluate weights under a different task/clock contract')
    parser.add_argument('--num-envs', type=int, default=1)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--audit-per-action', type=int, default=32, help='Native execution branches per action')
    parser.add_argument('--starts', nargs='+', default=[])
    parser.add_argument('--start-probability', type=float, default=0.0)
    parser.set_defaults(decision_only=False)
    return parser


def run(args):
    if args.phase in ('collect', 'bc', 'train'):
        unavailable()
    if getattr(args, 'config', None):
        base = Path(args.config).expanduser().resolve().parent
        args.starts = [str(base / Path(path).expanduser()) for path in args.starts]
        args.opponents = [name if name in ('cpu', 'classic') else str(base / Path(name).expanduser())
                          for name in args.opponents]
    if min(args.episodes, args.num_envs, args.threads, args.attack_frames) < 1:
        raise ValueError('Counts must be positive')
    if not 0 < args.gamma < 1 or not 0 <= args.seed < 2**32 - args.episodes:
        raise ValueError('Use a valid discount and uint32 seed range')
    if args.task != 'period' and (not args.starts or args.phase != 'evaluate'):
        raise ValueError('Attack tasks require --starts and evaluate mode')
    if args.task == 'opportunity' and args.control_scope != 'carry':
        raise ValueError('Use --control-scope carry for the opportunity task')
    torch.set_num_threads(args.threads)
    if args.phase == 'audit':
        from nhl94_ai.evaluation.offense_audit import audit
        return audit(args)
    return evaluate(args)



if __name__ == '__main__':
    run(build_parser().parse_args())
