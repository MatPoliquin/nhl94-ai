"""Train scratch goalie policies and evaluate identical native goalie starts."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from nhl94_ai.env.goalie import GoalieEnv, bank_rows
from nhl94_ai.evaluation.goalie import source_hashes
from nhl94_ai.models.goalie import GoaliePolicy, load_model, save_model
from nhl94_ai.training.decision_ppo import DecisionPPO


def make_env(banks, interval, horizon, gamma):
    return Monitor(GoalieEnv(banks, interval=interval, horizon=horizon, gamma=gamma))


class Checkpoints(BaseCallback):
    def __init__(self, path, every, details):
        super().__init__()
        self.path, self.every, self.details = Path(path), every, details
        self.last, self.rows = 0, []

    def _save(self):
        exposure = self.training_env.env_method('exposure')
        totals = {key: sum(row[key] for row in exposure) for key in exposure[0]}
        path = self.path.parent / (self.path.stem + '-checkpoints') / f'step-{self.num_timesteps}.zip'
        save_model(self.model, path, **self.details, exposure=totals)
        self.rows.append({'path': str(path.resolve()), 'timesteps': self.num_timesteps, 'exposure': totals})
        self.last = self.num_timesteps

    def _on_training_start(self):
        self._save()

    def _on_rollout_start(self):
        if self.every and self.num_timesteps - self.last >= self.every:
            self._save()

    def _on_step(self):
        return True

    def _on_training_end(self):
        if self.num_timesteps != self.last:
            self._save()


def train(args):
    if args.model:
        raise ValueError('Goalie training starts from scratch; --model is evaluation-only')
    output = Path(args.output).with_suffix('.zip')
    if output.exists() or output.with_name(output.stem + '-checkpoints').exists():
        raise FileExistsError('Use a fresh output path for a scratch goalie run')
    rows = bank_rows(args.banks)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    jobs = [partial(make_env, args.banks, args.interval, args.horizon, args.gamma) for _ in range(args.envs)]
    env = SubprocVecEnv(jobs, start_method='forkserver') if len(jobs) > 1 else DummyVecEnv(jobs)
    hashes, started = source_hashes(), time.monotonic()
    details = {'initialization': 'random', 'seed': args.seed, 'interval': args.interval,
               'horizon': args.horizon, 'architecture': args.architecture, 'options': vars(args),
               'training_sources': sorted({row['source'] for row in rows}),
               'training_snapshots': len(rows), 'source_hashes': hashes,
               'reward': {'goal': -1, 'confirmed_friendly_possession': 1, 'timeout_or_stoppage': 0},
               'discount_unit': f'{args.interval} native frames'}
    try:
        model = DecisionPPO(GoaliePolicy, env, learning_rate=args.learning_rate, n_steps=args.rollout_steps,
                            batch_size=args.batch_size, n_epochs=4, gamma=args.gamma, gae_lambda=0.95,
                            ent_coef=0.01, target_kl=0.03, seed=args.seed, device=args.device,
                            policy_kwargs={'features_extractor_kwargs': {
                                'architecture': args.architecture, 'hidden_dim': args.hidden_dim}}, verbose=0)
        logdir = str(Path(args.output).with_suffix('')) + '-logs'
        model.set_logger(configure(logdir, ['csv']))
        callback = Checkpoints(args.output, args.checkpoint_every, details)
        print('parameters', sum(p.numel() for p in model.policy.parameters()),
              'architecture', args.architecture, 'scratch seed', args.seed, flush=True)
        model.learn(args.timesteps, callback=callback)
        if hashes != source_hashes():
            raise RuntimeError('Goalie training source changed during the run')
        exposure = env.env_method('exposure')
        totals = {key: sum(row[key] for row in exposure) for key in exposure[0]}
        if totals['decisions'] != model.num_timesteps:
            raise RuntimeError('Goalie active-decision accounting mismatch')
        if not all(torch.isfinite(parameter).all() for parameter in model.policy.parameters()):
            raise RuntimeError('Goalie training produced nonfinite weights')
        saved = save_model(model, args.output, **details, exposure=totals,
                           elapsed_seconds=time.monotonic() - started, checkpoints=callback.rows)
        print('saved', saved, 'exposure', totals, flush=True)
    finally:
        env.close()


def evaluate_chunk(job):
    args, indices = job
    torch.set_num_threads(1)
    mode = 'learned' if args.model or args.controller == 'neutral' else args.controller
    env = GoalieEnv(args.banks, mode=mode, interval=args.interval, horizon=args.horizon, gamma=args.gamma)
    model = load_model(args.model, interval=args.interval) if args.model else None
    if model:
        sources = {row['source'] for row in env.rows}
        if sources.intersection(model.goalie_metadata['training_sources']) and not args.allow_training_starts:
            raise ValueError('Evaluation overlaps goalie training source games')
    results = []
    try:
        for index in indices:
            obs, _ = env.reset(seed=args.seed, options={'index': index})
            done, reward = False, 0.0
            while not done:
                action = int(model.predict(obs, deterministic=True)[0]) if model else 0
                obs, part, done, _, _ = env.step(action)
                reward += part
            results.append({**env.result(), 'reward': reward})
    finally:
        env.close()
    return results


def evaluate(args):
    rows = bank_rows(args.banks)
    indices = list(range(min(len(rows), args.limit or len(rows))))
    chunks = [indices[i::args.workers] for i in range(args.workers)]
    hashes = source_hashes()
    model_digest = hashlib.sha256(Path(args.model).read_bytes()).hexdigest() if args.model else None
    if args.workers == 1:
        results = evaluate_chunk((args, indices))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            results = [row for batch in executor.map(evaluate_chunk, ((args, chunk) for chunk in chunks if chunk))
                       for row in batch]
    if hashes != source_hashes():
        raise RuntimeError('Goalie evaluation source changed during the run')
    if args.model and hashlib.sha256(Path(args.model).read_bytes()).hexdigest() != model_digest:
        raise RuntimeError('Goalie checkpoint changed during evaluation')
    results.sort(key=lambda row: row['start'])
    report = {'options': vars(args), 'source_hashes': hashes, 'model_sha256': model_digest, 'results': results,
              'summary': {'attempts': len(results), 'goals_against': sum(row['goal'] for row in results),
                          'outcomes': dict(Counter(row['outcome'] for row in results)),
                          'mean_frames': float(np.mean([row['frames'] for row in results]))}}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report['summary']), flush=True)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('train', 'evaluate'))
    parser.add_argument('--banks', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model')
    parser.add_argument('--controller', choices=('classic', 'contact', 'neutral'), default='classic')
    parser.add_argument('--architecture', choices=('mlp', 'transformer'), default='transformer')
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--seed', type=int, default=103000)
    parser.add_argument('--interval', type=int, default=4)
    parser.add_argument('--horizon', type=int, default=180)
    parser.add_argument('--gamma', type=float, default=0.997)
    parser.add_argument('--timesteps', type=int, default=1048576)
    parser.add_argument('--envs', type=int, default=8)
    parser.add_argument('--rollout-steps', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=0.0003)
    parser.add_argument('--checkpoint-every', type=int, default=131072)
    parser.add_argument('--device', choices=('cpu', 'cuda', 'auto'), default='cpu')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--allow-training-starts', action='store_true')
    return parser


def run(args):
    if min(args.interval, args.horizon, args.envs, args.workers, args.timesteps,
           args.hidden_dim, args.rollout_steps, args.batch_size) < 1 or args.limit < 0:
        raise ValueError('Goalie dimensions and budgets must be positive')
    if args.operation == 'train':
        train(args)
    else:
        evaluate(args)


if __name__ == '__main__':
    run(build_parser().parse_args())
