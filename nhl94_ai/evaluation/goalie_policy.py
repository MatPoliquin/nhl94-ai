"""Evaluate existing goalie policies on native starts."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from nhl94_ai.env.goalie import GoalieEnv, bank_rows
from nhl94_ai.evaluation.goalie import source_hashes
from nhl94_ai.models.goalie import load_model
from nhl94_ai.training.unavailable import unavailable


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
    parser.add_argument('operation', choices=('evaluate',))
    parser.add_argument('--banks', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model')
    parser.add_argument('--controller', choices=('classic', 'contact', 'neutral'), default='classic')
    parser.add_argument('--seed', type=int, default=103000)
    parser.add_argument('--interval', type=int, default=4)
    parser.add_argument('--horizon', type=int, default=180)
    parser.add_argument('--gamma', type=float, default=0.997)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--allow-training-starts', action='store_true')
    return parser


def run(args):
    if args.operation == 'train':
        unavailable()
    if min(args.interval, args.horizon, args.workers) < 1 or args.limit < 0:
        raise ValueError('Goalie dimensions and budgets must be positive')
    evaluate(args)



if __name__ == '__main__':
    run(build_parser().parse_args())
