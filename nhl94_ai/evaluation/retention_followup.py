"""Follow changed retention decisions for 144 native frames with Classic continuation."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import pickle

from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.agents.retention_ranker import load_retention_filter
from nhl94_ai.env.factory import make_retro
from nhl94_ai.evaluation.cpu_benchmark import CPU_RAM
from nhl94_ai.evaluation.goalie import source_hashes
from nhl94_ai.evaluation.retention import branch
from nhl94_ai.evaluation.retention_model import load_rows, cluster_interval


def follow_game(job):
    state_name, rows, model_path = job
    policy = load_retention_filter(model_path)
    env = make_retro(game='NHL94-Genesis-v0', state=state_name, num_players=1, goalie_policy='off')
    results = []
    try:
        for key, (address, kind) in CPU_RAM.items():
            env.data.set_variable(key, {'address': address, 'type': kind})
        env.reset(seed=rows[0]['seed'])
        for row in rows:
            selected, _ = policy.choose([b['features'] for b in row['branches']], row['baseline'])
            if selected == row['baseline']:
                continue
            path = Path(row['path'])
            if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
                raise ValueError('Retention snapshot checksum mismatch')
            with gzip.open(path, 'rb') as stream:
                sample = pickle.load(stream)
            arms = {}
            for name, command in (('classic', row['baseline']), ('filtered', selected)):
                measured = branch(env, sample, command, row['side'], ClassicAIV1Model.predict_frame, horizon=144)
                previous = row['branches'][command]
                if measured['trace'][:len(previous['trace'])] != previous['trace']:
                    raise RuntimeError('Extended replay does not reproduce its measured 24-frame prefix')
                arms[name] = measured
            results.append(dict(id=row['id'], source=row['source'], baseline=row['baseline'], selected=selected, **arms))
    finally:
        env.close()
    return results


def summarize(results):
    summary = {}
    for name in ('classic', 'filtered'):
        rows = [r[name] for r in results]
        summary[name] = dict(losses=sum(r['loss'] for r in rows), turnovers_before_release=sum(r['turnover'] for r in rows),
                             ordinary_turnovers=sum(r['ordinary_turnovers'] for r in rows),
                             opponent_possession_observed=sum(r['first_opponent_possession_frame'] is not None for r in rows),
                             shots=sum(r['shots'] for r in rows),
                             goals=sum(any(event['outcome'] == 'goal' for event in r['shot_events']) for r in rows),
                             releases=sum(r['release_frame'] is not None for r in rows),
                             mean_progress=sum(r['progress'] for r in rows) / max(1, len(rows)))
    deltas = [r['filtered']['loss'] - r['classic']['loss'] for r in results]
    return dict(changed_situations=len(results), horizon=144, **summary,
                net_prevented=summary['classic']['losses'] - summary['filtered']['losses'],
                loss_rate_delta_interval95=cluster_interval(deltas, [r['source'] for r in results]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    _, rows = load_rows(args.test)
    policy = load_retention_filter(args.model)
    used = set(policy.artifact['training']['training_sources']) | set(policy.artifact['training']['development_sources'])
    if used & {row['source'] for row in rows}:
        raise ValueError('Follow-up test games must be independent of training/development')
    games = {}
    for row in rows:
        chosen, _ = policy.choose([b['features'] for b in row['branches']], row['baseline'])
        if chosen != row['baseline']:
            games.setdefault((row['state'], row['seed'], row['side']), []).append(row)
    hashes = source_hashes()
    results = []
    jobs = [(key[0], value, args.model) for key, value in games.items()]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for batch in executor.map(follow_game, jobs):
            results.extend(batch)
            print('followed', len(results), 'changed decisions', flush=True)
    if hashes != source_hashes() or hashlib.sha256(Path(args.model).read_bytes()).hexdigest() != policy.sha256:
        raise RuntimeError('Follow-up runtime or model changed during measurement')
    report = dict(model_sha256=policy.sha256, source_hashes=hashes,
                  summary=summarize(results), results=results,
                  limitation='One changed decision followed by Classic; repeated filter use requires full periods')
    Path(args.output).write_text(json.dumps(report) + '\n', encoding='utf-8')
    print(json.dumps(report['summary']), flush=True)


if __name__ == '__main__':
    main()
