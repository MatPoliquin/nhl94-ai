"""Bounded within-situation movement ranking from native retention replays."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from nhl94_ai.agents.retention_ranker import RetentionRanker, SCHEMA
from nhl94_ai.training.retention import assert_disjoint, load_rows, score_policy


DEPTHS = (2, 3, 4)
ROUNDS = (100, 300)
MARGINS = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)
PARAMETERS = dict(objective='rank:pairwise', tree_method='hist', device='cpu', n_jobs=4,
                  learning_rate=0.05, min_child_weight=5, reg_lambda=10, reg_alpha=0,
                  subsample=1.0, colsample_bytree=1.0, base_score=0.0, random_state=94,
                  lambdarank_pair_method='mean', lambdarank_num_pair_per_sample=8,
                  lambdarank_normalization=True, lambdarank_score_normalization=False)


def ranking_arrays(rows, names):
    """One query per situation, one label per uncensored candidate movement."""
    valid_rows = [(row, [b for b in row['branches'] if b['label_valid']]) for row in rows]
    valid_rows = [(row, branches) for row, branches in valid_rows if branches]
    if not valid_rows:
        raise ValueError('Ranking requires uncensored training outcomes')
    sources = Counter(row['source'] for row, _ in valid_rows)
    x, y, qid, weights = [], [], [], []
    informative = pairs = 0
    for query, (row, branches) in enumerate(valid_rows):
        kept = sum(not b['loss'] for b in branches)
        pairs += kept * (len(branches) - kept)
        informative += 0 < kept < len(branches)
        weights.append(1 / sources[row['source']])
        for branch in branches:
            if tuple(branch['features']) != tuple(names):
                raise ValueError('Ranking feature schema differs between branches')
            x.append([branch['features'][name] for name in names])
            y.append(1 - branch['loss'])
            qid.append(query)
    weights = np.asarray(weights, dtype=np.float64)
    weights *= len(weights) / weights.sum()
    return (np.asarray(x, dtype=np.float32), np.asarray(y), np.asarray(qid), weights,
            dict(labelled_branches=len(y), labelled_situations=len(weights),
                 informative_situations=int(informative), distinct_label_pairs=int(pairs),
                 censored_situations=len(rows) - len(weights)))


def export_nodes(root, names):
    nodes = []
    def visit(node):
        index = len(nodes)
        nodes.append(dict(left=-1, right=-1, feature=0, threshold=0.0, value=float(node.get('leaf', 0))))
        if 'leaf' not in node:
            children = {child['nodeid']: child for child in node['children']}
            split = node['split']
            feature = int(split[1:]) if split.startswith('f') else list(names).index(split)
            nodes[index].update(feature=feature, threshold=float(node['split_condition']),
                                left=visit(children[node['yes']]), right=visit(children[node['no']]))
        return index
    visit(root)
    return nodes


def export_ranker(model, names, *, minimum_margin=0.0):
    """Export the fixed-base-score gbtree model to plain ordered JSON trees."""
    config = json.loads(model.get_booster().save_config())['learner']
    base = np.asarray(json.loads(config['learner_model_param']['base_score'])).reshape(-1)
    if base.size != 1 or config['objective']['name'] != 'rank:pairwise':
        raise ValueError('Expected scalar pairwise ranker')
    trees = [export_nodes(json.loads(dumped), names) for dumped in model.get_booster().get_dump(dump_format='json')]
    return dict(schema=SCHEMA, interval=4, horizon=24, feature_names=list(names),
                base_score=float(base[0]), minimum_margin=minimum_margin, trees=trees)


def feature_matrix(rows, names):
    features = [b['features'] for row in rows for b in row['branches']]
    if any(tuple(row) != tuple(names) for row in features):
        raise ValueError('Ranking feature order differs between datasets')
    return np.asarray([[row[name] for name in names] for row in features], dtype=np.float32)


def ranking_metrics(rows, predictions):
    """Compare labels only within a situation; equal scores receive half credit."""
    correct = ties = pairs = informative = 0
    per_state = []
    for row, scores in zip(rows, predictions):
        kept = [i for i, b in enumerate(row['branches']) if b['label_valid'] and not b['loss']]
        lost = [i for i, b in enumerate(row['branches']) if b['label_valid'] and b['loss']]
        if not kept or not lost:
            continue
        difference = scores[kept, None] - scores[None, lost]
        wins, equal = int((difference > 0).sum()), int((difference == 0).sum())
        correct += wins; ties += equal; pairs += difference.size; informative += 1
        per_state.append((wins + 0.5 * equal) / difference.size)
    return dict(informative_situations=informative, distinct_label_pairs=pairs,
                correctly_ordered_pairs=correct, tied_pairs=ties,
                pair_accuracy=(correct + 0.5 * ties) / pairs if pairs else None,
                mean_situation_pair_accuracy=float(np.mean(per_state)) if per_state else None,
                situations_with_different_scores=int((np.ptp(predictions, axis=1) > 0).sum()))


def development_feasible(result):
    return (result['model_turnovers'] <= result['classic_turnovers']
            and result['mean_changed_progress_delta'] >= -2 and result['new_retreats'] == 0)


def compact(result):
    return {key: value for key, value in result.items() if key != 'choices'}


def train(args):
    import xgboost
    from nhl94_ai.evaluation.goalie import source_hashes
    started, hashes = time.monotonic(), source_hashes()
    output = Path(args.output)
    if output.exists() or output.with_suffix('.training.json').exists():
        raise ValueError('Refusing to overwrite a ranking experiment')
    data, rows = load_rows(args.train)
    dev_data, dev = load_rows(args.development)
    assert_disjoint(rows, dev)
    names = tuple(rows[0]['branches'][0]['features'])
    x, y, qid, weights, statistics = ranking_arrays(rows, names)
    if not statistics['informative_situations']:
        raise ValueError('No within-situation outcome differences to learn')
    train_x, dev_x = feature_matrix(rows, names), feature_matrix(dev, names)
    candidates, artifacts, diagnostics = [], {}, []
    for depth in DEPTHS:
        for rounds in ROUNDS:
            model = xgboost.XGBRanker(max_depth=depth, n_estimators=rounds, **PARAMETERS)
            model.fit(x, y, qid=qid, sample_weight=weights, verbose=False)
            artifact = export_ranker(model, names)
            policy = RetentionRanker(artifact=artifact)
            predictions = {}
            max_error = 0.0
            for split, matrix in (('training', train_x), ('development', dev_x)):
                native = model.predict(matrix, output_margin=True)
                exported = policy.predict_matrix(matrix)
                max_error = max(max_error, float(np.max(np.abs(native - exported))))
                if not np.allclose(native, exported, atol=1e-6, rtol=1e-6):
                    raise RuntimeError('Exported ranking scores differ from XGBoost')
                predictions[split] = exported.reshape(-1, 9)
            train_scores, dev_scores = predictions['training'], predictions['development']
            train_rank, dev_rank = ranking_metrics(rows, train_scores), ranking_metrics(dev, dev_scores)
            diagnostics.append(dict(depth=depth, rounds=rounds, nodes=sum(map(len, artifact['trees'])),
                                    export_max_error=max_error, training=train_rank, development=dev_rank))
            artifacts[depth, rounds] = artifact
            for margin in MARGINS:
                policy.minimum_margin = margin
                result = score_policy(dev, policy, predictions=dev_scores)
                train_result = score_policy(rows, policy, predictions=train_scores)
                candidates.append(dict(depth=depth, rounds=rounds, minimum_margin=margin,
                                       development=compact(result), training=compact(train_result)))
            print(json.dumps(diagnostics[-1]), flush=True)
    feasible = [c for c in candidates if development_feasible(c['development'])]
    def selection(candidate):
        result = candidate['development']
        return (-result['net_prevented'], result['model_turnovers'], result['overrides'],
                candidate['depth'], candidate['rounds'], -candidate['minimum_margin'])
    selected = min(feasible or candidates, key=selection)
    promising = bool(feasible and selected['development']['net_prevented'] > 0)
    artifact = dict(artifacts[selected['depth'], selected['rounds']], minimum_margin=selected['minimum_margin'])
    artifact['training'] = dict(
        algorithm='XGBRanker', parameters=PARAMETERS, depth=selected['depth'], rounds=selected['rounds'],
        training_starts=len(rows), **statistics, development_promising=promising,
        training_sources=sorted({r['source'] for r in rows}),
        development_sources=sorted({r['source'] for r in dev}),
        training_data_sha256=hashlib.sha256(Path(args.train).read_bytes()).hexdigest(),
        development_data_sha256=hashlib.sha256(Path(args.development).read_bytes()).hexdigest(),
        training_source_hashes=data['source_hashes'], development_source_hashes=dev_data['source_hashes'],
        fitting_source_hashes=hashes, versions=dict(xgboost=xgboost.__version__, numpy=np.__version__),
        source_weighting='equal total query weight per physical game, excluding fully censored situations',
        export_parity_rows=len(train_x) + len(dev_x), wall_seconds=time.monotonic() - started)
    policy = RetentionRanker(artifact=artifact)
    train_scores, dev_scores = (policy.predict_matrix(matrix).reshape(-1, 9) for matrix in (train_x, dev_x))
    report = dict(selected=selected, development_promising=promising, diagnostics=diagnostics, candidates=candidates,
                  training=score_policy(rows, policy, predictions=train_scores),
                  development=score_policy(dev, policy, predictions=dev_scores),
                  training_ranking=ranking_metrics(rows, train_scores), development_ranking=ranking_metrics(dev, dev_scores))
    if hashes != source_hashes():
        raise RuntimeError('Source code changed during ranking fit')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2) + '\n', encoding='utf-8')
    report['model_sha256'] = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix('.training.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('SELECTED', json.dumps(selected), 'PROMISING', promising, flush=True)


def evaluate(args):
    _, rows = load_rows(args.test)
    policy = RetentionRanker(args.model)
    used = set(policy.artifact['training']['training_sources']) | set(policy.artifact['training']['development_sources'])
    if used & {r['source'] for r in rows}:
        raise ValueError('Ranking test games overlap training/development')
    scores = policy.predict_matrix(feature_matrix(rows, policy.names)).reshape(-1, 9)
    result = dict(model_sha256=policy.sha256, test_sha256=hashlib.sha256(Path(args.test).read_bytes()).hexdigest(),
                  policy=score_policy(rows, policy, predictions=scores), ranking=ranking_metrics(rows, scores))
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(compact(result['policy'])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    fitting = commands.add_parser('train')
    fitting.add_argument('--train', required=True)
    fitting.add_argument('--development', required=True)
    fitting.add_argument('--output', required=True)
    testing = commands.add_parser('evaluate')
    testing.add_argument('--test', required=True)
    testing.add_argument('--model', required=True)
    testing.add_argument('--output', required=True)
    args = parser.parse_args()
    (train if args.command == 'train' else evaluate)(args)


if __name__ == '__main__':
    main()
