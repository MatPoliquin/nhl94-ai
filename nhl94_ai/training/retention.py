"""Train small supervised retention trees with source-game-separated evaluation."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from nhl94_ai.agents.retention_filter import RetentionFilter, SCHEMA
from nhl94_ai.evaluation.retention import SCHEMA as DATA_SCHEMA


DEPTHS = (3, 5, 7)
LEAF_SIZES = (32, 64)
MINIMUM_RISKS = (0.25, 0.5)
MINIMUM_GAPS = (0.15, 0.30)


def load_rows(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if data['schema'] != DATA_SCHEMA or (data['interval'], data['horizon']) != (4, 24):
        raise ValueError('Retention data schema/cadence mismatch')
    rows = [row for game in data['games'] for row in game['starts']]
    if not rows:
        raise ValueError('Retention dataset is empty')
    for row in rows:
        if len(row['branches']) != 9 or [b['command'] for b in row['branches']] != list(range(9)):
            raise ValueError('Dataset requires every ordered movement alternative')
    return data, rows


def assert_disjoint(*partitions):
    seen = set()
    for rows in partitions:
        sources = {row['source'] for row in rows}
        if sources & seen:
            raise ValueError('Training/development/test source games overlap')
        seen.update(sources)


def arrays(rows, names):
    sources = Counter(row['source'] for row in rows if any(b['label_valid'] for b in row['branches']))
    x, y, weights = [], [], []
    for row in rows:
        valid = [b for b in row['branches'] if b['label_valid']]
        for branch in valid:
            if tuple(branch['features']) != tuple(names):
                raise ValueError('Retention feature order differs between datasets')
            x.append([branch['features'][name] for name in names])
            y.append(branch['loss'])
            weights.append(1 / (sources[row['source']] * len(valid)))
    if not x:
        raise ValueError('Retention dataset contains no uncensored labels')
    weights = np.asarray(weights, dtype=np.float64)
    weights *= len(weights) / weights.sum()
    return np.asarray(x, dtype=np.float32), np.asarray(y), weights


def export_tree(model, names):
    tree = model.tree_
    loss = list(model.classes_).index(1)
    nodes = []
    for i in range(tree.node_count):
        values = tree.value[i, 0]
        nodes.append(dict(left=int(tree.children_left[i]), right=int(tree.children_right[i]),
                          feature=int(tree.feature[i]), threshold=float(tree.threshold[i]),
                          risk=float(values[loss] / values.sum())))
    return dict(schema=SCHEMA, interval=4, horizon=24, feature_names=list(names),
                tree=nodes, minimum_risk=0.25, minimum_gap=0.15)


def cluster_interval(values, sources, seed=94):
    groups = {}
    for value, source in zip(values, sources):
        groups.setdefault(source, []).append(value)
    if not groups:
        return None
    sums = np.asarray([sum(group) for group in groups.values()], dtype=float)
    counts = np.asarray([len(group) for group in groups.values()])
    indices = np.random.default_rng(seed).integers(len(sums), size=(10000, len(sums)))
    estimates = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
    return np.quantile(estimates, [0.025, 0.975]).tolist()


def score_policy(rows, policy, *, predictions=None):
    if predictions is not None and len(predictions) != len(rows):
        raise ValueError('Prediction groups do not match situations')
    counts = Counter(starts=len(rows))
    losses, sources, progress = [], [], []
    choices = []
    for index, row in enumerate(rows):
        features = [b['features'] for b in row['branches']]
        selected, risks = (policy.choose(features, row['baseline']) if predictions is None else
                           policy.choose_scores(features, row['baseline'], predictions[index]))
        base, chosen = row['branches'][row['baseline']], row['branches'][selected]
        counts['censored_pairs'] += not (base['label_valid'] and chosen['label_valid'])
        if not (base['label_valid'] and chosen['label_valid']):
            continue
        changed = selected != row['baseline']
        counts['evaluated'] += 1
        counts['overrides'] += changed
        counts['classic_losses'] += base['loss']
        counts['model_losses'] += chosen['loss']
        counts['prevented'] += bool(base['loss'] and not chosen['loss'])
        counts['introduced'] += bool(chosen['loss'] and not base['loss'])
        counts['classic_turnovers'] += base['turnover']
        counts['model_turnovers'] += chosen['turnover']
        counts['classic_releases'] += base['release_frame'] is not None
        counts['model_releases'] += chosen['release_frame'] is not None
        counts['new_retreats'] += chosen['progress'] < 0 <= base['progress']
        losses.append(chosen['loss'] - base['loss'])
        sources.append(row['source'])
        if changed:
            progress.append(chosen['progress'] - base['progress'])
        choices.append(dict(id=row['id'], source=row['source'], baseline=row['baseline'], selected=selected,
                            **{policy.score_name: risks}, baseline_loss=base['loss'], selected_loss=chosen['loss'],
                            progress_delta=chosen['progress'] - base['progress']))
    return dict(counts, net_prevented=counts['prevented'] - counts['introduced'],
                mean_changed_progress_delta=float(np.mean(progress)) if progress else 0.0,
                loss_rate_delta=float(np.mean(losses)) if losses else None,
                loss_rate_delta_interval95=cluster_interval(losses, sources),
                source_games=len(set(sources)), choices=choices)


def predictive_metrics(rows, policy):
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    y, probabilities = [], []
    for row in rows:
        for branch in row['branches']:
            if branch['label_valid']:
                y.append(branch['loss'])
                probabilities.append(policy.risk(branch['features']))
    return dict(labelled_branches=len(y), positive_branches=sum(y), prevalence=float(np.mean(y)),
                brier=float(brier_score_loss(y, probabilities)),
                roc_auc=float(roc_auc_score(y, probabilities)) if len(set(y)) == 2 else None,
                average_precision=float(average_precision_score(y, probabilities)) if sum(y) else None)


def train(args):
    from importlib.metadata import version
    from sklearn.tree import DecisionTreeClassifier
    from nhl94_ai.evaluation.goalie import source_hashes
    started = time.monotonic()
    fitting_sources = source_hashes()
    output = Path(args.output)
    if output.exists():
        raise ValueError('Refusing to overwrite a retention model')
    training, rows = load_rows(args.train)
    development, dev = load_rows(args.development)
    assert_disjoint(rows, dev)
    names = tuple(rows[0]['branches'][0]['features'])
    x, y, weights = arrays(rows, names)
    if set(y) != {0, 1}:
        raise ValueError('Training requires both retention outcomes')
    candidates = []
    parity_rows = [b['features'] for row in (*rows, *dev) for b in row['branches']]
    parity_x = np.asarray([[row[name] for name in names] for row in parity_rows], dtype=np.float32)
    for depth in DEPTHS:
        for leaf in LEAF_SIZES:
            model = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf, random_state=94)
            model.fit(x, y, sample_weight=weights)
            artifact = export_tree(model, names)
            policy = RetentionFilter(artifact=artifact)
            expected = model.predict_proba(parity_x)[:, list(model.classes_).index(1)]
            actual = np.asarray([policy.risk(row) for row in parity_rows])
            if not np.allclose(actual, expected, rtol=0, atol=1e-14):
                raise RuntimeError('Exported tree differs from sklearn inference')
            for risk in MINIMUM_RISKS:
                for gap in MINIMUM_GAPS:
                    configured = dict(artifact, minimum_risk=risk, minimum_gap=gap)
                    result = score_policy(dev, RetentionFilter(artifact=configured))
                    candidates.append(dict(depth=depth, min_samples_leaf=leaf, minimum_risk=risk,
                                           minimum_gap=gap, artifact=configured,
                                           development={k: v for k, v in result.items() if k != 'choices'}))
    # Development alone selects the configuration. Favor net prevented losses,
    # then fewer turnovers, then fewer overrides and a smaller model. Restrict
    # progress/turnover regressions when a feasible configuration is available.
    feasible = [c for c in candidates if c['development']['model_turnovers'] <= c['development']['classic_turnovers']
                and c['development']['mean_changed_progress_delta'] >= -2
                and c['development']['new_retreats'] == 0]
    def ranking(c):
        d = c['development']
        return (-d['net_prevented'], d['model_turnovers'], d['overrides'], c['depth'], -c['min_samples_leaf'])
    selected = min(feasible or candidates, key=ranking)
    artifact = selected['artifact']
    artifact['training'] = dict(
        task='supervised four-frame movement followed by Classic for a 24-frame outcome',
        algorithm='DecisionTreeClassifier', depth=selected['depth'], min_samples_leaf=selected['min_samples_leaf'],
        training_starts=len(rows), training_branches=len(y), positive_branches=int(sum(y)),
        training_sources=sorted({r['source'] for r in rows}),
        development_sources=sorted({r['source'] for r in dev}),
        training_data_sha256=hashlib.sha256(Path(args.train).read_bytes()).hexdigest(),
        development_data_sha256=hashlib.sha256(Path(args.development).read_bytes()).hexdigest(),
        training_source_hashes=training['source_hashes'], development_source_hashes=development['source_hashes'],
        fitting_source_hashes=fitting_sources,
        versions={name: version(name) for name in ('scikit-learn', 'numpy', 'stable-retro')},
        export_parity_rows=len(parity_rows), source_weighting='equal total weight per physical source game',
        development_feasible=bool(feasible), wall_seconds=time.monotonic() - started)
    if fitting_sources != source_hashes():
        raise RuntimeError('Retention fitting code changed during training')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2) + '\n', encoding='utf-8')
    policy = RetentionFilter(output)
    report = dict(model=str(output.resolve()), sha256=policy.sha256,
                  selected={k: v for k, v in selected.items() if k != 'artifact'},
                  candidates=[{k: v for k, v in c.items() if k != 'artifact'} for c in candidates],
                  development=score_policy(dev, policy), prediction= predictive_metrics(dev, policy))
    output.with_suffix('.training.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report['selected']), flush=True)


def evaluate(args):
    _, rows = load_rows(args.test)
    policy = RetentionFilter(args.model)
    used = set(policy.artifact['training']['training_sources']) | set(policy.artifact['training']['development_sources'])
    if used & {row['source'] for row in rows}:
        raise ValueError('Held-out source games overlap training or development')
    report = dict(model=str(Path(args.model).resolve()), model_sha256=policy.sha256,
                  test_sha256=hashlib.sha256(Path(args.test).read_bytes()).hexdigest(),
                  policy=score_policy(rows, policy), prediction=predictive_metrics(rows, policy))
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report['policy'].items() if k != 'choices'}), flush=True)


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
