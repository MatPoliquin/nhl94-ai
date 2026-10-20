"""Evaluate existing retention ranking models against recorded outcomes."""
import hashlib
import json
from pathlib import Path
import numpy as np
from nhl94_ai.agents.retention_ranker import RetentionRanker
from nhl94_ai.evaluation.retention_model import load_rows, score_policy


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


def compact(result):
    return {key: value for key, value in result.items() if key != 'choices'}


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



def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    evaluate(parser.parse_args(argv))


if __name__ == '__main__':
    main()
