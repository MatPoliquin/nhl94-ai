"""Fit/evaluate a portable logistic shot model using grouped native alternatives."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from nhl94_ai.agents.shot_placement import MODEL_VERSION, ShotPlacement


def read_cases(path):
    data = Path(path).read_bytes()
    report = json.loads(gzip.decompress(data) if data[:2] == b'\x1f\x8b' else data)
    if report['protocol'] != 'native-primary-shot-alternatives-v3':
        raise ValueError('Expected native primary-shot alternatives')
    cases, excluded = [], Counter()
    for match in report['matches']:
        if not match['completed']:
            raise ValueError('Do not fit from incomplete baseline periods')
        for case in match['cases']:
            for branch in case['branches']:
                result = branch['outcome']
                if result['held_c_frames'] != min(branch['hold'], result['frames']):
                    raise ValueError('Native shot alternative did not execute its requested hold')
            if not all(branch['outcome']['resolved'] for branch in case['branches']):
                excluded['censored-case'] += 1
                continue
            cases.append({**case, 'seed': match['seed'], 'matchup': match['matchup']})
    if not cases:
        raise ValueError('No completely resolved alternative sets')
    return cases, {'file': str(path), 'sha256': hashlib.sha256(data).hexdigest(),
                   'cases': len(cases), 'excluded': dict(excluded),
                   'seeds': sorted({match['seed'] for match in report['matches']})}


def fit_logistic(cases, ridge=10.0):
    names = list(cases[0]['branches'][0]['features'])
    rows, labels, weights = [], [], []
    for case in cases:
        for branch in case['branches']:
            if list(branch['features']) != names:
                raise ValueError('Inconsistent shot feature schema')
            rows.append([branch['features'][name] for name in names])
            labels.append(branch['outcome']['goal'])
            weights.append(1/len(case['branches']))
    values, labels, sample_weights = np.asarray(rows), np.asarray(labels), np.asarray(weights)
    if not np.isfinite(values).all() or not set(labels).issubset({0, 1}):
        raise ValueError('Shot features and binary goal labels must be finite')
    mean = np.average(values, axis=0, weights=sample_weights)
    scale = np.sqrt(np.average((values-mean)**2, axis=0, weights=sample_weights))
    scale = np.where(scale < 1e-6, 1, scale)
    design = np.column_stack((np.ones(len(values)), (values-mean)/scale))
    beta = np.zeros(design.shape[1])
    rate = float(np.average(labels, weights=sample_weights))
    beta[0] = math.log(max(1e-6, rate)/max(1e-6, 1-rate))
    penalty = np.diag([0.0, *([ridge]*len(names))])
    for _ in range(100):
        probability = 1/(1+np.exp(-np.clip(design @ beta, -40, 40)))
        curvature = sample_weights*np.maximum(probability*(1-probability), 1e-8)
        gradient = design.T @ (sample_weights*(probability-labels)) + penalty @ beta
        hessian = design.T @ (curvature[:, None]*design) + penalty + np.eye(len(beta))*1e-9
        step = np.linalg.solve(hessian, gradient)
        beta -= step
        if float(np.max(np.abs(step))) < 1e-8:
            break
    if not np.isfinite(beta).all():
        raise RuntimeError('Nonfinite shot model fit')
    return {'version': MODEL_VERSION, 'features': names, 'mean': mean.tolist(), 'scale': scale.tolist(),
            'weights': beta[1:].tolist(), 'bias': float(beta[0]), 'min_gain': 0.0,
            'fit': {'method': 'ridge-logistic', 'ridge': ridge, 'case_weight': 1,
                    'cases': len(cases), 'alternatives': len(rows), 'observed_goal_rate': rate}}


def estimate(model, features):
    row = np.asarray([features[name] for name in model['features']])
    z = float(((row-np.asarray(model['mean']))/np.asarray(model['scale'])) @ np.asarray(model['weights']) + model['bias'])
    return 1/(1+math.exp(-max(-40, min(40, z))))


def evaluate(model, cases, *, threshold=None):
    threshold = model['min_gain'] if threshold is None else threshold
    counts, predicted, observed, decisions = Counter(), [], [], []
    for case in cases:
        branches = case['branches']
        scores = [estimate(model, branch['features']) for branch in branches]
        index = max(range(len(scores)), key=scores.__getitem__)
        index = index if scores[index] > scores[0] + threshold else 0
        old, selected = branches[0], branches[index]
        counts.update(cases=1, legacy_goals=old['outcome']['goal'], selected_goals=selected['outcome']['goal'],
                      oracle_goals=max(branch['outcome']['goal'] for branch in branches), overrides=index != 0,
                      aim_overrides=selected['aim'] != old['aim'], hold_overrides=selected['hold'] != old['hold'])
        predicted.extend(scores)
        observed.extend(branch['outcome']['goal'] for branch in branches)
        decisions.append({'matchup': case['matchup'], 'seed': case['seed'], 'frame': case['frame'],
                          'legacy': (old['aim'], old['hold']), 'selected': (selected['aim'], selected['hold']),
                          'legacy_goal': old['outcome']['goal'], 'selected_goal': selected['outcome']['goal'],
                          'gain_estimate': scores[index]-scores[0]})
    predicted, observed = np.asarray(predicted), np.asarray(observed)
    return {'totals': dict(counts), 'brier_all_alternatives': float(np.mean((predicted-observed)**2)),
            'mean_estimate': float(predicted.mean()), 'observed_goal_rate': float(observed.mean()),
            'decisions': decisions, 'threshold': threshold}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    fit = sub.add_parser('fit')
    fit.add_argument('--train', required=True)
    fit.add_argument('--validation', required=True)
    fit.add_argument('--output', required=True)
    fit.add_argument('--report', required=True)
    evaluation = sub.add_parser('evaluate')
    evaluation.add_argument('--model', required=True)
    evaluation.add_argument('--data', required=True)
    evaluation.add_argument('--report', required=True)
    args = parser.parse_args()
    if args.command == 'fit':
        training, train_info = read_cases(args.train)
        validation, val_info = read_cases(args.validation)
        if set(train_info['seeds']) & set(val_info['seeds']):
            raise ValueError('Training and validation seeds must be disjoint, including both team assignments')
        # Fixed feature model and regularization; validation only chooses how much
        # estimated gain is required to override the established shot execution.
        model = fit_logistic(training, ridge=10)
        choices = [evaluate(model, validation, threshold=threshold) for threshold in (0, .02, .05, .1, .2, 1)]
        chosen = max(choices, key=lambda row: (row['totals']['selected_goals'], -row['totals']['overrides'], row['threshold']))
        model['min_gain'] = chosen['threshold']
        model['training'], model['validation'] = train_info, val_info
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(model, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        report = {'protocol': 'shot-placement-grouped-fit-v1', 'training': train_info, 'validation': val_info,
                  'threshold_search': choices, 'selected_threshold': chosen['threshold'],
                  'training_performance': evaluate(model, training), 'validation_performance': chosen,
                  'model_sha256': hashlib.sha256(Path(args.output).read_bytes()).hexdigest()}
        print(json.dumps({'selected_threshold': chosen['threshold'], 'validation': chosen['totals'],
                          'training': report['training_performance']['totals']}, indent=2))
    else:
        placement = ShotPlacement(args.model)
        cases, info = read_cases(args.data)
        used = set(placement.model['training']['seeds']) | set(placement.model['validation']['seeds'])
        if used & set(info['seeds']):
            raise ValueError('Held-out evaluation seeds overlap model development')
        report = {'protocol': 'shot-placement-heldout-v1', 'data': info,
                  'model_sha256': placement.sha256, 'evaluation': evaluate(placement.model, cases)}
        print(json.dumps(report['evaluation']['totals'], indent=2))
    report['limitations'] = ['Primary-shot goals on sampled legacy trajectories, not full-game strength.',
                            'Goal before new possession or the bounded horizon; timeout is zero and rebound recoveries are excluded.',
                            'All alternatives from any incomplete/censored case are excluded.',
                            'Scores are learned estimates, not established calibrated game-wide xG.']
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
