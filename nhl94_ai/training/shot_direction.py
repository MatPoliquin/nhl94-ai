"""Fit nine-way shot aiming on bounded attack returns, with grouped validation."""
import argparse
from collections import Counter
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path

from nhl94_ai.evaluation.shot_direction import MODEL_VERSION, PADS
from nhl94_ai.training.shot_placement import estimate, fit_logistic


def read_cases(path):
    data = Path(path).read_bytes()
    report = json.loads(gzip.decompress(data) if data[:2] == b'\x1f\x8b' else data)
    if report['protocol'] != 'native-shot-direction-v1':
        raise ValueError('Expected native directional-shot alternatives')
    rows = []
    for match in report['matches']:
        if not match['completed']:
            raise ValueError('Incomplete collection period')
        for case in match['cases']:
            if ({tuple(b['pad']) for b in case['branches']} != set(PADS)
                    or len(case['branches']) != 9 or case['branches'][0]['pad'] != case['baseline']):
                raise ValueError('Require all nine inputs and original baseline first')
            if any(b['outcome']['goal'] not in (0, 1) or b['outcome']['primary_goal'] not in (0, 1)
                   for b in case['branches']):
                raise ValueError('Expected binary original-shot and attack-goal labels')
            rows.append({**case, 'seed': match['seed'], 'matchup': match['matchup']})
    if not rows:
        raise ValueError('No shot cases')
    return rows, {'file': str(path), 'sha256': hashlib.sha256(data).hexdigest(),
                  'cases': len(rows), 'seeds': sorted({m['seed'] for m in report['matches']})}


def selected_index(model, case):
    if model['kind'] == 'fixed-height':
        pad = case['baseline'][0], model['height']
        return next(i for i, b in enumerate(case['branches']) if tuple(b['pad']) == pad)
    pool = [i for i, b in enumerate(case['branches'])
            if model['scope'] == 'nine' or b['pad'][0] == case['baseline'][0]]
    if any(case['branches'][i]['features'] is None for i in pool):
        return 0
    scores = {i: estimate(model, case['branches'][i]['features']) for i in pool}
    best = max(pool, key=scores.__getitem__)
    return best if scores[best] > scores[0] + model['min_gain'] else 0


def evaluate(model, cases):
    counts, decisions = Counter(), []
    for case in cases:
        index = selected_index(model, case)
        old, new = case['branches'][0], case['branches'][index]
        a, b = old['outcome'], new['outcome']
        counts.update(cases=1, legacy_attack_goals=a['goal'], selected_attack_goals=b['goal'],
                      legacy_primary_goals=a['primary_goal'], selected_primary_goals=b['primary_goal'],
                      legacy_against=a['goals_against'], selected_against=b['goals_against'],
                      overrides=index != 0, height_overrides=new['pad'][1] != 0,
                      horizontal_overrides=new['pad'][0] != old['pad'][0],
                      oracle_attack_goals=max(x['outcome']['goal'] for x in case['branches']),
                      oracle_primary_goals=max(x['outcome']['primary_goal'] for x in case['branches']))
        decisions.append({'seed': case['seed'], 'matchup': case['matchup'], 'frame': case['frame'],
                          'baseline': old['pad'], 'selected': new['pad'],
                          'baseline_attack_goal': a['goal'], 'selected_attack_goal': b['goal'],
                          'baseline_primary_goal': a['primary_goal'], 'selected_primary_goal': b['primary_goal']})
    return {'totals': dict(counts), 'decisions': decisions}


def fit(training, validation, train_info, val_info):
    if set(train_info['seeds']) & set(val_info['seeds']):
        raise ValueError('Keep complete seeds and both assignments out of the other split')
    usable = [c for c in training if all(b['features'] is not None for b in c['branches'])]
    if not usable:
        raise ValueError('No complete feature sets for learning')
    learned = fit_logistic(usable, ridge=10)
    learned.update(version=MODEL_VERSION, kind='logistic')
    candidates = []
    for height, name in ((0, 'legacy'), (1, 'fixed-high'), (-1, 'fixed-low')):
        candidates.append((name, {'version': MODEL_VERSION, 'kind': 'fixed-height', 'height': height}))
    for scope in ('height-only', 'nine'):
        for threshold in (0, .02, .05, .1, .2, 1):
            model = {**deepcopy(learned), 'scope': scope, 'min_gain': threshold}
            candidates.append((f'{scope}:{threshold}', model))
    rows = [{'name': name, 'model': model, 'validation': evaluate(model, validation)} for name, model in candidates]
    best = max(rows, key=lambda row: (row['validation']['totals']['selected_attack_goals']
                                     - row['validation']['totals']['selected_against'],
                                     -row['validation']['totals']['overrides']))
    model = {**best['model'], 'selected_name': best['name'], 'training': train_info, 'validation': val_info,
             'target': 'Goal during bounded attack, including friendly rebounds; not calibrated game-wide xG.'}
    report = {'protocol': 'shot-direction-grouped-fit-v1', 'training': train_info, 'validation': val_info,
              'training_feature_complete_cases': len(usable), 'selected_name': best['name'],
              'selected_model': model, 'training_performance': evaluate(model, training),
              'validation_performance': best['validation'],
              'alternatives': [{'name': row['name'], 'model': row['model'],
                                'training_totals': evaluate(row['model'], training)['totals'],
                                'validation_totals': row['validation']['totals']} for row in rows],
              'limitations': ['Fixed ridge strength 10; validation chooses scope/threshold or a fixed control.',
                              'Missing features retain original aim for learned policies and remain in evaluations.',
                              'Paired alternatives share state/RNG; nearby cases remain correlated.',
                              'Validation selection requires subsequent fresh complete-period testing.']}
    return model, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', required=True)
    parser.add_argument('--validation', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    training, train_info = read_cases(args.training)
    validation, val_info = read_cases(args.validation)
    model, report = fit(training, validation, train_info, val_info)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(model, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    Path(args.report).write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({'selected': report['selected_name'],
                      'training': report['training_performance']['totals'],
                      'validation': report['validation_performance']['totals']}, indent=2))


if __name__ == '__main__':
    main()
