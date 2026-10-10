"""Fit one auditable carry/finish rule using paired native attack outcomes.

This is a small treatment-effect experiment, not calibrated xG. Train/validation
splits keep both assignments and every case from a seed together. Validation
does not select thresholds or refit the rule; complete-period testing follows.
"""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

import numpy as np


FEATURES = ('animation_locked', 'b_down', 'c_down', 'depth', 'vy', 'defender_distance',
            'goalie_distance', 'shot_geometry_quality', 'timer_clearance')


def cases(report):
    return [{**case, 'seed': row['seed'], 'matchup': row['matchup']}
            for row in report['matches'] for case in row['cases']]


def select(case, rule):
    if rule is None:
        return case['baseline']
    value = case['features'].get(rule['feature'])
    applies = (case['baseline'] == rule['baseline'] and value is not None
               and (value <= rule['threshold']) == rule['less_equal'])
    return rule['action'] if applies else case['baseline']


def reward(branch):
    return branch['attack_goal'] - branch['goals_against']


def evaluate(rows, rule):
    counts, deltas = Counter(), {}
    for case in rows:
        selected = select(case, rule)
        old, new = case['branches'][case['baseline']], case['branches'][selected]
        counts.update(cases=1, overrides=selected != case['baseline'],
                      legacy_attack_goals=old['attack_goal'], selected_attack_goals=new['attack_goal'],
                      legacy_goals=old['goals_for'], selected_goals=new['goals_for'],
                      legacy_against=old['goals_against'], selected_against=new['goals_against'],
                      legacy_shots=old['shots'], selected_shots=new['shots'],
                      paired_reward_gain=reward(new)-reward(old))
        deltas.setdefault(case['seed'], []).append(reward(new)-reward(old))
    return {'totals': dict(counts), 'seed_reward_gains': {
        str(seed): sum(values) for seed, values in sorted(deltas.items())}}


def fit(rows, min_cases=20, min_seeds=5):
    candidates = []
    for baseline in ('shoot', 'one-timer'):
        group = [case for case in rows if case['baseline'] == baseline]
        for feature in FEATURES:
            values = [case['features'][feature] for case in group if feature in case['features']]
            if not values:
                continue
            thresholds = sorted(set(float(x) for x in np.quantile(values, (0, .25, .5, .75, 1))))
            for threshold in thresholds:
                for less_equal in (True, False):
                    rule = {'baseline': baseline, 'action': 'carry', 'feature': feature,
                            'threshold': threshold, 'less_equal': less_equal}
                    changed = [case for case in group if select(case, rule) != baseline]
                    if len(changed) < min_cases or len({case['seed'] for case in changed}) < min_seeds:
                        continue
                    result = evaluate(rows, rule)
                    gain = result['totals']['paired_reward_gain']
                    if gain > 0:
                        candidates.append({'rule': rule, 'result': result})
    # Prefer the largest total paired gain; smaller interventions break ties.
    candidates.sort(key=lambda item: (-item['result']['totals']['paired_reward_gain'],
                                      item['result']['totals']['overrides']))
    return candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--validation-seed', type=int, default=25915)
    parser.add_argument('--min-cases', type=int, default=20)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    data = Path(args.data).read_bytes()
    report = json.loads(gzip.decompress(data) if args.data.endswith('.gz') else data)
    if report['protocol'] != 'native-chance-selection-v1' or args.min_cases < 1:
        raise ValueError('Expected native chance replay and positive minimum coverage')
    rows = cases(report)
    train = [case for case in rows if case['seed'] < args.validation_seed]
    validation = [case for case in rows if case['seed'] >= args.validation_seed]
    if not train or not validation:
        raise ValueError('Need nonempty, disjoint training/validation seeds')
    candidates = fit(train, args.min_cases)
    rule = candidates[0]['rule'] if candidates else None
    result = {'version': 'native-chance-rule-v1', 'rule': rule, 'settings': vars(args),
              'features_searched': FEATURES, 'training': evaluate(train, rule),
              'validation': evaluate(validation, rule), 'top_training_rules': candidates[:10],
              'split': {'training_seeds': sorted({case['seed'] for case in train}),
                        'validation_seeds': sorted({case['seed'] for case in validation})},
              'limitations': ['One threshold, trained on attack goals minus bounded goals conceded.',
                              'Rule selected only on training; validation is reported without refitting.',
                              'Cases within a seed and nearby branches are correlated.',
                              'A positive local effect does not establish full-period benefit.']}
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({key: result[key] for key in ('rule', 'training', 'validation')}, indent=2))


if __name__ == '__main__':
    main()
