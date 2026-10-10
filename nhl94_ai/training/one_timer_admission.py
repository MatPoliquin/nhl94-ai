"""Fit one conservative admission threshold on paired native attack returns.

All rule selection uses training only. Validation evaluates the frozen winner;
it never chooses a different gate, receiver, feature, threshold or inequality.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


FEATURES = ('contact_depth', 'contact_x', 'flight', 'margin', 'shot_value_gain', 'lateral_pass',
            'receiver_vx', 'receiver_vy', 'goalie_vx', 'goalie_lateral_distance',
            'carrier_animation_locked', 'receiver_animation_locked', 'baseline_shoot', 'baseline_pass')
VERSION = 'native-one-timer-admission-rule-v1'


def read_cases(path):
    data = Path(path).read_bytes()
    report = json.loads(gzip.decompress(data) if data[:2] == b'\x1f\x8b' else data)
    if report['protocol'] != 'native-one-timer-admission-v1':
        raise ValueError('Expected single-gate native one-timer alternatives')
    cases = []
    for match in report['matches']:
        if not match['completed']:
            raise ValueError('Incomplete collection period')
        for case in match['cases']:
            if not case['branches'] or any(b['gate'] not in ('window', 'position') for b in case['branches']):
                raise ValueError('Expected positional-gate alternatives')
            for outcome in [case['baseline'], *(b['outcome'] for b in case['branches'])]:
                if any(outcome[key] not in (0, 1) for key in ('attack_goal', 'goals_for', 'goals_against')):
                    raise ValueError('Expected bounded binary native outcomes')
            cases.append({**case, 'seed': match['seed'], 'matchup': match['matchup']})
    if not cases:
        raise ValueError('No sampled admission alternatives')
    return cases, {'sha256': hashlib.sha256(data).hexdigest(), 'cases': len(cases),
                   'seeds': sorted({m['seed'] for m in report['matches']}),
                   'matchups': sorted({m['matchup'] for m in report['matches']})}


def candidate(branches, gate):
    pool = [row for row in branches if row['gate'] == gate]
    return min(pool, key=lambda row: (-row['option']['shot_value'], -row['option']['margin'], row['option']['slot'])) if pool else None


def select(case, rule):
    if rule is None:
        return None
    branch = candidate(case['branches'], rule['gate'])
    if branch is None or rule.get('feature') is None:
        return branch
    value = (branch['features'] or {}).get(rule['feature'])
    return branch if value is not None and (value <= rule['threshold']) == rule['less_equal'] else None


def reward(outcome):
    return outcome['attack_goal'] - outcome['goals_against']


def evaluate(cases, rule):
    counts, decisions = Counter(), []
    for case in cases:
        chosen = select(case, rule)
        old, new = case['baseline'], chosen['outcome'] if chosen is not None else case['baseline']
        counts.update(cases=1, overrides=chosen is not None, legacy_attack_goals=old['attack_goal'],
                      selected_attack_goals=new['attack_goal'], legacy_against=old['goals_against'],
                      selected_against=new['goals_against'], legacy_continuation_goals=old['goals_for'],
                      selected_continuation_goals=new['goals_for'], paired_reward_gain=reward(new)-reward(old),
                      selected_intended_one_timers=new['intended_one_timer'],
                      selected_intended_one_timer_goals=new['intended_one_timer_goal'])
        decisions.append({'seed': case['seed'], 'matchup': case['matchup'], 'frame': case['frame'],
                          'baseline': case['baseline_decision'],
                          'selected_slot': chosen['option']['slot'] if chosen else None,
                          'gate': chosen['gate'] if chosen else None,
                          'legacy_reward': reward(old), 'selected_reward': reward(new)})
    return {'totals': dict(counts), 'decisions': decisions}


def fit(training, *, min_cases=20, min_seeds=5, min_gain=3):
    proposals = []
    for gate in ('window', 'position'):
        for feature in FEATURES:
            values = sorted({b['features'][feature] for c in training
                             if (b := candidate(c['branches'], gate)) is not None
                             and b['features'] is not None and feature in b['features']})
            if not values:
                continue
            thresholds = sorted(set(float(x) for x in np.quantile(values, (.25, .5, .75))))
            # Binary readiness/context indicators need a separating threshold.
            if values == [0, 1]:
                thresholds = [.5]
            for threshold in thresholds:
                for less_equal in (True, False):
                    rule = {'gate': gate, 'feature': feature, 'threshold': threshold, 'less_equal': less_equal}
                    affected = [c for c in training if select(c, rule) is not None]
                    if len(affected) < min_cases or len({c['seed'] for c in affected}) < min_seeds:
                        continue
                    result = evaluate(training, rule)
                    if result['totals']['paired_reward_gain'] >= min_gain:
                        proposals.append({'rule': rule, 'training': result})
    proposals.sort(key=lambda row: (-row['training']['totals']['paired_reward_gain'],
                                    row['training']['totals']['overrides'], json.dumps(row['rule'], sort_keys=True)))
    return proposals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', required=True)
    parser.add_argument('--validation', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    training, train_info = read_cases(args.training)
    validation, val_info = read_cases(args.validation)
    if set(train_info['seeds']) & set(val_info['seeds']):
        raise ValueError('Keep complete seeds and both team assignments in separate splits')
    proposals = fit(training)
    rule = proposals[0]['rule'] if proposals else None
    supported = fit(training, min_gain=-2*len(training))
    result = {'version': VERSION, 'rule': rule, 'split': {'training': train_info, 'validation': val_info},
              'selection': {'features': FEATURES, 'quantiles': [.25, .5, .75], 'min_cases': 20,
                            'min_seeds': 5, 'min_training_reward_gain': 3,
                            'target': 'Attack goal minus opponent continuation goal within the bounded replay.',
                            'receiver_choice': 'Original shot-value, margin, slot ranking within the single relaxed gate.'},
              'training': evaluate(training, rule), 'validation': evaluate(validation, rule),
              'top_training_rules': [{'rule': p['rule'], 'totals': p['training']['totals']} for p in proposals[:10]],
              'supported_rule_count': len(supported),
              'best_supported_training_rules': [{'rule': p['rule'], 'totals': p['training']['totals']}
                                                for p in supported[:5]],
              'fixed_gate_controls': {split: {gate: evaluate(cases, {'gate': gate})['totals']
                                             for gate in ('window', 'position')}
                                      for split, cases in (('training', training), ('validation', validation))},
              'limitations': ['Training alone chooses one rule; validation is a fixed-policy check.',
                              'Threshold search can overfit small correlated samples; fresh complete periods are required.',
                              'The diagnostic action can replace a preferred normal shot, so this is not merely deleting a gate.',
                              'Missing features keep Classic. Native executability is verified separately from eventual return.']}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({'rule': rule, 'training': result['training']['totals'],
                      'validation': result['validation']['totals'], 'controls': result['fixed_gate_controls']}, indent=2))


if __name__ == '__main__':
    main()
