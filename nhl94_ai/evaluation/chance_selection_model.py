"""Evaluate existing chance selection models against recorded outcomes."""
from collections import Counter


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
