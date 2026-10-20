"""Evaluate existing one timer admission models against recorded outcomes."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path


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
