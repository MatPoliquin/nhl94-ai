"""Evaluate existing shot direction models against recorded outcomes."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
from nhl94_ai.evaluation.shot_direction import PADS
from nhl94_ai.models.shot_scoring import estimate


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
