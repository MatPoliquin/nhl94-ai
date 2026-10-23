"""Evaluate existing shot placement models against recorded outcomes."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from nhl94_ai.agents.shot_placement import ShotPlacement
from nhl94_ai.models.shot_scoring import estimate


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



def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--report', required=True)
    args = parser.parse_args(argv)
    placement = ShotPlacement(args.model)
    cases, info = read_cases(args.data)
    used = set(placement.model['training']['seeds']) | set(placement.model['validation']['seeds'])
    if used & set(info['seeds']):
        raise ValueError('Held-out evaluation seeds overlap model development')
    report = {'protocol': 'shot-placement-heldout-v1', 'data': info,
              'model_sha256': placement.sha256, 'evaluation': evaluate(placement.model, cases),
              'limitations': ['Primary-shot outcomes on sampled trajectories, not full-game strength.',
                              'Incomplete/censored alternatives are excluded; scores are not calibrated game-wide xG.']}
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(report['evaluation']['totals'], indent=2))


if __name__ == '__main__':
    main()
