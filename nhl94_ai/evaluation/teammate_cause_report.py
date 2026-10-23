"""Summarize retained fork evidence without substituting proxies for outcomes."""
from collections import Counter
import json
from pathlib import Path


def branch_outcome(branch, selected):
    expected = selected['event'].get('receiver')
    release = selected['event'].get('release_frame', selected['capture_frame'])
    passes = [e for e in branch['passes'] if e['release_frame'] >= release-3]
    event = passes[0] if passes else None
    cancellations = []
    for row in branch['rows']:
        for e in row['events']:
            if e.get('reason') == 'receiver-selection-lost':
                cancellations.append({'frame':row['frame'],'slots':row['slots'], 'commitment':e.get('commitment')})
    return {'actions_sha256':branch['actions_sha256'], 'final_ram_sha256':branch['final_ram_sha256'],
            'expected_receiver':expected, 'first_tracked_pass':event,
            'received':bool(event and event['received_frame'] is not None),
            'native_one_timer':bool(event and event['one_timer']),
            'receiver_shot':bool(event and event['shot_frame'] is not None),
            'selection_cancellations':cancellations,
            'known_controller_mismatch_frames':sum(r['receiver_controller_mismatch'] and all(s>=0 for s in r['slots'])
                                                  for r in branch['rows']),
            'score':branch['score'], 'danger_entries_against':branch['measurements']['counts'].get('danger_entries_against',0),
            'receiving_dispatch_frames':sum('receiving' in r['sources'] for r in branch['rows'])}


def summarize(root):
    cases={}
    for path in sorted((root/'native-cases').glob('*/report.json')):
        report=json.loads(path.read_text(encoding='utf-8'))
        assert report['prefix_parity']
        for name,case in report['cases'].items():
            branches={mode:branch_outcome(branch,case['selection']) for mode,branch in case['branches'].items()}
            assert branches['default']==branches['repeat-default']
            cases[name]={'artifact':str(path), 'category':case['selection']['category'],'branches':branches}
    engagement={}
    for label in ('O','D'):
        report=json.loads((root/f'regulation-{label}/report.json').read_text(encoding='utf-8'))
        groups={}
        for name,row in report['matches'].items():
            group=groups.setdefault(name.rsplit('-',1)[0],{'measurements':Counter(),'decision_chain':Counter(),'late_period_cancellations':0})
            result=row['result']
            group['measurements'].update(result['teammate_diagnostics']['counts'])
            group['decision_chain'].update(result['decision_evidence']['counts'])
        engagement[label]=groups
    arbitration_path=root/'arbitration-cases/report.json'
    arbitration={}
    if arbitration_path.exists():
        for name,case in json.loads(arbitration_path.read_text(encoding='utf-8'))['cases'].items():
            arbitration[name]={mode:branch_outcome(branch,case['selection']) for mode,branch in case['branches'].items()}
    return {'reception_cases':cases,'arbitration_cases':arbitration,'regulation_engagement':engagement,
            'limitations':['Chronological selected cases are mechanism probes, not independent strength samples.',
                          '240-frame branch horizon can leave shot follow-through unresolved.',
                          'Support/role execution counts identify applied dispatch, not native success.',
                          'Missing categories were not filled by searching more seeds.']}


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args(); report=summarize(args.root)
    (args.root/'cause-report.json').write_text(json.dumps(report,indent=2)+'\n')
