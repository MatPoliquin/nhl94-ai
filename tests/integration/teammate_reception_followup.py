"""Verify the scoped correction against retained full-history native forks."""
import argparse
import gzip
import json
from pathlib import Path
import pickle

from nhl94_ai.evaluation.teammate_cases import branch
from nhl94_ai.evaluation.teammates import DEFAULT_STATE, make_match_env, prepare_match, source_hash
from nhl94_ai.evaluation.cpu_benchmark import CPU_RAM
from nhl94_ai.agents.teammate_receiving import SharedReceiving


def verify(root, output):
    results = {}
    for path in sorted(root.glob('X-*/report.json')):
        report = json.loads(path.read_text(encoding='utf-8'))
        for name, case in report['cases'].items():
            snapshot = pickle.loads(gzip.decompress((path.parent/(name+'.pickle.gz')).read_bytes()))
            # Preserve every pending action/history; migrate only the old option
            # to the current default before replaying the retained counterexample.
            snapshot['agent'].receiving.preserve_transient = SharedReceiving().preserve_transient
            # These histories predate the optional anticipation feature.
            snapshot['agent'].anticipate_cue = False
            snapshot['agent'].receiving.anticipate_cue = False
            settings = case['selection']['settings']
            side = 1 if settings['side']=='home' else 2
            with make_match_env(DEFAULT_STATE, team_size=settings['team_size']) as env:
                for key,(address,kind) in CPU_RAM.items():
                    env.data.set_variable(key, {'address':address,'type':kind})
                prepare_match(env,seed=settings['seed'],seconds=settings['seconds'],side=side,team_size=settings['team_size'])
                actual = branch(env,snapshot,'default')
                repeat = branch(env,snapshot,'default')
            expected = case['branches']['preserve-transient-commitment']
            old = case['branches']['default']
            assert actual == repeat
            for key in ('actions_sha256','final_ram_sha256','passes','score'):
                assert actual[key] == expected[key], (name,key)
            for row in actual['rows']:
                if all(slot < 0 for slot in row['slots']):
                    assert not any(row['pads']), (name,row['frame'])
            results[name] = {'exact_counterfactual_inputs':True,'exact_counterfactual_final_ram':True,
                             'old_score':old['score'],'new_score':actual['score'],
                             'old_passes':old['passes'],'new_passes':actual['passes'],
                             'old_danger':old['measurements']['counts'].get('danger_entries_against',0),
                             'new_danger':actual['measurements']['counts'].get('danger_entries_against',0)}
    assert len(results) == 8
    key=results['X-three-home-controlled-failure']
    assert key['old_passes'][0]['received_frame'] is None
    assert key['new_passes'][0]['received_frame'] is not None
    assert key['new_passes'][0]['shot_frame'] is not None
    assert key['old_score']==[0,0] and key['new_score']==[1,0]
    output.write_text(json.dumps({'source_sha256':source_hash(),'cases':results},indent=2)+'\n')
    print('PASS',len(results),'native full-history corrections/controls; transient inputs remain neutral')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();verify(args.cases,args.output)
