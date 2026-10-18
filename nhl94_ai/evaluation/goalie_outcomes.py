"""Fresh-snapshot outcome audit and frozen full-match evaluation for goalie v2."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import hashlib
import json
import multiprocessing
from pathlib import Path
from types import SimpleNamespace

from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.agents.goalie import GoalieController
from nhl94_ai.agents.goalie_outcome_planner import OutcomeGoalie
from nhl94_ai.agents.goalie_planner import PlanningGoalie
from nhl94_ai.agents.learned_goalie import replace_goalie
from nhl94_ai.env.factory import make_retro
from nhl94_ai.evaluation.cpu_benchmark import CPU_RAM, cpu_view
from nhl94_ai.evaluation.goalie import load_start, source_hashes
from nhl94_ai.game.ram import pass_geometry_info, restore_away_control


def replay(row):
    source, meta = load_start(row['path'])
    env = make_retro(game='NHL94-Genesis-v0', state=row['state'], num_players=1, goalie_policy='selective')
    for name, (address, kind) in CPU_RAM.items():
        env.data.set_variable(name, {'address': address, 'type': kind})
    env.reset(seed=row['seed'])
    branches = {}
    try:
        for mode in ('baseline', 'search-24', 'outcome-24', 'baseline-repeat'):
            sample = deepcopy(source)
            env.em.set_state(sample['snapshot'])
            env.data.update_ram()
            assert hashlib.sha256(bytes(env.get_ram())).hexdigest() == row['ram_sha256']
            info = env.data.lookup_all()
            model = ClassicAIV1Model(SimpleNamespace(action_type='FILTERED', goalie_policy='selective'))
            model.__dict__.update(vars(sample['model']))
            replacement = (GoalieController('selective') if mode.startswith('baseline') else
                           OutcomeGoalie() if mode == 'outcome-24' else PlanningGoalie(planning=mode))
            model.goalie = replace_goalie(model.goalie, replacement)
            state = sample['root_state']
            view = cpu_view(state, pass_geometry_info(env, info), meta['side'])
            initial = view.team2.stats.score
            digest = hashlib.sha256()
            contact, caught, recovered, conceded = None, None, None, False
            predictions, events = [], []
            previous_owner = view.engine.puck_owner
            previous_touch = env.data.memory.extract(0xFFBF7A, '>i2')
            base = 0xFFB04A+view.team1.defense_goalie*0x80
            previous_cooldown = env.data.memory.extract(base+0x5E, '|u1')
            ended = 'timeout'
            for frame in range(180):
                before = bytes(env.get_ram())
                calls = model.goalie.metrics.get('planner_calls', 0)
                action = model.predict_frame(view, 4)[0]
                assert bytes(env.get_ram()) == before, 'Policy modified gameplay RAM'
                if model.goalie.metrics.get('planner_calls', 0) > calls:
                    predictions.append({'frame': frame, 'puck': [view.puck.x, view.puck.y, view.puck.height],
                                        **deepcopy(model.goalie.diagnostics.get('planner', {}))})
                digest.update(action.tobytes())
                *_, info = env.step(action)
                if meta['side'] == 2:
                    info = restore_away_control(env.data, info, controller_prefix='bench', player_prefix='cpu', slots=6)
                view = cpu_view(state, pass_geometry_info(env, info), meta['side'])
                owner = view.engine.puck_owner
                touch = env.data.memory.extract(0xFFBF7A, '>i2')
                cooldown = env.data.memory.extract(base+0x5E, '|u1')
                if (contact is None and touch == view.team1.defense_goalie
                        and (touch != previous_touch or cooldown > previous_cooldown or owner == touch)):
                    contact = frame+1
                    events.append({'frame': frame+1, 'event': 'goalie-touch', 'owner': owner})
                if owner == view.team1.defense_goalie and caught is None:
                    caught = frame+1
                    events.append({'frame': frame+1, 'event': 'goalie-catch'})
                if owner >= 0 and owner != previous_owner:
                    events.append({'frame': frame+1, 'event': 'possession', 'owner': owner,
                                   'team': 'own' if view.team1.owns_scnum(owner) else 'opponent'})
                    if (recovered is None and contact is not None and owner != view.team1.defense_goalie
                            and owner != view.team2.defense_goalie):
                        recovered = 'own' if view.team1.owns_scnum(owner) else 'opponent'
                previous_owner, previous_touch, previous_cooldown = owner, touch, cooldown
                if view.team2.stats.score > initial:
                    conceded, ended = True, 'goal-against'
                    break
                if view.engine.clock_stopped:
                    ended = 'stoppage'
                    break
            branches[mode] = {'frames': frame+1, 'ended': ended, 'goal_against': conceded,
                'first_contact_frame': contact, 'catch_frame': caught, 'first_skater_recovery': recovered,
                'events': events, 'predictions': predictions, 'actions_sha256': digest.hexdigest(),
                'planner_changes': model.goalie.metrics.get('planner_changes', 0)}
    finally:
        env.close()
    assert branches['baseline'] == branches['baseline-repeat']
    return {'start': row, 'branches': branches}


def audit(args):
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    banks = [json.loads(Path(path).read_text(encoding='utf-8')) for path in args.banks]
    starts = [row for bank in banks for row in bank['starts'] if row['released_threat']]
    assert len({row['path'] for row in starts}) == len(starts)
    hashes = source_hashes()
    results = []
    path = destination/'policy-replays.jsonl'
    if path.exists():
        raise ValueError('Refusing to overwrite native replay results')
    with path.open('w', encoding='utf-8') as stream, ProcessPoolExecutor(max_workers=args.workers,
            mp_context=multiprocessing.get_context('fork')) as executor:
        for row in executor.map(replay, starts):
            results.append(row)
            stream.write(json.dumps(row, sort_keys=True)+'\n')
            stream.flush()
            if len(results) % 16 == 0:
                print(json.dumps({'completed': len(results), 'total': len(starts)}), flush=True)
    assert hashes == source_hashes(), 'Sources changed during native audit'
    summary = {}
    for mode in ('baseline', 'search-24', 'outcome-24'):
        rows = [row['branches'][mode] for row in results]
        summary[mode] = {'starts': len(rows), 'goals_against': sum(r['goal_against'] for r in rows),
            'goalie_contacts': sum(r['first_contact_frame'] is not None for r in rows),
            'goalie_catches': sum(r['catch_frame'] is not None for r in rows),
            'own_skater_recoveries': sum(r['first_skater_recovery'] == 'own' for r in rows),
            'opponent_skater_recoveries': sum(r['first_skater_recovery'] == 'opponent' for r in rows),
            'planner_changes': sum(r['planner_changes'] for r in rows)}
    report = {'protocol': 'goalie-outcome-policy-replays-v2', 'sources': hashes,
        'banks': args.banks, 'starts': len(starts),
        'source_games': len({(r['matchup'], r['seed']) for r in starts}),
        'summary': summary, 'all_baseline_repeats_exact': True, 'policy_ram_unchanged': True,
        'scope': 'Paired 180-frame CPU snapshot diagnostics; overlapping starts are not independent games. Catches do not end the rollout; timeouts are not saves.'}
    (destination/'policy-replays-report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('replays',), default='replays')
    parser.add_argument('--banks', nargs='*', default=[])
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    audit(args)


if __name__ == '__main__':
    main()
