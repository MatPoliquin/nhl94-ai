"""Audit actual movement overrides against saved native counterfactual traces."""
import argparse
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import pickle

from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.agents.retention_features import attack_sign, features, movement_action
from nhl94_ai.agents.retention_filter import RetentionFilter, SCHEMA
from nhl94_ai.agents.retention_ranker import load_retention_filter
from nhl94_ai.env.factory import make_retro
from nhl94_ai.evaluation.cpu_benchmark import CPU_RAM, cpu_view
from nhl94_ai.evaluation.retention import restore
from nhl94_ai.game.ram import pass_geometry_info, restore_away_control


def check_game(game, model_path=None):
    names = list(game['starts'][0]['branches'][0]['features'])
    # Synthetic preference deliberately exercises overrides; it is not trained
    # or measured for playing strength. Real artifacts can use the same audit.
    artifact = dict(schema=SCHEMA, interval=4, horizon=24, feature_names=names,
                    minimum_risk=0.5, minimum_gap=0.2,
                    tree=[dict(left=1, right=2, feature=names.index('pad_x'), threshold=0.5, risk=0.5),
                          dict(left=-1, right=-1, feature=-2, threshold=-2, risk=0.9),
                          dict(left=-1, right=-1, feature=-2, threshold=-2, risk=0.1)])
    policy = load_retention_filter(model_path) if model_path else RetentionFilter(artifact=artifact)
    env = make_retro(game='NHL94-Genesis-v0', state=game['starts'][0]['state'], num_players=1, goalie_policy='off')
    checked, frames = 0, 0
    try:
        for name, (address, kind) in CPU_RAM.items():
            env.data.set_variable(name, {'address': address, 'type': kind})
        env.reset(seed=game['seed'])
        for row in game['starts']:
            chosen, _ = policy.choose([b['features'] for b in row['branches']], row['baseline'])
            if chosen == row['baseline']:
                continue
            path = Path(row['path'])
            assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
            with gzip.open(path, 'rb') as stream:
                sample = pickle.load(stream)
            restore(env, sample)
            sample = deepcopy(sample)
            model, state, view = (sample[key] for key in ('model', 'root_state', 'view'))
            for command in range(9):
                assert features(view, command) == row['branches'][command]['features']
            owner = view.engine.puck_owner
            before = (model.scheduler.frames, model.scheduler.decisions, model.scheduler.remaining)
            action = policy.apply(model, view, movement_action(row['baseline'], attack_sign(view))[None])[0]
            assert before == (model.scheduler.frames, model.scheduler.decisions, model.scheduler.remaining)
            for expected in row['branches'][chosen]['trace']:
                assert action.tolist() == expected['buttons']
                assert model._last_decision == expected['decision']
                *_, info = env.step(action)
                if row['side'] == 2:
                    info = restore_away_control(env.data, info, controller_prefix='bench', player_prefix='cpu', slots=6)
                view = cpu_view(state, pass_geometry_info(env, info), row['side'])
                player = view.team1.get_player_by_scnum(owner)
                assert view.engine.puck_owner == expected['owner']
                assert [view.puck.x, view.puck.y] == expected['puck_xy']
                assert [player.x, player.y] == expected['carrier_xy']
                frames += 1
                action = ClassicAIV1Model.predict_frame(model, view, frame_skip=4)[0]
            checked += 1
            if checked == 4:
                break
    finally:
        env.close()
    return dict(matchup=game['matchup'], seed=game['seed'], overrides=checked, frames=frames)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', required=True)
    parser.add_argument('--model')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    data = json.loads(Path(args.report).read_text(encoding='utf-8'))
    results = [check_game(game, args.model) for game in data['games']]
    assert sum(row['overrides'] for row in results) > 0, 'No actual overrides were exercised'
    report = dict(passed=True, synthetic=args.model is None, games=results)
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('PASS', sum(row['overrides'] for row in results), 'overrides;',
          sum(row['frames'] for row in results), 'native frames')


if __name__ == '__main__':
    main()
