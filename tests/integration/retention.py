"""Compare recorded baseline branches with the actual unbranched CPU games."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.evaluation import cpu_benchmark


def check_game(game):
    original = ClassicAIV1Model.predict_frame
    trace = []

    def predict(model, state, *args, **kwargs):
        before = dict(owner=state.engine.puck_owner, puck_xy=[state.puck.x, state.puck.y],
                      players={state.team1.skater_scnum_base() + i: [p.x, p.y]
                               for i, p in enumerate(state.team1.players)})
        action = original(model, state, *args, **kwargs)
        trace.append(dict(before, buttons=action[0].tolist(), decision=model._last_decision))
        return action

    with patch.object(ClassicAIV1Model, 'predict_frame', new=predict):
        period = cpu_benchmark.cpu_match(('classic-v1', game['matchup'], game['seed'],
                                         300, 4, 'FILTERED', 'off'))
    for field in ('frames', 'actions_sha256', 'goals', 'shots', 'lineup', 'teams'):
        if field in game['period']:
            assert period[field] == game['period'][field], (game['matchup'], field)
    checked = 0
    for start in game['starts']:
        index = start['frame'] - 1
        owner = trace[index]['owner']
        for offset, native in enumerate(start['branches'][start['baseline']]['trace']):
            before, after = trace[index + offset], trace[index + offset + 1]
            expected = dict(frame=offset + 1, owner=after['owner'], carrier_xy=after['players'][owner],
                            puck_xy=after['puck_xy'], decision=before['decision'], buttons=before['buttons'])
            assert expected == native, (start['id'], offset, expected, native)
            checked += 1
    return dict(matchup=game['matchup'], seed=game['seed'], starts=len(game['starts']),
                replay_frames_checked=checked, source_frames=period['frames'],
                actions_sha256=period['actions_sha256'], passed=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = json.loads(Path(args.report).read_text(encoding='utf-8'))
    seed = report['options']['seed']
    results = []
    for game in report['games']:
        if game['seed'] == seed:
            result = check_game(game)
            results.append(result)
            print(result, flush=True)
    Path(args.output).write_text(json.dumps(dict(passed=True, games=results), indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
