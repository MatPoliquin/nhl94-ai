"""Rescue the real seed-12000 pass from the watched away/manual-goalie command."""
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.config import load_hyperparams, resolve_hyperparams_for_model
from nhl94_ai.env.actions import HockeyActionController
from nhl94_ai.env.factory import build_single_nhl94_env
from nhl94_ai.evaluation.benchmark import away_view
from nhl94_ai.evaluation.play import parse_cmdline
from nhl94_ai.game.constants import GameConsts as Buttons
from nhl94_ai.game.state import NHL94GameState


def _view(info):
    state = NHL94GameState(5)
    state.BeginFrame(info, [0] * 6)
    return away_view(state)


def run():
    fixture = json.loads((Path(__file__).parents[1] / 'fixtures/classic-missed-reception.json').read_text())
    args = parse_cmdline(['--nn', 'ClassicAIV1', '--env', 'NHL94-Genesis-v0',
                         '--mode', 'model_vs_game', '--rf', 'PostPlay', '--state', fixture['state'],
                         '--side', 'away', '--goalie-policy', 'selective', '--seed', str(fixture['seed'])])
    hyperparams = resolve_hyperparams_for_model(load_hyperparams(args.hyperparams), args.nn)
    env = build_single_nhl94_env(args, hyperparams, use_frame_skip=False, episode_rom_seed=fixture['seed'])
    try:
        _, info = env.reset(seed=fixture['seed'])
        for mask, count in fixture['inputs']:
            action = np.array([(mask >> bit) & 1 for bit in range(12)], dtype=np.int8)
            for _ in range(count):
                *_, info = env.step(action)
        initial = _view(info)
        assert initial.engine.puck_owner == fixture['passer']
        assert (initial.puck.x, initial.puck.y) == tuple(fixture['puck'])
        snapshot = env.unwrapped.em.get_state()
        correction = ClassicAIV1Model._receive_pass
        results = {}
        for mode, schema, method in (('before', 'FILTERED', lambda *_: None),
                                     ('after', 'FILTERED', correction),
                                     ('intents', 'HOCKEY_INTENT_DPAD', correction)):
            env.unwrapped.em.set_state(snapshot)
            state = deepcopy(initial)
            model_args = deepcopy(args)
            model_args.action_type = schema
            if schema != 'FILTERED':
                model_args.goalie_policy = 'off'
            model = ClassicAIV1Model(model_args)
            model_args.game_state = state
            processor = HockeyActionController(model_args)
            macro = processor._new_action_state()
            release, first_owner, switches, steered = None, None, [], []
            with patch.object(ClassicAIV1Model, '_receive_pass', method):
                for frame in range(200):
                    model_args.game_state = state
                    action = processor._process_action(model.predict_frame(state)[0], macro)[0]
                    if frame == 0:
                        assert model.offense.pending['receiver'] == fixture['receiver']
                    if model._last_decision == 'receive-pass-switch':
                        switches.append(frame)
                        assert action[Buttons.INPUT_B] and not action[Buttons.INPUT_C]
                    if model._last_decision == 'receive-pass':
                        steered.append(frame)
                        assert state.team1.defense_control == fixture['receiver']
                        assert not action[Buttons.INPUT_B] and not action[Buttons.INPUT_C]
                    *_, info = env.step(action)
                    state = _view(info)
                    if release is None and state.engine.puck_owner < 0:
                        release = frame + 1
                    if release is not None and state.engine.puck_owner >= 0:
                        first_owner = state.engine.puck_owner
                        break
            results[mode] = {'owner': first_owner, 'loose_frames': frame + 1 - release,
                             'switches': switches, 'steered': steered}
        assert results['before']['owner'] == 1, results
        assert results['before']['loose_frames'] == 180, results
        for mode in ('after', 'intents'):
            assert results[mode]['owner'] == fixture['receiver'], results
            assert results[mode]['loose_frames'] <= 30, results
            assert results[mode]['switches'] and results[mode]['steered'], results
        print(f'PASS: native missed pass corrected with confirmed receiver control: {results}')
    finally:
        env.close()


if __name__ == '__main__':
    run()
