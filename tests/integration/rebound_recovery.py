"""Capture a real recovery, then verify the native handoff in both input formats."""
from copy import deepcopy
import hashlib
from types import SimpleNamespace

import numpy as np

from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.env.actions import HockeyActionController
from nhl94_ai.env.factory import make_retro
from nhl94_ai.evaluation.cpu_benchmark import CPU_RAM, select_side
from nhl94_ai.evaluation.refinement_replay import advance, read_view, restore
from nhl94_ai.game.state import NHL94GameState


def run():
    env = make_retro(game='NHL94-Genesis-v0', state='SabresVsMightyDucks.ManualGoalie.Start',
                     num_players=1, goalie_policy='selective')
    try:
        for name, (address, kind) in CPU_RAM.items():
            env.data.set_variable(name, {'address': address, 'type': kind})
        env.reset(seed=25702)
        env.data.update_ram()
        select_side(env.data, env.data.lookup_all(), 1)
        env.data.set_value('bench_rng', 25702)
        env.data.set_value('bench_clock', 300)
        env.step(np.zeros(12, dtype=np.int8))
        model = ClassicAIV1Model(SimpleNamespace(action_type='FILTERED', goalie_policy='selective'))
        state = NHL94GameState(5)
        for frame in range(1, 16000):
            view, info = read_view(env, state, 1)
            if (model.shot.same_shooter_recovery(view, model.scheduler.decisions)
                    and model.shot.recovery_ready(view)):
                break
            assert info['bench_clock'] > 0, 'No native recovery in the regression period'
            advance(env, model.predict_frame(view, 4)[0], 1)
        else:
            raise AssertionError('No native recovery in the regression frame budget')
        assert model.shot.release_observed and view.engine.puck_owner == model.shot.slot
        snapshot = env.em.get_state()
        digest = hashlib.sha256(env.get_ram().tobytes()).hexdigest()
        history = deepcopy(model)
        for interval in (1, 4, 8):
            outcomes = []
            for schema in ('FILTERED', 'HOCKEY_INTENT_DPAD'):
                restore(env, snapshot, digest)
                decoded, model = deepcopy(state), deepcopy(history)
                view, _ = read_view(env, decoded, 1)
                # Only the confirmed skater possession is replayed. No goalie
                # controller is invoked through the intent-only handoff.
                model.goalie = None
                model.shot.rebound_recovery = True
                model._intents = schema == 'HOCKEY_INTENT_DPAD'
                model._size = 6 if model._intents else 12
                processor = HockeyActionController(SimpleNamespace(action_type=schema, game_state=view))
                macro = processor._new_action_state()
                action = model.predict_frame(view, interval)[0]
                buttons = processor._process_action(action, macro)[0]
                assert model.shot.recoveries == history.shot.recoveries + 1
                assert model.scheduler.decisions == history.scheduler.decisions + 1
                assert model.scheduler.frames == history.scheduler.frames + 1
                assert model._last_decision != 'shot-follow-through'
                advance(env, buttons, 1)
                outcomes.append((buttons.tolist(), hashlib.sha256(env.get_ram().tobytes()).hexdigest()))
                print(f'PASS: native recovery frame {frame}, {schema}, interval {interval}, '
                      f'immediate {model._last_decision}')
            assert outcomes[0] == outcomes[1], (interval, 'Control formats differ at the native recovery handoff')
    finally:
        env.close()


if __name__ == '__main__':
    run()
