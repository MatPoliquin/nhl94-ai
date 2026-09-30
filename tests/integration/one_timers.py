"""Replay a naturally occurring one-timer through raw buttons and intent macros.

Requires the NHL94 ROM. The captured emulator state stays in memory; no ROM or
save-state assets are written. Run: python tests/integration/one_timers.py
"""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np

from nhl94_ai.env.actions import HockeyActionController
from nhl94_ai.env.factory import make_retro
from nhl94_ai.evaluation.benchmark import RAM, away_view, make_agent, update_state
from nhl94_ai.game.state import NHL94GameState


def environment():
    env = make_retro(game='NHL94-Genesis-v0', num_players=2)
    for name, (address, kind) in RAM.items():
        env.data.set_variable(name, {'address': address, 'type': kind})
    env.reset(seed=3000)
    return env


def capture_setup(seed=3000):
    env = environment()
    try:
        env.data.set_value('bench_rng', seed)
        env.data.set_value('bench_clock', 300)
        *_, info = env.step(np.zeros(24, dtype=np.int8))
        state = NHL94GameState(5)
        agents = [make_agent('classic-v1'), make_agent('classic-v1-direct')]
        for agent in agents:
            agent.frame_skip = 4
        snapshot = None
        for tick in range(16000):
            update_state(state, info)
            new_decision = agents[0]._frame_remaining == 0
            before = deepcopy(agents) if new_decision else None
            actions = np.concatenate([agent.predict_game_state(view)[0]
                                      for agent, view in zip(agents, (state, away_view(state)))])
            if new_decision and agents[0]._last_decision == 'one-timer-pass':
                snapshot = (env.em.get_state(), before, info['bench_one_timers1'])
                saved_tick = tick
            _, _, terminated, truncated, info = env.step(actions)
            if snapshot and info['bench_one_timers1'] > snapshot[2] and tick - saved_tick < 56:
                return snapshot
            if terminated or truncated or info['bench_clock'] == 0:
                break
        return None
    finally:
        env.close()


def replay(snapshot, schema):
    env = environment()
    try:
        data, saved_agents, attempts = snapshot
        agents = deepcopy(saved_agents)
        env.em.set_state(data)
        env.data.update_ram()
        info = env.data.lookup_all()
        state = NHL94GameState(5)
        intents = schema == 'HOCKEY_INTENT_DPAD'
        if intents:
            # Retain the exact controller timing; change only its action encoding.
            agents[0].controller._intents = True
            agents[0].controller._size = 6
            processor = HockeyActionController(SimpleNamespace(action_type=schema, game_state=state))
            macro = processor._new_action_state()
        for tick in range(56):
            update_state(state, info)
            actions = [agent.predict_game_state(view)[0]
                       for agent, view in zip(agents, (state, away_view(state)))]
            if tick == 0:
                assert agents[0]._last_decision == 'one-timer-pass'
            if intents:
                first = processor._process_action(actions[0], macro)[0]
            else:
                first = actions[0]
            *_, info = env.step(np.concatenate((first, actions[1])))
            if info['bench_one_timers1'] > attempts:
                print(f'PASS: {schema} converted the same live pass into a ROM-counted one-timer')
                return
        raise AssertionError(f'{schema} failed to produce a one-timer')
    finally:
        env.close()


if __name__ == '__main__':
    setup = None
    for seed in range(3000, 3004):
        setup = capture_setup(seed)
        if setup is not None:
            break
    assert setup is not None, 'No successful one-timer setup found in four bounded seeded periods'
    replay(setup, 'FILTERED')
    replay(setup, 'HOCKEY_INTENT_DPAD')
