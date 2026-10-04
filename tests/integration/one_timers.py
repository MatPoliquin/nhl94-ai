"""Replay a naturally occurring one-timer through raw buttons and intent macros.

Requires the NHL94 ROM. The captured emulator state stays in memory; no ROM or
save-state assets are written. Run: python tests/integration/one_timers.py
"""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np

from nhl94_ai.env.actions import HockeyActionController
from nhl94_ai.env.factory import make_retro
from nhl94_ai.agents.offense import FEINT_FRAMES
from nhl94_ai.evaluation.benchmark import RAM, away_view, make_agent, update_state
from nhl94_ai.game.state import NHL94GameState


def environment(*, cpu=False):
    env = (make_retro(game='NHL94-Genesis-v0', state='PenguinsVsSenators.start', num_players=1)
           if cpu else make_retro(game='NHL94-Genesis-v0', num_players=2))
    for name, (address, kind) in RAM.items():
        env.data.set_variable(name, {'address': address, 'type': kind})
    env.reset(seed=3000)
    return env


def capture_setup(seed=3000, *, from_cut=False, cpu=False, frame_skip=4):
    env = environment(cpu=cpu)
    try:
        env.data.set_value('bench_rng', seed)
        env.data.set_value('bench_clock', 300)
        *_, info = env.step(np.zeros(12 if cpu else 24, dtype=np.int8))
        if cpu:
            assert (info['bench_team1'], info['bench_team2']) == (1, 0)
        state = NHL94GameState(5)
        agents = [make_agent('classic-v1')]
        if not cpu:
            agents.append(make_agent('classic-v1-direct'))
        for agent in agents:
            agent.frame_skip = frame_skip
        snapshot = None
        saved_tick = -56
        cut_snapshot, cut_owner, cut_tick = None, None, -FEINT_FRAMES
        requested_receiver = None
        for tick in range(16000):
            update_state(state, info, env)
            new_decision = agents[0]._frame_remaining == 0
            before = deepcopy(agents) if new_decision else None
            actions = np.concatenate([agent.predict_game_state(view)[0]
                                      for agent, view in zip(agents, (state, away_view(state)))])
            fresh_cut = (cut_snapshot is None or cut_owner != state.engine.puck_owner
                         or tick - cut_tick > FEINT_FRAMES)
            if from_cut and new_decision and agents[0]._last_decision == 'one-timer-setup' and fresh_cut:
                cut_snapshot = (env.em.get_state(), before, info['bench_one_timers1'])
                cut_owner, cut_tick = state.engine.puck_owner, tick
            if new_decision and agents[0]._last_decision == 'one-timer-pass':
                if not from_cut or (cut_snapshot is not None and cut_owner == state.engine.puck_owner
                                    and tick - cut_tick <= FEINT_FRAMES + 4):
                    snapshot = cut_snapshot if from_cut else (env.em.get_state(), before, info['bench_one_timers1'])
                    requested_receiver = agents[0]._one_timer[1]
                    saved_tick = tick
            _, _, terminated, truncated, info = env.step(actions)
            if snapshot and info['bench_one_timers1'] > snapshot[2] and tick - saved_tick < 56:
                if not from_cut or info['shot_player'] == requested_receiver:
                    return snapshot
            if terminated or truncated or info['bench_clock'] == 0:
                break
        return None
    finally:
        env.close()


def replay(snapshot, schema, *, fast_pass=False, from_cut=False, frame_skip=4):
    env = environment(cpu=len(snapshot[1]) == 1)
    try:
        data, saved_agents, attempts = snapshot
        agents = deepcopy(saved_agents)
        for agent in agents:
            agent.frame_skip = frame_skip
        env.em.set_state(data)
        env.data.update_ram()
        info = env.data.lookup_all()
        if fast_pass:
            memory = env.data.memory
            memory.assign(0xFFB04A + info['puck_owner'] * 0x80 + 0x6E, '|u1', 30)
            for slot in range(5):
                memory.assign(0xFFB04A + slot * 0x80 + 0x71, '|u1', 0)
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
        requested_receiver = None
        for tick in range(56 + (FEINT_FRAMES if from_cut else 0)):
            update_state(state, info, env)
            actions = [agent.predict_game_state(view)[0]
                       for agent, view in zip(agents, (state, away_view(state)))]
            if tick == 0:
                assert agents[0]._last_decision == ('one-timer-setup' if from_cut else 'one-timer-pass')
            if from_cut and agents[0]._last_decision == 'one-timer-setup':
                assert agents[0]._last_target == agents[0].offense.feint_target
            if agents[0]._last_decision == 'one-timer-pass' and agents[0]._one_timer is not None:
                requested_receiver = agents[0]._one_timer[1]
            if intents:
                first = processor._process_action(actions[0], macro)[0]
            else:
                first = actions[0]
            *_, info = env.step(np.concatenate((first, *actions[1:])))
            if info['bench_one_timers1'] > attempts:
                if from_cut:
                    assert requested_receiver is not None and info['shot_player'] == requested_receiver
                if fast_pass:
                    assert info[f"defense_{info['shot_player']}_stick"] == 0
                update_state(state, info, env)
                agents[0].predict_game_state(state)
                assert agents[0]._one_timer is None, ('one-timer remains active after native release', frame_skip)
                assert agents[0].one_timer_metrics.get('shot-released', 0) > 0, agents[0].one_timer_metrics
                print(f'PASS: {schema} converted the live pass into a ROM-counted one-timer'
                      + f' at decision interval {frame_skip} and completed on native release'
                      + (' after an actually executed goalie-safe setup cut' if from_cut else '')
                      + (' despite zero receiver stick handling and maximum pass speed' if fast_pass else ''))
                return
        raise AssertionError(f'{schema} failed to produce a one-timer')
    finally:
        env.close()


if __name__ == '__main__':
    snapshots = {}
    for interval in (1, 4, 8, 10):
        setup = None
        for seed in range(3000, 3004):
            setup = capture_setup(seed, frame_skip=interval)
            if setup is not None:
                break
        assert setup is not None, ('No successful one-timer setup in four bounded periods', interval)
        # Receiver activation changes its motion; capture at the replay cadence.
        snapshots[interval] = setup
        replay(setup, 'FILTERED', frame_skip=interval)
        replay(setup, 'HOCKEY_INTENT_DPAD', frame_skip=interval)
    setup = snapshots[4]
    replay(setup, 'FILTERED', fast_pass=True)
    replay(setup, 'HOCKEY_INTENT_DPAD', fast_pass=True)
    from tests.integration.classic_offense import one_timer_setup_cut
    one_timer_setup_cut()
