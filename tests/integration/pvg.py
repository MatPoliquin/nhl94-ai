"""ROM-backed PvG resets, native goals/saves/completion and frame-skip boundaries."""
from pathlib import Path

import numpy as np

from nhl94_ai.cli import parse_configured
from nhl94_ai.env.factory import build_single_nhl94_env
from nhl94_ai.evaluation.metrics import LiveTeamTotals
from nhl94_ai.game.constants import GameConsts as Buttons
from nhl94_ai.game.ram import GOALIE_INPUT_ATTRIBUTES, ROM_RNG_ADDRESS, SKATER_INPUT_ATTRIBUTES
from nhl94_ai.tasks.defense_setup import _place_object, _rebuild_object_order
from nhl94_ai.tasks.pvg import init_pvg, isdone_pvg
from nhl94_ai.tasks.pvg_setup import (
    GOALIE_POSITION_JITTER, PLAYER_POSITION_JITTER, PLAYER_VELOCITY_JITTER,
)
from nhl94_ai.training.datasets import get_game_state
from nhl94_ai.training.live import build_parser, prepare_args


def settings():
    path = Path(__file__).resolve().parents[2] / 'configs/training/pvg.json'
    return prepare_args(parse_configured(build_parser(), ['--config', str(path)]))


def finish(env, action, max_steps=1200):
    total = 0.0
    for _ in range(max_steps):
        observation, reward, terminated, truncated, info = env.step(action)
        assert np.isfinite(observation).all()
        total += reward
        if terminated or truncated:
            assert terminated and not truncated
            return total, info
    raise AssertionError('Native shootout attempt did not finish within the bounded check.')


def feature_values(args, observation):
    names = [f'{group}.{field}' for group, fields in args.hyperparams_dict['model_input']['groups'].items()
             for field in fields]
    return dict(zip(names, observation))


def contact_fixture(env, goal):
    memory = env.unwrapped.data.memory
    if goal:
        _place_object(memory, 11, (-100, 247))
        _place_object(memory, 14, (0, 252))
        velocity = 14000
    else:
        _place_object(memory, 11, (0, 247))
        _place_object(memory, 14, (0, 240))
        velocity = 2000
    memory.assign(0xFFB74A + 0x2A, '>i2', velocity)
    memory.assign(0xFFB7AA, '>i2', -256)
    memory.assign(0xFFBED8, '>i2', 0)
    memory.assign(0xFFC2F2, '>u2', 0x0400)
    _rebuild_object_order(memory)


def read_motion(memory, slot):
    base = 0xFFB04A + slot * 0x80
    return np.array([memory.extract(base + offset, '>i4') / 65536 for offset in (0, 0x14)]
                    + [memory.extract(base + offset, '>i2') for offset in (0x28, 0x2A)])


def check_randomized_starts(env, args):
    raw = env.unwrapped
    raw.reset(seed=0)
    memory = raw.data.memory
    saved = {slot: read_motion(memory, slot) for slot in (0, 11, 14)}
    attributes = {
        slot: tuple(memory.extract(0xFFB04A + slot * 0x80 + offset, '|u1')
                    for offset in range(0x66, 0x80))
        for slot in (0, 11)
    }
    starts, reference = [], None
    neutral = np.zeros(12, dtype=np.int8)
    for seed in (*range(32), None, 0):
        raw.reset(seed=seed)
        init_pvg(raw, args.env)
        initialized = {slot: read_motion(memory, slot) for slot in saved}
        for slot, bounds in ((0, PLAYER_POSITION_JITTER), (11, GOALIE_POSITION_JITTER),
                             (14, PLAYER_POSITION_JITTER)):
            base = 0xFFB04A + slot * 0x80
            delta = initialized[slot][:2] - saved[slot][:2]
            assert np.all(np.abs(delta) <= bounds), (seed, slot, delta)
            np.testing.assert_array_equal(delta, np.round(delta))
            for current, previous in ((0, 0x1C), (0x14, 0x20), (0x18, 0x24)):
                assert memory.extract(base + current, '>i4') == memory.extract(base + previous, '>i4')
            assert memory.extract(base + 0x18, '>i4') == memory.extract(base + 0x2C, '>i2') == 0
        np.testing.assert_array_equal(initialized[14][:2] - initialized[0][:2],
                                      saved[14][:2] - saved[0][:2])
        velocity_noise = initialized[0][2:] - saved[0][2:]
        assert np.all(np.abs(velocity_noise) <= PLAYER_VELOCITY_JITTER)
        np.testing.assert_array_equal(initialized[14][2:] - saved[14][2:], velocity_noise)
        np.testing.assert_array_equal(initialized[11][2:], saved[11][2:])
        if saved[0][3] > PLAYER_VELOCITY_JITTER:
            assert initialized[0][3] > 0
        assert initialized[0][1] < initialized[11][1] < 270
        assert initialized[11][1] - initialized[0][1] > 58
        order = [memory.extract(0xFFB88A + index, '|u1') // 2 for index in range(16)]
        assert set(order) == set(range(16))
        axis = 0 if memory.extract(0xFFC2EC, '|u1') & 0x80 else 0x14
        coordinates = []
        for index, slot in enumerate(order):
            position = memory.extract(0xFFB04A + slot * 0x80 + axis, '>i2')
            assert memory.extract(0xFFB84A + slot * 2, '>i2') == position
            assert memory.extract(0xFFB86A + slot * 2, '>u2') == index
            coordinates.append(position)
        assert coordinates == sorted(coordinates)

        observation, info = env.reset(seed=seed)
        actual = {slot: read_motion(memory, slot) for slot in saved}
        if seed is not None:
            for slot in saved:
                assert np.linalg.norm(actual[slot][:2] - initialized[slot][:2]) < 2
        starts.append(tuple(np.concatenate((actual[0], actual[11][:2]))))
        if seed == 0:
            if reference is None:
                reference = np.asarray(observation).copy()
            else:
                np.testing.assert_array_equal(observation, reference)
        for slot, expected in attributes.items():
            assert tuple(memory.extract(0xFFB04A + slot * 0x80 + offset, '|u1')
                         for offset in range(0x66, 0x80)) == expected
        assert info['puck_owner'] == info['defense_control1'] == 0
        assert (info['defense_team1'], info['defense_team2']) == (1, 0)
        assert not get_game_state(env).has_released_shot
        previous = actual
        for _ in range(32):
            observation, _, terminated, truncated, info = env.step(neutral)
            assert not terminated and not truncated
            assert info['puck_owner'] == info['defense_control1'] == 0
            assert np.isfinite(observation).all()
            actual = {slot: read_motion(memory, slot) for slot in saved}
            for slot in saved:
                assert np.linalg.norm(actual[slot][:2] - previous[slot][:2]) < 2
            assert np.linalg.norm(actual[14][:2] - actual[0][:2]) < 24
            previous = actual
    assert len(set(starts[:32])) == 32
    for axis in range(6):
        assert len({start[axis] for start in starts[:32]}) > 4
    print(f'PASS: {args.state}; 32 seeded starts plus ordinary/repeated resets; '
          'bounded full-word jitter, attached puck, preserved ratings/control, '
          'collision ordering and 32-frame neutral physics.')


def check_windup_opening(env, args, frame_skip):
    env.reset(seed=3)
    assert not get_game_state(env)._has_viable_shooting_opening()
    action = np.zeros(12, dtype=np.int8)
    action[Buttons.INPUT_C] = 1
    held = 0
    for _ in range(4):
        observation, reward, done, _, _ = env.step(action)
        held += frame_skip
        values = feature_values(args, observation)
        assert values['player.c_pressed'] == 1
        assert np.isclose(values['player.c_frames_held'], held / 60)
        assert reward == 0 and not done
        if get_game_state(env).is_shooting:
            break
    else:
        raise AssertionError('C input did not enter actual ROM windup.')
    memory = env.unwrapped.data.memory
    _place_object(memory, 11, (-100, 247))
    _rebuild_object_order(memory)
    observation, reward, done, _, _ = env.step(action)
    held += frame_skip
    state = get_game_state(env)
    assert reward == 0.05 and not done
    assert state.is_shooting and not state.has_released_shot
    assert state.c_frames_held == held
    values = feature_values(args, observation)
    assert values['player.c_pressed'] == 1
    assert np.isclose(values['player.c_frames_held'], held / 60)
    action[Buttons.INPUT_C] = 0
    remaining, _ = finish(env, action)
    state = get_game_state(env)
    assert remaining in (0.0, 1.0)
    assert state.has_released_shot and not state.c_pressed and state.c_frames_held == 0
    observation, _ = env.reset(seed=3)
    values = feature_values(args, observation)
    assert values['player.c_pressed'] == values['player.c_frames_held'] == 0
    assert not get_game_state(env).has_released_shot


def main():
    args = settings()
    goal_frames = None
    for frame_skip in (1, 4, 10):
        params = {**args.hyperparams_dict, 'frame_skip': frame_skip}
        env = build_single_nhl94_env(args, params, num_players=1, monitor=True)
        try:
            seeds = []
            reference = None
            for seed in (0, None, 0):
                observation, info = env.reset(seed=seed)
                state = get_game_state(env)
                assert np.asarray(observation).shape == (52,)
                assert env.observation_space.contains(np.asarray(observation, dtype=np.float32))
                assert np.isfinite(observation).all()
                assert not isdone_pvg(state)
                assert state.is_shootout_active is True
                assert info['puck_owner'] == info['defense_control1'] == 0
                assert (info['defense_team1'], info['defense_team2']) == (1, 0)
                assert not info['g2_control_flags'] & 8
                assert info['g2_control_roster'] == 0
                memory = env.unwrapped.data.memory
                values = feature_values(args, observation)
                assert values['player.is_controlled'] == values['goalie.is_present'] == 1
                player, goalie = 0xFFB04A, 0xFFB04A + 11 * 0x80
                for group, base, attributes in (
                    ('player', player, SKATER_INPUT_ATTRIBUTES),
                    ('goalie', goalie, GOALIE_INPUT_ATTRIBUTES),
                ):
                    for name, (offset, scale) in attributes.items():
                        expected = np.clip(memory.extract(base + offset, '|u1') / scale, -1, 1)
                        assert np.isclose(values[f'{group}.{name}'], expected), (group, name, values)
                assert np.isclose(values['goalie.x'],
                                  (memory.extract(goalie, '>i2') - memory.extract(player, '>i2')) / 240)
                assert np.isclose(values['goalie.y'],
                                  (memory.extract(goalie + 0x14, '>i2') - memory.extract(player + 0x14, '>i2')) / 540)
                assert np.isclose(values['goalie.vx'],
                                  (memory.extract(goalie + 0x28, '>i2') - memory.extract(player + 0x28, '>i2')) / 65536)
                seeds.append(memory.extract(ROM_RNG_ADDRESS, '>u4'))
                for side, address in ((1, 0xFFD574), (2, 0xFFD576)):
                    assert env.unwrapped.data.get_variable(f'p{side}_score') == {
                        'address': address, 'type': '>u2',
                    }
                if seed == 0:
                    if reference is None:
                        reference = np.asarray(observation).copy()
                    else:
                        np.testing.assert_array_equal(observation, reference)
            assert seeds[0] == seeds[2] and seeds[0] != seeds[1]

            if frame_skip == 1:
                check_randomized_starts(env, args)
            neutral = np.zeros(12, dtype=np.int8)
            env.reset(seed=0)
            memory.assign(player + 0x6C, '|u1', 30)
            memory.assign(goalie + 0x6C, '|u1', 5)
            observation, _, terminated, _, _ = env.step(neutral)
            assert not terminated
            values = feature_values(args, observation)
            assert values['player.shot_power'] == 1
            assert np.isclose(values['goalie.puck_control'], 5 / 30)
            check_windup_opening(env, args, frame_skip)
            for goal in (False, True):
                _, baseline = env.reset(seed=0)
                ordinary_score = env.unwrapped.data.memory.extract(0xFFC6DA, '>u2')
                contact_fixture(env, goal)
                reward, info = finish(env, neutral, max_steps=100)
                assert reward == float(goal), (
                    frame_skip, goal, reward, info['p1_score'], info['ba_ps_flags'], info['puck_owner'],
                )
                assert info['p1_score'] - baseline['p1_score'] == int(goal)
                assert get_game_state(env).is_shootout_active is False
                totals = LiveTeamTotals()
                totals.add_delta(LiveTeamTotals.from_info(baseline, 1),
                                 LiveTeamTotals.from_info(info, 1))
                assert totals.goals == int(goal)
                if goal:
                    if goal_frames is None:
                        goal_frames = info['episode']['l']
                    assert info['episode']['l'] == goal_frames
                    assert env.unwrapped.data.memory.extract(0xFFC6DA, '>u2') == ordinary_score
                else:
                    assert info['puck_owner'] == 11
            env.reset(seed=0)
            reward, info = finish(env, neutral)
            assert reward == 0.0
            assert info['puck_owner'] != 6, 'Continued into the away attempt.'
            print(f'PASS: PvG frame skip {frame_skip}; windup opening +0.05, native goal +1, '
                  'catch/timeout 0; C timing, release gate, goal telemetry and seeded resets.')
        finally:
            env.close()

    args.state = 'MightyDucksVsAllStarCampbell.Shootout.Start'
    env = build_single_nhl94_env(args, {**args.hyperparams_dict, 'frame_skip': 1}, num_players=1)
    try:
        check_randomized_starts(env, args)
        env.reset(seed=0)
        assert not isdone_pvg(get_game_state(env))
        reward, _ = finish(env, np.zeros(12, dtype=np.int8))
        assert reward == 0.0
    finally:
        env.close()
    print('PASS: Longer-approach save supported.')


if __name__ == '__main__':
    main()
