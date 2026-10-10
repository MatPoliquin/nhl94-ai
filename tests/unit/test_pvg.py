"""Sparse shootout rewards, native completion, resets and configuration."""
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from nhl94_ai.cli import parse_configured
from nhl94_ai.config import EnvironmentConfig
from nhl94_ai.env.encoding import init_model, set_model_input
from nhl94_ai.env.observation import NHL94Observation2PEnv
from nhl94_ai.env.wrappers import StochasticFrameSkip
from nhl94_ai.evaluation.metrics import LiveTeamTotals
from nhl94_ai.game.ram import ROM_RNG_ADDRESS
from nhl94_ai.game.state import NHL94GameState
from nhl94_ai.tasks.legacy import register_functions
from nhl94_ai.tasks.pvg import init_pvg, isdone_pvg, rf_pvg
from nhl94_ai.tasks.pvg_setup import (
    GOALIE_POSITION_JITTER, PLAYER_POSITION_JITTER, PLAYER_VELOCITY_JITTER,
)
from nhl94_ai.tasks.registry import get_task
from nhl94_ai.training.live import build_parser, prepare_args
from tests.ram_fixture import FixtureMemory
from tests.unit.test_environment_contracts import FixtureEnv


class ShootoutData:
    def __init__(self, memory):
        self.memory = memory

    def set_variable(self, name, definition):
        self.memory.fields[name] = definition['address'], definition['type']
        self.memory.info[name] = self.memory.extract(definition['address'], definition['type'])


def shootout_data(info):
    memory = FixtureMemory(info)
    memory.fields.update({
        'ba_ps_flags': (0xFFC2F2, '>u2'),
        'p1_score': (0xFFC6DA, '>u2'), 'p2_score': (0xFFCA3E, '>u2'),
    })
    for slot in range(12):
        memory.assign(0xFFB04A + slot * 0x80 + 0x66, '|u1', 255)
    memory.assign(0xFFB04A + 0x66, '|u1', 7)
    memory.assign(0xFFB04A + 11 * 0x80 + 0x66, '|u1', 0)
    memory.assign(0xFFC320, '>i2', 0)
    memory.assign(0xFFB7AA, '>i2', 0)
    memory.assign(0xFFC2F2, '>u2', 0x0400)
    memory.assign(0xFFC6DA, '>u2', 3)
    memory.assign(0xFFCA3E, '>u2', 2)
    memory.assign(0xFFD574, '>u2', 7)
    memory.assign(0xFFD576, '>u2', 4)
    for slot, position, velocity in (
        (0, (0, 177.90028381347656), (0, 1694)),
        (11, (0, 247.3582763671875), (0, 0)),
        (14, (-4, 197.14772033691406), (0, 1720)),
    ):
        base = 0xFFB04A + slot * 0x80
        for current, previous, value in zip((0, 0x14), (0x1C, 0x20), position):
            memory.assign(base + current, '>i4', int(value * 65536))
            memory.assign(base + previous, '>i4', int(value * 65536))
        for offset, value in zip((0x28, 0x2A), velocity):
            memory.assign(base + offset, '>i2', value)
    return ShootoutData(memory)


class ShootoutFixtureEnv(FixtureEnv):
    def reset(self, **kwargs):
        observation, _ = super().reset(**kwargs)
        self.data = shootout_data(self.info)
        return observation, copy.deepcopy(self.info)


def shootout_state():
    state = NHL94GameState(5)
    state.is_shootout_active = True
    state.engine.puck_owner = 0
    state.team1.stats.score = 7
    state.team2.stats.score = 4
    state.EndFrame()
    return state


class PvGRewardTests(unittest.TestCase):
    def test_new_home_goals_reward_and_end_the_episode(self):
        state = shootout_state()
        self.assertEqual(rf_pvg(state), 0.0)
        self.assertFalse(isdone_pvg(state))
        state.team1.stats.score += 1
        state.is_shootout_active = False
        state.created_pre_shot_opening = True
        self.assertEqual(rf_pvg(state), 1.0)
        self.assertTrue(isdone_pvg(state))
        state.EndFrame()
        self.assertEqual(rf_pvg(state), 0.0)

    def test_saved_or_decreased_goals_are_not_new_rewards(self):
        state = shootout_state()
        state.team1.stats.score = 0
        self.assertEqual(rf_pvg(state), 0.0)
        self.assertFalse(isdone_pvg(state))
        state.EndFrame()
        state.team1.stats.score = 2
        self.assertEqual(rf_pvg(state), 1.0)

    def test_shots_catches_loose_pucks_and_positions_have_no_shaping(self):
        for owner in (0, 11, -256):
            with self.subTest(owner=owner):
                state = shootout_state()
                state.engine.puck_owner = owner
                state.team1.stats.shots = 1
                state.team1.stats.passing = 1
                state.team1.stats.bodychecks = 1
                state.engine.shot_taken = state.engine.shot_mode_active = 1.0
                state.puck.x, state.puck.y = 20, 270
                self.assertEqual(rf_pvg(state), 0.0)
                self.assertFalse(isdone_pvg(state))

    def test_native_completion_ends_failures_without_a_penalty(self):
        state = shootout_state()
        state.is_shootout_active = False
        self.assertTrue(isdone_pvg(state))
        self.assertEqual(rf_pvg(state), 0.0)

    def test_away_goals_do_not_reward_the_home_shooter(self):
        state = shootout_state()
        state.team2.stats.score += 1
        self.assertTrue(isdone_pvg(state))
        self.assertEqual(rf_pvg(state), 0.0)

    def test_stopped_game_clock_does_not_end_a_live_attempt(self):
        state = shootout_state()
        state.time = 0
        self.assertFalse(isdone_pvg(state))
        self.assertEqual(rf_pvg(state), 0.0)


class PvGResetTests(unittest.TestCase):
    def env(self, seed=0):
        return SimpleNamespace(data=shootout_data({}), np_random=np.random.default_rng(seed))

    def test_initializer_changes_only_positions_motion_contact_sorting_and_rom_seed(self):
        env = self.env()
        before = bytearray(env.data.memory.buffer)
        init_pvg(env, 'NHL94-Genesis-v0')
        after = bytearray(env.data.memory.buffer)
        allowed = [(ROM_RNG_ADDRESS, 4), (0xFFB84A, 80)]
        for slot in (0, 11, 14):
            base = 0xFFB04A + slot * 0x80
            allowed.extend(((base, 4), (base + 0x14, 32)))
        for address, size in allowed:
            offset = address & 0xFFFF
            before[offset:offset + size] = after[offset:offset + size]
        self.assertEqual(before, after)
        self.assertNotEqual(env.data.memory.extract(ROM_RNG_ADDRESS, '>u4'), 0)
        self.assertEqual(env.data.memory.info['p1_score'], 7)
        self.assertEqual(env.data.memory.info['p2_score'], 4)
        self.assertEqual(env.data.memory.extract(0xFFC6DA, '>u2'), 3)

    def test_rom_seed_is_reproducible_and_varies_across_episodes(self):
        first, second, different = self.env(7), self.env(7), self.env(8)
        for env in (first, second, different):
            init_pvg(env, 'NHL94-Genesis-v0')
        def read(env):
            return env.data.memory.extract(ROM_RNG_ADDRESS, '>u4')
        self.assertEqual(read(first), read(second))
        self.assertNotEqual(read(first), read(different))
        self.assertEqual(first.data.memory.buffer, second.data.memory.buffer)
        self.assertNotEqual(first.data.memory.buffer, different.data.memory.buffer)
        initial = read(first)
        init_pvg(first, 'NHL94-Genesis-v0')
        self.assertNotEqual(read(first), initial)

    def test_offsets_are_small_and_preserve_puck_geometry_and_forward_approach(self):
        sampled = []
        for seed in range(128):
            env = self.env(seed)
            memory = env.data.memory
            saved_positions = {
                slot: tuple(memory.extract(0xFFB04A + slot * 0x80 + offset, '>i4')
                            for offset in (0, 0x14))
                for slot in (0, 11, 14)
            }
            init_pvg(env, 'NHL94-Genesis-v0')
            positions = {}
            for slot, bounds in ((0, PLAYER_POSITION_JITTER), (11, GOALIE_POSITION_JITTER),
                                 (14, PLAYER_POSITION_JITTER)):
                base = 0xFFB04A + slot * 0x80
                positions[slot] = tuple(memory.extract(base + offset, '>i4') for offset in (0, 0x14))
                for value, saved, radius in zip(positions[slot], saved_positions[slot], bounds):
                    self.assertLessEqual(abs(value - saved), radius * 65536)
                    self.assertEqual((value - saved) % 65536, 0)
                for current, previous in ((0, 0x1C), (0x14, 0x20), (0x18, 0x24)):
                    self.assertEqual(memory.extract(base + current, '>i4'),
                                     memory.extract(base + previous, '>i4'))
                self.assertEqual(memory.extract(base + 0x18, '>i4'), 0)
                self.assertEqual(memory.extract(base + 0x2C, '>i2'), 0)
            for axis in range(2):
                self.assertEqual(positions[14][axis] - positions[0][axis],
                                 saved_positions[14][axis] - saved_positions[0][axis])
            vx, vy = (memory.extract(0xFFB04A + offset, '>i2') for offset in (0x28, 0x2A))
            self.assertLessEqual(abs(vx), PLAYER_VELOCITY_JITTER)
            self.assertLessEqual(abs(vy - 1694), PLAYER_VELOCITY_JITTER)
            self.assertGreater(vy, 0)
            self.assertEqual(memory.extract(0xFFB74A + 0x28, '>i2'), vx)
            self.assertEqual(memory.extract(0xFFB74A + 0x2A, '>i2') - 1720, vy - 1694)
            self.assertGreater(positions[11][1] - positions[0][1], 58 * 65536)
            self.assertLess(positions[0][1], 270 * 65536)
            self.assertEqual(memory.extract(0xFFB7AA, '>i2'), 0)
            sampled.append((*positions[0], *positions[11], vx, vy))
        self.assertEqual(len(set(sampled)), 128)
        for axis in range(6):
            self.assertGreater(len({sample[axis] for sample in sampled}), 4)

    def test_position_jitter_preserves_goalie_velocity(self):
        env = self.env()
        base = 0xFFB04A + 11 * 0x80
        env.data.memory.assign(base + 0x28, '>i2', 123)
        env.data.memory.assign(base + 0x2A, '>i2', -456)
        init_pvg(env, 'NHL94-Genesis-v0')
        self.assertEqual(env.data.memory.extract(base + 0x28, '>i2'), 123)
        self.assertEqual(env.data.memory.extract(base + 0x2A, '>i2'), -456)

    def test_reset_rebuilds_collision_positions_and_both_sort_indices(self):
        for horizontal in (False, True):
            env = self.env()
            memory = env.data.memory
            memory.assign(0xFFC2EC, '|u1', 0x80 if horizontal else 0)
            init_pvg(env, 'NHL94-Genesis-v0')
            order = [memory.extract(0xFFB88A + index, '|u1') // 2 for index in range(16)]
            self.assertEqual(set(order), set(range(16)))
            coordinates = []
            for index, slot in enumerate(order):
                self.assertEqual(memory.extract(0xFFB86A + slot * 2, '>u2'), index)
                actual = memory.extract(0xFFB04A + slot * 0x80 + (0 if horizontal else 0x14), '>i2')
                self.assertEqual(memory.extract(0xFFB84A + slot * 2, '>i2'), actual)
                coordinates.append(actual)
            self.assertEqual(coordinates, sorted(coordinates))

    def test_initializer_does_not_inspect_the_selected_skater(self):
        env = self.env()
        env.data.memory.assign(0xFFC320, '>i2', -1)
        init_pvg(env, 'NHL94-Genesis-v0')
        self.assertEqual(env.data.memory.info['p1_score'], 7)
        self.assertEqual(env.data.memory.extract(0xFFC320, '>i2'), -1)


class PvGEnvironmentTests(unittest.TestCase):
    def wrapped(self, frame_skip):
        args = SimpleNamespace(env='NHL94-Genesis-v0', nn='MlpPolicy',
                               action_type='FILTERED', hyperparams_dict={})
        inner = NHL94Observation2PEnv(ShootoutFixtureEnv(), args, 1, 'PvG')
        env = StochasticFrameSkip(inner, frame_skip, -1)
        self.addCleanup(env.close)
        return env, inner

    def test_goal_and_native_miss_stop_repetition_on_the_outcome_frame(self):
        for frame_skip in (1, 4, 10):
            for goal in (False, True):
                with self.subTest(frame_skip=frame_skip, goal=goal):
                    env, inner = self.wrapped(frame_skip)
                    for _ in range(2):
                        observation, info = env.reset(seed=0)
                        observation = np.asarray(observation)
                        self.assertEqual(observation.shape, (52,))
                        self.assertTrue(np.isfinite(observation).all())
                        self.assertEqual(info['p1_score'], 7)
                        self.assertEqual(inner.game_state.team1.last_stats.score, 7)
                        action = np.zeros(12, dtype=np.int8)
                        _, reward, done, _, _ = env.step(action)
                        self.assertEqual(reward, 0.0)
                        self.assertFalse(done)
                        memory = inner.env.data.memory
                        if goal:
                            memory.assign(0xFFD574, '>u2', 8)
                        memory.assign(0xFFC2F2, '>u2', 0)
                        before = inner.env.steps
                        _, reward, done, _, info = env.step(action)
                        self.assertEqual(reward, 1.0 if goal else 0.0)
                        self.assertTrue(done)
                        self.assertEqual(inner.env.steps, before + 1)
                        totals = LiveTeamTotals()
                        totals.add_delta(LiveTeamTotals(goals=7), LiveTeamTotals.from_info(info, 1))
                        self.assertEqual(totals.goals, int(goal))

    def test_registration_preserves_the_existing_input_and_button_contracts(self):
        task = get_task('PvG')
        functions = register_functions('PvG')
        self.assertEqual((task.initialize, task.reward, task.done), functions[:3])
        self.assertIs(task.observation_size, init_model)
        self.assertIs(task.encode, set_model_input)
        action = np.ones(12, dtype=np.int8)
        task.restrict_action(action)
        np.testing.assert_array_equal(action, np.ones(12, dtype=np.int8))

    def test_training_example_resolves_mlp_and_cpu_state(self):
        path = Path(__file__).resolve().parents[2] / 'configs/training/pvg.json'
        args = prepare_args(parse_configured(
            build_parser(), ['--config', str(path), '--num_env', '8', '--num_timesteps', '1000000'],
        ))
        EnvironmentConfig.from_args(args)
        self.assertEqual((args.rf, args.nn, args.action_type), ('PvG', 'MlpPolicy', 'FILTERED'))
        self.assertEqual(args.state, 'MightyDucksVsAllStarCampbell.Shootout.NearGoalie.Start')
        self.assertEqual((args.num_players, args.num_env, args.num_timesteps), (1, 8, 1000000))
        self.assertEqual(args.hyperparams_dict['net_arch'], {'pi': [128, 128], 'vf': [128, 128]})
        self.assertEqual(args.hyperparams_dict['frame_skip'], 4)

    def test_invalid_pvg_options_fail_before_emulator_creation(self):
        for override in ({'num_players': 2}, {'selfplay': True}, {'side': 'away'},
                         {'env': 'NHL941on1-Genesis-v0'}, {'nn': 'CnnPolicy'}):
            with self.subTest(override=override):
                values = {'env': 'NHL94-Genesis-v0', 'nn': 'MlpPolicy', 'rf': 'PvG'}
                values.update(override)
                with self.assertRaisesRegex(ValueError, 'PvG requires'):
                    EnvironmentConfig.from_args(SimpleNamespace(**values))


if __name__ == '__main__':
    unittest.main()
