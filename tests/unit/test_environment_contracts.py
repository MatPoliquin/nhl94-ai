"""State, action, and reset contracts without an emulator or ROM."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import gymnasium as gym
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
from nhl94_ai.env.wrappers import StochasticFrameSkip
from nhl94_ai.game.state import NHL94GameState
from nhl94_ai.env.observation import NHL94Observation2PEnv
from nhl94_ai.tasks.legacy import register_functions
from tests.ram_fixture import FixtureMemory


class FixtureEnv(gym.Env):
    def __init__(self, game='NHL94-Genesis-v0', truncate=False):
        self.info = json.loads((ROOT / f'tests/fixtures/{game}.json').read_text(encoding='utf-8'))
        self.observation_space = gym.spaces.Box(0, 255, (224, 256, 3), dtype=np.uint8)
        self.action_space = gym.spaces.MultiBinary(12)
        self.steps = 0
        self.truncate = truncate

    def reset(self, **kwargs):
        super().reset(seed=kwargs.get('seed'))
        self.steps = 0
        self.data = SimpleNamespace(memory=FixtureMemory(self.info))
        return np.zeros(self.observation_space.shape, dtype=np.uint8), copy.deepcopy(self.info)

    def render(self):
        return np.zeros(self.observation_space.shape, dtype=np.uint8)

    def set_value(self, name, value):
        self.info[name] = value

    def step(self, action):
        self.steps += 1
        return np.zeros(self.observation_space.shape, dtype=np.uint8), 0.0, False, self.truncate, copy.deepcopy(self.info)


def wrapped(nn='MlpPolicy', action_type='FILTERED', rf='PostPlay'):
    args = SimpleNamespace(env='NHL94-Genesis-v0', nn=nn, action_type=action_type,
                           seq_len=4, alg='ppo2', hyperparams_dict={})
    base = FixtureEnv()
    if rf in ('DefenseZone', 'SelfPlayDefenseFinetune'):
        base.info.update(puck_owner=6, period=0, time=274)
    return NHL94Observation2PEnv(base, args, 1, rf)


class EnvironmentContracts(unittest.TestCase):
    def test_observation_and_sequence_reset(self):
        env = wrapped('GRUMlpPolicy')
        observation, _ = env.reset()
        self.assertEqual(observation.shape, (4, env.NUM_PARAMS))
        np.testing.assert_array_equal(observation[0], observation[-1])
        env.frame_buffer[0][:] = 99
        reset, _ = env.reset()
        np.testing.assert_array_equal(reset[0], reset[-1])

    def test_macro_reset_and_controller_independence(self):
        env = wrapped(action_type='HOCKEY_INTENT_DPAD')
        env.reset()
        env.learner_action_state['one_timer_shot_frames'] = 9
        self.assertEqual(env.opponent_action_state['one_timer_shot_frames'], 0)
        env.reset()
        self.assertEqual(env.learner_action_state['one_timer_shot_frames'], 0)
        self.assertEqual(env.action_space.shape, (6,))

    def test_opponent_view_does_not_mutate_learner_state(self):
        env = wrapped()
        env.reset()
        original = copy.deepcopy(env.game_state)
        view = env._build_opponent_view_state()
        self.assertIsNot(view, env.game_state)
        self.assertEqual(env.game_state.team1.players[0].x, original.team1.players[0].x)
        self.assertEqual(env.game_state.puck.y, original.puck.y)
        self.assertEqual(view.team1.stats.score, original.team2.stats.score)

    def test_mirroring_actions_twice_is_identity(self):
        env = wrapped()
        action = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0])
        np.testing.assert_array_equal(env._mirror_env_action(env._mirror_env_action(action)), action)

    def test_postplay_reward_and_completion(self):
        _, reward, done, *_ = register_functions('PostPlay')
        state = NHL94GameState(5)
        state.time = 100
        self.assertEqual(reward(state), 0.0)
        self.assertFalse(done(state))
        state.time = 0
        self.assertTrue(done(state))

    def test_defense_tasks_stop_frame_skip_on_recovery_or_conceding(self):
        for task in ('DefenseZone', 'SelfPlayDefenseFinetune'):
            for recovered in (False, True):
                with self.subTest(task=task, recovered=recovered):
                    inner = wrapped(rf=task)
                    env = StochasticFrameSkip(inner, 4, -1)
                    self.addCleanup(env.close)
                    env.reset()
                    if recovered:
                        inner.env.info['puck_owner'] = 0
                    else:
                        inner.env.info['p2_score'] += 1
                    _, reward, terminated, _, _ = env.step(np.zeros(12, dtype=np.int8))
                    self.assertEqual(reward, 1.0 if recovered else -1.0)
                    self.assertTrue(terminated)
                    self.assertEqual(inner.env.steps, 2)  # One reset frame, one terminal frame.

    def test_defense_shots_stop_frame_skip_without_counting_saved_shots(self):
        for task in ('DefenseZone', 'SelfPlayDefenseFinetune'):
            for frame_skip in (1, 4, 10):
                with self.subTest(task=task, frame_skip=frame_skip):
                    inner = wrapped(rf=task)
                    env = StochasticFrameSkip(inner, frame_skip, -1)
                    self.addCleanup(env.close)
                    info = inner.env.info
                    info['p2_shots'] = 7
                    action = np.zeros(12, dtype=np.int8)
                    for _ in range(2):
                        info['puck_owner'] = 6
                        env.reset()
                        info['puck_owner'] = -256
                        _, reward, terminated, _, _ = env.step(action)
                        self.assertEqual(reward, 0.0)
                        self.assertFalse(terminated)
                        info['p2_shots'] += 1
                        before = inner.env.steps
                        _, reward, terminated, _, _ = env.step(action)
                        self.assertEqual(reward, -1.0)
                        self.assertTrue(terminated)
                        self.assertEqual(inner.env.steps, before + 1)

    def test_defense_outcome_frame_shot_penalties(self):
        for task in ('DefenseZone', 'SelfPlayDefenseFinetune'):
            for outcome in ('recovery', 'save', 'conceded', 'timeout'):
                with self.subTest(task=task, outcome=outcome):
                    inner = wrapped(rf=task)
                    env = StochasticFrameSkip(inner, 4, -1)
                    self.addCleanup(env.close)
                    env.reset()
                    info = inner.env.info
                    info['p2_shots'] += 1
                    info['puck_owner'] = 5 if outcome == 'save' else 0
                    if outcome == 'conceded':
                        info['p2_score'] += 1
                    elif outcome == 'timeout':
                        info.update(puck_owner=-256, time=199)
                    _, reward, terminated, _, _ = env.step(np.zeros(12, dtype=np.int8))
                    self.assertEqual(reward, -1.0)
                    self.assertTrue(terminated)
                    self.assertEqual(inner.env.steps, 2)

    def test_defense_goalie_hold_continues_until_skater_recovers(self):
        for task in ('DefenseZone', 'SelfPlayDefenseFinetune'):
            for frame_skip in (1, 4, 10):
                with self.subTest(task=task, frame_skip=frame_skip):
                    inner = wrapped(rf=task)
                    env = StochasticFrameSkip(inner, frame_skip, -1)
                    self.addCleanup(env.close)
                    env.reset()
                    info = inner.env.info
                    info['puck_owner'] = 5
                    action = np.zeros(12, dtype=np.int8)
                    _, reward, terminated, _, _ = env.step(action)
                    self.assertEqual(reward, 0.0)
                    self.assertFalse(terminated)
                    _, reward, terminated, _, _ = env.step(action)
                    self.assertEqual(reward, 0.0)
                    self.assertFalse(terminated)
                    self.assertEqual(inner.env.steps, 1 + 2 * frame_skip)
                    info['puck_owner'] = 3
                    _, reward, terminated, _, _ = env.step(action)
                    self.assertEqual(reward, 1.0)
                    self.assertTrue(terminated)
                    self.assertEqual(inner.env.steps, 2 + 2 * frame_skip)

    def test_defense_no_event_frames_stay_zero_across_frame_skip_and_reset(self):
        for task in ('DefenseZone', 'SelfPlayDefenseFinetune'):
            for frame_skip in (1, 4, 10):
                with self.subTest(task=task, frame_skip=frame_skip):
                    inner = wrapped(rf=task)
                    env = StochasticFrameSkip(inner, frame_skip, -1)
                    self.addCleanup(env.close)
                    for _ in range(2):
                        env.reset()
                        info = inner.env.info
                        for team in ('p1', 'p2'):
                            for slot in ('', '_2', '_3', '_4', '_5'):
                                info[f'{team}{slot}_x'] = 100
                                info[f'{team}{slot}_y'] = 0
                        info.update(puck_owner=6, puck_x=0, puck_y=-180,
                                    p2_x=0, p2_y=-180, g1_x=100, g1_y=-264)
                        _, reward, terminated, _, _ = env.step(np.zeros(12, dtype=np.int8))
                        self.assertFalse(terminated)
                        self.assertEqual(reward, 0.0)

    def test_period_direction_and_goalie_control(self):
        env = wrapped()
        info = env.env.info
        info['p1_emptystar_x'] = info['g1_x']
        info['p1_emptystar_y'] = info['g1_y']
        info['puck_owner'] = 5
        for period in (1, 2, 3):
            info['period'] = period
            env.reset()
            self.assertEqual(env.game_state.team1.control, 0)
            self.assertIs(env.game_state.team1.get_controlled_player(), env.game_state.team1.goalie)
            self.assertEqual(env._should_flip_zones_for_opponent(), period == 2)
            view = env._build_opponent_view_state()
            expected_x = env.game_state.team2.players[0].x * (-1 if period == 2 else 1)
            self.assertEqual(view.team1.players[0].x, expected_x)

    @unittest.expectedFailure
    def test_frame_skip_stops_on_truncation(self):
        # Existing behavior: frame skipping only breaks on termination.
        base = FixtureEnv(truncate=True)
        env = StochasticFrameSkip(base, 4, -1)
        env.reset()
        _, _, _, truncated, _ = env.step(np.zeros(12))
        self.assertTrue(truncated)
        self.assertEqual(base.steps, 1)


if __name__ == '__main__':
    unittest.main()
