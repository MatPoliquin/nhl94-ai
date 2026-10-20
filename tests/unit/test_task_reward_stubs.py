"""Zero rewards preserve task progression and cannot masquerade as evaluation scores."""
from dataclasses import replace
import importlib
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from nhl94_ai.config import EvaluationConfig
from nhl94_ai.env.wrappers import StochasticFrameSkip
from nhl94_ai.evaluation.metrics import evaluate_policy_with_totals
from nhl94_ai.evaluation.runner import evaluate
from nhl94_ai.game.state import NHL94GameState
from nhl94_ai.tasks.registry import TASKS, get_task
from tests.unit.test_environment_contracts import wrapped
from tests.unit.test_episode_statistics import StatisticsEnv


def attack_state():
    state = NHL94GameState(5)
    state.time = 300
    state.update_nets()
    state.team1.control = 1
    state.team1.player_haspuck = True
    state.puck.x, state.puck.y = 0, 220
    state.team1.players[0].x, state.team1.players[0].y = 0, 220
    return state


class TaskRewardStubTests(unittest.TestCase):
    def test_every_reward_entrypoint_is_a_stateless_zero_stub(self):
        root = Path(__file__).resolve().parents[2] / 'nhl94_ai/tasks'
        for path in root.glob('*.py'):
            module = importlib.import_module('nhl94_ai.tasks.' + path.stem)
            for name, function in vars(module).items():
                if name.startswith('rf_') and callable(function):
                    with self.subTest(module=module.__name__, function=name):
                        # None cannot supply telemetry or receive tracker side effects.
                        self.assertEqual(function(None), 0.0)
        for name, task in TASKS.items():
            with self.subTest(task=name):
                self.assertFalse(task.reward_available)
                self.assertEqual(task.reward(None), 0.0)

    def test_progression_precedes_completion_once_per_native_frame(self):
        advances = []
        for skip in (1, 4, 10):
            with self.subTest(skip=skip):
                inner = wrapped()
                env = StochasticFrameSkip(inner, skip, -1)
                self.addCleanup(env.close)
                inner.task = replace(inner.task, advance=advances.append)
                inner.done_function = lambda state: len(advances) >= 3
                for _ in range(2):
                    advances.clear()
                    env.reset()
                    self.assertFalse(advances)
                    done = False
                    while not done:
                        _, reward, done, _, info = env.step(np.zeros(12, dtype=np.int8))
                        self.assertEqual(reward, 0.0)
                        self.assertEqual(info['reward_status'], 'unavailable')
                    self.assertEqual(len(advances), 3)

    def test_stall_deadlines_remain_exact_without_reward_calls(self):
        for task_name, deadline in [('ScoreGoalV2', 45), ('CrossCreaseV2', 35)]:
            with self.subTest(task=task_name):
                state = attack_state()
                task = get_task(task_name)
                for frame in range(1, deadline + 1):
                    task.advance(state)
                    self.assertEqual(task.done(state), frame == deadline)
                    state.EndFrame()

    def test_moving_possession_deadlines_and_shot_release_are_preserved(self):
        for task_name, deadline in [('ScoreGoalV2', 240), ('CrossCreaseV2', 220)]:
            for shoot in (False, True):
                with self.subTest(task=task_name, shoot=shoot):
                    state = attack_state()
                    state.team1.players[0].vx = 6
                    task = get_task(task_name)
                    for frame in range(1, deadline + 1):
                        state.engine.shot_taken = shoot and frame == 5
                        task.advance(state)
                        self.assertEqual(task.done(state), frame == deadline and not shoot)
                        state.EndFrame()

    def test_cross_pass_window_still_ends_on_its_original_frame(self):
        task = get_task('CrossCreaseV2')
        state = attack_state()
        state.team1.players[0].x = state.puck.x = 50
        task.advance(state)
        state.EndFrame()
        state.team1.players[0].x = state.puck.x = -10
        state.team1.stats.passing += 1
        for frame in range(30):
            task.advance(state)
            self.assertEqual(task.done(state), frame == 29)
            state.EndFrame()

    def test_createopp_completes_on_the_pass_after_an_admitted_opportunity(self):
        state = attack_state()
        task = get_task('CreateOpp')
        with patch('nhl94_ai.tasks.createopp._best_opportunity_scores', return_value=(0.8, 0, 0)):
            task.advance(state)
            self.assertFalse(task.done(state))
            state.EndFrame()
            state.team1.stats.passing += 1
            task.advance(state)
            self.assertTrue(task.done(state))
        self.assertEqual(task.reward(None), 0.0)

    def test_headless_evaluation_marks_rewards_unavailable_but_keeps_episode_lengths(self):
        env = wrapped()
        self.addCleanup(env.close)
        agent = SimpleNamespace(reset=lambda: None, act=lambda *args: SimpleNamespace(action=np.zeros(12, dtype=np.int8)))
        result = evaluate(agent, EvaluationConfig(episodes=2, max_steps=2, seed=0), env=env)
        self.assertEqual(result.rewards, [None, None])
        self.assertEqual(result.lengths, [2, 2])
        self.assertEqual(result.budget_exhausted, [True, True])
        self.assertEqual(result.metadata['reward_status'], 'unavailable')

    def test_live_evaluation_preserves_goal_totals_with_unavailable_rewards(self):
        env = StatisticsEnv({'p1_score': 1})
        original_step = env.step

        def step(action):
            obs, reward, done, info = original_step(action)
            info[0]['reward_status'] = 'unavailable'
            return obs, reward, done, info

        env.step = step
        reporter = Mock()
        model = SimpleNamespace(predict=lambda *args, **kwargs: (np.zeros((1, 12)), None))
        mean, std, home, away = evaluate_policy_with_totals(
            model, env, n_eval_episodes=2, deterministic=True, env_name='NHL94-Genesis-v0',
            num_players=1, episode_reporter=reporter)
        self.assertEqual((mean, std), (None, None))
        self.assertEqual((home.goals, away.goals), (2, 0))
        self.assertTrue(all(call.args[1] is None for call in reporter.call_args_list))

    def test_shootout_state_no_longer_computes_bonus_only_geometry(self):
        from tests.unit.test_environment_contracts import FixtureEnv
        state = NHL94GameState(5)
        info = FixtureEnv().info
        info['ba_ps_flags'] = 0x0400
        with patch.object(state, '_has_viable_shooting_opening', side_effect=AssertionError('reward-only work')):
            state.BeginFrame(info, [0] * 6)
            state.EndFrame()
