"""Episode totals exclude the save's counters and retain first-action events."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from nhl94_ai.config import EvaluationConfig
from nhl94_ai.evaluation.metrics import LiveTeamTotals, evaluate_policy_with_totals, reset_team_totals
from nhl94_ai.game.state import NHL94GameState, Stats
from nhl94_ai.game.specs import GAMES
from nhl94_ai.tasks.defensezone import isdone_defensezone, rf_defensezone


def saved_info():
    path = Path(__file__).resolve().parents[1] / 'fixtures/NHL94-Genesis-v0.json'
    info = json.loads(path.read_text(encoding='utf-8'))
    for team in (1, 2):
        info.update({f'p{team}_score': 3, f'p{team}_passing': 19,
                     f'p{team}_bodychecks': 7, f'p{team}_shots': (123 << 16) | 12,
                     f'p{team}_onetimer': (321 << 16) | 4})
    return info


class StatisticsEnv:
    def __init__(self, events=None, terminal=True):
        self.events = events or {}
        self.terminal = terminal
        self.reset_infos = []

    def reset(self):
        self.reset_infos = [saved_info()]
        return np.zeros((1, 310), dtype=np.float32)

    def step(self, action):
        info = saved_info()
        for key, value in self.events.items():
            info[key] += value
        # A VecEnv replaces reset_infos before returning a terminal transition.
        self.reset_infos = [{key: 0 for key in info}]
        return np.zeros((1, 310)), np.array([1.0]), np.array([self.terminal]), [info]


class EpisodeStatisticsTests(unittest.TestCase):
    def evaluate(self, events=None, terminal=True):
        model = SimpleNamespace(predict=lambda *args, **kwargs: (np.zeros((1, 12)), None))
        return evaluate_policy_with_totals(
            model, StatisticsEnv(events, terminal), n_eval_episodes=20,
            config=EvaluationConfig(episodes=20, max_steps=1, deterministic=True),
            deterministic=True, env_name='NHL94-Genesis-v0', num_players=1,
        )

    def test_saved_counters_are_not_counted_again_each_episode(self):
        for terminal in (False, True):
            mean, std, home, away = self.evaluate(terminal=terminal)
            self.assertEqual((mean, std), (1.0, 0.0))
            self.assertEqual(home, LiveTeamTotals())
            self.assertEqual(away, LiveTeamTotals())

    def test_first_action_events_count_even_when_that_action_ends_the_episode(self):
        _, _, home, away = self.evaluate({'p1_passing': 1, 'p2_shots': 1, 'p2_score': 1, 'p2_onetimer': 1})
        self.assertEqual(home, LiveTeamTotals(passes=20))
        self.assertEqual(away, LiveTeamTotals(goals=20, shots=20, one_timers=20))

    def test_neighboring_high_words_are_not_shot_or_one_timer_counts(self):
        totals = LiveTeamTotals.from_game_stats(Stats(shots=9, onetimer=(123 << 16) | 3))
        self.assertEqual(totals, LiveTeamTotals(shots=9, one_timers=3))
        baseline = LiveTeamTotals.from_info(saved_info(), 1)
        changed = saved_info()
        changed['p1_shots'] += 1 << 16
        changed['p1_onetimer'] -= 1 << 16
        events = LiveTeamTotals()
        events.add_delta(baseline, LiveTeamTotals.from_info(changed, 1))
        self.assertEqual(events, LiveTeamTotals())

    def test_frame_stacking_does_not_hide_reset_counter_baselines(self):
        from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
        from tests.unit.test_environment_contracts import wrapped
        env = wrapped()
        env.unwrapped.render_mode = 'rgb_array'
        env.unwrapped.info.update(saved_info())
        vector = VecFrameStack(DummyVecEnv([lambda: env]), n_stack=4)
        self.addCleanup(vector.close)
        vector.reset()
        self.assertEqual(reset_team_totals(vector),
                         (LiveTeamTotals.from_info(saved_info(), 1), LiveTeamTotals.from_info(saved_info(), 2)))

    @unittest.skipUnless(importlib.util.find_spec('pygame'), 'optional display extra is not installed')
    def test_live_replay_baseline_is_reset_without_inflating_totals(self):
        from nhl94_ai.ui.live import LiveTrainingDisplay
        display = LiveTrainingDisplay.__new__(LiveTrainingDisplay)
        display.total_sessions_played = 0
        display.uses_nhl94_gamestate = True
        display.target_agent = None
        display.env = SimpleNamespace(reset_infos=[saved_info()])
        display.total_team1_stats = LiveTeamTotals()
        display.total_team2_stats = LiveTeamTotals()
        display.game_state = NHL94GameState(5)
        for _ in range(100):
            display._begin_new_replay_session()
            info = copy.deepcopy(display.env.reset_infos[0])
            info['p2_shots'] += 1
            display.game_state.BeginFrame(info, [0] * 6)
            display.game_state.EndFrame()
            display._update_total_game_stats()
            display._update_total_game_stats()
        self.assertEqual(display.total_team1_stats, LiveTeamTotals())
        self.assertEqual(display.total_team2_stats, LiveTeamTotals(shots=100))


class ShotDecodingTests(unittest.TestCase):
    def test_raw_shot_counts_decode_once_for_both_teams_and_all_variants(self):
        root = Path(__file__).resolve().parents[1] / 'fixtures'
        for game, spec in GAMES.items():
            with self.subTest(game=game):
                info = json.loads((root / f'{game}.json').read_text(encoding='utf-8'))
                state = NHL94GameState(spec.skaters_per_team)
                for high in (0, 1, 65535, 0):
                    info.update(p1_shots=(high << 16) | 9, p2_shots=(high << 16) | 12)
                    original = info.copy()
                    state.BeginFrame(info, [0] * 6)
                    self.assertEqual(state.team1.stats.shots, 9)
                    self.assertEqual(state.team2.stats.shots, 12)
                    self.assertEqual(info, original)
                    for team_id, team in ((1, state.team1), (2, state.team2)):
                        self.assertEqual(LiveTeamTotals.from_game_stats(team.stats),
                                         LiveTeamTotals.from_info(info, team_id))
                    state.EndFrame()
                    self.assertEqual(state.team1.last_stats.shots, 9)
                    self.assertEqual(state.team2.last_stats.shots, 12)
                info['p2_shots'] += 1
                state.BeginFrame(info, [0] * 6)
                self.assertEqual(state.team2.last_stats.shots, 12)
                self.assertEqual(state.team2.stats.shots, 13)

    def test_neighbor_changes_leave_observations_and_defense_outcomes_unchanged(self):
        from tests.unit.test_environment_contracts import wrapped
        for task in ('DefenseZone', 'SelfPlayDefenseFinetune'):
            with self.subTest(task=task):
                env = wrapped(rf=task)
                self.addCleanup(env.close)
                info = env.env.info
                info.update(p1_shots=9, p2_shots=12)
                env.reset(seed=0)
                action = np.zeros(12, dtype=np.int8)
                baseline, _, _, _, _ = env.step(action)
                for high in (1, 65535, 0):
                    info.update(p1_shots=(high << 16) | 9, p2_shots=(high << 16) | 12)
                    observation, reward, done, _, _ = env.step(action)
                    self.assertEqual(reward, 0.0)
                    self.assertFalse(done)
                    self.assertEqual(rf_defensezone(env.game_state), 0.0)
                    self.assertFalse(isdone_defensezone(env.game_state))
                    np.testing.assert_array_equal(observation, baseline)
                info['p2_shots'] += 1
                _, reward, done, _, _ = env.step(action)
                self.assertEqual(reward, -1.0)
                self.assertTrue(done)

    def test_missing_shot_field_fails_at_decoding_boundary(self):
        info = saved_info()
        del info['p1_shots']
        with self.assertRaisesRegex(KeyError, 'p1_shots'):
            NHL94GameState(5).BeginFrame(info, [0] * 6)

    @unittest.skipUnless(importlib.util.find_spec('pygame'), 'optional display extra is not installed')
    def test_raw_info_display_decodes_shots_without_showing_neighbor_words(self):
        from nhl94_ai.ui.game import NHL94GameDisplayEnv
        display = NHL94GameDisplayEnv.__new__(NHL94GameDisplayEnv)
        display.positions = {'stats': (0, 0)}
        display.info_font = None
        display.draw_string = Mock()
        display.draw_game_stats([saved_info()])
        labels = [call.args[1] for call in display.draw_string.call_args_list]
        self.assertIn('P1 SHOTS: 12', labels)
        self.assertIn('P2 SHOTS: 12', labels)


if __name__ == '__main__':
    unittest.main()
