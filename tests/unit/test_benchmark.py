"""Fairness and accounting contracts for the paired NHL94 benchmark."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from nhl94_ai.evaluation.benchmark import away_view, build_parser, make_agent, run, summarize, update_state
from nhl94_ai.game.state import NHL94GameState


class BenchmarkContracts(unittest.TestCase):
    def test_both_teams_receive_control_and_possession_from_slots(self):
        path = Path(__file__).resolve().parents[1] / 'fixtures/NHL94-Genesis-v0.json'
        info = json.loads(path.read_text(encoding='utf-8'))
        info.update(bench_control1=1, bench_control2=8, bench_shots1=3, bench_shots2=7, puck_owner=8)
        original = copy.deepcopy(info)
        state = NHL94GameState(5)
        update_state(state, info)
        self.assertEqual(state.team1.control, 2)
        self.assertEqual(state.team2.control, 3)
        self.assertFalse(state.team1.player_haspuck)
        self.assertTrue(state.team2.player_haspuck)
        self.assertEqual(state.team1.stats.shots, 3)
        self.assertEqual(state.team2.stats.shots, 7)
        self.assertEqual(info, original)
        info['puck_owner'] = -256
        update_state(state, info)
        self.assertFalse(state.team1.player_haspuck or state.team2.player_haspuck)

    def test_away_view_preserves_physical_coordinates_and_net_direction(self):
        state = NHL94GameState(5)
        state.team1.net.y, state.team2.net.y = -264, 264
        state.team2.players[0].y = 219
        view = away_view(state)
        self.assertIsNot(view, state)
        self.assertIs(view.team1, state.team2)
        self.assertEqual(view.team1.get_controlled_player().y, state.team2.goalie.y)
        self.assertEqual(view.team1.players[0].y, 219)
        self.assertEqual(view.team2.net.y, -264)
        self.assertEqual(state.team1.net.y, -264)

    def test_summary_uses_both_sides_and_excludes_incomplete_scores(self):
        common = {'completed': True, 'decision_ns': [1000, 2000], 'decisions': [1, 1]}
        results = [dict(common, agents=['candidate', 'baseline'], goals=[3, 1]),
                   dict(common, agents=['baseline', 'candidate'], goals=[0, 0]),
                   dict(common, agents=['baseline', 'candidate'], goals=[2, 1]),
                   dict(common, agents=['candidate', 'baseline'], goals=[99, 0], completed=False)]
        row = summarize(results, 'candidate', ['baseline'])['baseline']
        self.assertEqual([row[key] for key in ('games', 'wins', 'draws', 'losses', 'goals_for', 'goals_against')],
                         [3, 1, 1, 1, 4, 3])

    def test_each_seed_is_played_on_each_side(self):
        args = build_parser().parse_args(['--seed', '7', '--pairs', '2'])
        fixtures = []
        def fake_match(fixture):
            fixtures.append(fixture)
            home, away, seed, _, _ = fixture
            return {'agents': [home, away], 'seed': seed, 'completed': True,
                    'goals': [0, 0], 'decision_ns': [1, 1], 'decisions': [1, 1]}
        with patch('nhl94_ai.evaluation.benchmark.match', side_effect=fake_match), patch('builtins.print'):
            run(args)
        self.assertEqual(args.opponents, ['classic-v1-direct'])
        self.assertEqual(len(fixtures), 4)
        for seed in (7, 8):
            for rival in args.opponents:
                self.assertIn(('classic-v1', rival, seed, 300, 4), fixtures)
                self.assertIn((rival, 'classic-v1', seed, 300, 4), fixtures)

    def test_invalid_protocol_is_rejected_before_loading_rom(self):
        for flags in (['--pairs', '0'], ['--seed', '-1'], ['--frame-skip', '0'],
                      ['--opponents', 'classic-v1'], ['--opponents', 'classic'],
                      ['--agent', 'classic', '--opponents', 'classic-v1'],
                      ['--opponents', 'classic-v1-direct', 'classic-v1-direct'],
                      ['--agent', 'classic-v1-direct', '--opponents', 'classic', 'classic-v1']):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                run(build_parser().parse_args(flags))

    def test_incomplete_matches_are_saved_but_fail_the_run(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / 'report.json'
            args = build_parser().parse_args(['--pairs', '1', '--output', str(output)])
            def unfinished(fixture):
                return {'agents': list(fixture[:2]), 'completed': False, 'goals': [99, 0]}
            with patch('nhl94_ai.evaluation.benchmark.match', side_effect=unfinished), patch('builtins.print'):
                with self.assertRaisesRegex(RuntimeError, 'Incomplete matches'):
                    run(args)
            report = json.loads(output.read_text(encoding='utf-8'))
            self.assertEqual(len(report['matches']), 2)
            self.assertTrue(all(row['games'] == 0 for row in report['summary'].values()))

    def test_ablation_only_disables_the_one_timer_branch(self):
        candidate, baseline = make_agent('classic-v1'), make_agent('classic-v1-direct')
        self.assertIs(type(candidate.controller), type(baseline.controller))
        self.assertTrue(candidate._one_timers)
        self.assertFalse(baseline._one_timers)

    def test_one_timer_counters_follow_the_agents_when_sides_swap(self):
        common = {'completed': True, 'goals': [0, 0], 'decision_ns': [1, 1], 'decisions': [1, 1]}
        rows = [dict(common, agents=['candidate', 'base'], one_timers=[2, 1], one_timer_goals=[1, 0]),
                dict(common, agents=['base', 'candidate'], one_timers=[0, 3], one_timer_goals=[0, 2])]
        summary = summarize(rows, 'candidate', ['base'])['base']
        self.assertEqual(summary['one_timers'], {'candidate': 5, 'base': 1})
        self.assertEqual(summary['one_timer_goals'], {'candidate': 3, 'base': 0})


if __name__ == '__main__':
    unittest.main()
