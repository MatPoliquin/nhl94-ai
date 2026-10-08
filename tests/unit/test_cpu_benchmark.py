"""CPU ownership, roster identity, and complete-period accounting."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from nhl94_ai.evaluation.cpu_benchmark import MATCHUPS, build_parser, lineup, run, select_side, summarize
from nhl94_ai.game.ram import register_goalie_motion, register_pass_state, register_skater_ratings, restore_away_control
from nhl94_ai.game.state import NHL94GameState


class CpuBenchmarkContracts(unittest.TestCase):
    def test_lookahead_flag_reaches_trials_and_preserves_other_options(self):
        fixtures = []
        def match(fixture):
            fixtures.append(fixture)
            return dict(matchup=fixture[1], side=MATCHUPS[fixture[1]][1], completed=True,
                        goals=[0, 0], one_timers=[0, 0], one_timer_goals=[0, 0], decisions={})
        for flags in ([], ['--chance-creation', '--uncertain-carry', '--deke', '--cross-crease']):
            args = build_parser().parse_args(['--offense-lookahead', '--trials', '1', *flags])
            fixtures.clear()
            with patch('nhl94_ai.evaluation.cpu_benchmark.cpu_match', side_effect=match), patch('builtins.print'):
                report = run(args)
            self.assertTrue(report['settings']['offense_lookahead'])
            self.assertTrue(all(fixture[-5:] == (bool(flags),) * 4 + (True,) for fixture in fixtures))
        self.assertFalse(build_parser().parse_args([]).offense_lookahead)

    def test_chance_creation_flag_reaches_each_trial_without_losing_other_options(self):
        args = build_parser().parse_args(
            ['--chance-creation', '--uncertain-carry', '--deke', '--cross-crease', '--trials', '1'])
        fixtures = []
        def match(fixture):
            fixtures.append(fixture)
            return dict(matchup=fixture[1], side=MATCHUPS[fixture[1]][1], completed=True,
                        goals=[0, 0], one_timers=[0, 0], one_timer_goals=[0, 0], decisions={})
        with patch('nhl94_ai.evaluation.cpu_benchmark.cpu_match', side_effect=match), patch('builtins.print'):
            report = run(args)
        self.assertTrue(report['settings']['chance_creation'])
        self.assertTrue(all(fixture[-4:] == (True, True, True, True) for fixture in fixtures))
        self.assertFalse(build_parser().parse_args([]).chance_creation)

    def test_uncertain_carry_flag_reaches_every_trial(self):
        args = build_parser().parse_args(['--uncertain-carry', '--trials', '1'])
        fixtures = []
        def match(fixture):
            fixtures.append(fixture)
            return dict(matchup=fixture[1], side=MATCHUPS[fixture[1]][1], completed=True,
                        goals=[0, 0], one_timers=[0, 0], one_timer_goals=[0, 0], decisions={})
        with patch('nhl94_ai.evaluation.cpu_benchmark.cpu_match', side_effect=match), patch('builtins.print'):
            report = run(args)
        self.assertTrue(report['settings']['uncertain_carry'])
        self.assertTrue(all(fixture[-3:] == (False, False, True) for fixture in fixtures))
        self.assertFalse(build_parser().parse_args([]).uncertain_carry)

    def test_campbell_matchups_are_opt_in_and_share_the_single_controller_save(self):
        parser = build_parser()
        self.assertEqual(parser.parse_args([]).matchups,
                         ['penguins-senators', 'senators-penguins', 'nordiques-canadiens'])
        for matchup, side in (('ducks-campbell-manual', 1), ('campbell-ducks-manual', 2)):
            with self.subTest(matchup=matchup):
                self.assertEqual(MATCHUPS[matchup],
                                 ('MightyDucksVsAllStarCampbell.ManualGoalie.Start', side))
                self.assertEqual(parser.parse_args(['--matchups', matchup]).matchups, [matchup])

    def test_campbell_trials_use_matching_seeds_and_selective_goalies_on_both_sides(self):
        matchups = ['ducks-campbell-manual', 'campbell-ducks-manual']
        args = build_parser().parse_args(
            ['--matchups', *matchups, '--trials', '2', '--seed', '7', '--goalie-policy', 'selective'])
        fixtures = []
        def match(fixture):
            fixtures.append(fixture)
            return dict(matchup=fixture[1], side=MATCHUPS[fixture[1]][1], completed=True,
                        goals=[0, 0], one_timers=[0, 0], one_timer_goals=[0, 0], decisions={})
        with patch('nhl94_ai.evaluation.cpu_benchmark.cpu_match', side_effect=match), patch('builtins.print'):
            report = run(args)
        self.assertEqual(fixtures, [
            ('classic-v1', matchup, seed, 300, 4, 'FILTERED', 'selective')
            for matchup in matchups for seed in (7, 8)])
        self.assertEqual([report['summary'][matchup]['periods'] for matchup in matchups], [2, 2])

    def test_cross_crease_option_reaches_each_trial_and_report(self):
        args = build_parser().parse_args(['--cross-crease', '--trials', '1'])
        fixtures = []
        def match(fixture):
            fixtures.append(fixture)
            return dict(matchup=fixture[1], side=MATCHUPS[fixture[1]][1], completed=True,
                        goals=[0, 0], one_timers=[0, 0], one_timer_goals=[0, 0], decisions={},
                        cross_crease_metrics={'attempts': 2, 'recorded_shots': 1})
        with patch('nhl94_ai.evaluation.cpu_benchmark.cpu_match', side_effect=match), patch('builtins.print'):
            report = run(args)
        self.assertTrue(all(fixture[-1] is True for fixture in fixtures))
        self.assertTrue(report['settings']['cross_crease'])
        for row in report['summary'].values():
            self.assertEqual(row['cross_crease_metrics'], {'attempts': 2, 'recorded_shots': 1})

    def test_optional_goalie_motion_and_pass_fields_do_not_replace_legacy_velocity(self):
        env = Mock()
        register_pass_state(env)
        register_goalie_motion(env)
        fields = dict(call.args for call in env.data.set_variable.call_args_list)
        self.assertEqual(fields['pass_target'], {'address': 0xFFBEE0, 'type': '>i2'})
        self.assertEqual(fields['p1_one_timer_attempts'], {'address': 0xFFCA2A, 'type': '>u2'})
        self.assertEqual(fields['p2_one_timer_attempts'], {'address': 0xFFCD8E, 'type': '>u2'})
        self.assertEqual(fields['g2_live_vel_x'], {'address': 0xFFB04A + 11 * 0x80 + 0x28, 'type': '>i2'})
        self.assertNotIn('g2_vel_x', fields)
        info = json.loads((Path(__file__).resolve().parents[1] /
                           'fixtures/NHL94-Genesis-v0.json').read_text(encoding='utf-8'))
        info.update(g2_live_vel_x=-4096, g2_live_vel_y=2048, pass_target=7,
                    p1_one_timer_attempts=0, p2_one_timer_attempts=3)
        state = NHL94GameState(5)
        state.BeginFrame(info, [0] * 6)
        self.assertEqual(state.team2.goalie.motion_x, -4096 * 17 / 65536)
        self.assertEqual(state.team2.goalie.motion_y, 2048 * 17 / 65536)
        self.assertEqual(state.team2.goalie.vx, info.get('g2_vel_x', 0))
        self.assertEqual(state.engine.pass_target, 7)
        self.assertEqual(state.team1.one_timer_attempts, 0)
        self.assertEqual(state.team2.one_timer_attempts, 3)
        del info['g2_live_vel_x']
        state.BeginFrame(info, [0] * 6)
        self.assertIsNone(state.team2.goalie.motion_x)

    def test_away_assignment_releases_home_player_to_the_cpu(self):
        data = Mock()
        info = dict(bench_team1=1, bench_team2=0, period=0, bench_control1=2,
                    cpu_2_flags=0x88, cpu_8_flags=0x40)
        select_side(data, info, 2)
        changes = dict(call.args for call in data.set_value.call_args_list)
        self.assertEqual(changes, {'bench_team1': 2, 'bench_control1': 8,
                                   'cpu_2_flags': 0x82, 'cpu_8_flags': 0x48})
        # Human-vs-human saves must never silently stand in for the CPU.
        with self.assertRaises(ValueError):
            select_side(data, dict(info, bench_team2=2), 2)

    def test_away_restoration_uses_benchmark_aliases_and_refreshes_shared_feedback(self):
        info = dict(bench_team1=2, bench_team2=0, bench_control1=3,
                    cpu_3_flags=0xA8, cpu_9_flags=0x40, defense_control1=3)
        data = Mock()
        data.set_value.side_effect = info.__setitem__
        data.update_ram.side_effect = lambda: info.update(defense_control1=info['bench_control1'])
        data.lookup_all.side_effect = lambda: dict(info)
        with self.assertWarns(RuntimeWarning):
            corrected = restore_away_control(data, info, controller_prefix='bench', player_prefix='cpu', slots=6)
        self.assertEqual(corrected['bench_control1'], 9)
        self.assertEqual(corrected['defense_control1'], 9)
        self.assertEqual(corrected['cpu_3_flags'], 0xA2)
        self.assertEqual(corrected['cpu_9_flags'], 0x48)

    def test_missing_accuracy_is_distinct_from_a_real_zero(self):
        info = json.loads((Path(__file__).resolve().parents[1] /
                           'fixtures/NHL94-Genesis-v0.json').read_text(encoding='utf-8'))
        state = NHL94GameState(5)
        info.update(p1_shot_accuracy=0, p1_2_shot_accuracy=30, p2_shot_accuracy=6)
        state.BeginFrame(info, [0] * 6)
        self.assertEqual(state.team1.players[0].shot_accuracy, 0)
        self.assertEqual(state.team1.players[1].shot_accuracy, 30)
        self.assertEqual(state.team2.players[0].shot_accuracy, 6)
        self.assertIsNone(state.team1.players[2].shot_accuracy)
        del info['p1_shot_accuracy']
        state.BeginFrame(info, [0] * 6)
        self.assertIsNone(state.team1.players[0].shot_accuracy)

    def test_rating_registration_skips_goalies_and_unused_slots(self):
        env = Mock()
        register_skater_ratings(env, 2)
        fields = dict(call.args for call in env.data.set_variable.call_args_list)
        self.assertEqual(set(fields), {'p1_shot_accuracy', 'p1_2_shot_accuracy',
                                       'p2_shot_accuracy', 'p2_2_shot_accuracy'})
        self.assertEqual(fields['p2_2_shot_accuracy'],
                         {'address': 0xFFB04A + 7 * 0x80 + 0x6D, 'type': '|u1'})

    def test_reduced_variant_goalie_motion_uses_live_goalie_slot(self):
        env = Mock()
        register_goalie_motion(env, 2)
        fields = dict(call.args for call in env.data.set_variable.call_args_list)
        self.assertEqual(fields['g1_live_vel_x'], {'address': 0xFFB04A + 2 * 0x80 + 0x28, 'type': '>i2'})
        self.assertEqual(fields['g2_live_vel_y'], {'address': 0xFFB04A + 8 * 0x80 + 0x2A, 'type': '>i2'})

    def test_lineup_names_follow_roster_indices(self):
        rom = bytearray(128)
        rom[10:12] = (10).to_bytes(2, 'big')
        for address, name in ((20, b'First'), (35, b'Second')):
            rom[address:address + 2] = (len(name) + 2).to_bytes(2, 'big')
            rom[address + 2:address + 2 + len(name)] = name
        info = dict(home_roster=10, away_roster=10)
        for slot in range(12):
            info[f'cpu_{slot}_roster'] = int(slot >= 6)
            info[f'cpu_{slot}_accuracy'] = 7
        players = lineup(info, rom)
        self.assertEqual(players[0]['name'], 'First')
        self.assertEqual(players[6]['name'], 'Second')
        self.assertEqual(players[6]['shot_accuracy'], 7)
        self.assertIsNone(players[11]['shot_accuracy'])

    def test_away_scores_and_one_timers_are_attributed_to_the_candidate(self):
        common = dict(matchup='senators-penguins', side=2, completed=True,
                      decisions={'one-timer-pass': 3}, one_timers=[1, 2], one_timer_goals=[0, 1])
        rows = [dict(common, goals=[0, 2]), dict(common, goals=[1, 1]),
                dict(common, goals=[0, 99], completed=False)]
        result = summarize(rows)['senators-penguins']
        self.assertEqual([result[key] for key in ('periods', 'wins', 'draws', 'losses',
                                                 'goals_for', 'goals_against')], [2, 1, 1, 0, 3, 1])
        self.assertEqual(result['one_timers'], 4)
        self.assertEqual(result['one_timer_goals'], 2)

    def test_carrier_defense_metrics_only_summarize_completed_periods(self):
        common = dict(matchup='senators-penguins', side=2, completed=True, goals=[0, 0],
                      decisions={}, one_timers=[0, 0], one_timer_goals=[0, 0])
        rows = [dict(common, carrier_defense_metrics={'neutral': {'frames': 8, 'cutoff_frames': 2}}),
                dict(common, carrier_defense_metrics={'neutral': {'frames': 4, 'cutoff_frames': 3}}),
                dict(common, completed=False, carrier_defense_metrics={'neutral': {'frames': 99}})]
        self.assertEqual(summarize(rows)['senators-penguins']['carrier_defense_metrics'],
                         {'neutral': {'frames': 12, 'cutoff_frames': 5}})

    def test_invalid_protocol_and_partial_periods_fail(self):
        for flags in (['--trials', '0'], ['--seed', '-1'], ['--seconds', '65536'],
                      ['--matchups', 'senators-penguins', 'senators-penguins']):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                run(build_parser().parse_args(flags))
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'partial.json'
            args = build_parser().parse_args(['--trials', '1', '--output', str(path)])
            def partial(fixture):
                return {'matchup': fixture[1], 'completed': False}
            with patch('nhl94_ai.evaluation.cpu_benchmark.cpu_match', side_effect=partial), patch('builtins.print'):
                with self.assertRaisesRegex(RuntimeError, 'Incomplete CPU'):
                    run(args)
            report = json.loads(path.read_text(encoding='utf-8'))
            self.assertTrue(all(row['periods'] == 0 for row in report['summary'].values()))


if __name__ == '__main__':
    unittest.main()
