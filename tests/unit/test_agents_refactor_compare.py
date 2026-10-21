"""A refactor gate cannot pass by dropping seeds, settings, or failed periods."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from nhl94_ai.evaluation.agents_refactor_compare import CORE_OPTIONS, SIDES, compare


class AgentsRefactorComparisonTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()  # pylint: disable=consider-using-with
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.before, self.after = self.root / 'before', self.root / 'after'
        for directory, count in ((self.before, 20), (self.after, 2)):
            directory.mkdir()
            manifest = {'revision': 'baseline', 'sources': {'agent.py': 'a' * 64},
                        'assets': {'rom.md': 'b' * 64,
                                   'SabresVsMightyDucks.ManualGoalie.Start.state': 'c' * 64},
                        'dependencies': {'nhl94-ai': '1', 'stable-retro': '2', 'numpy': '3'},
                        'python': '3.12', 'machine': {'machine': 'test'}}
            self.write(directory / 'run-manifest.json', manifest)
            for cadence in (1, 4):
                settings = dict(CORE_OPTIONS, agent='classic-v1', matchups=list(SIDES), seed=8000,
                                seconds=300, frame_skip=cadence, workers=1, trials=count,
                                action_type='FILTERED', output=str(directory / f'native-{cadence}.json'))
                rows = []
                for matchup, side in SIDES.items():
                    for seed in range(8000, 8000 + count):
                        rows.append(dict(CORE_OPTIONS, matchup=matchup, side=side, seed=seed,
                                         agent='classic-v1', completed=True, clock_remaining=0, frames=9000,
                                         teams=[2, 0], lineup=[{'slot': i, 'name': str(i)} for i in range(12)],
                                         actions_sha256='d' * 64, initial_state_sha256='e' * 64,
                                         goals=[1, 2], shots=[3, 5], one_timers=[2, 4], one_timer_goals=[0, 1],
                                         one_timer_accounting={'started': 1, 'ended': 1}, decisions={'carry': 5}))
                report = {'protocol': 'nhl94-cpu-first-period-v2', 'settings': settings,
                          'sources': manifest['sources'], 'versions': manifest['dependencies'],
                          'shot_placement_sha256': None, 'goalie_model_sha256': None,
                          'retention_model_sha256': None, 'matches': rows}
                self.write(directory / f'native-{cadence}.json', report)

    @staticmethod
    def write(path, data):
        path.write_text(json.dumps(data), encoding='utf-8')

    def change(self, update, filename='native-1.json'):
        path = self.after / filename
        report = json.loads(path.read_text())
        update(report)
        self.write(path, report)

    def compare(self, **options):
        return compare(self.before, self.after, tier='check', **options)

    def test_exact_checkpoint_selects_eight_reference_periods_and_attributes_away_scores(self):
        result = self.compare()
        self.assertTrue(result['passed'])
        self.assertEqual(len(result['paired']), 8)
        ducks = next(row['after'] for row in result['summaries']
                     if row['matchup'] == 'ducks-sabres-manual')
        self.assertEqual((ducks['wins'], ducks['goals_for'], ducks['goals_against']), (2, 4, 2))
        self.assertEqual((ducks['shots_for'], ducks['shots_against']), (10, 6))

    def test_missing_duplicate_and_unexpected_seeds_are_rejected(self):
        path = self.after / 'native-1.json'
        original = path.read_text()
        changes = ((lambda r: r['matches'].pop(), 'Missing trials'),
                   (lambda r: r['matches'].append(deepcopy(r['matches'][0])), 'Duplicate trial'),
                   (lambda r: r['matches'][0].update(seed=8020), 'Unexpected trial'))
        for update, error in changes:
            with self.subTest(error=error):
                path.write_text(original)
                self.change(update)
                with self.assertRaisesRegex(ValueError, error):
                    self.compare()

    def test_wrong_profile_and_unknown_effective_setting_are_rejected(self):
        for key, value in (('frame_skip', 8), ('goalie_policy', 'off'), ('new_gameplay_flag', True)):
            with self.subTest(key=key):
                path = self.after / 'native-1.json'
                original = path.read_text()
                self.change(lambda r, key=key, value=value: r['settings'].update({key: value}))
                with self.assertRaisesRegex(ValueError, 'profile|settings'):
                    self.compare()
                path.write_text(original)

    def test_changed_assets_dependencies_and_missing_asset_provenance_fail(self):
        path = self.after / 'run-manifest.json'
        original = path.read_text()
        for update in (lambda r: r['assets'].update({'rom.md': 'f' * 64}),
                       lambda r: r['assets'].pop('SabresVsMightyDucks.ManualGoalie.Start.state'),
                       lambda r: r['dependencies'].update(numpy='other')):
            path.write_text(original)
            self.change(update, 'run-manifest.json')
            with self.assertRaises(ValueError):
                self.compare()

    def test_incomplete_failed_and_wrong_side_trials_fail(self):
        path = self.after / 'native-1.json'
        original = path.read_text()
        for values in ({'completed': False}, {'clock_remaining': 1}, {'error': 'crash'}, {'side': 2}):
            with self.subTest(values=values):
                path.write_text(original)
                self.change(lambda r, values=values: r['matches'][0].update(values))
                with self.assertRaisesRegex(ValueError, 'Incomplete|side'):
                    self.compare()

    def test_behavior_mode_reports_changes_without_automatically_accepting_them(self):
        self.change(lambda r: r['matches'][0].update(goals=[4, 2], actions_sha256='f' * 64))
        self.assertFalse(self.compare()['passed'])
        result = self.compare(mode='behavior')
        self.assertIsNone(result['passed'])
        self.assertTrue(result['requires_behavior_review'])
        self.assertEqual(result['summaries'][0]['goal_difference_delta'], 3)
        self.assertIn('goals', result['paired'][0]['gameplay'])

    def test_diagnostics_are_reported_separately_and_still_fail_exact_mode(self):
        self.change(lambda r: r['matches'][0].update(decisions={'carry': 6}))
        result = self.compare()
        self.assertFalse(result['passed'])
        self.assertFalse(result['paired'][0]['gameplay'])
        self.assertIn('decisions', result['paired'][0]['diagnostics'])

    def test_full_mode_cannot_accept_checkpoint_or_missing_cadence(self):
        with self.assertRaisesRegex(ValueError, 'trials'):
            compare(self.before, self.after)
        (self.after / 'native-4.json').unlink()
        with self.assertRaisesRegex(ValueError, 'cadence'):
            self.compare()

    def test_source_changes_are_allowed_only_when_the_run_manifest_agrees(self):
        self.change(lambda r: r['sources'].update({'agent.py': 'f' * 64}))
        with self.assertRaisesRegex(ValueError, 'Source/manifest'):
            self.compare()
        self.change(lambda r: r['sources'].update({'agent.py': 'f' * 64}), 'run-manifest.json')
        self.change(lambda r: r['sources'].update({'agent.py': 'f' * 64}), 'native-4.json')
        self.assertTrue(self.compare()['passed'])
