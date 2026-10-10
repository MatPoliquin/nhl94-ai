"""Native alternatives stay grouped, finite, and separate from model evaluation."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from nhl94_ai.training.shot_placement import estimate, evaluate, fit_logistic, main, read_cases


def dataset(seed):
    case = {'frame': 100, 'branches': [
        {'aim': aim, 'hold': hold, 'features': {'aim': aim, 'hold': hold/12},
         'outcome': {'goal': int(hold == 12), 'resolved': True, 'held_c_frames': hold, 'frames': 60}}
        for aim, hold in ((-1, 4), (1, 4), (-1, 12), (1, 12))]}
    return {'protocol': 'native-primary-shot-alternatives-v3',
            'matches': [{'seed': seed, 'matchup': 'test', 'completed': True, 'cases': [case]}]}


class ShotTrainingTests(unittest.TestCase):
    def test_rejects_incorrect_execution_even_when_outcome_claims_resolved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'data.json'
            data = dataset(1)
            data['matches'][0]['cases'][0]['branches'][0]['outcome']['held_c_frames'] = 0
            path.write_text(json.dumps(data), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'requested hold'):
                read_cases(path)

    def test_regularized_fit_learns_from_native_labels_and_threshold_preserves_fallback(self):
        data = dataset(1)
        cases = [{**data['matches'][0]['cases'][0], 'seed': 1, 'matchup': 'test'}]*20
        model = fit_logistic(cases)
        branches = cases[0]['branches']
        self.assertGreater(estimate(model, branches[2]['features']), estimate(model, branches[0]['features']))
        self.assertEqual(evaluate(model, cases, threshold=0)['totals']['selected_goals'], 20)
        self.assertEqual(evaluate(model, cases, threshold=1)['totals']['overrides'], 0)

    def test_incomplete_alternative_excludes_entire_case(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'data.json'
            data = dataset(1)
            data['matches'][0]['cases'][0]['branches'][1]['outcome']['resolved'] = False
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'No completely resolved'):
                read_cases(path)

    def test_fit_and_evaluate_reject_seed_leakage_across_team_assignments(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train, val, model, report = (str(root/name) for name in ('train.json', 'val.json', 'model.json', 'report.json'))
            Path(train).write_text(json.dumps(dataset(1)), encoding='utf-8')
            Path(val).write_text(json.dumps(dataset(1)), encoding='utf-8')
            argv = ['shot-placement', 'fit', '--train', train, '--validation', val,
                    '--output', model, '--report', report]
            with patch('sys.argv', argv), self.assertRaisesRegex(ValueError, 'disjoint'):
                main()
            Path(val).write_text(json.dumps(dataset(2)), encoding='utf-8')
            with patch('sys.argv', argv), patch('builtins.print'):
                main()
            with patch('sys.argv', ['shot-placement', 'evaluate', '--model', model, '--data', train, '--report', report]), \
                    self.assertRaisesRegex(ValueError, 'overlap'):
                main()
