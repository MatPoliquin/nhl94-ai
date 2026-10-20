"""Native alternatives stay grouped, finite, and separate from model evaluation."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from nhl94_ai.evaluation.shot_placement_model import read_cases


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


    def test_incomplete_alternative_excludes_entire_case(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'data.json'
            data = dataset(1)
            data['matches'][0]['cases'][0]['branches'][1]['outcome']['resolved'] = False
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'No completely resolved'):
                read_cases(path)
