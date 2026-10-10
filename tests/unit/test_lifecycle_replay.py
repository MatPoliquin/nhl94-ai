"""Equivalence evidence must contain complete traces and expose first divergence."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from nhl94_ai.evaluation.lifecycle_replay import compare_reports


class LifecycleReplayTests(unittest.TestCase):
    def reports(self, root):
        report = {'protocol': 'classic-lifecycle-replay-v1', 'sources': {'source.py': 'a' * 64},
                  'cases': {'sample': {'fixture': ['fixed'], 'predicted_frames': 1,
                                       'prediction_seconds': 0.1, 'trace': 'sample.jsonl.gz',
                                       'result': {'completed': True, 'clock_remaining': 0,
                                                  'frames': 2, 'actions_sha256': 'b' * 64}}}}
        paths = []
        for name in ('before', 'after'):
            directory = root / name
            directory.mkdir()
            with gzip.open(directory / 'sample.jsonl.gz', 'wt') as trace:
                trace.write('[[[0]],"carry",[0,0]]\n')
            path = directory / 'report.json'
            path.write_text(json.dumps(report))
            paths.append(path)
        return paths, report

    def test_complete_comparison_reports_each_case(self):
        with TemporaryDirectory() as directory:
            paths, _ = self.reports(Path(directory))
            result = compare_reports(paths[0], paths[1])
            self.assertTrue(result['passed'])
            self.assertEqual(result['cases'][0]['frames'], 1)

    def test_changed_input_identifies_first_divergent_frame(self):
        with TemporaryDirectory() as directory:
            paths, _ = self.reports(Path(directory))
            with gzip.open(paths[1].parent / 'sample.jsonl.gz', 'wt') as trace:
                trace.write('[[[1]],"carry",[0,0]]\n')
            with self.assertRaisesRegex(ValueError, 'first divergent frame 1'):
                compare_reports(paths[0], paths[1])

    def test_equal_but_truncated_traces_cannot_pass(self):
        with TemporaryDirectory() as directory:
            paths, _ = self.reports(Path(directory))
            for path in paths:
                with gzip.open(path.parent / 'sample.jsonl.gz', 'wt'):
                    pass
            with self.assertRaisesRegex(ValueError, 'complete period'):
                compare_reports(paths[0], paths[1])

    def test_matching_inputs_still_require_matching_outcomes(self):
        with TemporaryDirectory() as directory:
            paths, report = self.reports(Path(directory))
            for path, goals in zip(paths, ([1, 0], [0, 1])):
                changed = deepcopy(report)
                changed['cases']['sample']['result']['goals'] = goals
                path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, 'outcome/metric mismatch'):
                compare_reports(paths[0], paths[1])
