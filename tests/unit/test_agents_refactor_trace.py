"""Only complete, hash-verified applied input streams locate native divergence."""
import gzip
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from nhl94_ai.evaluation.agents_refactor_trace import PROTOCOL, compare


class NativeRefactorTraceTests(unittest.TestCase):
    def fixture(self, root, name, *, button=False, diagnostic=False, truncate=False):
        directory = root / name
        directory.mkdir()
        rows = [{'frame': i + 1, 'applied': [int(button and i == 1)] + [0] * 11,
                 'input': {'owner': 0}, 'decision': 'new' if diagnostic else 'carry'} for i in range(2)]
        digest = hashlib.sha256(b''.join(bytes(row['applied']) for row in rows)).hexdigest()
        report = {'protocol': PROTOCOL, 'uninstrumented_match_reproduced': True,
                  'instrumentation_sha256': 'fixed', 'settings': {'frame_skip': 4},
                  'trace': 'frames.jsonl.gz', 'traced_frames': 2,
                  'result': {'completed': True, 'clock_remaining': 0, 'frames': 3,
                             'matchup': 'ducks-sabres-manual', 'seed': 8000, 'side': 2,
                             'lineup': ['fixed'], 'initial_state_sha256': 'fixed', 'actions_sha256': digest}}
        with gzip.open(directory / report['trace'], 'wt') as stream:
            for row in rows[:1] if truncate else rows:
                stream.write(json.dumps(row) + '\n')
        path = directory / 'report.json'
        path.write_text(json.dumps(report))
        return path

    def test_first_applied_divergence_contains_both_input_states(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = compare(self.fixture(root, 'before'), self.fixture(root, 'after', button=True))
            self.assertFalse(result['actions_identical'])
            first = result['first_action_difference']
            self.assertEqual(first['frame'], 2)
            self.assertEqual(first['before']['input'], first['after']['input'])
            self.assertIsNone(result['first_input_difference_frame'])

    def test_diagnostic_changes_do_not_hide_or_invent_applied_action_changes(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = compare(self.fixture(root, 'before'), self.fixture(root, 'after', diagnostic=True))
            self.assertTrue(result['actions_identical'])
            self.assertEqual(result['first_diagnostic_difference_frame'], 1)

    def test_equal_truncated_traces_fail_hash_and_completion_validation(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                compare(self.fixture(root, 'before', truncate=True), self.fixture(root, 'after', truncate=True))

    def test_different_instrumentation_is_rejected(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            before, after = self.fixture(root, 'before'), self.fixture(root, 'after')
            report = json.loads(after.read_text())
            report['instrumentation_sha256'] = 'changed'
            after.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, 'instrumentation'):
                compare(before, after)
