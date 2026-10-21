"""Only complete, hash-verified applied input streams locate native divergence."""
import gzip
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from nhl94_ai.evaluation.agents_refactor_trace import PROTOCOL, compare, verify_result


class NativeRefactorTraceTests(unittest.TestCase):
    def test_raw_native_contact_tuples_match_saved_json_but_changed_coordinates_fail(self):
        measured = {'frames': 3, 'actions_sha256': 'fixed',
                    'goalie_contact_events': [{'preceding_action': {'target': (-10.0, -199.0)}}]}
        saved = json.loads(json.dumps(measured))
        self.assertEqual(verify_result(measured, saved, 2, 'fixed'), saved)
        measured['goalie_contact_events'][0]['preceding_action']['target'] = (-11.0, -199.0)
        with self.assertRaisesRegex(RuntimeError, 'goalie_contact_events'):
            verify_result(measured, saved, 2, 'fixed')

    def test_matching_results_still_require_the_independently_hashed_applied_stream(self):
        result = {'frames': 3, 'actions_sha256': 'saved'}
        for frames, digest in ((1, 'saved'), (2, 'different')):
            with self.subTest(frames=frames, digest=digest), self.assertRaisesRegex(RuntimeError, 'Applied trace'):
                verify_result(result, result, frames, digest)

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
