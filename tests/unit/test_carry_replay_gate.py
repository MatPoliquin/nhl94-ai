"""The cadence gate fails before expensive replay or success output on invalid evidence."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from nhl94_ai.evaluation.carry_replay_gate import (
    REQUIRED_SOURCES, build_parser, run, validate_benchmark, validate_prefix, validate_sources,
)


ROOT = Path(__file__).resolve().parents[2]


def benchmark():
    return {
        'protocol': 'nhl94-cpu-first-period-v2',
        'settings': {'agent': 'classic-v1', 'seconds': 300, 'frame_skip': 4,
                     'action_type': 'FILTERED', 'goalie_policy': 'off', 'cross_crease': False,
                     'deke': False, 'uncertain_carry': False, 'seed': 42, 'trials': 1,
                     'matchups': ['sabres-ducks-manual', 'ducks-sabres-manual']},
        'sources': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                    for name in REQUIRED_SOURCES},
        'matches': [{'side': side, 'seed': 42, 'matchup': matchup, 'completed': True,
                     'clock_remaining': 0, 'frames': 100, 'actions_sha256': 'a' * 64}
                    for side, matchup in ((1, 'sabres-ducks-manual'), (2, 'ducks-sabres-manual'))],
    }


def prefix():
    return {'seed': 42, 'status': 'prefix-only', 'cases': [], 'missing_strata': [],
            'clock_remaining': 0, 'benchmark_frames': 100, 'applied_action_frames': 99,
            'searched_frames': 99, 'actions_sha256': 'a' * 64}


class CarryReplayGateTests(unittest.TestCase):
    def test_current_standard_default_report_is_accepted(self):
        self.assertEqual(set(validate_benchmark(benchmark(), ROOT)), {(1, 42), (2, 42)})

    def test_wrong_protocol_cadence_and_experimental_flags_are_rejected(self):
        for name, value in (('frame_skip', 1), ('seconds', 60), ('uncertain_carry', True),
                            ('chance_creation', True), ('offense_lookahead', True),
                            ('classic_refinements', ['finishing']),
                            ('deke', True), ('cross_crease', True), ('goalie_policy', 'selective'),
                            ('action_type', 'HOCKEY_INTENT_DPAD')):
            with self.subTest(setting=name):
                report = benchmark()
                report['settings'][name] = value
                with self.assertRaisesRegex(ValueError, name):
                    validate_benchmark(report, ROOT)
        report = benchmark()
        report['protocol'] = 'unknown'
        with self.assertRaisesRegex(ValueError, 'protocol|first-period'):
            validate_benchmark(report, ROOT)

    def test_stale_source_and_missing_dispatch_coverage_are_rejected(self):
        report = benchmark()
        report['sources']['nhl94_ai/agents/base.py'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'fingerprint mismatch'):
            validate_benchmark(report, ROOT)
        report = benchmark()
        del report['sources']['nhl94_ai/agents/base.py']
        with self.assertRaisesRegex(ValueError, 'coverage is incomplete'):
            validate_benchmark(report, ROOT)

    def test_source_paths_cannot_escape_the_repository(self):
        for name in ('../outside.py', '/tmp/outside.py'):
            with self.subTest(path=name), self.assertRaisesRegex(ValueError, 'inside the repository'):
                validate_sources({name: 'a' * 64}, ROOT)

    def test_incomplete_duplicate_and_missing_seed_trials_are_rejected(self):
        for change in ('incomplete', 'duplicate', 'seed'):
            with self.subTest(change=change):
                report = benchmark()
                if change == 'incomplete':
                    report['matches'][0]['completed'] = False
                elif change == 'duplicate':
                    report['matches'][1] = deepcopy(report['matches'][0])
                else:
                    report['matches'][1]['seed'] = 43
                with self.assertRaises(ValueError):
                    validate_benchmark(report, ROOT)

    def test_prefix_requires_full_frames_clock_and_applied_action_parity(self):
        expected = benchmark()['matches'][1]
        validate_prefix(prefix(), expected)
        for name, value in (('clock_remaining', 1), ('benchmark_frames', 99),
                            ('applied_action_frames', 98), ('actions_sha256', 'b' * 64),
                            ('status', 'captured'), ('cases', [{}])):
            with self.subTest(field=name):
                result = prefix()
                result[name] = value
                with self.assertRaises(ValueError):
                    validate_prefix(result, expected)

    def test_success_runs_current_native_dispatch_without_a_frozen_reference(self):
        with TemporaryDirectory() as directory:
            reference = Path(directory) / 'benchmark.json'
            reference.write_text(json.dumps(benchmark()), encoding='utf-8')
            output = Path(directory) / 'gate.json'
            args = build_parser().parse_args(['--benchmark', str(reference), '--output', str(output)])
            with patch('nhl94_ai.evaluation.carry_replay_gate.replay_seed', return_value=prefix()) as native:
                report = run(args)
            self.assertTrue(report['passed'])
            self.assertTrue(output.is_file())
            self.assertEqual(native.call_args.args[0], 42)
            self.assertEqual(native.call_args.kwargs['strata'], ())
            self.assertEqual(native.call_args.kwargs['search_frames'], 42000)

    def test_native_mismatch_never_writes_success_output(self):
        with TemporaryDirectory() as directory:
            reference = Path(directory) / 'benchmark.json'
            reference.write_text(json.dumps(benchmark()), encoding='utf-8')
            output = Path(directory) / 'gate.json'
            args = build_parser().parse_args(['--benchmark', str(reference), '--output', str(output)])
            wrong = prefix()
            wrong['actions_sha256'] = 'b' * 64
            with patch('nhl94_ai.evaluation.carry_replay_gate.replay_seed', return_value=wrong):
                with self.assertRaisesRegex(ValueError, 'applied actions differ'):
                    run(args)
            self.assertFalse(output.exists())

    def test_source_changes_during_native_measurement_never_write_success(self):
        with TemporaryDirectory() as directory:
            reference = Path(directory) / 'benchmark.json'
            reference.write_text(json.dumps(benchmark()), encoding='utf-8')
            output = Path(directory) / 'gate.json'
            args = build_parser().parse_args(['--benchmark', str(reference), '--output', str(output)])
            checks = [None, None, ValueError('Source fingerprint mismatch during native replay')]
            with patch('nhl94_ai.evaluation.carry_replay_gate.validate_sources', side_effect=checks), \
                    patch('nhl94_ai.evaluation.carry_replay_gate.replay_seed', return_value=prefix()):
                with self.assertRaisesRegex(ValueError, 'fingerprint mismatch'):
                    run(args)
            self.assertFalse(output.exists())

    def test_new_helper_hashes_cannot_replace_measured_policy_fingerprints(self):
        with TemporaryDirectory() as directory:
            reference = Path(directory) / 'benchmark.json'
            report = benchmark()
            reference.write_text(json.dumps(report), encoding='utf-8')
            args = build_parser().parse_args(['--benchmark', str(reference)])
            with patch('nhl94_ai.evaluation.carry_replay_gate.source_hashes', return_value={
                    'nhl94_ai/agents/base.py': 'b' * 64}), \
                    patch('nhl94_ai.evaluation.carry_replay_gate.replay_seed', return_value=prefix()):
                result = run(args)
            self.assertEqual(result['sources']['nhl94_ai/agents/base.py'],
                             report['sources']['nhl94_ai/agents/base.py'])

    def test_output_cannot_overwrite_the_reference_or_policy_sources(self):
        with TemporaryDirectory() as directory:
            reference = Path(directory) / 'benchmark.json'
            original = json.dumps(benchmark())
            reference.write_text(original, encoding='utf-8')
            for output in (reference, ROOT / 'nhl94_ai/agents/base.py'):
                with self.subTest(output=output):
                    args = build_parser().parse_args(['--benchmark', str(reference), '--output', str(output)])
                    with patch('nhl94_ai.evaluation.carry_replay_gate.replay_seed') as native:
                        with self.assertRaisesRegex(ValueError, 'must not overwrite'):
                            run(args)
                    native.assert_not_called()
            self.assertEqual(reference.read_text(encoding='utf-8'), original)


if __name__ == '__main__':
    unittest.main()
