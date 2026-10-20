"""Shared replay artifacts preserve serialization and source provenance."""
import gzip
import hashlib
from pathlib import Path
import tempfile
import unittest

from nhl94_ai.evaluation import benchmark, cpu_benchmark, possession_ablation_replay, refinement_replay
from nhl94_ai.evaluation.replay_support import (
    advance, observation, read_view, replay_source_hashes, restore, write_replay_report,
)
from nhl94_ai.game import genesis_match


class ReplaySupportTests(unittest.TestCase):
    def test_report_bytes_and_compression_are_stable(self):
        report = {'protocol': 'test', 'matches': [{'seed': 7, 'goals': [1, 0]}]}
        expected = b'{"protocol":"test","matches":[{"seed":7,"goals":[1,0]}]}\n'
        with tempfile.TemporaryDirectory() as directory:
            plain = Path(directory) / 'plain' / 'report.json'
            first = Path(directory) / 'compressed' / 'first.json.gz'
            second = Path(directory) / 'second.json.gz'
            for path in (plain, first, second):
                write_replay_report(path, report)
            self.assertEqual(plain.read_bytes(), expected)
            self.assertEqual(gzip.decompress(first.read_bytes()), expected)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(first.read_bytes()[4:8], b'\0' * 4)

    def test_nonfinite_report_does_not_overwrite_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'report.json'
            path.write_bytes(b'original')
            with self.assertRaises(ValueError):
                write_replay_report(path, {'value': float('nan')})
            self.assertEqual(path.read_bytes(), b'original')

    def test_fingerprints_cover_extracted_helpers_and_nested_platforms(self):
        root = Path(__file__).resolve().parents[2]
        sources = replay_source_hashes('evaluation/shot_direction.py')
        for name in ('evaluation/replay_support.py', 'evaluation/shot_direction.py',
                     'game/genesis_match.py', 'game/platforms/genesis.py', 'game/platforms/snes.py'):
            path = root / 'nhl94_ai' / name
            self.assertEqual(sources['nhl94_ai/' + name], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(sources, replay_source_hashes('evaluation/shot_direction.py'))

    def test_historical_imports_remain_available(self):
        for name in ('RAM', 'away_view', 'update_state'):
            self.assertIs(getattr(benchmark, name), getattr(genesis_match, name))
        for name in ('CPU_RAM', 'select_side', 'lineup', 'cpu_view'):
            self.assertIs(getattr(cpu_benchmark, name), getattr(genesis_match, name))
        for name, value in (('advance', advance), ('read_view', read_view), ('restore', restore)):
            self.assertIs(getattr(refinement_replay, name), value)
        self.assertIs(possession_ablation_replay.observation, observation)
        self.assertEqual(possession_ablation_replay.source_hashes(), replay_source_hashes())
