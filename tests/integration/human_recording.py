"""Real Genesis input replays, including snapshot seeks in another process.

Run explicitly with the local stable-retro integration and imported ROM installed:
    python -m tests.integration.human_recording
"""
import gzip
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from nhl94_ai.evaluation.human_session import build_parser, run
from nhl94_ai.evaluation.human_replay import replay


class NativeHumanRecording(unittest.TestCase):
    def viewer(self, *, quit_after=None):
        count = 0
        def action():
            nonlocal count
            count += 1
            buttons = [0] * 12
            buttons[4 + count // 20 % 4] = 1
            buttons[8] = int(count % 17 < 3)
            buttons[0] = int(count % 31 == 0)
            return buttons
        return SimpleNamespace(before_frame=lambda: quit_after is None or count < quit_after,
                               human_action=action, show=lambda *_: None, close=lambda: None)

    def test_multiple_periods_both_sides_and_cross_process_seek(self):
        for side in ('home', 'away'):
            with self.subTest(side=side), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'session'
                args = build_parser().parse_args(['--output', str(path), '--side', side,
                                                  '--periods', '2', '--seconds', '5',
                                                  '--snapshot-interval', '60'])
                with patch('nhl94_ai.evaluation.external.MatchViewer', side_effect=lambda *_a, **_k: self.viewer()):
                    report = run(args)
                self.assertEqual(report['summary']['completed_periods'], 2)
                self.assertEqual([row['seed'] for row in report['matches']], [94, 95])
                for row in report['matches']:
                    trace = path / row['directory'] / 'inputs.jsonl.gz'
                    verified = replay(trace)
                    self.assertTrue(verified['verified'])
                    self.assertEqual(verified['goals'], row['goals'])
                    self.assertEqual(verified['actions_sha256'], row['actions_sha256'])
                    process = subprocess.run([sys.executable, '-m', 'nhl94_ai', 'replay-human', str(trace),
                                              '--start-frame', '120'], check=True,
                                             capture_output=True, text=True)
                    sought = json.loads(process.stdout)
                    self.assertTrue(sought['verified'])
                    self.assertEqual(sought['verified_from_frame'], 120)
                    self.assertEqual(sought['final_ram_sha256'], row['final_ram_sha256'])
                    with gzip.open(trace, 'rt') as stream:
                        next(stream)
                        first = json.loads(next(stream))
                    human = first['buttons'][0:12] if side == 'home' else first['buttons'][12:24]
                    self.assertEqual({index for index, value in enumerate(human) if value}, {4, 8})

    def test_quit_during_period_retains_verified_partial_run(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'session'
            args = build_parser().parse_args(['--output', str(path), '--periods', '10'])
            with patch('nhl94_ai.evaluation.external.MatchViewer', return_value=self.viewer(quit_after=75)):
                report = run(args)
            self.assertEqual(len(report['matches']), 1)
            self.assertEqual(report['summary']['completed_periods'], 0)
            self.assertEqual(report['summary']['incomplete_periods'], 1)
            verified = replay(path / 'period-001' / 'inputs.jsonl.gz')
            self.assertTrue(verified['verified'])
            self.assertFalse(verified['completed'])
            self.assertEqual(verified['frames'], 75)

    def test_slow_classic_and_automatic_goalie_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'session'
            config = Path(directory) / 'classic.json'
            config.write_text(json.dumps({'decision_hz': 10, 'ai_goalie': False}))
            args = build_parser().parse_args(['--output', str(path), '--periods', '1', '--seconds', '1',
                                              '--classic-config', str(config)])
            with patch('nhl94_ai.evaluation.external.MatchViewer', return_value=self.viewer()):
                report = run(args)
            self.assertEqual(report['summary']['completed_periods'], 1)
            self.assertIsNone(report['matches'][0]['goalie_metrics']['opponent'])
            self.assertTrue(replay(path / 'period-001' / 'inputs.jsonl.gz')['verified'])


if __name__ == '__main__':
    unittest.main()
