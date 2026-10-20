"""Real SNES native input replays, including snapshot seeks in another process.

Run explicitly with the local stable-retro NHL94-Snes-v0 integration and the
imported SNES ROM installed:
    python -m tests.integration.snes_recording
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

from nhl94_ai.evaluation.snes_recording import SnesRecording
from nhl94_ai.evaluation.snes_replay import replay
from nhl94_ai.evaluation.snes_session import build_parser, classic_options, play_snes_period, run


class NativeSnesRecording(unittest.TestCase):
    def viewer(self, *, quit_after):
        count = 0
        def action():
            nonlocal count
            count += 1
            buttons = [0] * 12
            buttons[4 + count // 20 % 4] = 1
            buttons[8] = int(count % 17 < 3)
            buttons[0] = int(count % 31 == 0)
            return buttons
        return SimpleNamespace(before_frame=lambda: count < quit_after,
                               human_action=action, show=lambda *_: None, close=lambda: None)

    def test_human_quit_retains_verified_partial_periods_on_both_sides(self):
        for side in ('home', 'away'):
            with self.subTest(side=side), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'session'
                args = build_parser().parse_args(['--output', str(path), '--side', side, '--human',
                                                  '--periods', '3', '--snapshot-interval', '60'])
                with patch('nhl94_ai.evaluation.external.MatchViewer',
                           side_effect=lambda *_a, **_k: self.viewer(quit_after=240)):
                    report = run(args)
                self.assertEqual(report['status'], 'stopped')
                self.assertEqual(report['summary']['incomplete_periods'], 1)
                trace = path / 'period-001' / 'inputs.jsonl.gz'
                verified = replay(trace)
                self.assertTrue(verified['verified'])
                self.assertEqual(verified['frames'], 240)
                process = subprocess.run([sys.executable, '-m', 'nhl94_ai', 'replay-snes', str(trace),
                                          '--start-frame', '120'],
                                         check=True, capture_output=True, text=True)
                sought = json.loads(process.stdout)
                self.assertTrue(sought['verified'])
                self.assertEqual(sought['verified_from_frame'], 120)
                self.assertEqual(sought['final_ram_sha256'], verified['final_ram_sha256'])
                with gzip.open(trace, 'rt', encoding='utf-8') as stream:
                    header = json.loads(next(stream))
                    first = json.loads(next(stream))
                self.assertEqual(header['human_side'], 1 if side == 'home' else 2)
                human = first['buttons'][0:12] if side == 'home' else first['buttons'][12:24]
                # Human channels map onto the native SNES button layout.
                self.assertEqual({index for index, value in enumerate(human) if value}, {4, 8})

    def test_native_classic_recording_replays_without_decision_code(self):
        with tempfile.TemporaryDirectory() as directory:
            args = build_parser().parse_args(['--output', str(Path(directory) / 'unused')])
            options = classic_options(args)
            job = {'side': 'away', 'human': False, 'goalie_policy': options['goalie_policy'],
                   'frame_skip': 4, 'classic_options': options, 'max_frames': 600}
            recorder = SnesRecording(Path(directory) / 'period', job, snapshot_interval=120)
            try:
                result = recorder.finish(play_snes_period({**job, '_recorder': recorder}))
            finally:
                recorder.close()
            self.assertEqual(result['frames'], 600)
            self.assertFalse(result['completed'])
            self.assertTrue(result['decision_frames']['home'])
            self.assertTrue(result['decision_frames']['away'])
            self.assertTrue(result['goalie_metrics']['home'] is None
                            or isinstance(result['goalie_metrics']['home'], dict))
            verified = replay(Path(directory) / 'period' / 'inputs.jsonl.gz')
            self.assertTrue(verified['verified'])
            self.assertEqual(verified['frames'], 600)
            self.assertEqual(verified['actions_sha256'], result['actions_sha256'])


if __name__ == '__main__':
    unittest.main()
