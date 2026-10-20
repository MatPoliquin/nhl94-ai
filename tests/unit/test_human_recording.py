"""Native input identity, replay integrity, and interrupted session accounting."""
from contextlib import ExitStack
import gzip
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from nhl94_ai.cli import main
from nhl94_ai.evaluation import human_replay, human_session
from nhl94_ai.evaluation.external import BUTTONS
from nhl94_ai.evaluation.human_recording import HumanRecording, json_value


class ReplayEnv:
    """Deterministic stand-in whose state depends on both controllers' inputs."""

    def __init__(self, rom):
        self.pass_geometry_rom = str(rom)
        self.buttons = BUTTONS
        self.frames, self.value = 0, 0
        self.close = Mock()
        self.em = SimpleNamespace(get_state=self.get_state, set_state=self.set_state)
        self.data = SimpleNamespace(set_variable=Mock(), reset=Mock(), update_ram=Mock(),
                                    lookup_all=self.info)

    def info(self):
        return {'period': 0, 'bench_clock': max(0, 4 - self.frames),
                'p1_score': self.frames // 4, 'p2_score': 0,
                'bench_shots1': self.frames // 2, 'bench_shots2': 0,
                'puck_owner': 0 if self.frames < 3 else 6}

    def reset(self):
        self.frames, self.value = 0, 0

    def get_state(self):
        return bytes([self.frames, self.value])

    def set_state(self, value):
        self.frames, self.value = value
        return True

    def get_ram(self):
        return np.array([self.frames, self.value], dtype=np.uint8)

    def step(self, buttons):
        self.frames += 1
        self.value = (self.value + sum((index + 1) * int(value) for index, value in enumerate(buttons))) % 256
        return None, 0, False, False, self.info()


def fake_classic():
    return SimpleNamespace(_last_decision='carry', _last_target=(np.int64(1), 2),
                           input_owner='offense', scheduler=SimpleNamespace(decisions=1),
                           offense_diagnostics={'estimate': float('inf')},
                           defense_diagnostics={}, goalie_diagnostics={})


class RecordingTests(unittest.TestCase):
    def setUp(self):
        self.resources = ExitStack()
        self.addCleanup(self.resources.close)
        self.root = Path(self.resources.enter_context(tempfile.TemporaryDirectory()))
        self.rom = self.root / 'rom'
        self.rom.write_bytes(b'test-rom')
        for target, value in (
            ('nhl94_ai.evaluation.human_recording.runtime_provenance', {'test': 'core'}),
            ('nhl94_ai.evaluation.human_recording.source_provenance', {}),
            ('nhl94_ai.evaluation.human_replay.runtime_provenance', {'test': 'core'}),
        ):
            self.resources.enter_context(patch(target, return_value=value))
        self.resources.enter_context(patch('nhl94_ai.game.ram.pass_geometry_info', side_effect=lambda _, info: info))
        self.environments = []
        def make(**_):
            env = ReplayEnv(self.rom)
            self.environments.append(env)
            return env
        self.resources.enter_context(patch.object(human_replay, 'make_retro', side_effect=make))

    def record(self, *, frames=4, completed=True, human=True, side=2):
        env = ReplayEnv(self.rom)
        directory = self.root / 'period'
        recorder = HumanRecording(directory, {'side': side, 'seed': 94, 'human': human}, snapshot_interval=2)
        recorder.start(env, env.info(), opponent_metadata={'name': 'Test bot'} if not human else None)
        actions = []
        for frame in range(frames):
            buttons = np.zeros(24, dtype=np.int8)
            buttons[[frame % 12, 12 + (frame + 3) % 12]] = 1
            actions.append(buttons.tolist())
            recorder.before_step(env, env.info(), fake_classic())
            *_, info = env.step(buttons)
            recorder.after_step(env, buttons, info)
        result = recorder.finish({'completed': completed})
        return directory / 'inputs.jsonl.gz', result, actions

    def read_rows(self, path):
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            return [json.loads(line) for line in stream]

    def rewrite(self, path, rows):
        with gzip.open(path, 'wt', encoding='utf-8') as stream:
            stream.write(''.join(json.dumps(row) + '\n' for row in rows))

    def test_both_inputs_observations_decisions_and_results_round_trip(self):
        path, result, actions = self.record()
        rows = self.read_rows(path)
        self.assertEqual([row['buttons'] for row in rows[1:-1]], actions)
        self.assertEqual(rows[1]['observations']['bench_clock'], 4)
        self.assertEqual(rows[1]['after']['clock'], 3)
        self.assertIsNone(rows[1]['classic']['offense']['estimate'])
        self.assertEqual(rows[1]['classic']['target'], [1, 2])
        self.assertEqual(result['goals'], [1, 0])
        self.assertEqual([row['type'] for row in result['events']],
                         ['shot', 'possession_change', 'goal', 'shot'])
        for start in (0, 2, 3, 4):
            replayed = human_replay.replay(path, start_frame=start)
            self.assertTrue(replayed['verified'])
            self.assertEqual(replayed['actions_sha256'], result['actions_sha256'])
            self.assertEqual(replayed['final_ram_sha256'], result['final_ram_sha256'])
            self.assertEqual(replayed['verified_from_frame'], start - start % 2)
        for env in self.environments:
            env.close.assert_called_once_with()

    def test_input_tampering_fails_at_first_diverging_frame(self):
        path, _, _ = self.record()
        rows = self.read_rows(path)
        rows[2]['buttons'][0] = 1
        self.rewrite(path, rows)
        with self.assertRaisesRegex(RuntimeError, 'frame 2'):
            human_replay.replay(path)
        self.environments[-1].close.assert_called_once_with()

    def test_external_recording_labels_classic_and_adapter_without_a_human(self):
        path, result, _ = self.record(human=False, side=1)
        header = self.read_rows(path)[0]
        self.assertEqual(header['schema'], 'nhl94-native-inputs-v1')
        self.assertIsNone(header['human_side'])
        self.assertEqual(header['classic_side'], 1)
        self.assertEqual(header['opponent_metadata'], {'name': 'Test bot'})
        self.assertIsNone(result['human_side'])
        self.assertEqual(result['classic_side'], 1)
        self.assertTrue(human_replay.replay(path)['verified'])

    def test_missing_footer_duplicate_frames_bad_buttons_and_footer_checksum(self):
        path, _, _ = self.record()
        originals = self.read_rows(path)
        for defect in ('footer', 'sequence', 'buttons', 'hash', 'trailing'):
            rows = json.loads(json.dumps(originals))
            if defect == 'footer':
                rows.pop()
            elif defect == 'sequence':
                rows[2]['frame'] = 1
            elif defect == 'buttons':
                rows[1]['buttons'][0] = 2
            elif defect == 'hash':
                rows[-1]['actions_sha256'] = 'wrong'
            else:
                rows.append(rows[1])
            self.rewrite(path, rows)
            with self.subTest(defect=defect), self.assertRaises((ValueError, RuntimeError)):
                human_replay.replay(path)
            self.environments[-1].close.assert_called_once_with()

    def test_runtime_rom_and_snapshot_mismatch_are_rejected(self):
        path, _, _ = self.record()
        with patch.object(human_replay, 'runtime_provenance', return_value={'test': 'different'}):
            with self.assertRaisesRegex(ValueError, 'core'):
                human_replay.replay(path)
        self.rom.write_bytes(b'different-rom')
        with self.assertRaisesRegex(ValueError, 'ROM'):
            human_replay.replay(path)
        self.rom.write_bytes(b'test-rom')
        initial = path.parent / 'frame-0000000.state.gz'
        initial.write_bytes(gzip.compress(b'wrong'))
        with self.assertRaisesRegex(ValueError, 'Snapshot checksum'):
            human_replay.replay(path)

    def test_early_quit_and_zero_frame_replays_stay_incomplete(self):
        path, _, _ = self.record(frames=0, completed=False)
        replayed = human_replay.replay(path)
        self.assertTrue(replayed['verified'])
        self.assertFalse(replayed['completed'])
        self.assertEqual(replayed['frames'], 0)
        with self.assertRaisesRegex(ValueError, 'beyond'):
            human_replay.replay(path, start_frame=1)

    def test_never_overwrite_a_period(self):
        path, _, _ = self.record()
        with self.assertRaises(FileExistsError):
            HumanRecording(path.parent, {'side': 1})


class SessionTests(unittest.TestCase):
    def test_cli_commands_dispatch(self):
        with patch.object(human_session, 'run') as record:
            main(['record-human', '--periods', '10', '--output', '/tmp/test-session'])
        self.assertEqual(record.call_args.args[0].periods, 10)
        with patch.object(human_replay, 'run') as replay:
            main(['replay-human', 'period/inputs.jsonl.gz', '--start-frame', '600'])
        self.assertEqual(replay.call_args.args[0].start_frame, 600)
        with patch.object(human_replay, 'run') as replay:
            main(['replay-match', 'match/inputs.jsonl.gz'])
        self.assertEqual(replay.call_args.args[0].replay, 'match/inputs.jsonl.gz')

    def test_completed_only_summary_uses_human_side(self):
        rows = [dict(completed=True, human_side=1, goals=[2, 1]),
                dict(completed=True, human_side=2, goals=[3, 1]),
                dict(completed=True, human_side=2, goals=[1, 1]),
                dict(completed=False, human_side=1, goals=[20, 0])]
        self.assertEqual(human_session.summarize(rows), {
            'completed_periods': 3, 'incomplete_periods': 1,
            'human_wins': 1, 'classic_wins': 1, 'draws': 1})

    def test_invalid_options_do_not_create_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'session'
            for extra in (['--periods', '0'], ['--seconds', '0'], ['--snapshot-interval', '0'],
                          ['--seed', '-1'], ['--seed', str(2**32 - 1), '--periods', '2']):
                args = human_session.build_parser().parse_args(['--output', str(path), *extra])
                with self.subTest(extra=extra), self.assertRaises(ValueError):
                    human_session.run(args)
                self.assertFalse(path.exists())

    def test_periods_increment_seeds_and_stop_on_quit_without_overwriting(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'session'
            args = human_session.build_parser().parse_args(['--output', str(path), '--periods', '5'])
            jobs = []
            def match(job):
                jobs.append(job)
                return {'completed': len(jobs) == 1, 'user_quit': len(jobs) == 2, 'goals': [0, 1]}
            with patch.object(human_session, 'play_match', side_effect=match):
                result = human_session.run(args)
            self.assertEqual([job['seed'] for job in jobs], [94, 95])
            self.assertTrue(all(job['human'] and job['opponent'] == 'classic' for job in jobs))
            self.assertEqual(result['status'], 'stopped')
            self.assertEqual(result['summary']['completed_periods'], 1)
            self.assertEqual(result['summary']['incomplete_periods'], 1)
            with self.assertRaises(FileExistsError):
                human_session.run(args)

    def test_interrupt_and_failure_preserve_incomplete_results(self):
        for error, status in ((KeyboardInterrupt(), 'interrupted'), (RuntimeError('test failure'), 'error')):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'session'
                args = human_session.build_parser().parse_args(['--output', str(path)])
                with patch.object(human_session, 'play_match', side_effect=error):
                    if status == 'error':
                        with self.assertRaisesRegex(RuntimeError, 'test failure'):
                            human_session.run(args)
                    else:
                        human_session.run(args)
                report = json.loads((path / 'session.json').read_text())
                self.assertEqual(report['status'], status)
                self.assertEqual(report['summary']['incomplete_periods'], 1)
                self.assertFalse(report['matches'][0]['completed'])

    def test_classic_options_respect_play_config(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'classic.json'
            config.write_text(json.dumps({'decision_hz': 10, 'ai_goalie': False,
                                          'player_ratings_consideration': False}))
            args = human_session.build_parser().parse_args(['--output', 'unused', '--classic-config', str(config)])
            options = human_session.classic_options(args)
            self.assertEqual(options['classic_control_interval'], 6)
            self.assertEqual(options['goalie_policy'], 'off')
            self.assertFalse(options['player_ratings_consideration'])


if __name__ == '__main__':
    unittest.main()
