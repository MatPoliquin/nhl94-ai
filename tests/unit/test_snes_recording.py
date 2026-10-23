"""SNES native input identity, replay integrity, and session accounting without a ROM."""
from contextlib import ExitStack
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from nhl94_ai.cli import main
from nhl94_ai.evaluation import snes_replay, snes_session
from nhl94_ai.evaluation.human_recording import sha256
from nhl94_ai.evaluation.snes_recording import SnesRecording, scoreboard
from nhl94_ai.game.platforms import get_backend
from tests.support.recording import InputReplayEnv, fake_classic

SNES_BUTTONS = get_backend('snes').buttons


class ReplayEnv(InputReplayEnv):
    """Deterministic stand-in whose state depends on both controllers' inputs."""

    def __init__(self):
        super().__init__(SNES_BUTTONS)
        self.initial_state = b'test-state'

    def info(self):
        return {'nhl94_platform': 'snes', 'period': 0, 'time': max(0, 4 - self.frames),
                'p1_score': self.frames // 4, 'p2_score': 0,
                'p1_shots': self.frames // 2, 'p2_shots': 0,
                'puck_owner': 0 if self.frames < 3 else 6,
                'defense_team1': 1, 'defense_team2': 2,
                'defense_control1': 2, 'defense_control2': 8,
                'p1_control_slot': 2, 'p2_control_slot': 8}


class RecordingTests(unittest.TestCase):
    def setUp(self):
        self.resources = ExitStack()
        self.addCleanup(self.resources.close)
        self.root = Path(self.resources.enter_context(tempfile.TemporaryDirectory()))
        for target, value in (
            ('nhl94_ai.evaluation.snes_recording.runtime_provenance', {'test': 'core'}),
            ('nhl94_ai.evaluation.snes_recording.source_provenance', {}),
            ('nhl94_ai.evaluation.snes_recording.rom_sha256', 'test-rom'),
            ('nhl94_ai.evaluation.snes_replay.runtime_provenance', {'test': 'core'}),
            ('nhl94_ai.evaluation.snes_replay.rom_sha256', 'test-rom'),
        ):
            self.resources.enter_context(patch(target, return_value=value))
        self.environments = []
        self.save_state = b'test-state'
        def make(**_):
            env = ReplayEnv()
            env.initial_state = self.save_state
            self.environments.append(env)
            return env
        self.resources.enter_context(patch.object(snes_replay, 'make_retro', side_effect=make))

    def record(self, *, frames=4, completed=True, human=True, side='away'):
        env = ReplayEnv()
        directory = self.root / 'period'
        recorder = SnesRecording(directory, {'side': side, 'human': human,
                                             'goalie_policy': 'selective', 'frame_skip': 4},
                                 snapshot_interval=2)
        recorder.start(env, env.info())
        actions = []
        for frame in range(frames):
            buttons = np.zeros(24, dtype=np.int8)
            buttons[[frame % 12, 12 + (frame + 3) % 12]] = 1
            actions.append(buttons.tolist())
            recorder.before_step(env, env.info(), {'home': fake_classic(),
                                                   'away': None if human else fake_classic()})
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

    def test_inputs_observations_decisions_and_results_round_trip(self):
        path, result, actions = self.record()
        rows = self.read_rows(path)
        header = rows[0]
        self.assertEqual(header['schema'], 'nhl94-snes-human-inputs-v1')
        self.assertEqual(header['game'], 'NHL94-Snes-v0')
        self.assertEqual(header['platform'], 'snes')
        self.assertEqual(header['buttons'], list(SNES_BUTTONS))
        self.assertEqual(header['human_side'], 2)
        self.assertEqual(header['controllers'], {'home': 'classic', 'away': 'human'})
        self.assertEqual(header['controller_assignment']['defense_team2'], 2)
        self.assertEqual(header['goalie_policy'], 'selective')
        self.assertEqual(header['action_schema'], 'FILTERED')
        self.assertEqual(header['decision_cadence'], {'frame_skip': 4, 'classic_control_interval': 0})
        self.assertEqual(header['initial_state_sha256'], sha256(b'test-state'))
        self.assertEqual(header['initial_scoreboard'], {'goals': [0, 0], 'shots': [0, 0]})
        self.assertEqual([row['buttons'] for row in rows[1:-1]], actions)
        self.assertEqual(rows[1]['observations']['time'], 4)
        self.assertEqual(rows[1]['after']['clock'], 3)
        self.assertIsNone(rows[1]['classic']['away'])
        self.assertIsNone(rows[1]['classic']['home']['offense']['estimate'])
        self.assertEqual(rows[1]['classic']['home']['target'], [1, 2])
        self.assertEqual(result['goals'], [1, 0])
        self.assertEqual(result['human_side'], 2)
        self.assertEqual(result['decision_frames']['home'], {'carry': 4})
        self.assertEqual(result['decision_frames']['away'], {})
        self.assertEqual([row['type'] for row in result['events']],
                         ['shot', 'possession_change', 'goal', 'shot'])
        for start in (0, 2, 3, 4):
            replayed = snes_replay.replay(path, start_frame=start)
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
            snes_replay.replay(path)
        self.environments[-1].close.assert_called_once_with()

    def test_native_recording_labels_both_sides_classic_without_a_human(self):
        path, result, _ = self.record(human=False, side='home')
        header = self.read_rows(path)[0]
        self.assertEqual(header['schema'], 'nhl94-snes-native-inputs-v1')
        self.assertIsNone(header['human_side'])
        self.assertEqual(header['controllers'], {'home': 'classic', 'away': 'classic'})
        self.assertIsNone(result['human_side'])
        self.assertEqual(result['controllers'], {'home': 'classic', 'away': 'classic'})
        self.assertEqual(result['decision_frames']['away'], {'carry': 4})
        self.assertTrue(snes_replay.replay(path)['verified'])

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
                snes_replay.replay(path)
            self.environments[-1].close.assert_called_once_with()

    def test_runtime_rom_save_and_snapshot_mismatch_are_rejected(self):
        path, _, _ = self.record()
        with patch.object(snes_replay, 'runtime_provenance', return_value={'test': 'different'}):
            with self.assertRaisesRegex(ValueError, 'core'):
                snes_replay.replay(path)
        with patch.object(snes_replay, 'rom_sha256', return_value='different-rom'):
            with self.assertRaisesRegex(ValueError, 'ROM'):
                snes_replay.replay(path)
        self.save_state = b'different-state'
        with self.assertRaisesRegex(ValueError, 'save state'):
            snes_replay.replay(path)
        self.save_state = b'test-state'
        initial = path.parent / 'frame-0000000.state.gz'
        initial.write_bytes(gzip.compress(b'wrong'))
        with self.assertRaisesRegex(ValueError, 'Snapshot checksum'):
            snes_replay.replay(path)

    def test_early_quit_and_zero_frame_replays_stay_incomplete(self):
        path, _, _ = self.record(frames=0, completed=False)
        replayed = snes_replay.replay(path)
        self.assertTrue(replayed['verified'])
        self.assertFalse(replayed['completed'])
        self.assertEqual(replayed['frames'], 0)
        with self.assertRaisesRegex(ValueError, 'beyond'):
            snes_replay.replay(path, start_frame=1)

    def test_never_overwrite_a_period(self):
        path, _, _ = self.record()
        with self.assertRaises(FileExistsError):
            SnesRecording(path.parent, {'side': 'home'})

    def test_recording_requires_a_supported_side_and_positive_interval(self):
        with self.assertRaisesRegex(ValueError, 'side'):
            SnesRecording(self.root / 'bad-side', {'side': 'east'})
        with self.assertRaisesRegex(ValueError, 'snapshot_interval'):
            SnesRecording(self.root / 'bad-interval', {'side': 'home'}, snapshot_interval=0)
        self.assertFalse((self.root / 'bad-side').exists())
        self.assertFalse((self.root / 'bad-interval').exists())

    def test_scoreboard_uses_snes_variables(self):
        self.assertEqual(scoreboard({'p1_score': 2, 'p2_score': 1, 'p1_shots': 9, 'p2_shots': 4}),
                         {'goals': [2, 1], 'shots': [9, 4]})


class SessionTests(unittest.TestCase):
    def test_cli_commands_dispatch(self):
        with patch.object(snes_session, 'run') as record:
            main(['record-snes', '--periods', '10', '--output', '/tmp/test-snes-session'])
        self.assertEqual(record.call_args.args[0].periods, 10)
        with patch.object(snes_replay, 'run') as replay:
            main(['replay-snes', 'period/inputs.jsonl.gz', '--start-frame', '600'])
        self.assertEqual(replay.call_args.args[0].start_frame, 600)

    def test_completed_only_summaries_for_human_and_native_sessions(self):
        rows = [dict(completed=True, human_side=1, goals=[2, 1]),
                dict(completed=True, human_side=2, goals=[3, 1]),
                dict(completed=True, human_side=2, goals=[1, 1]),
                dict(completed=False, human_side=1, goals=[20, 0])]
        self.assertEqual(snes_session.summarize(rows), {
            'completed_periods': 3, 'incomplete_periods': 1,
            'human_wins': 1, 'classic_wins': 1, 'draws': 1})
        native = [dict(completed=True, human_side=None, goals=[2, 1]),
                  dict(completed=True, human_side=None, goals=[0, 0]),
                  dict(completed=False, human_side=None, goals=[0, 9])]
        self.assertEqual(snes_session.summarize(native), {
            'completed_periods': 2, 'incomplete_periods': 1,
            'home_wins': 1, 'away_wins': 0, 'draws': 1})

    def test_invalid_options_do_not_create_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'session'
            for extra in (['--periods', '0'], ['--snapshot-interval', '0'], ['--frame-skip', '0']):
                args = snes_session.build_parser().parse_args(['--output', str(path), *extra])
                with self.subTest(extra=extra), self.assertRaises(ValueError):
                    snes_session.run(args)
                self.assertFalse(path.exists())

    def test_periods_and_stop_on_quit_without_overwriting(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'session'
            args = snes_session.build_parser().parse_args(['--output', str(path), '--periods', '5'])
            jobs = []
            def match(job):
                jobs.append(job)
                return {'completed': len(jobs) == 1, 'user_quit': len(jobs) == 2, 'goals': [0, 1]}
            with patch.object(snes_session, 'play_snes_period', side_effect=match):
                result = snes_session.run(args)
            self.assertEqual(len(jobs), 2)
            self.assertTrue(all(job['side'] == 'away' and not job['human'] for job in jobs))
            self.assertTrue(all(job['goalie_policy'] == 'selective' for job in jobs))
            self.assertEqual(result['status'], 'stopped')
            self.assertEqual(result['summary']['completed_periods'], 1)
            self.assertEqual(result['summary']['incomplete_periods'], 1)
            with self.assertRaises(FileExistsError):
                snes_session.run(args)

    def test_interrupt_and_failure_preserve_incomplete_results(self):
        for error, status in ((KeyboardInterrupt(), 'interrupted'), (RuntimeError('test failure'), 'error')):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'session'
                args = snes_session.build_parser().parse_args(['--output', str(path)])
                with patch.object(snes_session, 'play_snes_period', side_effect=error):
                    if status == 'error':
                        with self.assertRaisesRegex(RuntimeError, 'test failure'):
                            snes_session.run(args)
                    else:
                        snes_session.run(args)
                report = json.loads((path / 'session.json').read_text())
                self.assertEqual(report['status'], status)
                self.assertEqual(report['schema'], 'nhl94-snes-session-v1')
                self.assertEqual(report['summary']['incomplete_periods'], 1)
                self.assertFalse(report['matches'][0]['completed'])

    def test_classic_options_respect_play_config(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'classic.json'
            config.write_text(json.dumps({'decision_hz': 10, 'ai_goalie': False,
                                          'player_ratings_consideration': False}))
            args = snes_session.build_parser().parse_args(['--output', 'unused',
                                                           '--classic-config', str(config)])
            options = snes_session.classic_options(args)
            self.assertEqual(options['classic_control_interval'], 6)
            self.assertEqual(options['classic_decision_interval'], 4)
            self.assertEqual(options['goalie_policy'], 'off')
            self.assertEqual(options['goalie_positioning'], 'classic')
            self.assertFalse(options['player_ratings_consideration'])


if __name__ == '__main__':
    unittest.main()
