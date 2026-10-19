"""External adapters remain optional and receive every native observation."""
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from nhl94_ai.evaluation import external


def dummy_adapter(**kwargs):
    return SimpleNamespace(ram={'clock': (0xFF0000, '>u2')}, metadata={'version': 'test'},
                           reset=Mock(), act=Mock(return_value=np.zeros(12)), close=Mock())


class AdapterTests(unittest.TestCase):
    def test_load_reset_physical_observations_and_close(self):
        adapter = dummy_adapter()
        with patch(__name__+'.dummy_adapter', return_value=adapter) as factory:
            session = external.load_adapter(__name__+':dummy_adapter', side=2, seed=123)
        factory.assert_called_once_with(protocol_version=1, side=2, seed=123, config=None)
        adapter.reset.assert_called_once_with()
        memory = SimpleNamespace(extract=Mock(return_value=42))
        env = SimpleNamespace(unwrapped=SimpleNamespace(data=SimpleNamespace(memory=memory)))
        for phase in ('play', 'intermission', 'play'):
            session.action(env, phase)
        frames = [call.args[0] for call in adapter.act.call_args_list]
        self.assertEqual([frame.number for frame in frames], [0, 1, 2])
        self.assertEqual([frame.phase for frame in frames], ['play', 'intermission', 'play'])
        self.assertTrue(all(frame.side == 2 and frame.observations == {'clock': 42} for frame in frames))
        session.close()
        session.close()
        adapter.close.assert_called_once_with()

    def test_invalid_declarations_close_adapter(self):
        for declaration in ((0, '>u2'), (0xFFFFFF, '>u2'), (0xFF0000, 'bad'), None):
            adapter = dummy_adapter()
            adapter.ram = {'invalid': declaration}
            with self.subTest(declaration=declaration), self.assertRaises(ValueError):
                external.AdapterSession(adapter, 1)
            adapter.close.assert_called_once_with()

    def test_bad_factories_and_actions_fail(self):
        for spec in ('unqualified', ':factory', 'missing_external_test_module:create'):
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                external.load_adapter(spec, side=1, seed=0)
        for action in (np.zeros(11), np.zeros((1, 12)), [0]*11+[2], [0]*11+[float('nan')]):
            with self.subTest(action=action), self.assertRaises(ValueError):
                external.validate_action(action)


class NativeMatchTests(unittest.TestCase):
    def match(self, side, *, quit_immediately=False, fail_step=False):
        base = dict(bench_team1=1, bench_team2=2, period=0, bench_clock=300,
                    bench_control1=2, bench_control2=8, goalie_mode_1=0, goalie_mode_2=0,
                    p1_score=0, p2_score=0, sflags=0, home_team=2, away_team=0,
                    bench_shots1=0, bench_shots2=0, bench_one_timers1=0, bench_one_timers2=0,
                    bench_one_timer_goals1=0, bench_one_timer_goals2=0)
        snapshots = [base, {**base, 'bench_clock': 0},
                     {**base, 'period': 1, 'bench_clock': 0},
                     {**base, 'period': 1, 'sflags': 32768},
                     {**base, 'period': 1, 'bench_control1': -1, 'bench_control2': -1},
                     {**base, 'period': 1, 'bench_clock': 299},
                     {**base, 'period': 1, 'bench_clock': 0},
                     {**base, 'period': 2, 'sflags': 32768},
                     {**base, 'period': 2, 'bench_clock': 299},
                     {**base, 'period': 2, 'bench_clock': 0}]
        sent, observed = [], []
        env = SimpleNamespace(buttons=external.BUTTONS, initial_state=b'test', close=Mock())
        def extract(address, kind):
            return 0 if (address-0xFFB04A-0x34) % (6*0x80) == 5*0x80 else 1
        env.data = SimpleNamespace(set_variable=Mock(), set_value=Mock(), update_ram=Mock(),
                                   lookup_all=lambda: dict(base), memory=SimpleNamespace(extract=extract))
        env.em = SimpleNamespace(get_state=lambda: b'test')
        env.get_ram = lambda: np.zeros(32, dtype=np.uint8)
        env.reset = Mock()
        def step(buttons):
            if fail_step:
                raise RuntimeError('emulator failure')
            sent.append(buttons.copy())
            return None, 0, False, False, dict(snapshots[len(sent)])
        env.step = step
        other = SimpleNamespace(frames=0, metadata={}, close=Mock())
        def action(_, phase):
            observed.append((len(sent), phase))
            other.frames += 1
            buttons = np.zeros(12, dtype=np.int8)
            buttons[4] = 1
            return buttons
        other.action = action
        candidate = SimpleNamespace(goalie=SimpleNamespace(metrics={}), scheduler=SimpleNamespace(decisions=0),
                                    _last_decision='test', end_one_timer=Mock())
        candidate.predict_frame = Mock(return_value=(np.zeros(12, dtype=np.int8),))
        def update(state, info, _):
            sign = -1 if info['period'] % 2 == 0 else 1
            state.team1.net.y, state.team2.net.y = sign*264, -sign*264
        viewer = SimpleNamespace(before_frame=lambda: not quit_immediately, close=Mock(), show=Mock())
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            rom = Path(directory)/'rom'
            rom.write_bytes(b'rom')
            for target, kwargs in (
                ('nhl94_ai.evaluation.external.load_adapter', {'return_value': other}),
                ('nhl94_ai.agents.classic_v1.ClassicAIV1Model', {'return_value': candidate}),
                ('nhl94_ai.env.factory.make_retro', {'return_value': env}),
                ('nhl94_ai.evaluation.cpu_benchmark.lineup', {'return_value': []}),
                ('stable_retro.data.get_romfile_path', {'return_value': str(rom)}),
                ('nhl94_ai.evaluation.benchmark.update_state', {'side_effect': update}),
                ('nhl94_ai.evaluation.benchmark.away_view', {'side_effect': lambda state: state}),
                ('nhl94_ai.evaluation.external.MatchViewer', {'return_value': viewer}),
            ):
                stack.enter_context(patch(target, **kwargs))
            job = dict(side=side, seed=123, periods=3, opponent='external',
                       adapter='dummy:create', fixture='ducks-campbell', watch=quit_immediately)
            if fail_step:
                with self.assertRaisesRegex(RuntimeError, 'emulator failure'):
                    external.play_match(job)
                result = None
            else:
                result = external.play_match(job)
        env.close.assert_called_once_with()
        other.close.assert_called_once_with()
        if quit_immediately:
            viewer.close.assert_called_once_with()
        return result, sent, observed

    def test_native_cadence_sides_intermission_and_periods(self):
        for side in (1, 2):
            with self.subTest(side=side):
                result, sent, observed = self.match(side)
                self.assertTrue(result['completed'])
                self.assertEqual(result['external_frames'], 9)
                self.assertEqual(len(result['period_results']), 3)
                self.assertEqual([row[0] for row in observed], list(range(9)))
                self.assertEqual(sum(phase == 'intermission' for _, phase in observed), 6)
                offset = 12*(2-side)
                self.assertTrue(all(buttons[offset+4] == 1 for buttons in sent))
                self.assertEqual(sent[3][3], 1)
                raw = np.zeros(12, dtype=np.int8)
                raw[4] = 1
                self.assertEqual(result['external_actions_sha256'], hashlib.sha256(raw.tobytes()*9).hexdigest())
                if side == 2:
                    self.assertEqual(sent[3][offset+3], 1)

    def test_early_quit_is_incomplete_and_closes_resources(self):
        result, sent, _ = self.match(1, quit_immediately=True)
        self.assertFalse(result['completed'])
        self.assertTrue(result['user_quit'])
        self.assertEqual(sent, [])
        self.assertEqual(result['controller_latency_ms']['calls'], 0)

    def test_emulator_error_closes_resources(self):
        self.match(1, fail_step=True)


class CommandTests(unittest.TestCase):
    def test_cli_registration_and_optional_adapter(self):
        from nhl94_ai.cli import main
        with patch('nhl94_ai.evaluation.external.load_adapter', side_effect=AssertionError('unexpected import')):
            with self.assertRaises(SystemExit) as stopped:
                main(['external', '--help'])
            self.assertEqual(stopped.exception.code, 0)

    def test_pairing_report_and_incomplete_scores(self):
        args = external.build_parser().parse_args(['--adapter', 'dummy:create', '--pairs', '2', '--seed', '10'])
        def match(job):
            return dict(seed=job['seed'], side=job['side'], goals=[1, 0], frames=1,
                        completed=job['side'] == 1, user_quit=False, goal_difference=1)
        with tempfile.TemporaryDirectory() as directory:
            args.output = str(Path(directory)/'results.json')
            with patch.object(external, 'play_match', side_effect=match) as play:
                with self.assertRaisesRegex(RuntimeError, 'Incomplete matches'):
                    external.run(args)
            self.assertEqual([(c.args[0]['seed'], c.args[0]['side']) for c in play.call_args_list],
                             [(10, 1), (10, 2), (11, 1), (11, 2)])
            report = json.loads(Path(args.output).read_text(encoding='utf-8'))
            self.assertEqual(report['summary'], dict(completed=2, incomplete=2, local_wins=2, draws=0, external_wins=0))

    def test_config_resolves_adapter_configuration(self):
        from nhl94_ai.cli import parse_configured
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'options.json'
            path.write_text(json.dumps(dict(adapter='dummy:create', adapter_config='adapter.json')))
            args = parse_configured(external.build_parser(), ['--config', str(path)])
            self.assertEqual(args.adapter_config, str(Path(directory)/'adapter.json'))

    def test_invalid_match_options(self):
        for flags in (['--pairs', '0'], ['--seed', '-1'], ['--periods', '3', '--seconds', '30'],
                      ['--frame-skip', '0'], ['--seed', str(2**32-1), '--pairs', '2']):
            args = external.build_parser().parse_args(['--adapter', 'dummy:create', *flags])
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                external.validate_options(args)


if __name__ == '__main__':
    unittest.main()
