"""Counterfactual replays clone actual policy history and isolate imported snapshots."""
from copy import deepcopy
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.evaluation.carry_replay import (
    build_parser, frame_action, forced_policy, load_reference, policy_from_history, sampling_strata,
)
from tests.unit.test_classic_offense import offense_state


class CarryReplayTests(unittest.TestCase):
    def test_policy_history_is_equal_but_not_shared(self):
        history = ClassicAIV1Model()
        history._tick, history._frame_remaining = 17, 0
        history.offense.pass_at = 42
        history.offense.last_pass = {'outcome': 'received', 'owner': 6}
        before = deepcopy(history.offense.last_pass)
        policy = policy_from_history(history, ClassicAIV1Model, experimental=True)
        self.assertEqual(policy._tick, 17)
        self.assertEqual(policy.offense.pass_at, 42)
        self.assertTrue(policy.offense.allow_uncertified)
        policy.offense.last_pass['owner'] = 7
        self.assertEqual(history.offense.last_pass, before)
        self.assertFalse(history.offense.allow_uncertified)

    def test_reference_import_restores_live_modules_on_success_and_failure(self):
        names = [f'nhl94_ai.agents.{name}' for name in ('carry', 'offense', 'classic_v1')]
        before = {name: sys.modules[name] for name in names}
        with TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / 'nhl94_ai' / 'agents'
            package.mkdir(parents=True)
            (package / 'carry.py').write_text('VALUE = 1\n', encoding='utf-8')
            (package / 'offense.py').write_text('VALUE = 2\n', encoding='utf-8')
            (package / 'classic_v1.py').write_text('class ClassicAIV1Model:\n    pass\n', encoding='utf-8')
            self.assertEqual(load_reference(root).__name__, 'ClassicAIV1Model')
            self.assertTrue(all(sys.modules[name] is before[name] for name in names))
            (package / 'classic_v1.py').write_text('raise RuntimeError("broken snapshot")\n', encoding='utf-8')
            with self.assertRaisesRegex(RuntimeError, 'broken snapshot'):
                load_reference(root)
            self.assertTrue(all(sys.modules[name] is before[name] for name in names))

    def test_replay_defaults_are_bounded_and_offensive_zone_focused(self):
        arguments = build_parser().parse_args(['--reference-runtime=frozen'])
        self.assertEqual(arguments.min_depth, 140)
        self.assertEqual(arguments.horizon, 180)
        self.assertEqual(arguments.seeds, 6)
        self.assertEqual(arguments.search_frames, 6000)
        self.assertEqual(arguments.strata, ['general'])
        self.assertEqual(arguments.workers, 1)

    def test_outcome_and_sampling_controls_are_explicit(self):
        arguments = build_parser().parse_args([
            '--reference-runtime=frozen', '--outcome-ended', '--horizon=900',
            '--strata', 'general', 'one-timer', 'safe-pass', '--workers=2'])
        self.assertTrue(arguments.outcome_ended)
        self.assertEqual(arguments.horizon, 900)
        self.assertEqual(arguments.strata, ['general', 'one-timer', 'safe-pass'])
        self.assertEqual(arguments.workers, 2)

    def test_strata_are_predetermined_and_include_recent_one_timer_creation(self):
        self.assertEqual(sampling_strata(100, 0, [], [], ['carry']), {'general'})
        self.assertEqual(sampling_strata(100, 68, [], [object()], ['carry']),
                         {'general', 'one-timer', 'safe-pass'})
        self.assertEqual(sampling_strata(100, 67, [], [], ['carry']), {'general'})
        self.assertEqual(sampling_strata(100, 0, [], [], ['one-timer-setup']), {'general', 'one-timer'})

    def test_opportunity_controls_are_not_excluded_when_policies_agree(self):
        self.assertEqual(sampling_strata(100, 0, [object()], [object()], ['one-timer-pass'],
                                         divergent=False), {'one-timer', 'safe-pass'})
        self.assertEqual(sampling_strata(100, 68, [], [], ['carry'], divergent=False), set())

    def test_standalone_replays_use_the_native_four_frame_dispatch(self):
        policy = ClassicAIV1Model()
        state = offense_state()
        with patch.object(policy, 'predict_frame', wraps=policy.predict_frame) as dispatch, \
                patch.object(policy, 'predict_game_state', side_effect=AssertionError('wrong cadence')):
            for _ in range(4):
                frame_action(policy, state)
        self.assertEqual(dispatch.call_count, 4)
        self.assertEqual(policy._tick, 1)
        self.assertEqual(policy.defense.frames, 4)
        self.assertEqual(policy._frame_remaining, 0)

    def test_forced_choices_advance_first_frame_clocks_without_a_phantom_plan(self):
        history = ClassicAIV1Model()
        history._tick, history.defense.frames = 17, 64
        state = offense_state()
        policy = forced_policy(history, state)
        self.assertEqual(policy._tick, 18)
        self.assertEqual(policy.defense.frames, 65)
        self.assertIsNone(policy._one_timer)
        self.assertIsNone(policy.offense.pending)
        self.assertEqual(history._tick, 17)
        self.assertEqual(history.defense.frames, 64)


if __name__ == '__main__':
    unittest.main()
