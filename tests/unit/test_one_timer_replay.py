"""Counterfactual execution keeps native input history and primary-shot attribution."""
from copy import deepcopy
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np

from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.agents.lifecycle import ONE_TIMER_EXECUTION_INCOMPATIBLE
from nhl94_ai.evaluation.one_timer_replay import configured, fork_timer
from nhl94_ai.game.constants import GameConsts as Buttons
from tests.unit.test_classic_offense import live_one_timer_state


def started():
    state, model = live_one_timer_state(), ClassicAIV1Model()
    state.team1.one_timer_attempts = 0
    model.predict_frame(state, 4)
    assert model.one_timer.pending is not None
    assert model.scheduler.action[0, Buttons.INPUT_B]
    return state, model


class OneTimerReplayTests(TestCase):
    def test_execution_is_opt_in_and_rejects_unsupported_combinations(self):
        self.assertIsNone(ClassicAIV1Model().one_timer.execution)
        for mode in ('early-cue', 'release-retry'):
            args = {'one_timer_execution': mode}
            self.assertEqual(ClassicAIV1Model(SimpleNamespace(**args)).one_timer.execution, mode)
            for setting in ({'action_type': 'HOCKEY_INTENT_DPAD'}, {'env': 'other'}, {'selfplay': True},
                            *({flag: ['finishing'] if flag == 'classic_refinements' else True}
                              for flag in ONE_TIMER_EXECUTION_INCOMPATIBLE)):
                with self.subTest(mode=mode, setting=setting), self.assertRaises(ValueError):
                    ClassicAIV1Model(SimpleNamespace(**args, **setting))
        with self.assertRaises(ValueError):
            ClassicAIV1Model(SimpleNamespace(one_timer_execution='unknown'))

    def test_branches_preserve_initial_pass_and_isolate_controller_history(self):
        _, history = started()
        saved = deepcopy(history)
        for variant in ('legacy', 'frame-cue', 'early-cue', 'release-retry'):
            model = configured(history, variant)
            np.testing.assert_array_equal(model.scheduler.action, saved.scheduler.action)
            self.assertTrue(model.scheduler.action[0, Buttons.INPUT_B])
            self.assertFalse(model.scheduler.action[0, Buttons.INPUT_C])
            self.assertEqual(model.buttons, saved.buttons)
            self.assertEqual(model.scheduler.frames, saved.scheduler.frames)
            model.one_timer.pending.launched = True
            self.assertFalse(history.one_timer.pending.launched)
        with self.assertRaises(ValueError):
            configured(history, 'unknown')

    def test_early_cue_requires_observed_flight_and_never_overlaps_b(self):
        state, history = started()
        model = configured(history, 'early-cue')
        waiting = model.predict_frame(state, 4)[0]
        self.assertTrue(waiting[Buttons.INPUT_B])
        self.assertFalse(waiting[Buttons.INPUT_C])
        state.engine.puck_owner = -1
        state.team1.pass_attempts = 1
        state.engine.pass_target = model.one_timer.pending.receiver
        cue = model.predict_frame(state, 4)[0]
        self.assertTrue(cue[Buttons.INPUT_C])
        self.assertFalse(cue[Buttons.INPUT_B])
        self.assertEqual(model.scheduler.frames, history.scheduler.frames + 2)

    def test_frame_cue_preserves_four_frame_pass_hold(self):
        state, history = started()
        model = configured(history, 'frame-cue')
        state.engine.puck_owner = -1
        state.engine.pass_target = model.one_timer.pending.receiver
        state.team1.pass_attempts = 1
        frames = [model.scheduler.action[0]] + [model.predict_frame(state, 4)[0] for _ in range(5)]
        self.assertEqual([bool(a[Buttons.INPUT_B]) for a in frames], [True]*4 + [False]*2)
        self.assertEqual([bool(a[Buttons.INPUT_C]) for a in frames], [False]*4 + [True, False])

    def test_release_retry_has_fresh_b_edges_and_a_fixed_deadline(self):
        state, history = started()
        model = configured(history, 'release-retry')
        deadline = model.one_timer.pending.deadline
        actions = [model.scheduler.action[0]] + [model.predict_frame(state, 4)[0] for _ in range(19)]
        self.assertEqual([bool(a[Buttons.INPUT_B]) for a in actions],
                         [True]*4 + [False]*4 + [True]*4 + [False]*4 + [True]*4)
        self.assertFalse(any(a[Buttons.INPUT_C] for a in actions))
        self.assertEqual(model.one_timer.pending.deadline, deadline)
        # Flight ends all retries even though the receiver has not activated yet.
        state.engine.puck_owner = -1
        state.engine.pass_target = model.one_timer.pending.receiver
        state.team1.pass_attempts += 1
        in_flight = [model.predict_frame(state, 4)[0] for _ in range(8)]
        self.assertFalse(any(a[Buttons.INPUT_B] for a in in_flight))
        self.assertTrue(any(a[Buttons.INPUT_C] for a in in_flight))

    def test_clock_stoppage_releases_buttons_and_accounts_for_one_ending(self):
        state, history = started()
        for variant in ('early-cue', 'release-retry'):
            model = configured(history, variant)
            state.engine.clock_stopped = True
            action = model.predict_frame(state, 4)[0]
            self.assertFalse(action.any())
            self.assertIsNone(model.one_timer.pending)
            self.assertEqual(model.one_timer.metrics, {'play-stopped': 1})

    def test_early_cue_does_not_activate_a_mismatched_native_receiver(self):
        state, history = started()
        model = configured(history, 'early-cue')
        state.engine.puck_owner = -1
        state.engine.pass_target = model.one_timer.pending.receiver + 1
        state.team1.pass_attempts += 1
        self.assertFalse(model.predict_frame(state, 4)[0].any())
        self.assertEqual(model.one_timer.metrics, {'receiver-mismatch': 1})

    def test_later_attempt_and_activation_do_not_relabel_failed_primary_sequence(self):
        state, history = started()
        model = deepcopy(history)
        later = deepcopy(state)
        later.team1.get_player_by_scnum(model.one_timer.pending.receiver).is_one_timer = 1
        later.engine.shot_player = model.one_timer.pending.receiver
        later.team1.one_timer_attempts = 1
        info = {'p1_score': 0, 'p2_score': 0, 'bench_clock': 100}

        def ended(*_args, **_kwargs):
            model.one_timer.finish('possession-changed')
            return np.zeros((1, Buttons.INPUT_MAX), dtype=np.int8)

        model.predict_frame = Mock(side_effect=ended)
        with (patch('nhl94_ai.evaluation.one_timer_replay.restore'),
              patch('nhl94_ai.evaluation.one_timer_replay.configured', return_value=model),
              patch('nhl94_ai.evaluation.one_timer_replay.advance'),
              patch('nhl94_ai.evaluation.one_timer_replay.trace_state', return_value={}),
              patch('nhl94_ai.evaluation.one_timer_replay.read_view', side_effect=[
                  (state, info), (state, info), (later, info), (later, info)])):
            result = fork_timer(Mock(), b'', '', state, 1, history, 'legacy', 3)
        self.assertEqual(result['primary_end'], {'reason': 'possession-changed', 'frame': 1})
        self.assertIsNone(result['activation_frame'])
        self.assertIsNone(result['first_attempt'])
        self.assertEqual(result['continuation_native_attempts'], 1)
