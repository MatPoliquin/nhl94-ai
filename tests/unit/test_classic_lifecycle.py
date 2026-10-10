"""Timing and transition contracts retained by the lifecycle extraction."""
from copy import deepcopy
from types import SimpleNamespace
import unittest

from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.agents.lifecycle import PassLifecycle, PassState, ReceptionCorrection
from nhl94_ai.evaluation.lifecycle_history import cache_input, copy_history
from nhl94_ai.game.constants import GameConsts as Buttons
from tests.unit.test_classic_offense import offense_state
from tests.unit.test_classic_reception import reception_state, waiting_model
from tests.unit.test_classic_v1 import shooting_state


class LifecycleContracts(unittest.TestCase):
    def test_same_shooter_recovery_preserves_existing_follow_through(self):
        # Characterize the known behavior; changing it requires a separate fix.
        state, model = shooting_state(), ClassicAIV1Model()
        model.predict_frame(state, 4)
        state.engine.puck_owner, state.engine.shot_player = -1, 0
        state.team1.stats.shots += 1
        for _ in range(8):
            model.predict_frame(state, 4)
        state.engine.puck_owner = 0
        action = model.predict_frame(state, 4)[0]
        self.assertEqual(model._last_decision, 'shot-follow-through')
        self.assertFalse(action[Buttons.INPUT_C])
        self.assertEqual(model.input_owner, 'shot')

    def test_deadline_keeps_priority_over_simultaneous_reception(self):
        state = offense_state()
        state.engine.puck_owner = 1
        for frame, outcome in ((19, 'received'), (20, 'flight-timeout'), (21, 'flight-timeout')):
            with self.subTest(frame=frame):
                lifecycle = PassLifecycle(PassState(0, 1, 0, 20, launched=True, flight_observed=True))
                event = lifecycle.observe(state, frame)
                self.assertEqual(event.reason, outcome)
                saved = deepcopy(lifecycle.last_pass)
                self.assertIsNone(lifecycle.finish(state, frame + 1, 'duplicate'))
                self.assertEqual(lifecycle.last_pass, saved)

    def test_receiver_correction_advances_frames_without_advancing_decisions(self):
        state, model = reception_state(), waiting_model()
        initial = model.scheduler.frames
        for _ in range(4):
            model.predict_frame(state, 4)
        self.assertEqual(model.scheduler.frames, initial + 4)
        self.assertEqual(model.scheduler.decisions, 0)
        self.assertEqual(model.defense.frames, model.scheduler.frames)
        self.assertEqual(model.input_owner, 'pass')

    def test_native_and_legacy_callers_keep_distinct_elapsed_contracts(self):
        for native, frames in ((False, 12), (True, 3)):
            with self.subTest(native=native):
                state, model = shooting_state(), ClassicAIV1Model()
                for _ in range(3):
                    if native:
                        model.predict_frame(state, 4)
                    else:
                        model.predict_game_state(state)
                self.assertEqual(model.scheduler.frames, frames)
                self.assertEqual(model.scheduler.decisions, 1 if native else 3)

    def test_pass_snapshot_and_fork_do_not_share_reception_correction(self):
        lifecycle = PassLifecycle(PassState(0, 1, 0, 20))
        lifecycle.pending.reception_correction = ReceptionCorrection(4, {'frames': 9}, 0)
        cloned = deepcopy(lifecycle)
        cloned.pending.reception_correction.switch_frame = 8
        self.assertIsNone(lifecycle.pending.reception_correction.switch_frame)
        self.assertEqual(cloned.pending.snapshot()['reception_correction']['switch_frame'], 8)

    def test_one_timer_completion_order_and_once_only_accounting(self):
        state, model = offense_state(), ClassicAIV1Model()
        model.one_timer.start(state, 1, 5)
        pending = model.one_timer.pending
        pending.launched = True
        state.engine.shot_player = 1
        state.team1.stats.shots += 1
        # Release wins even at the deadline with the passer owning the rebound.
        result = model.one_timer.observe(state, pending.deadline)
        self.assertEqual(result.reason, 'shot-released')
        self.assertIsNone(model.one_timer.finish('timeout'))
        self.assertEqual(model.one_timer.starts, sum(model.one_timer.metrics.values()))


class ArchivedHistoryContracts(unittest.TestCase):
    def history(self):
        state, model = offense_state(), ClassicAIV1Model()
        model.scheduler.frames, model.scheduler.decisions = 65, 17
        model.scheduler.remaining = 3
        model.shot.start(state, 17, 65, 4)
        model.one_timer.start(state, 1, 65)
        model.offense.pass_action.pending = PassState(0, 1, 60, 90, launched=True)
        return model

    def test_current_copy_retains_target_executor_implementations(self):
        history, target = self.history(), ClassicAIV1Model()
        shot, one_timer, passing = target.shot, target.one_timer, target.offense.pass_action
        copy_history(history, target)
        self.assertIs(target.shot, shot)
        self.assertIs(target.one_timer, one_timer)
        self.assertIs(target.offense.pass_action, passing)
        self.assertEqual(target.scheduler.frames, 65)
        target.one_timer.pending.launched = True
        self.assertFalse(history.one_timer.pending.launched)

    def test_legacy_export_preserves_both_clocks_and_pending_actions(self):
        history = self.history()
        legacy = SimpleNamespace(offense=SimpleNamespace())
        copy_history(history, legacy)
        self.assertEqual((legacy._tick, legacy.defense.frames, legacy._frame_remaining), (17, 65, 3))
        self.assertEqual(legacy._one_timer, (0, 1, 137))
        self.assertEqual(legacy._one_timer_started, 65)
        self.assertEqual(legacy._shot_until, 24)
        self.assertEqual(legacy.offense.pending['deadline'], 90)
        legacy.offense.pending['launched'] = False
        self.assertTrue(history.offense.pass_action.pending.launched)
        cache_input(legacy, [1], 2)
        self.assertEqual((legacy._frame_action, legacy._frame_remaining), ([1], 2))
