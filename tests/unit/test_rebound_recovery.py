"""A new possession can end shot follow-through only after an observed release."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.agents.lifecycle import ShotLifecycle, ONE_TIMER_EXECUTION_INCOMPATIBLE
from nhl94_ai.game.constants import GameConsts as Buttons
from nhl94_ai.evaluation.rebound_replay import fork_rebound
from tests.unit.test_classic_v1 import shooting_state as base_shooting_state


def shooting_state():
    state = base_shooting_state()
    state.team1.players[0].selection_flags = 8
    return state


class ReboundRecoveryTests(unittest.TestCase):
    def shot(self):
        state = shooting_state()
        shot = ShotLifecycle(rebound_recovery=True)
        shot.start(state, 5, 20, 4)
        return state, shot

    def test_same_owner_during_windup_or_unconfirmed_loose_puck_does_not_cancel(self):
        state, shot = self.shot()
        self.assertFalse(shot.observe(state, 6))
        state.engine.puck_owner = -1
        self.assertFalse(shot.observe(state, 6))
        state.engine.puck_owner = shot.slot
        self.assertFalse(shot.observe(state, 6))
        self.assertTrue(shot.active(6))
        self.assertEqual(shot.recoveries, 0)

    def test_confirmed_release_stays_in_follow_through_until_actual_recovery(self):
        state, shot = self.shot()
        state.engine.puck_owner, state.engine.shot_player = -1, shot.slot
        state.team1.stats.shots += 1
        self.assertFalse(shot.observe(state, 6))
        self.assertTrue(shot.release_observed)
        self.assertTrue(shot.active(6))
        state.engine.puck_owner = shot.slot
        self.assertTrue(shot.observe(state, 6))
        self.assertFalse(shot.active(6))
        self.assertFalse(shot.observe(state, 6))
        self.assertEqual(shot.recoveries, 1)

    def test_recovered_shooter_waits_for_native_input_unlock_and_known_feedback(self):
        for flags in (None, 0x28):
            state, shot = self.shot()
            state.engine.puck_owner, state.engine.shot_player = -1, shot.slot
            state.team1.stats.shots += 1
            shot.observe(state, 6)
            state.engine.puck_owner = shot.slot
            state.team1.players[0].selection_flags = flags
            self.assertFalse(shot.observe(state, 6))
            self.assertTrue(shot.active(6))
            state.team1.players[0].selection_flags = 8
            self.assertTrue(shot.observe(state, 6))
            self.assertEqual(shot.recoveries, 1)

    def test_stale_shot_count_wrong_shooter_and_clock_stoppage_cannot_create_recovery(self):
        for reason in ('stale-count', 'wrong-shooter', 'stopped-clock'):
            state, shot = self.shot()
            state.engine.puck_owner, state.engine.shot_player = -1, shot.slot
            state.team1.stats.shots += int(reason != 'stale-count')
            if reason == 'wrong-shooter':
                state.engine.shot_player = 2
            if reason == 'stopped-clock':
                shot.observe(state, 6)
                state.engine.clock_stopped = True
            self.assertFalse(shot.observe(state, 6))
            state.engine.clock_stopped = False
            state.engine.puck_owner = shot.slot
            self.assertFalse(shot.observe(state, 6), reason)

    def test_each_new_shot_and_cancellation_clear_previous_release_evidence(self):
        state, shot = self.shot()
        shot.release_observed = True
        shot.start(state, 6, 24, 4)
        self.assertFalse(shot.release_observed)
        shot.release_observed = True
        shot.cancel()
        self.assertFalse(shot.release_observed)

    def test_other_teammate_recovery_keeps_legacy_cancellation(self):
        state, shot = self.shot()
        state.engine.puck_owner = 1
        self.assertTrue(shot.observe(state, 6))
        self.assertEqual(shot.recoveries, 0)

    def test_native_dispatch_replans_immediately_in_both_schemas_at_each_interval(self):
        for schema in ('FILTERED', 'HOCKEY_INTENT_DPAD'):
            for interval in (1, 4, 8):
                state = shooting_state()
                model = ClassicAIV1Model(SimpleNamespace(action_type=schema, rebound_recovery=True))
                model.predict_frame(state, interval)
                state.engine.puck_owner, state.engine.shot_player = -1, 0
                state.team1.stats.shots += 1
                model.predict_frame(state, interval)
                previous_decisions, previous_frames = model.scheduler.decisions, model.scheduler.frames
                state.engine.puck_owner = 0
                model.predict_frame(state, interval)
                self.assertEqual(model.shot.recoveries, 1, (schema, interval))
                self.assertNotEqual(model._last_decision, 'shot-follow-through')
                self.assertEqual(model.scheduler.decisions, previous_decisions + 1)
                self.assertEqual(model.scheduler.frames, previous_frames + 1)

    def test_release_tracking_alone_keeps_default_inputs_identical(self):
        state = shooting_state()
        old, new = ClassicAIV1Model(), ClassicAIV1Model(SimpleNamespace(rebound_recovery=False))
        for owner in (0, -1, -1, 0, 0):
            state.engine.puck_owner, state.engine.shot_player = owner, 0
            if owner < 0:
                state.team1.stats.shots = 1
            np.testing.assert_array_equal(old.predict_frame(state, 4), new.predict_frame(state, 4))
        self.assertEqual(new.shot.recoveries, 0)

    def test_recovery_releases_old_c_before_another_shot_can_start(self):
        state = shooting_state()
        model = ClassicAIV1Model(SimpleNamespace(rebound_recovery=True))
        self.assertTrue(model.predict_frame(state, 4)[0, Buttons.INPUT_C])
        state.engine.puck_owner, state.engine.shot_player = -1, 0
        state.team1.stats.shots += 1
        model.predict_frame(state, 4)
        self.assertTrue(model.buttons.c_down)
        state.engine.puck_owner = 0
        self.assertFalse(model.predict_frame(state, 4)[0, Buttons.INPUT_C])
        self.assertFalse(model.buttons.c_down)
        self.assertEqual(model.shot.recoveries, 1)

    def test_replay_changes_only_the_initial_recovery_and_preserves_original_history(self):
        state = shooting_state()
        state.team1.one_timer_attempts = 0
        history = ClassicAIV1Model()
        history.shot.start(state, 0, 0, 4)
        history.shot.release_observed = True
        info = {'p1_score': 0, 'p2_score': 0, 'bench_clock': 100, 'bench_shots1': 1}

        def trace(_env, _state, model, _action, frame, _slot):
            self.assertFalse(model.shot.rebound_recovery)
            return {'frame': frame, 'recoveries': model.shot.recoveries}

        with (patch('nhl94_ai.evaluation.rebound_replay.restore'),
              patch('nhl94_ai.evaluation.rebound_replay.advance'),
              patch('nhl94_ai.evaluation.rebound_replay.trace_state', side_effect=trace),
              patch('nhl94_ai.evaluation.rebound_replay.read_view', return_value=(state, info))):
            result = fork_rebound(Mock(), b'', '', state, 1, history, True, 3)
        self.assertEqual(result['first_replan']['frame'], 0)
        self.assertEqual([t['recoveries'] for t in result['trace']], [1, 1, 1])
        self.assertTrue(history.shot.release_observed)
        self.assertEqual(history.shot.recoveries, 0)

    def test_recovery_evidence_is_copied_without_sharing_and_expires_with_commitment(self):
        state, shot = self.shot()
        shot.release_observed = True
        clone = deepcopy(shot)
        clone.cancel()
        self.assertTrue(shot.release_observed)
        self.assertFalse(shot.same_shooter_recovery(state, shot.until_decision))

    def test_experiment_validation_rejects_mixed_policies(self):
        self.assertFalse(ClassicAIV1Model().shot.rebound_recovery)
        for key in (*ONE_TIMER_EXECUTION_INCOMPATIBLE, 'one_timer_execution'):
            value = ['finishing'] if key == 'classic_refinements' else True
            with self.subTest(key=key), self.assertRaises(ValueError):
                ClassicAIV1Model(SimpleNamespace(rebound_recovery=True, **{key: value}))
        for values in ({'selfplay': True}, {'action_type': 'TARGET_POSITION'}, {'env': 'other'}):
            with self.assertRaises(ValueError):
                ClassicAIV1Model(SimpleNamespace(rebound_recovery=True, **values))
