"""Learned placement must preserve the shot trigger and exact input ownership."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.agents.shot_placement import CONTROL_VERSION, INCOMPATIBLE, MODEL_VERSION, ShotPlacement, shot_actions, shot_features
from nhl94_ai.evaluation.shot_placement_replay import placed
from nhl94_ai.game.constants import GameConsts as Buttons
from tests.unit.test_classic_offense import offense_state, live_one_timer_state
from tests.unit.test_possession_value import finish_state


def forced(side, hold):
    return SimpleNamespace(choose=Mock(return_value=(side, hold)), diagnostics={'status': 'test'})


class ShotPlacementTests(unittest.TestCase):
    def test_fixed_hold_control_preserves_legacy_aim_in_both_directions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'control.json'
            path.write_text(json.dumps({'version': CONTROL_VERSION, 'hold': 12}))
            control = ShotPlacement(path)
            state = finish_state()
            state.team1.players[0].shot_power = 20
            for goalie_x, aim in ((20, -1), (-20, 1)):
                state.team2.goalie.x = goalie_x
                self.assertEqual(control.choose(state, 4), (aim, 12))
            path.write_text(json.dumps({'version': CONTROL_VERSION, 'hold': True}))
            with self.assertRaises(ValueError):
                ShotPlacement(path)

    def test_shot_placement_rejects_combined_offensive_experiments(self):
        self.assertIsNone(ClassicAIV1Model().shot_placement)
        for flag in INCOMPATIBLE:
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                ClassicAIV1Model(SimpleNamespace(shot_placement='unused.json',
                                                **{flag: ['finishing'] if flag == 'classic_refinements' else True}))

    def test_holds_cross_decision_boundaries_and_release_on_the_requested_frame(self):
        for interval in (1, 4, 8):
            for hold in (1, 4, 8, 12):
                state = finish_state()
                model = ClassicAIV1Model()
                model.shot_placement = forced(0, hold)
                actions = [model.predict_frame(state, interval)[0] for _ in range(hold + 2)]
                self.assertEqual([bool(action[Buttons.INPUT_C]) for action in actions],
                                 [True]*hold + [False]*2, (interval, hold))
                self.assertTrue(all(not action[Buttons.INPUT_LEFT] and not action[Buttons.INPUT_RIGHT]
                                    for action in actions[:hold]))
                self.assertEqual(model.shot_placement.choose.call_count, 1)

    def test_existing_pass_and_one_timer_decisions_do_not_consult_shot_model(self):
        for factory in (offense_state, live_one_timer_state):
            state = factory()
            old, new = ClassicAIV1Model(), ClassicAIV1Model()
            new.shot_placement = forced(-1, 12)
            np.testing.assert_array_equal(old.predict_frame(state), new.predict_frame(state))
            new.shot_placement.choose.assert_not_called()

    def test_cancellation_cannot_suppress_future_defensive_c_inputs(self):
        model = ClassicAIV1Model()
        model.shot_placement = forced(1, 12)
        state = finish_state()
        model.predict_frame(state)
        self.assertTrue(model.shot.extended_hold)
        model.shot.cancel()
        self.assertFalse(model.shot.extended_hold)

    def test_restored_placement_changes_only_aim_and_hold_of_existing_shot(self):
        model = ClassicAIV1Model()
        model.predict_frame(finish_state())
        old = deepcopy(model)
        branch = placed(model, 0, 12)
        self.assertEqual(branch._last_decision, model._last_decision)
        self.assertEqual(branch.scheduler.frames, model.scheduler.frames)
        self.assertEqual(branch.buttons, model.buttons)
        self.assertEqual(branch.shot.hold_until_frame, model.scheduler.frames + 12)
        self.assertTrue(branch.scheduler.action[0, Buttons.INPUT_C])
        np.testing.assert_array_equal(model.scheduler.action, old.scheduler.action)
        self.assertEqual(model.shot, old.shot)

    def test_finite_model_and_motion_features_drive_action_selection(self):
        state = finish_state()
        state.team1.players[0].shot_power = 20
        model_data = {'version': MODEL_VERSION, 'features': ['aim', 'hold'], 'mean': [0, 0],
                      'scale': [1, 1], 'weights': [1, 1], 'bias': 0, 'min_gain': 0}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.json'
            path.write_text(json.dumps(model_data))
            placement = ShotPlacement(path)
            self.assertEqual(placement.choose(state, 4), (1, 12))
            baseline = (-1, 4)
            self.assertEqual(shot_actions(state, 4)[0], baseline)
            a = shot_features(state, -1, 4)
            state.team2.goalie.motion_x = 1
            b = shot_features(state, -1, 4)
            self.assertNotEqual(a, b)
            model_data['scale'] = [0, 1]
            path.write_text(json.dumps(model_data))
            with self.assertRaises(ValueError):
                ShotPlacement(path)

    def test_missing_feedback_falls_back_to_original_aim_and_cadence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.json'
            path.write_text(json.dumps({'version': MODEL_VERSION, 'features': ['aim'], 'mean': [0],
                                        'scale': [1], 'weights': [1], 'bias': 0, 'min_gain': 0}))
            placement = ShotPlacement(path)
            self.assertEqual(placement.choose(finish_state(), 4), (-1, 4))
            self.assertEqual(placement.metrics['missing-feedback'], 1)
