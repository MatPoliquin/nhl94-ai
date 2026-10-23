"""The filter must abstain safely, preserve action lifecycles and isolate games."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.agents.retention_filter import RetentionFilter, SCHEMA
from nhl94_ai.agents.retention_features import features, movement_action, movement_id
from tests.support.offense import offense_state


def artifact():
    return dict(schema=SCHEMA, interval=4, horizon=24,
                feature_names=['projected_dy', 'pad_y', 'pad_x'], minimum_risk=0.5, minimum_gap=0.15,
                tree=[dict(left=1, right=2, feature=2, threshold=0.5, risk=0.5),
                      dict(left=-1, right=-1, feature=-2, threshold=-2, risk=0.8),
                      dict(left=-1, right=-1, feature=-2, threshold=-2, risk=0.1)])


class RetentionFilterTests(unittest.TestCase):
    def test_current_features_choose_lower_risk_admissible_action(self):
        policy = RetentionFilter(artifact=artifact())
        rows = [dict(projected_dy=3, pad_y=1, pad_x=x) for x in (0, 1)]
        self.assertEqual(policy.choose(rows, 0)[0], 1)
        rows[1]['projected_dy'] = 1
        self.assertEqual(policy.choose(rows, 0)[0], 0)

    def test_ties_and_insufficient_risk_gap_keep_classic(self):
        policy = RetentionFilter(artifact=artifact())
        rows = [dict(projected_dy=3, pad_y=1, pad_x=1) for _ in range(2)]
        self.assertEqual(policy.choose(rows, 1)[0], 1)
        data = artifact(); data['minimum_gap'] = 0.9
        policy = RetentionFilter(artifact=data)
        rows[0]['pad_x'] = 0
        self.assertEqual(policy.choose(rows, 0)[0], 0)

    def test_invalid_tree_and_schema_fail_closed(self):
        for change in (dict(interval=1), dict(schema='old'), dict(minimum_gap=0)):
            with self.assertRaises(ValueError):
                RetentionFilter(artifact={**artifact(), **change})
        data = artifact(); data['tree'][0]['left'] = 0
        with self.assertRaises(ValueError):
            RetentionFilter(artifact=data)

    def test_feature_order_is_part_of_contract(self):
        policy = RetentionFilter(artifact=artifact())
        with self.assertRaises(ValueError):
            policy.risk(dict(pad_x=1, pad_y=1, projected_dy=3))


    def test_filter_is_only_called_at_an_available_new_decision(self):
        model = ClassicAIV1Model(SimpleNamespace(action_type='FILTERED'))
        state = offense_state()
        policy = SimpleNamespace(apply=lambda *args: np.ones((1, 12), dtype=np.int8))
        model.retention_filter = policy
        neutral = np.zeros((1, 12), dtype=np.int8)
        def decision(*args):
            model.scheduler.decisions += 1
            return neutral
        with patch('nhl94_ai.agents.retention_features.available', return_value=True), \
                patch.object(model, '_predict_native_frame', side_effect=decision):
            self.assertTrue(model.predict_frame(state).all())
        with patch('nhl94_ai.agents.retention_features.available', return_value=True), \
                patch.object(model, '_predict_native_frame', return_value=neutral):
            self.assertFalse(model.predict_frame(state).any())
        with self.assertRaises(ValueError):
            model.predict_frame(state, frame_skip=1)

    def test_pass_and_shot_inputs_are_never_overridden(self):
        model = ClassicAIV1Model(SimpleNamespace(action_type='FILTERED'))
        model.input_owner = 'shot'
        action = np.zeros((1, 12), dtype=np.int8)
        policy = RetentionFilter(artifact=artifact())
        self.assertIs(policy.apply(model, offense_state(), action), action)
        model.input_owner = 'pass'
        self.assertIs(policy.apply(model, offense_state(), action), action)
        self.assertEqual(policy.metrics['eligible_decisions'], 0)


    def test_override_replaces_only_direction_and_cached_input(self):
        model = ClassicAIV1Model(SimpleNamespace(action_type='FILTERED'))
        model.input_owner = 'offense'
        model.scheduler.remaining = 3
        model.scheduler.frames = 41
        model.scheduler.decisions = 11
        policy = RetentionFilter(artifact=artifact())
        state = offense_state()
        rows = [dict(projected_dy=3, pad_y=1, pad_x=int(i == 1)) for i in range(9)]
        with patch('nhl94_ai.agents.retention_filter.features', side_effect=lambda state, i: rows[i]):
            action = policy.apply(model, state, np.zeros((1, 12), dtype=np.int8))
        self.assertEqual(movement_id(action[0], 1 if state.team2.net.y > state.team1.net.y else -1), 1)
        np.testing.assert_array_equal(action, model.scheduler.action)
        self.assertEqual((model.scheduler.remaining, model.scheduler.frames, model.scheduler.decisions), (3, 41, 11))
        self.assertFalse(model.buttons.b_down or model.buttons.c_down)
        self.assertEqual(policy.metrics['overrides'], 1)


if __name__ == '__main__':
    unittest.main()
