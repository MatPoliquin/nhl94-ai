"""Within-state supervision, export boundaries, and conservative rank decisions."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.agents.retention_ranker import RetentionRanker, SCHEMA, load_retention_filter
from nhl94_ai.evaluation.retention_ranking_model import ranking_metrics
from tests.support.offense import offense_state


def artifact():
    return dict(schema=SCHEMA, interval=4, horizon=24, base_score=0,
                feature_names=['projected_dy', 'pad_y', 'pad_x'], minimum_margin=0.2,
                trees=[[dict(left=1, right=2, feature=2, threshold=0.25, value=0),
                        dict(left=-1, right=-1, feature=0, threshold=0, value=-1),
                        dict(left=-1, right=-1, feature=0, threshold=0, value=1)]])


def branch(value, valid=True):
    return dict(features=dict(x=value), label_valid=valid, loss=value)


class RetentionRankingTests(unittest.TestCase):

    def test_state_only_prediction_gets_no_pairwise_credit_above_chance(self):
        rows = [dict(branches=[branch(0), branch(1)]) for _ in range(2)]
        scores = np.asarray([[7, 7], [-4, -4]], dtype=float)
        measured = ranking_metrics(rows, scores)
        self.assertEqual(measured['pair_accuracy'], 0.5)
        self.assertEqual(measured['tied_pairs'], 2)
        measured = ranking_metrics(rows, np.asarray([[3, 2], [-3, -4]]))
        self.assertEqual(measured['pair_accuracy'], 1)

    def test_strict_float32_split_and_margin_admission(self):
        policy = RetentionRanker(artifact=artifact())
        boundary = np.float32(0.25)
        below = np.nextafter(boundary, np.float32(-np.inf))
        rows = [dict(projected_dy=3, pad_y=1, pad_x=float(x)) for x in (below, boundary)]
        self.assertEqual(policy.scores(rows), [-1, 1])
        self.assertEqual(policy.choose(rows, 0)[0], 1)
        policy.minimum_margin = 3
        self.assertEqual(policy.choose(rows, 0)[0], 0)
        policy.minimum_margin = 0.2
        rows[1]['projected_dy'] = 0
        self.assertEqual(policy.choose(rows, 0)[0], 0)

    def test_ties_keep_classic_at_zero_margin(self):
        data = artifact(); data['minimum_margin'] = 0
        policy = RetentionRanker(artifact=data)
        rows = [dict(projected_dy=3, pad_y=1, pad_x=1) for _ in range(9)]
        self.assertEqual(policy.choose(rows, 7)[0], 7)

    def test_ranking_outputs_cannot_be_misread_as_probabilities(self):
        policy = RetentionRanker(artifact=artifact())
        with self.assertRaises(ValueError):
            policy.risk(dict(projected_dy=3, pad_y=1, pad_x=1))
        self.assertEqual(policy.score_name, 'ranking_scores')

    def test_bad_schema_nonfinite_inputs_and_cycles_fail_closed(self):
        for changes in (dict(interval=1), dict(schema='old'), dict(minimum_margin=-1),
                        dict(minimum_margin=float('inf')), dict(trees=[])):
            with self.assertRaises(ValueError):
                RetentionRanker(artifact={**artifact(), **changes})
        data = artifact(); data['trees'][0][0]['left'] = 0
        with self.assertRaises(ValueError):
            RetentionRanker(artifact=data)
        policy = RetentionRanker(artifact=artifact())
        with self.assertRaises(ValueError):
            policy.predict_matrix([[0, 0, float('nan')]])
        with self.assertRaises(ValueError):
            policy.scores([dict(pad_y=0, projected_dy=0, pad_x=0)])

    def test_versioned_loader_wires_classic_without_training_library(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'ranker.json'
            path.write_text(json.dumps(artifact()), encoding='utf-8')
            policy = load_retention_filter(path)
            self.assertIsInstance(policy, RetentionRanker)
            model = ClassicAIV1Model(SimpleNamespace(action_type='FILTERED', retention_model=str(path)))
            self.assertEqual(model.retention_filter.sha256, policy.sha256)

    def test_rank_override_preserves_scheduler_and_pass_priority(self):
        model = ClassicAIV1Model(SimpleNamespace(action_type='FILTERED'))
        model.input_owner = 'offense'
        model.scheduler.remaining, model.scheduler.decisions, model.scheduler.frames = 3, 11, 41
        policy = RetentionRanker(artifact=artifact())
        rows = [dict(projected_dy=3, pad_y=1, pad_x=int(i == 1)) for i in range(9)]
        neutral = np.zeros((1, 12), dtype=np.int8)
        with patch('nhl94_ai.agents.retention_filter.features', side_effect=lambda state, i: rows[i]):
            action = policy.apply(model, offense_state(), neutral)
        self.assertEqual(policy.metrics['overrides'], 1)
        self.assertIn('ranking_scores', model.offense_diagnostics['retention'])
        self.assertNotIn('risks', model.offense_diagnostics['retention'])
        np.testing.assert_array_equal(action, model.scheduler.action)
        self.assertEqual((model.scheduler.remaining, model.scheduler.decisions, model.scheduler.frames), (3, 11, 41))
        model.input_owner = 'pass'
        self.assertIs(policy.apply(model, offense_state(), neutral), neutral)


if __name__ == '__main__':
    unittest.main()
