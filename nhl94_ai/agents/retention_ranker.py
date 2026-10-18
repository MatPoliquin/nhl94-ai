"""Dependency-free boosted-tree ranking for the optional retention filter."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from nhl94_ai.agents.retention_features import INTERVAL, admissible
from nhl94_ai.agents.retention_filter import RetentionFilter


SCHEMA = 'native-carry-retention-ranker-v1'


def load_retention_filter(path):
    """Dispatch versioned artifacts without importing training dependencies."""
    schema = json.loads(Path(path).read_text(encoding='utf-8')).get('schema')
    return RetentionRanker(path) if schema == SCHEMA else RetentionFilter(path)


class RetentionRanker(RetentionFilter):
    """Scores order movements within one state; they are not probabilities."""
    score_name = 'ranking_scores'

    # Reuse action application, but validate the ensemble's distinct schema.
    def __init__(self, path=None, *, artifact=None):  # pylint: disable=super-init-not-called
        self.path = str(Path(path).resolve()) if path else None
        raw = Path(path).read_bytes() if path else None
        self.sha256 = hashlib.sha256(raw).hexdigest() if raw is not None else None
        self.artifact = artifact if artifact is not None else json.loads(raw)
        data = self.artifact
        if (data.get('schema') != SCHEMA or data.get('interval') != INTERVAL
                or data.get('horizon') != 24 or not data.get('feature_names')):
            raise ValueError('Incompatible retention ranker schema or cadence')
        self.names = tuple(data['feature_names'])
        self.minimum_margin = data['minimum_margin']
        if not np.isfinite(self.minimum_margin) or self.minimum_margin < 0:
            raise ValueError('Ranking margin must be finite and nonnegative')
        self.base_score = np.float32(data['base_score'])
        if not np.isfinite(self.base_score) or not 1 <= len(data['trees']) <= 1024:
            raise ValueError('Invalid ranking ensemble')
        self.trees = []
        for tree in data['trees']:
            if not 1 <= len(tree) <= 511:
                raise ValueError('Invalid ranking tree size')
            for i, node in enumerate(tree):
                if not np.isfinite(node['value']):
                    raise ValueError('Invalid ranking leaf value')
                if node['left'] == -1:
                    if node['right'] != -1:
                        raise ValueError('Invalid ranking leaf children')
                elif not (i < node['left'] < len(tree) and i < node['right'] < len(tree)
                          and 0 <= node['feature'] < len(self.names) and np.isfinite(node['threshold'])):
                    raise ValueError('Invalid ranking split')
            self.trees.append({key: np.asarray([node[key] for node in tree], dtype=dtype)
                               for key, dtype in (('left', int), ('right', int), ('feature', int),
                                                  ('threshold', np.float32), ('value', np.float32))})
        self.metrics = Counter()

    def risk(self, row):
        raise ValueError('Ranking scores are not calibrated loss probabilities')

    def predict_matrix(self, values):
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != len(self.names) or not np.isfinite(values).all():
            raise ValueError('Invalid ranking feature matrix')
        scores = np.full(len(values), self.base_score, dtype=np.float32)
        for tree in self.trees:
            indices = np.zeros(len(values), dtype=int)
            active = np.flatnonzero(tree['left'][indices] >= 0)
            while len(active):
                nodes = indices[active]
                # XGBoost uses float32 input and strict <, unlike sklearn's <=.
                left = values[active, tree['feature'][nodes]] < tree['threshold'][nodes]
                indices[active] = np.where(left, tree['left'][nodes], tree['right'][nodes])
                active = active[tree['left'][indices[active]] >= 0]
            scores += tree['value'][indices]
        return scores

    def scores(self, rows):
        if any(tuple(row) != self.names for row in rows):
            raise ValueError('Ranking feature order/schema differs from trained artifact')
        return self.predict_matrix([[row[name] for name in self.names] for row in rows]).tolist()

    def choose_scores(self, rows, baseline, scores):
        if len(scores) != len(rows) or not np.isfinite(scores).all():
            raise ValueError('Invalid candidate ranking scores')
        allowed = admissible([{'features': row} for row in rows], baseline)
        chosen = min(allowed, key=lambda i: (-scores[i], i != baseline, i))
        if scores[chosen] <= scores[baseline] or scores[chosen] - scores[baseline] < self.minimum_margin:
            chosen = baseline
        return chosen, list(map(float, scores))

    def choose(self, rows, baseline):
        return self.choose_scores(rows, baseline, self.scores(rows))
