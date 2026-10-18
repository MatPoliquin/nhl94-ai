"""Optional shallow-tree correction of ordinary Classic movement decisions."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from nhl94_ai.agents.retention_features import (
    INTERVAL, PADS, admissible, attack_sign, features, movement_action, movement_id, ordinary_movement,
)


SCHEMA = 'native-carry-retention-tree-v1'


class RetentionFilter:
    score_name = 'risks'

    def __init__(self, path=None, *, artifact=None):
        self.path = str(Path(path).resolve()) if path else None
        self.sha256 = hashlib.sha256(Path(path).read_bytes()).hexdigest() if path else None
        self.artifact = artifact if artifact is not None else json.loads(Path(path).read_text(encoding='utf-8'))
        data = self.artifact
        if (data.get('schema') != SCHEMA or data.get('interval') != INTERVAL
                or data.get('horizon') != 24 or not data.get('feature_names')):
            raise ValueError('Incompatible retention tree schema or decision cadence')
        self.names = tuple(data['feature_names'])
        self.tree = data['tree']
        self.minimum_risk, self.minimum_gap = data['minimum_risk'], data['minimum_gap']
        if not 0 <= self.minimum_risk <= 1 or not 0 < self.minimum_gap <= 1:
            raise ValueError('Retention risk thresholds must be within [0, 1] with a positive gap')
        if not self.tree or len(self.tree) > 255:
            raise ValueError('Retention artifact requires a shallow nonempty tree')
        for i, node in enumerate(self.tree):
            if not np.isfinite(node['risk']) or not 0 <= node['risk'] <= 1:
                raise ValueError('Invalid retention leaf risk')
            if node['left'] != -1 and not (i < node['left'] < len(self.tree) and i < node['right'] < len(self.tree)
                                           and 0 <= node['feature'] < len(self.names) and np.isfinite(node['threshold'])):
                raise ValueError('Invalid retention tree split')
        self.metrics = Counter()

    def risk(self, row):
        if tuple(row) != self.names:
            raise ValueError('Retention feature order/schema differs from the trained artifact')
        values = np.asarray([row[name] for name in self.names], dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError('Non-finite retention input')
        index = 0
        while self.tree[index]['left'] != -1:
            node = self.tree[index]
            index = node['left'] if values[node['feature']] <= node['threshold'] else node['right']
        return self.tree[index]['risk']

    def choose(self, rows, baseline):
        """Only current-state feature rows participate in action selection."""
        risks = [self.risk(row) for row in rows]
        allowed = admissible([{'features': row} for row in rows], baseline)
        chosen = min(allowed, key=lambda i: (risks[i], i != baseline, i))
        if risks[baseline] < self.minimum_risk or risks[baseline] - risks[chosen] < self.minimum_gap:
            chosen = baseline
        return chosen, risks

    def apply(self, model, state, action):
        if not ordinary_movement(model, action[0]):
            return action
        baseline = movement_id(action[0], attack_sign(state))
        rows = [features(state, command) for command in range(len(PADS))]
        chosen, risks = self.choose(rows, baseline)
        self.metrics['eligible_decisions'] += 1
        self.metrics['overrides'] += chosen != baseline
        model.offense_diagnostics['retention'] = dict(baseline=baseline, chosen=chosen,
                                                     **{self.score_name: risks})
        if chosen == baseline:
            return action
        replacement = movement_action(chosen, attack_sign(state))[None]
        model.scheduler.action = replacement.copy()
        model._last_action_preferences = replacement.astype(np.float32)
        return replacement
