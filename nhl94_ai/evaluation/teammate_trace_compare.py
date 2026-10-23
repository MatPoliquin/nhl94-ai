"""Validate complete recordings; distinguish first action and diagnostic changes.

After the first intended action change, live observations are allowed to diverge.
This is policy trace review, not input-only replay or an exact policy parity claim.
"""
import gzip
import hashlib
from itertools import zip_longest
import json
from pathlib import Path

import numpy as np


def trace_rows(path, result):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        first = stream.readline()
        if not first:
            raise ValueError('Empty trace')
        header = json.loads(first)
        for key in ('seed', 'seconds', 'frame_skip', 'rom_sha256', 'initial_ram_sha256'):
            if header[key] != result[key]:
                raise ValueError('Trace header mismatch: ' + key)
        digest, count = hashlib.sha256(), 0
        for line in stream:
            row = json.loads(line)
            count += 1
            buttons = np.asarray(row['buttons'])
            if row['frame'] != count or buttons.shape != (result['controllers'] * 12,) or not np.isin(buttons, (0, 1)).all():
                raise ValueError('Malformed trace frame')
            digest.update(buttons.astype(np.int8).tobytes())
            yield row
        if count != result['frames'] or digest.hexdigest() != result['actions_sha256']:
            raise ValueError('Truncated or corrupt trace')


def first_divergences(before_path, before_result, after_path, after_result):
    action, diagnostic = None, None
    for old, new in zip_longest(trace_rows(before_path, before_result), trace_rows(after_path, after_result)):
        if diagnostic is None and (old or {}).get('diagnostics') != (new or {}).get('diagnostics'):
            diagnostic = (old or new)['frame']
        if action is None and (old or {}).get('buttons') != (new or {}).get('buttons'):
            action = {'frame': (old or new)['frame'], 'before': old, 'after': new,
                      'same_observed_owner_clock': old is not None and new is not None
                      and all(old.get(key) == new.get(key) for key in ('owner', 'clock', 'period'))}
    return {'first_action_difference': action, 'first_diagnostic_difference': diagnostic,
            'exact_actions': action is None}


def compare_fixture_matrix(reference, candidate):
    before = json.loads(reference.read_text())
    after = json.loads(candidate.read_text())
    results = {}
    for name, old in before['teammates'].items():
        new = after['teammates'][name]
        if old['settings'] != new['settings']:
            raise ValueError('Fixture settings differ: ' + name)
        results[name] = first_divergences(reference.parent/(name+'.jsonl'), old['result'],
                                         candidate.parent/(name+'.jsonl'), new['result'])
    return results
