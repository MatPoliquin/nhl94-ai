"""Observational decision-chain counters and bounded native pass flight evidence."""
from collections import Counter, deque
from copy import deepcopy
import gzip
import json

from nhl94_ai.game.motion import velocity


class PassEvidence:
    """Retain 24 pre-release and at most 120 post-release observations per pass.

    Endings can occur later; each record includes that ending and an explicit
    truncated-window flag. Native attempts and tracked events are separate.
    """
    def __init__(self, path=None):
        self.stream = gzip.open(path, 'wt') if path else None
        self.ring = deque(maxlen=24)
        self.flights = {}
        self.finished = 0
        self.counts = Counter()
        self.targets = {}
        self.target_since = {}
        self.lifetimes = []
        self.guard = None

    def observe(self, frame, state, diagnostics, action, receptions):
        rows = diagnostics['controllers']
        for row in rows:
            i = row['controller']
            support = row.get('support', {})
            source = row.get('input_source', 'unclassified')
            target = (row['slot'], tuple(row.get('target', ())), support.get('target_until'))
            if target != self.targets.get(i):
                if i in self.target_since:
                    self.lifetimes.append(frame - self.target_since[i])
                self.target_since[i] = frame
                self.targets[i] = target
                self.counts['target_changes'] += 1
                self.counts['support_candidates_at_change'] += len(support.get('candidates', []))
            self.counts['executed:' + source] += 1
            self.counts['selected_role:' + row['role']] += 1
            if row.get('decision') == 'cpu-carrier-support':
                self.counts['cpu_support_selected_frames'] += 1
                self.counts['cpu_support_executed_frames'] += source == 'classic'
            requested = row.get('requested_buttons', row['buttons'])
            if requested[0] and not action[(i - 1)*12]:
                self.counts['veto:' + row.get('b_arbitration', 'baseline')] += 1
        roles = diagnostics.get('roles', {})
        guard = roles.get('guard')
        if guard != self.guard:
            self.counts['guard_reassignments'] += 1
            self.guard = guard
        for event in diagnostics.get('events', []):
            self.counts['commitment:' + event['phase']] += 1
            if event.get('reason'):
                self.counts['invalidation:' + event['reason']] += 1
        if self.stream is None:
            return
        pending = receptions.pending
        slots = {row['slot'] for row in rows}
        slots.add(state.engine.pass_target)
        if pending:
            slots.add(pending['receiver'])
        actors = {}
        for slot in slots:
            player = state.team1.get_player_by_scnum(slot) if slot is not None else None
            if player is not None:
                actors[slot] = {'position': [player.precise_x, player.precise_y],
                                'velocity': list(velocity(player)), 'animation': [player.live_anim, player.live_anim_frame]}
        sample = {'frame': frame, 'period': state.period, 'owner': state.engine.puck_owner,
                  'native_target': state.engine.pass_target, 'actors': actors,
                  'puck': [state.puck.x, state.puck.y], 'pads': action.tolist(),
                  'controllers': [{'slot': r['slot'], 'role': r['role'], 'source': r.get('input_source'),
                                   'target': r.get('target'), 'requested': r.get('requested_buttons'),
                                   'b_intent': r.get('b_intent'), 'veto': r.get('b_arbitration'),
                                   'support': r.get('support'), 'pass': r.get('pass')} for r in rows],
                  'commitment': diagnostics.get('pending_pass'), 'events': diagnostics.get('events', [])}
        # Copy mutable policy diagnostics, never modify them or feed them back.
        sample = deepcopy(sample)
        if pending and pending['release_frame'] not in self.flights:
            self.flights[pending['release_frame']] = list(self.ring)
        for release, samples in self.flights.items():
            if frame - release <= 120:
                samples.append(sample)
        self.ring.append(sample)
        self.flush(receptions)

    def flush(self, receptions):
        if self.stream:
            for event in receptions.events[self.finished:]:
                samples = self.flights.pop(event['release_frame'], [])
                self.stream.write(json.dumps({'event': event, 'samples': samples,
                                              'window_truncated': event['end_frame'] - event['release_frame'] > 120}) + '\n')
        self.finished = len(receptions.events)

    def close(self, receptions):
        self.flush(receptions)
        if self.stream:
            self.stream.close()
        return {'counts': dict(self.counts), 'completed_target_lifetimes': self.lifetimes,
                'window': {'before_release': 24, 'after_release': 120},
                'event_count': self.finished}
