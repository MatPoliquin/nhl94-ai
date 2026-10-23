"""Observed outcomes around coordinated traps; not causal tackle attribution."""
from collections import Counter
import math

from nhl94_ai.game.actors import owns_puck


class PressureMetrics:
    def __init__(self):
        self.counts = Counter()
        self.reasons = Counter()
        self.active = None
        self.follow = []
        self.shot_until = -1
        self.events = []
        self.last_shots = None

    def observe(self, frame, state, diagnostics):
        pressure = diagnostics.get('pressure', {})
        rows = diagnostics.get('controllers', [])
        trapping = pressure.get('phase') in ('contain', 'close')
        owner = state.engine.puck_owner
        shots = state.team2.stats.shots
        if self.last_shots is not None and shots > self.last_shots and (self.active or frame <= self.shot_until):
            self.counts['opponent_shots_during_or_after_trap'] += shots - self.last_shots
        self.last_shots = shots
        if trapping:
            self.counts['trap_frames'] += 1
            roles = tuple((row['slot'], row['role']) for row in rows)
            if self.active is None:
                self.counts['attempts'] += 1
                self.active = {'frame': frame, 'carrier': owner, 'roles': roles}
            elif roles != self.active['roles']:
                self.counts['role_changes'] += 1
                self.active['roles'] = roles
            players = [state.team1.get_player_by_scnum(row['slot']) for row in rows]
            self.counts['close_spacing_frames'] += any(
                math.dist((a.x, a.y), (b.x, b.y)) < 18
                for i, a in enumerate(players) if a is not None for b in players[i + 1:] if b is not None)
        else:
            if pressure:
                self.reasons[pressure.get('reason', 'unknown')] += 1
            if self.active is not None:
                event = {**self.active, 'ended': frame, 'reason': pressure.get('reason', 'transition')}
                self.events.append(event)
                self.follow.append(event)
                self.shot_until = frame + 60
                self.active = None
        for event in self.follow[:]:
            outcome = None
            if owns_puck(state.team1, owner):
                self.counts['team_recoveries_within_60_frames'] += 1
                outcome = 'team-recovery'
            elif owns_puck(state.team2, owner) and owner != event['carrier']:
                self.counts['opponent_receptions_within_60_frames'] += 1
                outcome = 'opponent-reception'
            elif frame >= event['ended'] + 60:
                self.counts['no_recovery_within_60_frames'] += 1
                outcome = 'no-recovery'
            if outcome is not None:
                event.update(outcome=outcome, outcome_frame=frame)
                self.follow.remove(event)

    def summary(self):
        return {'counts': dict(self.counts), 'fallback_frames': dict(self.reasons), 'events': self.events}
