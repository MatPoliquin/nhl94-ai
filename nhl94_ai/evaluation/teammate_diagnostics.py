"""Read-only live denominators and native ownership/input evidence.

Danger entries count opponent skater possession entering |x| < 70, within 90
rink units of our net. They are a geometry proxy, not expected goals.
"""
from collections import Counter

import numpy as np

from nhl94_ai.evaluation.teammate_opportunities import opportunity_diagnostics
from nhl94_ai.game.constants import GameConsts as Buttons


class TeammateDiagnostics:
    def __init__(self):
        self.counts = Counter()
        self.roles = Counter()
        self.requests = Counter()
        self.request_frames = Counter()
        self.request_edges = Counter()
        self.previous_requested = None
        self.edges = Counter()
        self.opportunities = Counter()
        self.previous = None
        self.danger = False
        self.owner = None
        self.loose = 0
        self.last_side = None
        self.goalie_hold = 0
        self.goalie_events = []
        self.outlet = None
        self.ticks = {}
        self.faceoffs = None
        self.last_faceoffs = None

    def observe(self, frame, state, diagnostics, action, *, highlight=False):
        rows = diagnostics['controllers']
        pads = np.asarray(action).reshape(-1, 12)[:len(rows)]
        previous = self.previous if self.previous is not None else np.zeros_like(pads)
        previous_requested = self.previous_requested if self.previous_requested is not None else np.zeros_like(pads)
        requested_pads = np.asarray([row.get('requested_buttons', pads[i]) for i, row in enumerate(rows)])
        live = not (highlight or state.engine.clock_stopped or state.time <= 0 or state.engine.sflags & 0x100)
        owner, team = state.engine.puck_owner, state.team1
        slots = [row['slot'] for row in rows]
        transient = bool(diagnostics.get('transient'))
        known = not transient and all(slot >= 0 for slot in slots)
        cpu_carrier = known and owner not in slots and team.owns_scnum(owner) and owner != team.goalie_scnum()
        if not highlight:
            counters = [state.team1.stats.faceoffwon, state.team2.stats.faceoffwon]
            if all(value is not None for value in counters):
                if self.faceoffs is None:
                    self.faceoffs = counters
                self.last_faceoffs = counters
        if live:
            self.counts['live_frames'] += 1
            self.counts['live_transient_frames'] += transient
            self.counts['invalid_assignment_frames'] += any(slot >= 0 and not team.owns_scnum(slot) for slot in slots)
            self.counts['cpu_carrier_frames'] += cpu_carrier
            self.counts['skater_possession_frames'] += team.owns_scnum(owner) and owner != team.goalie_scnum()
            self.counts['goalie_possession_frames'] += owner == team.goalie_scnum()
            phase = ('goalie' if owner == team.goalie_scnum() else 'attack' if team.owns_scnum(owner)
                     else 'defense' if state.team2.owns_scnum(owner) else 'loose')
            for row in rows:
                self.roles[f"{phase}:p{row['controller']}:{row['role']}"] += 1
            opponent = state.team2.get_player_by_scnum(owner)
            danger = (opponent is not None and opponent is not state.team2.goalie
                      and abs(opponent.x) < 70 and abs(opponent.y - team.net.y) < 90)
            self.counts['danger_entries_against'] += danger and not self.danger
            self.counts['danger_frames_against'] += danger
            self.danger = danger
            side = 1 if team.owns_scnum(owner) else 2 if state.team2.owns_scnum(owner) else None
            if side is not None:
                self.counts['defensive_recoveries'] += side == 1 and self.last_side == 2 and self.loose <= 90
                self.last_side, self.loose = side, 0
            else:
                self.loose += 1
            if frame % 32 == 0:
                sample = opportunity_diagnostics(state, {**diagnostics, 'frame': frame}, previous)
                self.opportunities['sampled_live_frames'] += 1
                for candidate in (sample or {}).get('candidates', []):
                    key = f"{candidate['purpose']}:{'controlled' if candidate['controlled_partner'] else 'cpu'}"
                    self.opportunities[key + ':candidates'] += 1
                    self.opportunities[key + ':feasible'] += bool(candidate['feasible'])
        else:
            self.danger, self.last_side, self.loose = False, None, 0
        # Keep request ticks separate from applied rising edges; held labels are not requests.
        for index, row in enumerate(rows):
            tick = row.get('decision_tick')
            intent = row.get('b_intent', 'unclassified')
            if requested_pads[index, Buttons.INPUT_B]:
                self.request_frames[intent] += 1
                self.request_edges[intent] += not previous_requested[index, Buttons.INPUT_B]
                if row.get('proposal_frame') is not None:
                    self.counts['b_proposal_frames'] += 1
            if tick is not None and tick != self.ticks.get(index):
                self.counts['decision_ticks'] += 1
                if row.get('requested_buttons', pads[index])[Buttons.INPUT_B]:
                    self.requests[intent] += 1
                    if live and cpu_carrier:
                        self.counts['cpu_carrier_b_request_ticks'] += 1
                if live and cpu_carrier and row.get('decision') in ('recover', 'switch-request'):
                    self.counts['cpu_carrier_recovery_ticks'] += 1
            self.ticks[index] = tick
            if pads[index, Buttons.INPUT_B] and not previous[index, Buttons.INPUT_B]:
                self.edges['b:' + intent] += 1
            if pads[index, Buttons.INPUT_C] and not previous[index, Buttons.INPUT_C]:
                self.edges['c:' + row.get('input_source', row['role'])] += 1
        for event in diagnostics.get('events', []):
            if event.get('phase') == 'one-timer-cue':
                self.counts['one_timer_cues'] += 1
                self.counts['one_timer_cues_' + event.get('basis', 'live')] += 1
            elif event.get('phase') == 'one-timer-armed':
                self.counts['one_timer_armed'] += 1
            elif event.get('phase') == 'ended' and event.get('cue_whipt'):
                self.counts['one_timer_cues_whipt'] += 1
        if not highlight and not state.engine.clock_stopped and state.time > 0:
            if owner == team.goalie_scnum():
                self.goalie_hold += 1
                self.counts['max_goalie_hold_frames'] = max(self.counts['max_goalie_hold_frames'], self.goalie_hold)
            elif self.owner == team.goalie_scnum():
                self.outlet = {'release_frame': frame, 'hold_frames': self.goalie_hold,
                               'evidence': 'native-goalie-owner-exit', 'receiver': None}
                self.goalie_events.append(self.outlet)
                self.goalie_hold = 0
            if self.outlet is not None:
                if owner >= 0:
                    outcome = ('returned-to-goalie' if owner == team.goalie_scnum() else
                               'received' if team.owns_scnum(owner) else 'intercepted')
                    self.outlet.update(receiver=owner, reception_frame=frame, outcome=outcome)
                    self.outlet = None
                elif frame - self.outlet['release_frame'] >= 90:
                    self.outlet['outcome'] = 'unresolved'
                    self.outlet = None
        else:
            if self.outlet is not None:
                self.outlet['outcome'] = 'stoppage'
            self.outlet, self.goalie_hold = None, 0
        self.owner = None if highlight or state.engine.clock_stopped else owner
        self.previous = pads.copy()
        self.previous_requested = requested_pads.copy()

    def summary(self):
        live = self.counts['live_frames']
        return {'counts': dict(self.counts), 'live_role_frames': dict(self.roles),
                'one_timer_cues': {'predicted': self.counts['one_timer_cues_predicted'],
                                   'live': self.counts['one_timer_cues_live']},
                'one_timer_cues_whipt': self.counts['one_timer_cues_whipt'],
                'requested_b_decisions': dict(self.requests), 'applied_edges': dict(self.edges),
                'requested_b_frames': dict(self.request_frames), 'requested_b_edges': dict(self.request_edges),
                'opportunities': dict(self.opportunities), 'opportunity_sample_interval': 32,
                'native_faceoffs_won': ([end - start for start, end in zip(self.faceoffs, self.last_faceoffs)]
                                        if self.faceoffs is not None else None),
                'goalie_releases': self.goalie_events,
                'live_skater_possession_share': self.counts['skater_possession_frames'] / live if live else None,
                'live_goalie_possession_share': self.counts['goalie_possession_frames'] / live if live else None,
                'live_transient_rate': self.counts['live_transient_frames'] / live if live else None}
