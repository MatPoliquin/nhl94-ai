"""Policy-independent native pass, reception and subsequent-shot attribution."""
from collections import Counter

from nhl94_ai.game.motion import velocity
from nhl94_ai.evaluation.carry_outcomes import normal_shot_released

_UNOBSERVED = object()


class NativeShotTracker:
    """Fresh native release evidence, independent of delayed recorded shots."""
    def __init__(self):
        self.previous = None
        self.events = []

    def observe(self, frame, state, *, highlight=False):
        if highlight:
            self.previous = None
            return None
        previous = self.previous
        owner, shooter = state.engine.puck_owner, state.engine.shot_player
        player = state.team1.get_player_by_scnum(owner)
        animation = (player.live_anim, player.live_anim_frame) if player is not None else (None, None)
        attempts = state.team1.one_timer_attempts
        event = None
        if previous is not None:
            one_timer = attempts is not None and previous['attempts'] is not None and attempts > previous['attempts']
            normal = normal_shot_released(state, previous['owner'], previous['motion'], previous['animation'])
            if (one_timer or normal) and state.team1.owns_scnum(shooter) and not state.engine.clock_stopped:
                event = {'frame': frame, 'shooter': shooter, 'one_timer': one_timer,
                         'evidence': 'native-one-timer-counter' if one_timer else 'native-animation-release-and-impulse'}
                self.events.append(event)
        self.previous = {'owner': owner, 'attempts': attempts, 'animation': animation, 'motion': velocity(state.puck)}
        return event


class ReceptionOutcomes:
    def __init__(self):
        self.pending = None
        self.events = []
        self.previous = None
        self.shots = NativeShotTracker()

    def finish(self, frame, reason):
        if self.pending is not None:
            self.events.append({**self.pending, 'ending': reason, 'end_frame': frame})
            self.pending = None

    def observe(self, frame, state, selected, *, highlight=False, release=_UNOBSERVED):
        if release is _UNOBSERVED:
            release = self.shots.observe(frame, state, highlight=highlight)
        if highlight:
            self.finish(frame, 'highlight')
            self.previous = None
            return
        owner = state.engine.puck_owner
        counters = (state.team1.pass_attempts, state.team1.stats.shots,
                    state.team1.one_timer_attempts, state.team1.stats.score)
        previous = self.previous
        event = self.pending
        if event is not None:
            if ((event['receiver'] is None or event['receiver'] < 0) and release is not None
                    and release['shooter'] != event['passer']):
                event['receiver'] = release['shooter']
                event['controlled_pair'] = (event['receiver'] in event['selected_at_release']
                                            and event['passer'] in event['selected_at_release'])
            shot = (release is not None and release['shooter'] == event['receiver']
                    or counters[1] > event['shots_before']
                    or counters[2] is not None and event['attempts_before'] is not None
                    and counters[2] > event['attempts_before'])
            if shot and state.engine.shot_player == event['receiver'] and event['shot_frame'] is None:
                event['shot_frame'] = frame
                event['expiry'] = frame + 150
                event['one_timer'] = counters[2] is not None and counters[2] > (event['attempts_before'] or 0)
                event['shot_evidence'] = release['evidence'] if release is not None else 'recorded-counter-and-shooter'
            if event['shot_frame'] is not None:
                if counters[3] > event['goals_before']:
                    self.finish(frame, 'goal')
                elif owner >= 0 or state.engine.clock_stopped:
                    self.finish(frame, 'shot-recovery' if owner >= 0 else 'shot-stoppage')
            elif owner == event['receiver']:
                event['loose_since'] = None
                if event['received_frame'] is None:
                    event['received_frame'], event['expiry'] = frame, frame + 600
            elif state.team2.owns_scnum(owner):
                self.finish(frame, 'turnover-after-reception' if event['received_frame'] is not None else 'intercepted')
            elif owner >= 0 and owner != event['passer']:
                if event['receiver'] is None or event['receiver'] < 0:
                    # Fresh native ownership identifies an actual reception;
                    # retain the cleared target word instead of inventing intent.
                    event['receiver'] = owner
                    event['received_frame'], event['expiry'] = frame, frame + 600
                    event['controlled_pair'] = owner in event['selected_at_release'] and event['passer'] in event['selected_at_release']
                else:
                    self.finish(frame, 'other-receiver')
            elif state.engine.clock_stopped:
                self.finish(frame, 'stoppage')
            elif event['received_frame'] is not None and owner < 0:
                if event['loose_since'] is None:
                    event['loose_since'] = frame
                elif frame - event['loose_since'] >= 24:
                    self.finish(frame, 'unresolved-after-reception')
            elif owner == event['passer'] and frame > event['release_frame']:
                self.finish(frame, 'returned-to-passer')
            if self.pending is not None and frame >= event['expiry']:
                self.finish(frame, 'unresolved-shot' if event['shot_frame'] is not None else 'unresolved-reception'
                            if event['received_frame'] is not None else 'unresolved-flight')
        native_pass = (previous is not None and counters[0] is not None and previous['passes'] is not None
                       and counters[0] > previous['passes'])
        if native_pass and state.team1.owns_scnum(previous['owner']) and not state.engine.clock_stopped:
            self.finish(frame, 'next-pass')
            receiver = state.engine.pass_target
            self.pending = {'passer': previous['owner'], 'receiver': receiver, 'release_frame': frame,
                            'native_target_word': receiver, 'selected_at_release': previous['selected'],
                            'controlled_pair': (previous['owner'] in previous['selected']
                                                and receiver in previous['selected'] and receiver != previous['owner']),
                            'received_frame': frame if owner == receiver else None, 'shot_frame': None,
                            'one_timer': False, 'loose_since': None,
                            'expiry': frame + (600 if owner == receiver else 90), 'shots_before': counters[1],
                            'attempts_before': counters[2], 'goals_before': counters[3]}
        self.previous = {'owner': owner, 'passes': counters[0], 'selected': list(selected)}

    def summary(self):
        rows = self.events
        pair = [event for event in rows if event['controlled_pair']]
        return {'passes': len(rows), 'receptions': sum(e['received_frame'] is not None for e in rows),
                'controlled_pair_passes': len(pair),
                'controlled_pair_receptions': sum(e['received_frame'] is not None for e in pair),
                'reception_to_shots': sum(e['received_frame'] is not None and e['shot_frame'] is not None for e in rows),
                'controlled_pair_reception_to_shots': sum(e['received_frame'] is not None and e['shot_frame'] is not None for e in pair),
                'direct_one_timers': sum(e['received_frame'] is None and e['one_timer'] for e in rows),
                'receiver_shots': sum(e['shot_frame'] is not None for e in rows),
                'receiver_goals': sum(e['ending'] == 'goal' for e in rows),
                'endings': dict(Counter(e['ending'] for e in rows))}
