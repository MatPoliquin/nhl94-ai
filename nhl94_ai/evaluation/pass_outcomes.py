"""Observe deliberate V4 setup passes without changing controller decisions."""
from collections import Counter


class PassOutcomes:
    """Frame-level outcomes; direction mistakes are separate from final outcomes.

    A fresh team pass-attempt counter validates pass_target, which otherwise may
    refer to an old pass. Team one-timer counters confirm execution, not C presses.
    """
    def __init__(self, side):
        self.side = side
        self.pending = None
        self.events = []

    def start(self, frame, info, passer, intended):
        if self.pending is not None:
            self.finish(frame, info, 'superseded')
        self.pending = {
            'frame': frame, 'passer': passer, 'intended': intended, 'actual': None,
            'launched': False,
            'passes_before': info[f'p{self.side}_pass_attempts'],
            'one_timers_before': info[f'bench_one_timers{self.side}'],
        }

    def finish(self, frame, info, outcome):
        if self.pending is not None:
            event = self.pending
            event.update(outcome=outcome, elapsed_frames=frame - event['frame'],
                         final_owner=info['puck_owner'])
            self.events.append(event)
            self.pending = None

    def observe(self, frame, info):
        event = self.pending
        if event is None:
            return
        owner = info['puck_owner']
        if info[f'p{self.side}_pass_attempts'] > event['passes_before']:
            if event['actual'] is None:
                event['actual'] = info.get('pass_target')
            event['launched'] = True
        elif owner < 0 and info.get('last_puck_player') == event['passer']:
            event['launched'] = True
        if (info[f'bench_one_timers{self.side}'] > event['one_timers_before']
                and info['shot_player'] in (event['actual'], event['intended'])):
            self.finish(frame, info, 'one_timer_shot')
        elif owner >= 0 and not (self.side - 1) * 6 <= owner < self.side * 6:
            self.finish(frame, info, 'intercepted' if event['launched'] else 'lost_before_release')
        elif owner >= 0 and owner != event['passer']:
            self.finish(frame, info, 'received_without_shot')
        elif event['launched'] and owner == event['passer']:
            self.finish(frame, info, 'returned_to_passer')
        elif frame - event['frame'] >= 120:
            self.finish(frame, info, 'flight_timeout' if event['launched'] else 'not_released')

    def summary(self):
        return {
            'outcomes': dict(Counter(event['outcome'] for event in self.events)),
            'wrong_recipient': sum(event['actual'] is not None and event['actual'] != event['intended']
                                   for event in self.events),
        }
