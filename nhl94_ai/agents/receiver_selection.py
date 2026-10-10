"""Rerank admitted receivers only after Classic has committed to a pass.

The legacy profile observes without changing inputs. The value profile uses the
existing, uncalibrated possession utility within a single action category. The
all-safe profile also admits ordinary receivers without legacy progress/benefit
thresholds. Native recipient, contact, offside, catch and race gates stay strict.
"""
import math

from nhl94_ai.agents.motion import velocity
from nhl94_ai.agents.passing import shot_value
from nhl94_ai.agents.possession_value import PossessionValuePlanner, VALUE_MODEL


PROFILES = ('legacy', 'value', 'all-safe')
INCOMPATIBLE = ('possession_value', 'possession_ablation', 'offense_lookahead',
                'uncertain_carry', 'chance_creation', 'cross_crease', 'deke', 'classic_refinements')


class ReceiverSelector(PossessionValuePlanner):
    def __init__(self, profile):
        if profile not in PROFILES:
            raise ValueError('Unknown receiver selection profile: ' + str(profile))
        super().__init__()
        self.profile = profile
        self.expand_pool = profile == 'all-safe'
        self.offense = None
        # Evaluation forks may force one admitted option; never expands the pool.
        self.forced_slot = None

    def _passes(self, state, purpose):
        return self.offense.passes(state, purpose, continuations=False)

    def options(self, state, purpose):
        choices, _ = self._passes(state, 'position' if self.expand_pool and purpose != 'one-timer' else purpose)
        if purpose != 'one-timer' and not self.expand_pool:
            player = state.team1.get_player_by_scnum(state.engine.puck_owner)
            sign = 1 if state.team2.net.y > state.team1.net.y else -1
            current = shot_value(state, player)
            choices = [option for option in choices
                       if self.offense._immediate_gain(option, player, sign, purpose, current)]
        return choices

    def choose_receiver(self, model, state, baseline, purpose):
        if not self.available(state):
            self.diagnostics = {'status': 'missing-feedback-use-legacy'}
            return baseline
        self.offense = model.offense
        self.interval, self.one_timers = model.scheduler.interval, model.offense.one_timers
        self.one_timer_at, self.pass_timing = model.one_timer.retry_at_frame, model.offense.pass_timing
        frame = model.scheduler.frames
        choices = self.options(state, purpose)
        if baseline not in choices:
            raise ValueError('The original pass must remain in the admitted receiver pool')
        rows = []
        for option in choices:
            if purpose == 'one-timer':
                candidate = self._one_timer(state, option)
                score = candidate.score if candidate is not None else None
            else:
                score = self.score_pass(state, option, frame)
            rows.append({
                'slot': option.slot, 'point': option.point, 'direction': option.direction,
                'legacy_value': option.value, 'legacy_shot_value': option.shot_value,
                'forward_gain': option.forward_gain, 'bypassed': option.bypassed,
                'margin': option.margin, 'robustness': option.robustness,
                'receiver_velocity': velocity(state.team1.players[option.index]),
                'status': 'scored' if score is not None else 'unforecastable',
                **(score.snapshot() if score is not None else {}),
            })
        complete = all(row['status'] == 'scored' and math.isfinite(row['value']) for row in rows)
        valued_slot = max(rows, key=lambda row: (row['value'], row['slot'] == baseline.slot))['slot'] if complete else baseline.slot
        slot = valued_slot if self.profile != 'legacy' else baseline.slot
        if self.forced_slot is not None:
            if self.forced_slot not in {option.slot for option in choices}:
                raise ValueError('Replay can only select an admitted receiver')
            slot = self.forced_slot
        selected = next(option for option in choices if option.slot == slot)
        self.metrics['passes'] += 1
        self.metrics['multi-receiver'] += len(choices) > 1
        self.metrics['overrides'] += slot != baseline.slot
        self.metrics[purpose + ':overrides'] += slot != baseline.slot
        self.metrics['unforecastable-use-legacy'] += not complete
        self.diagnostics = {
            'profile': self.profile, 'pool': 'all-safe' if self.expand_pool else 'legacy',
            'value_model': VALUE_MODEL, 'frame': frame, 'purpose': purpose,
            'baseline_slot': baseline.slot, 'value_slot': valued_slot, 'selected_slot': slot,
            'candidates': rows, 'status': 'ranked' if complete else 'unforecastable-use-legacy',
        }
        return selected
