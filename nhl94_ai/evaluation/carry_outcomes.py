"""Read-only native attack outcomes; loose pucks and shot rebounds are not turnovers."""
from collections import Counter
import math

from nhl94_ai.agents.defense import controlled_slot
from nhl94_ai.agents.motion import velocity
from nhl94_ai.game.constants import GameConsts


POSSESSION_CONFIRM_FRAMES = 4
SHOT_SETTLE_FRAMES = 24
NORMAL_SHOT_ANIMATIONS = (0x7FC, 0x92E)


class CarryOutcomes:
    def __init__(self, state, goals, touch):
        if state.engine.clock_stopped is None:
            raise ValueError('Replay requires native clock/stoppage feedback.')
        self.goals_before = goals
        self.shots_before = self.last_shots = state.team1.stats.shots
        self.attempts_before = self.last_attempts = state.team1.one_timer_attempts
        if self.last_attempts is None:
            raise ValueError('Replay requires native one-timer counters.')
        self.possession = Counter()
        self.shot_events = []
        self.pending = None
        self.last_owner = state.engine.puck_owner
        self.last_touch = touch
        self.last_motion = velocity(state.puck)
        self.last_animation = self._animation(state, self.last_owner)
        self.first_opponent_possession = None
        self.owner_run = self.stopped_run = 0
        self.ordinary_turnovers = 0
        self.end_reason = None
        self.elapsed = 0
        self.option_frames = self.option_windows = 0
        self.had_option = False
        self.best_option_value = 0.0
        self.controlled_attack_frames = self.team_attack_frames = 0
        self.max_controlled_depth = self.max_team_depth = None

    @staticmethod
    def _animation(state, slot):
        player = state.team1.get_player_by_scnum(slot)
        return (player.live_anim, player.live_anim_frame) if player is not None else (None, None)

    @staticmethod
    def possession_kind(state):
        owner = state.engine.puck_owner
        if owner < 0:
            return 'loose'
        if owner == state.team2.goalie_scnum():
            return 'opponent-goalie'
        if state.team2.owns_scnum(owner):
            return 'opponent-skater'
        if owner == state.team1.goalie_scnum():
            return 'friendly-goalie'
        if state.team1.owns_scnum(owner):
            return 'controlled-skater' if owner == controlled_slot(state.team1) else 'friendly-skater'
        raise ValueError(f'Unexpected puck owner in full-team replay: {owner}')

    def record_options(self, options):
        self.option_frames += bool(options)
        self.option_windows += bool(options) and not self.had_option
        self.had_option = bool(options)
        self.best_option_value = max([self.best_option_value, *(item['value'] for item in options)])

    def _release(self, frame, state, evidence, *, release_known=True):
        if self.pending is not None:
            self.pending.update(outcome='unresolved-before-next-shot', resolution_frame=frame)
        event = {'observed_frame': frame, 'release_frame': frame if release_known else None,
                 'shooter': state.engine.shot_player, 'evidence': evidence,
                 'recorded_shot': False, 'outcome': None, 'resolution_frame': None}
        self.shot_events.append(event)
        self.pending = event

    def _resolve(self, frame, outcome):
        self.pending.update(outcome=outcome, resolution_frame=frame)
        self.pending = None
        self.end_reason = 'shot-resolved'

    def observe(self, frame, state, goals, touch):
        owner = state.engine.puck_owner
        kind = self.possession_kind(state)
        self.elapsed = frame
        self.possession[kind] += 1
        if kind in ('controlled-skater', 'friendly-skater'):
            carrier = state.team1.get_player_by_scnum(owner)
            depth = carrier.y * (1 if state.team2.net.y > state.team1.net.y else -1)
            self.team_attack_frames += depth >= GameConsts.ATACKZONE_POS_Y
            self.max_team_depth = depth if self.max_team_depth is None else max(self.max_team_depth, depth)
            if kind == 'controlled-skater':
                self.controlled_attack_frames += depth >= GameConsts.ATACKZONE_POS_Y
                self.max_controlled_depth = depth if self.max_controlled_depth is None else max(
                    self.max_controlled_depth, depth)
        self.owner_run = self.owner_run + 1 if owner >= 0 and owner == self.last_owner else 1
        self.stopped_run = self.stopped_run + 1 if state.engine.clock_stopped else 0
        if state.team2.owns_scnum(owner) and self.first_opponent_possession is None:
            self.first_opponent_possession = frame
        shots, attempts = state.team1.stats.shots, state.team1.one_timer_attempts
        if shots < self.last_shots or attempts is None or attempts < self.last_attempts:
            raise ValueError('Native shot counters decreased or disappeared within a replay.')
        animation, index = self.last_animation
        current_animation, current_index = self._animation(state, self.last_owner)
        normal = (animation in NORMAL_SHOT_ANIMATIONS
                  and max(index or 0, current_index or 0) >= 0x1C
                  and current_animation in NORMAL_SHOT_ANIMATIONS
                  and state.team1.owns_scnum(self.last_owner)
                  and state.engine.shot_player == self.last_owner and owner < 0
                  and math.dist(velocity(state.puck), self.last_motion) > 0.25)
        if attempts > self.last_attempts and state.team1.owns_scnum(state.engine.shot_player):
            self._release(frame, state, 'native-one-timer-counter-and-shooter')
        elif normal:
            self._release(frame, state, 'native-shot-animation-release-and-puck-impulse')
        if shots > self.last_shots:
            if self.pending is None:
                self._release(frame, state, 'recorded-shot-counter; release-time-unknown', release_known=False)
            self.pending['recorded_shot'] = True
        if goals > self.goals_before:
            if self.pending is None:
                self._release(frame, state, 'goal-counter; release-time-unknown', release_known=False)
            self._resolve(frame, 'goal')
            self.end_reason = 'goal'
        elif self.pending is not None:
            event = self.pending
            if touch != self.last_touch and touch != event['shooter'] and touch >= 0:
                if touch == state.team2.goalie_scnum():
                    event.setdefault('goalie_touch_frame', frame)
                else:
                    event.setdefault('skater_touch_frame', frame)
                    event.setdefault('touch_slot', touch)
            sign = 1 if state.team2.net.y > state.team1.net.y else -1
            if state.puck.y * sign > state.team2.net.y * sign:
                event.setdefault('passed_net_plane_frame', frame)
            if owner >= 0 and self.owner_run >= POSSESSION_CONFIRM_FRAMES:
                outcome = ('save-held' if kind == 'opponent-goalie' else
                           'save-rebound-recovered' if 'goalie_touch_frame' in event else
                           'block-or-deflection-recovered' if 'skater_touch_frame' in event else
                           'shot-caught-or-recovered')
                event['recovery_owner'] = owner
                self._resolve(frame, outcome)
            elif self.stopped_run >= POSSESSION_CONFIRM_FRAMES:
                self._resolve(frame, 'save-stoppage' if 'goalie_touch_frame' in event else
                              'stoppage-with-unresolved-shot')
            elif ('passed_net_plane_frame' in event
                  and frame - event['passed_net_plane_frame'] >= SHOT_SETTLE_FRAMES):
                self._resolve(frame, 'passed-net-plane-without-goal')
        elif self.stopped_run >= POSSESSION_CONFIRM_FRAMES:
            self.end_reason = 'stoppage'
        elif kind == 'opponent-skater' and self.owner_run >= POSSESSION_CONFIRM_FRAMES:
            self.ordinary_turnovers += self.end_reason is None
            self.end_reason = 'ordinary-turnover'
        elif kind == 'opponent-goalie' and self.owner_run >= POSSESSION_CONFIRM_FRAMES:
            self.end_reason = 'opponent-goalie-possession-without-confirmed-shot'
        self.last_owner, self.last_touch = owner, touch
        self.last_motion = velocity(state.puck)
        self.last_animation = self._animation(state, owner)
        self.last_shots, self.last_attempts = shots, attempts
        return self.end_reason is not None

    def summary(self):
        if self.elapsed < 1:
            raise ValueError('Replay outcome summary requires an observed emulator frame.')
        if self.pending is not None:
            self.pending.update(outcome='unresolved-at-timeout', resolution_frame=None)
        return {
            'elapsed_frames': self.elapsed, 'end_reason': self.end_reason or 'timeout',
            'possession_frames': dict(self.possession),
            'possession_fractions': {kind: frames / self.elapsed for kind, frames in self.possession.items()},
            'ordinary_turnovers': self.ordinary_turnovers,
            'first_opponent_possession_frame': self.first_opponent_possession,
            'shots': self.last_shots - self.shots_before,
            'one_timer_attempts': self.last_attempts - self.attempts_before,
            'frames_with_one_timer_option': self.option_frames,
            'one_timer_option_windows': self.option_windows,
            'best_one_timer_value': self.best_option_value,
            'controlled_offensive_zone_frames': self.controlled_attack_frames,
            'team_offensive_zone_frames': self.team_attack_frames,
            'maximum_controlled_attack_depth': self.max_controlled_depth,
            'maximum_team_attack_depth': self.max_team_depth,
            'observed_shot_releases': sum(event['release_frame'] is not None for event in self.shot_events),
            'shot_events': self.shot_events,
        }
