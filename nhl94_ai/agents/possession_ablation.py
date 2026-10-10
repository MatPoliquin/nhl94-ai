"""Controlled outer-ranking experiments over Classic's existing proposals.

Each category still uses its legacy generator and executor. The experiment
changes only outer ranking, endpoint continuation credit, and (separately) the
two estimated pass-risk gates. It does not expand routes or pass directions.
"""
from collections import Counter
import math

from nhl94_ai.agents.carry import body_clearance, carry_path
from nhl94_ai.agents.finishing import shot_speed
from nhl94_ai.agents.offense import carry_future, normal_shot_release_frames
from nhl94_ai.agents.passing import pressure_margin
from nhl94_ai.agents.possession_value import (
    CARRY_HORIZON, VALUE_MODEL, PossessionValuePlanner, position_value, shot_quality, transition_score,
)


PROFILES = ('legacy', 'rank', 'continuations', 'risk', 'continuations-risk')


class LegacyValueRanker(PossessionValuePlanner):
    def __init__(self, profile):
        if profile not in PROFILES:
            raise ValueError('Unknown possession ablation: ' + str(profile))
        super().__init__()
        self.profile = profile
        self.continuations = profile in ('continuations', 'continuations-risk')
        self.relaxed_risk = profile in ('risk', 'continuations-risk')
        self.offense = None
        self.metrics = Counter()

    def _passes(self, state, purpose):
        # Preserve the legacy one-timer's additional worthwhile-position gate.
        return self.offense.passes(state, purpose, continuations=False)

    def _terminal(self, state, frame):
        if self.continuations:
            return super()._terminal(state, frame)
        player = state.team1.get_player_by_scnum(state.engine.puck_owner)
        return position_value(state, player), 'retain', None

    def _shot_score(self, state, finish):
        player = state.team1.get_player_by_scnum(state.engine.puck_owner)
        hold = self.interval if finish is None else finish.hold_frames
        side = (-1 if state.team2.goalie.x > 0 else 1) if finish is None else finish.side
        release = normal_shot_release_frames(player, hold)
        path = carry_path(player, None, release)
        if path is None:
            return None
        point = state.puck.x + path[-1].x - player.x, state.puck.y + path[-1].y - player.y
        terminal = shot_quality(state, player, point, side, release, shot_speed(player, hold))
        return transition_score(state, terminal, point, release,
                                pressure_margin(state.team2, point, release), continuation='shot')

    def _carry_score(self, state, target, frame):
        player = state.team1.get_player_by_scnum(state.engine.puck_owner)
        horizon = max(CARRY_HORIZON, self.interval)
        path = carry_path(player, target, horizon, decision_interval=self.interval)
        if path is None:
            return None
        future = carry_future(state, player, path[-1], frames=horizon)
        future.puck.motion_x, future.puck.motion_y = path[-1].motion_x, path[-1].motion_y
        terminal, continuation, slot = self._terminal(future, frame + horizon)
        margin = min(pressure_margin(state.team2, (path[t].x, path[t].y), t)
                     for t in (4, 8, 12, horizon))
        return transition_score(state, terminal, (path[-1].x, path[-1].y), horizon, margin,
                                uncertainty=max(0, -body_clearance(state, player, path)),
                                continuation=continuation, continuation_slot=slot)

    def choose_legacy(self, model, state, plan, shot_allowed, prefer_timer, timer_target, finish):
        self.offense = model.offense
        self.interval, self.one_timers = model.scheduler.interval, model.offense.one_timers
        self.one_timer_at, self.pass_timing = model.one_timer.retry_at_frame, model.offense.pass_timing
        self.refine_finishing = False
        frame = model.scheduler.frames
        player = state.team1.get_player_by_scnum(state.engine.puck_owner)
        fallback, _ = self.offense.fallback_carry(state, player)
        baseline = ('one-timer' if prefer_timer else 'shoot' if shot_allowed else
                    'plan' if plan is not None else 'fallback')
        proposals = [('fallback', 'carry', fallback, None, self._carry_score(state, fallback, frame))]
        if plan is not None:
            mode, target, option = plan
            # _carry(..., breakaway=True) executes the fallback route, not the nominal plan target.
            target = fallback if mode == 'carry-breakaway' else target
            if option is None and mode not in ('feint', 'one-timer-setup', 'carry-opportunity', 'create-chance',
                                                'carry-breakaway'):
                escape = model._goalie_escape(state, player, target)
                target = escape[0] if escape is not None else target
            score = (self.score_pass(state, option, frame) if option is not None else
                     self._carry_score(state, target, frame))
            proposals.append(('plan', 'pass' if option is not None else 'carry', target, option, score))
        if shot_allowed:
            side = (-1 if state.team2.goalie.x > 0 else 1) if finish is None else finish.side
            proposals.append(('shoot', 'shoot', (side * 13, state.team2.net.y), None,
                              self._shot_score(state, finish)))
        if timer_target is not None:
            option = model.one_timer.option
            candidate = self._one_timer(state, option) if option is not None else None
            proposals.append(('one-timer', 'one-timer', option.point if option else None, option,
                              candidate.score if candidate else None))
        rows = [{'proposal': key, 'kind': kind, 'target': target,
                 'slot': option.slot if option else state.engine.puck_owner,
                 'status': 'eligible' if score is not None else 'unscorable',
                 **(score.snapshot() if score is not None else {})}
                for key, kind, target, option, score in proposals]
        # A forecast failure must not silently remove a legacy-admissible action.
        complete = all(score is not None and math.isfinite(score.value) for *_, score in proposals)
        selected = baseline
        if complete and self.profile != 'legacy':
            selected = max(proposals, key=lambda item: (item[-1].value, item[0] == baseline))[0]
        chosen = next(row for row in rows if row['proposal'] == selected)
        self.metrics['decisions'] += 1
        self.metrics['selected:' + chosen['kind']] += 1
        self.metrics['overrides'] += selected != baseline
        self.metrics['unscorable-fallback'] += not complete
        self.metrics[baseline + '->' + selected] += 1
        self.metrics['next:' + chosen.get('continuation', 'unscorable')] += 1
        self.diagnostics = {
            'possession_ablation': self.profile, 'value_model': VALUE_MODEL,
            'evaluation_frame': frame, 'evaluation_carrier': state.engine.puck_owner,
            'possession_candidates': rows, 'baseline_proposal': baseline,
            'selected_proposal': selected, 'possession_score': chosen,
            'status': 'ranked' if complete else 'unscorable-use-legacy',
        }
        return selected
