"""Evaluation-only finish selection over Classic's admitted actions.

The baseline and executors are unchanged. A carry alternative delays a finish
by one tactical decision, using Classic's existing safe carry route. Features
are available before acting; native outcomes must never enter this hook.
"""
from collections import Counter
import math

from nhl94_ai.agents.defense import on_ice
from nhl94_ai.agents.finishing import one_timer_finish
from nhl94_ai.agents.motion import velocity
from nhl94_ai.agents.shot_placement import legacy_action, shot_features


def readiness(player):
    flags = player.selection_flags
    return {'animation_known': float(flags is not None),
            'animation_locked': float(bool((flags or 0) & 0x20)),
            'falling': float(bool(player.is_falling)),
            'animation_timer': float(player.animation_timer or 0)}


def features(model, state, baseline, shot_allowed, timer_target):
    player = state.team1.get_player_by_scnum(state.engine.puck_owner)
    sign = 1 if state.team2.net.y > state.team1.net.y else -1
    goalie = state.team2.goalie
    vx, vy = velocity(player)
    gx, gy = velocity(goalie)
    values = {
        **readiness(player), 'baseline_shoot': float(baseline == 'shoot'),
        'shot_available': float(shot_allowed), 'timer_available': float(timer_target is not None),
        'b_down': float(model.buttons.b_down), 'c_down': float(model.buttons.c_down),
        'x': float(player.x), 'depth': float(player.y * sign), 'vx': vx, 'vy': vy * sign,
        'goalie_x': float(goalie.x), 'goalie_depth': float(goalie.y * sign),
        'goalie_vx': gx, 'goalie_vy': gy * sign,
        'goalie_distance': math.dist((player.x, player.y), (goalie.x, goalie.y)),
        'defender_distance': min((math.dist((player.x, player.y), (p.x, p.y))
                                  for p in state.team2.players if on_ice(p)), default=300),
    }
    shot = shot_features(state, *legacy_action(state, model.scheduler.interval))
    if shot is not None:
        values.update({'shot_' + key: value for key, value in shot.items()})
    option = model.one_timer.option if timer_target is not None else None
    if option is not None:
        receiver = state.team1.get_player_by_scnum(option.slot)
        values.update({'receiver_' + key: value for key, value in readiness(receiver).items()})
        values.update(timer_value=option.value, timer_shot_value=option.shot_value,
                      timer_margin=option.margin, timer_flight=option.flight_frames,
                      receiver_vx=velocity(receiver)[0], receiver_vy=velocity(receiver)[1] * sign)
        finish = one_timer_finish(state, option)
        if finish is not None:
            values.update(timer_clearance=finish.clearance, timer_finish_value=finish.value)
    return {name: float(value) for name, value in values.items() if math.isfinite(value)}


class ChanceSelection:
    """Use the existing outer-arbitration hook without introducing a gameplay flag."""
    def __init__(self, forced=None, rule=None):
        if forced not in (None, 'carry', 'shoot', 'one-timer'):
            raise ValueError('Unknown chance alternative')
        self.forced, self.rule = forced, rule
        self.diagnostics, self.metrics = {}, Counter()
        self.serial = 0
        self.resume_once = False

    @staticmethod
    def available(state):
        return state.engine.puck_owner_known

    def choose_legacy(self, model, state, plan, shot_allowed, prefer_timer, timer_target, finish):
        self.serial += 1
        baseline = ('one-timer' if prefer_timer else 'shoot' if shot_allowed else
                    'plan' if plan is not None else 'fallback')
        self.diagnostics = {}
        if self.resume_once:
            self.resume_once = False
            self.metrics['resume-legacy'] += 1
            return baseline
        if baseline not in ('shoot', 'one-timer'):
            return baseline
        alternatives = ['carry']
        if shot_allowed:
            alternatives.append('shoot')
        if timer_target is not None:
            alternatives.append('one-timer')
        values = features(model, state, baseline, shot_allowed, timer_target)
        selected = self.forced or baseline
        if self.rule is not None:
            value = values.get(self.rule['feature'])
            applies = (value is not None and baseline == self.rule['baseline']
                       and (value <= self.rule['threshold']) == self.rule['less_equal'])
            selected = self.rule['action'] if applies else baseline
            # A learned deferral lasts one decision. Let Classic reconsider the
            # new state before this rule can postpone another opportunity.
            self.resume_once = selected == 'carry'
        if selected not in alternatives:
            raise ValueError('Cannot force a finish excluded by Classic admission')
        self.metrics['chances'] += 1
        self.metrics['overrides'] += selected != baseline
        self.metrics[baseline + '->' + selected] += 1
        self.diagnostics = {'chance': {'baseline': baseline, 'selected': selected,
                                       'alternatives': alternatives, 'features': values}}
        if selected == 'carry':
            # Keep an existing carry plan; exclude an ordinary pass from this arm.
            return 'plan' if plan is not None and plan[2] is None else 'fallback'
        return selected
