"""Audit single-gate rejected one-timers and execute one diagnostic alternative.

Only the shooting rectangle or comparative-position heuristic may be removed.
The diagnostic branch forces that otherwise admissible pass for one decision;
it is not a claim that merely removing a gate would beat Classic's shot priority.
"""
from copy import deepcopy
import math

from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.agents.motion import facing, velocity
from nhl94_ai.agents.passing import evaluate_pass, one_timer_position_ok, pass_release_frames, shot_value
from nhl94_ai.evaluation.chance_selection import readiness
from nhl94_ai.game.constants import GameConsts as Buttons


GATES = ('window', 'position')


def rejection_audit(model, state):
    player = state.team1.get_player_by_scnum(state.engine.puck_owner)
    sign = 1 if state.team2.net.y > state.team1.net.y else -1
    if (not model._one_timers or player is None or player.passing is None
            or player.y * sign < 100 or model.scheduler.frames < model.one_timer.retry_at_frame):
        return {'status': 'outer-ineligible', 'rows': [], 'alternatives': []}
    rows, alternatives = [], []
    current = shot_value(state, player)
    for index, receiver in enumerate(state.team1.players):
        if receiver is player:
            continue
        kwargs = dict(decision_interval=model.scheduler.interval,
                      release_prediction=model.offense.pass_timing, rank_risk=False)
        option, details = evaluate_pass(state, player, index, receiver, 'one-timer', **kwargs)
        original = details['status']
        if option is not None and not one_timer_position_ok(option, player, current):
            original = 'weaker-one-timer-position'
        row = {'slot': details['slot'], 'original_status': original, 'details': details}
        gate = None
        if original == 'outside-one-timer-window':
            option, relaxed = evaluate_pass(state, player, index, receiver, 'one-timer',
                                            one_timer_window=False, **kwargs)
            row['after_window'] = relaxed
            if option is not None:
                if one_timer_position_ok(option, player, current):
                    gate = 'window'
                else:
                    row['remaining_gate'] = 'weaker-one-timer-position'
        elif original == 'weaker-one-timer-position':
            gate = 'position'
        if gate is not None:
            row['single_gate'] = gate
            alternatives.append({'gate': gate, 'option': option,
                                 'features': candidate_features(state, option, gate)})
        rows.append(row)
    return {'status': 'button-held' if model.buttons.b_down else 'audited',
            'rows': rows, 'alternatives': alternatives}


def candidate_features(state, option, gate):
    player = state.team1.get_player_by_scnum(state.engine.puck_owner)
    receiver, goalie = state.team1.players[option.index], state.team2.goalie
    sign = 1 if state.team2.net.y > state.team1.net.y else -1
    px, py = velocity(player)
    rx, ry = velocity(receiver)
    gx, gy = velocity(goalie)
    fx, fy = facing(receiver)
    current = shot_value(state, player)
    values = {'gate_window': float(gate == 'window'), 'gate_position': float(gate == 'position'),
              'carrier_x': float(player.x), 'carrier_depth': player.y * sign,
              'carrier_vx': px, 'carrier_vy': py * sign,
              'receiver_x': float(receiver.x), 'receiver_depth': receiver.y * sign,
              'receiver_vx': rx, 'receiver_vy': ry * sign, 'receiver_facing_x': fx, 'receiver_facing_y': fy * sign,
              'contact_x': option.point[0], 'contact_depth': option.point[1] * sign,
              'flight': option.flight_frames, 'margin': option.margin, 'robustness': option.robustness,
              'shot_value': option.shot_value, 'shot_value_gain': option.shot_value - current,
              'carrier_shot_value': current, 'accuracy': float(receiver.shot_accuracy or 0),
              'goalie_x': float(goalie.x), 'goalie_depth': goalie.y * sign,
              'goalie_vx': gx, 'goalie_vy': gy * sign,
              'goalie_lateral_distance': abs(option.point[0] - goalie.x),
              'lateral_pass': abs(option.point[0] - player.x), 'forward_gain': option.forward_gain}
    for prefix, who in (('carrier', player), ('receiver', receiver), ('goalie', goalie)):
        values.update({prefix + '_' + name: value for name, value in readiness(who).items()})
    return {name: float(value) for name, value in values.items()} if all(
        math.isfinite(value) for value in values.values()) else None


class ForceOneTimer:
    """One-decision diagnostic arbitration through the original action executor."""
    diagnostics = {}

    @staticmethod
    def available(state):
        return state.engine.puck_owner_known

    @staticmethod
    def choose_legacy(model, state, plan, shot_allowed, prefer_timer, timer_target, finish):
        if timer_target is None:
            raise ValueError('Cannot force a pass without a validated one-timer target')
        return 'one-timer'


class AdmissionModel(ClassicAIV1Model):  # pylint: disable=abstract-method
    def __init__(self, args=None, env=None):
        super().__init__(args, env)
        self.admission_audit = None
        self.forced_candidate = None

    def predict_frame(self, state, frame_skip=4, deterministic=True):
        self.admission_audit = None
        return super().predict_frame(state, frame_skip, deterministic)

    def _attack(self, state, action, player, actual, attack):
        self.admission_audit = rejection_audit(self, state)
        if self.forced_candidate is not None:
            candidate = self.forced_candidate
            valid = next((row for row in self.admission_audit['alternatives']
                          if row['gate'] == candidate['gate'] and row['option'] == candidate['option']), None)
            if valid is None or self.admission_audit['status'] != 'audited':
                raise ValueError('Counterfactual must remove exactly one heuristic and have a fresh B edge')
            self.possession_ablation = ForceOneTimer()
        result = super()._attack(state, action, player, actual, attack)
        self.admission_audit['baseline_decision'] = self._last_decision
        for row in self.admission_audit['alternatives']:
            if row['features'] is not None:
                row['features'].update(baseline_shoot=float(self._last_decision == 'shoot'),
                                       baseline_one_timer=float(self._last_decision == 'one-timer-pass'),
                                       baseline_pass=float(self._last_decision in ('advance-pass', 'position-pass')))
        return result

    def _one_timer_target(self, team, opponents, attack, prepare=False, state=None):
        candidate = self.forced_candidate
        if candidate is None or prepare or state is None:
            return super()._one_timer_target(team, opponents, attack, prepare, state)
        option = candidate['option']  # pylint: disable=unsubscriptable-object
        self.one_timer.option = option
        self.one_timer.flight_frames = option.flight_frames + pass_release_frames(team.players[option.index])
        indices = [i for i, p in enumerate(team.players) if p is not team.get_player_by_scnum(state.engine.puck_owner)]
        return indices.index(option.index), option.slot


def original_controller(history):
    model = deepcopy(history)
    model.__class__ = ClassicAIV1Model
    model.possession_ablation = None
    for name in ('admission_audit', 'forced_candidate'):
        model.__dict__.pop(name, None)
    return model


def configured(history, state, candidate=None):
    model = original_controller(history)
    if candidate is not None:
        model.__class__ = AdmissionModel
        model.admission_audit, model.forced_candidate = None, candidate
    action = model.predict_frame(state, frame_skip=4)[0]
    if candidate is not None:
        assert model._last_decision == 'one-timer-pass' and action[Buttons.INPUT_B] and not action[Buttons.INPUT_C]
        assert model.one_timer.pending.receiver == candidate['option'].slot
    return original_controller(model), action
