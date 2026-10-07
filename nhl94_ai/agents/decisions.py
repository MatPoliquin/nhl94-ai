"""Display-only decision evidence, independent of policy action-space IDs."""
from dataclasses import dataclass
import math
from typing import Literal


@dataclass(frozen=True)
class DecisionScore:
    kind: str
    value: float

    def __post_init__(self):
        if not self.kind or not math.isfinite(self.value):
            raise ValueError('Decision scores require a named kind and finite value.')


@dataclass(frozen=True)
class ActionCandidate:
    action_id: str
    label: str
    group: str
    target_slot: int | None = None
    scores: tuple[DecisionScore, ...] = ()
    probability: float | None = None
    raw_probability: float | None = None
    probability_scope: str | None = None
    source: str = ''
    status: Literal['not-evaluated', 'eligible', 'rejected', 'unavailable', 'disabled'] = 'not-evaluated'
    reason: str = ''
    evaluated_frame: int | None = None

    def __post_init__(self):
        if not self.action_id or not self.label or not self.group:
            raise ValueError('Decision candidates require an ID, label and group.')
        if self.status not in ('not-evaluated', 'eligible', 'rejected', 'unavailable', 'disabled'):
            raise ValueError(f'Unknown candidate status: {self.status}')
        for value in (self.probability, self.raw_probability):
            if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError('Action probabilities must be finite values in [0, 1].')
        if (self.probability is not None or self.raw_probability is not None) and not self.probability_scope:
            raise ValueError('Action probabilities require their network/head or conditional scope.')
        if self.evaluated_frame is not None and (
                not isinstance(self.evaluated_frame, int) or isinstance(self.evaluated_frame, bool) or self.evaluated_frame < 0):
            raise ValueError('Candidate evaluation frames must be nonnegative integers.')


@dataclass(frozen=True)
class DecisionSnapshot:
    source: str
    frame: int
    candidates: tuple[ActionCandidate, ...]
    plan_id: str | None = None
    execution_id: str | None = None
    phase: str = ''
    reason: str = ''
    action_schema: str = 'FILTERED'
    action: tuple[float, ...] = ()
    evaluation_frame: int | None = None
    evaluation_carrier: int | None = None
    goalie_policy: str | None = None

    def __post_init__(self):
        ids = [candidate.action_id for candidate in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError('Decision candidate IDs must be unique within a snapshot.')
        if any(selected is not None and selected not in ids for selected in (self.plan_id, self.execution_id)):
            raise ValueError('Selected decisions must identify a candidate in the snapshot.')
        if (not isinstance(self.frame, int) or isinstance(self.frame, bool) or self.frame < 0
                or any(not math.isfinite(value) for value in self.action)):
            raise ValueError('Decision frames and applied action values must be valid.')


CATALOGUE = (
    ('carry', 'Carry / keep puck', 'Offense'),
    ('carry-breakaway', 'Breakaway carry', 'Offense'),
    ('carry-opportunity', 'Carry to improve opportunity', 'Offense'),
    ('carry-escape', 'Escape pressure', 'Offense'),
    ('goalie-avoid', 'Avoid goalie contact', 'Offense'),
    ('feint', 'Feint / open a lane', 'Offense'),
    ('one-timer-setup', 'Set up a one-timer', 'Offense'),
    ('create-chance', 'Create a chance', 'Offense'),
    ('shoot', 'Normal shot', 'Finishing'),
    ('deke', 'Deke finish', 'Finishing'),
    ('cross-crease', 'Cross-crease finish', 'Finishing'),
    ('slapshot', 'Slapshot', 'Finishing'),
    ('behind-net', 'Go behind net / wraparound', 'Finishing'),
    ('recover-safe', 'Recover loose puck', 'Defense'),
    ('intercept-pass', 'Intercept pass', 'Defense'),
    ('protect-lane', 'Protect shooting lane', 'Defense'),
    ('deny-reception', 'Deny reception', 'Defense'),
    ('contain-boards', 'Contain along boards', 'Defense'),
    ('deny-goalie-outlet', 'Deny goalie outlet', 'Defense'),
    ('switch-request', 'Switch skater', 'Defense'),
    ('poke-request', 'Poke check', 'Defense'),
    ('check-request', 'Body check', 'Defense'),
    ('boost-request', 'Boost / skate', 'Defense'),
    ('skating', 'Skate toward target', 'Defense'),
    ('holding', 'Hold position', 'Defense'),
    ('goalie-takeover', 'Take goalie control', 'Goalie'),
    ('goalie-positioning', 'Position / align goalie', 'Goalie'),
    ('goalie-save', 'Goalie save', 'Goalie'),
    ('goalie-dive', 'Goalie dive', 'Goalie'),
    ('goalie-hold', 'Hold puck for outlet', 'Goalie'),
    ('goalie-outlet', 'Goalie outlet pass', 'Goalie'),
    ('goalie-clear', 'Goalie clear', 'Goalie'),
    ('goalie-return', 'Return to skater control', 'Goalie'),
    ('goalie-fallback', 'CPU goalie fallback', 'Goalie'),
    ('wait', 'Wait / neutral', 'Execution'),
)
POSITIONS = {1: 'LD', 2: 'RD', 3: 'LW', 4: 'C', 5: 'RW'}


def _pass_label(player, slot, one_timer):
    return f'{"One-timer" if one_timer else "Pass"} to {POSITIONS.get(player.role, "skater")} [{slot}]'


def catalogue_snapshot(team, source, frame, action_schema='FILTERED'):
    candidates = [ActionCandidate(action_id, label, group, source=source)
                  for action_id, label, group in CATALOGUE]
    for one_timer in (False, True):
        for index, player in enumerate(team.players):
            slot = team.skater_scnum_base() + index
            candidates.append(ActionCandidate(
                f'{"one-timer" if one_timer else "pass"}:{slot}', _pass_label(player, slot, one_timer),
                'One-timers' if one_timer else 'Passing', slot, source=source))
    order = ('Offense', 'Passing', 'One-timers', 'Finishing', 'Defense', 'Goalie', 'Execution')
    candidates.sort(key=lambda candidate: order.index(candidate.group))
    return DecisionSnapshot(source, frame, tuple(candidates), phase='catalogue only; no tactical producer',
                            action_schema=action_schema)


def _scores(details, fields):
    return tuple(DecisionScore(kind, float(details[field])) for kind, field in fields
                 if details.get(field) is not None)


def _pass_candidate(player, slot, details, *, one_timer, owner, controlled, frame):
    kind = 'one-timer' if one_timer else 'pass'
    status = details.get('status', 'not-evaluated')
    eligible = status == 'safe'
    reason = '' if eligible else status
    candidate_status = 'eligible' if eligible else 'not-evaluated' if status == 'not-evaluated' else 'rejected'
    if eligible and details.get('worthwhile') is False:
        candidate_status, reason = 'rejected', 'insufficient tactical gain'
    if slot == owner:
        candidate_status, reason = 'unavailable', 'current carrier'
    elif owner != controlled:
        candidate_status, reason = 'unavailable', 'no controlled possession'
    fields = (('one-timer', 'shot_value'),) if one_timer else (
        ('pass', 'value'), ('position', 'shot_value'), ('carry', 'continuation_position_value'),
        ('finish', 'continuation_finish_value'))
    return ActionCandidate(
        f'{kind}:{slot}', _pass_label(player, slot, one_timer),
        'One-timers' if one_timer else 'Passing', slot, _scores(details, fields),
        source='Classic rules', status=candidate_status, reason=reason,
        evaluated_frame=frame if status != 'not-evaluated' else None)


def _selected_ids(decision, offense, defense, goalie, controller):
    target = offense.get('desired_slot')
    if decision in ('advance-pass', 'position-pass', 'pass-release', 'pass-flight'):
        pending = controller.offense.pending
        target = pending['receiver'] if pending else target
        return f'pass:{target}', None
    if decision.startswith('one-timer-') and decision not in ('one-timer-setup', 'one-timer-ended'):
        target = controller._one_timer[1] if controller._one_timer else target
        return f'one-timer:{target}', None
    if decision.startswith('deke-'):
        return 'deke', None
    if decision.startswith('cross-crease-'):
        return 'cross-crease', None
    if decision == 'shot-follow-through':
        return 'shoot', None
    if decision == 'pass-button-release':
        return 'wait', None
    if decision.startswith('goalie-') and decision not in ('goalie-avoid', 'goalie-outlet'):
        phase = goalie.get('mode', controller.goalie.phase if controller.goalie else '')
        if phase == 'possession-outlet' or phase == 'outlet-flight':
            outlet = controller.goalie.outlet if controller.goalie else None
            return ('goalie-clear' if outlet and outlet['receiver'] is None else
                    'goalie-outlet' if outlet else 'goalie-hold'), None
        groups = {
            'prepare-goalie': 'goalie-takeover', 'request-goalie': 'goalie-takeover',
            'release-goalie': 'goalie-takeover', 'positioning': 'goalie-positioning',
            'save-request': 'goalie-save', 'dive-request': 'goalie-dive',
            'save-recovery': 'goalie-save', 'cpu-fallback': 'goalie-fallback',
            'request-skater': 'goalie-return', 'neutral-handoff': 'goalie-return',
            'backoff': 'goalie-fallback', 'stoppage': 'wait', 'skater': 'wait',
        }
        return groups.get(phase, f'unknown:{decision}'), None
    if defense:
        mode = defense.get('mode')
        execution = {'check-follow-through': 'check-request',
                     'waiting-for-selection': 'wait'}.get(mode, mode)
        return defense.get('decision', decision), execution
    return {'recover': 'recover-safe', 'switch': 'switch-request', 'init': 'wait',
            'one-timer-ended': 'wait'}.get(decision, decision), None


def classic_decision_snapshot(controller, state, action):
    """Adapt existing evidence without evaluating alternatives or touching history."""
    offense = controller.offense_diagnostics
    defense = controller.defense_diagnostics
    goalie = controller.goalie_diagnostics
    decision = controller._last_decision
    frame = controller.defense.frames
    evaluated = offense.get('evaluation_frame')
    controlled = state.team1.controlled_scnum() if state.team1.defense_control is None else state.team1.defense_control
    source_rows = {row['slot']: row for row in offense.get('teammate_scores', ())}
    candidates = []
    for action_id, label, group in CATALOGUE:
        status, reason = 'not-evaluated', ''
        scores = ()
        measured = None
        if action_id in ('slapshot', 'behind-net'):
            status, reason = 'disabled', 'not a standalone Classic tactic'
        elif action_id in ('deke', 'cross-crease', 'create-chance', 'one-timer-setup'):
            enabled = {'deke': controller.deke is not None, 'cross-crease': controller.cross_crease is not None,
                       'create-chance': controller.offense.chance_creation,
                       'one-timer-setup': controller._one_timers}[action_id]
            if not enabled:
                status, reason = 'disabled', 'feature off'
        elif group == 'Goalie' and controller.goalie is None and action_id != 'goalie-outlet':
            status, reason = 'disabled', 'manual-goalie AI off'
        if action_id == 'carry':
            scores = _scores(offense, (('keep', 'retained_value'),))
            measured = evaluated if scores else None
        if action_id == decision and action_id in ('carry-opportunity', 'carry-escape', 'carry', 'goalie-avoid'):
            scores += _scores(offense, (('position', 'carry_position_value'), ('finish', 'carry_shot_value')))
            measured = evaluated if scores else measured
        if action_id == 'create-chance':
            scores = _scores(offense, (('window', 'chance_value'),))
            measured = evaluated if scores else None
        candidates.append(ActionCandidate(action_id, label, group, scores=scores, source='Classic rules',
                                          status=status, reason=reason, evaluated_frame=measured))
    for one_timer in (False, True):
        for index, player in enumerate(state.team1.players):
            slot = state.team1.skater_scnum_base() + index
            row = source_rows.get(slot, {})
            details = (row.get('one_timer') if one_timer else row.get('pass')) or {}
            candidate = _pass_candidate(player, slot, details, one_timer=one_timer,
                                        owner=state.engine.puck_owner, controlled=controlled, frame=evaluated)
            if one_timer and not controller._one_timers:
                candidate = ActionCandidate(candidate.action_id, candidate.label, candidate.group, slot,
                                            source='Classic rules', status='disabled', reason='feature off')
            candidates.append(candidate)
    plan, execution = _selected_ids(decision, offense, defense, goalie, controller)
    ids = {candidate.action_id for candidate in candidates}
    for selected in (plan, execution):
        if selected is not None and selected not in ids:
            candidates.append(ActionCandidate(selected, selected, 'Execution',
                                              source='Classic rules', reason='unmapped decision'))
            ids.add(selected)
    group_order = {'Offense': 0, 'Passing': 1, 'One-timers': 2, 'Finishing': 3, 'Defense': 4, 'Goalie': 5, 'Execution': 6}
    candidates.sort(key=lambda candidate: group_order[candidate.group])
    return DecisionSnapshot(
        'Classic rules', frame, tuple(candidates), plan, execution, phase=decision,
        reason=goalie.get('reason', '') if decision.startswith('goalie-') else (
            defense.get('reason', '') if defense else offense.get('reason', offense.get('status', ''))),
        action_schema='HOCKEY_INTENT_DPAD' if controller._intents else 'FILTERED',
        action=tuple(float(value) for value in action), evaluation_frame=evaluated,
        evaluation_carrier=offense.get('evaluation_carrier'),
        goalie_policy=getattr(controller.args, 'goalie_policy', 'off'))
