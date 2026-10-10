"""State and feedback transitions for Classic's committed offensive actions.

These executors preserve the established priority of simultaneous observations.
They do not rank opportunities or change tactical thresholds.
"""
from collections import Counter
from dataclasses import dataclass, field
import math

import numpy as np

from nhl94_ai.agents.defense import (
    DefenseController, controlled_slot, defensive_steering, eligible, owns_puck,
)
from nhl94_ai.agents.receiving import passive_reception, reception_target
from nhl94_ai.env.actions import HOCKEY_PASS_PRESS_FRAMES
from nhl94_ai.env.intents import HOCKEY_INTENT_CHANGE_PLAYER, HOCKEY_INTENT_NOOP
from nhl94_ai.game.constants import GameConsts as Buttons
from nhl94_ai.game.geometry import aim_pass


ONE_TIMER_TIMEOUT_FRAMES = 72
ONE_TIMER_CONTACT_GRACE = 20
ONE_TIMER_RETRY_FRAMES = 96
ONE_TIMER_EXECUTION_MODES = ('early-cue', 'release-retry')
ONE_TIMER_EXECUTION_INCOMPATIBLE = (
    'possession_value', 'possession_ablation', 'receiver_selection', 'shot_placement',
    'reception_control', 'offense_lookahead', 'uncertain_carry', 'chance_creation',
    'cross_crease', 'deke', 'classic_refinements',
)


def validate_one_timer_execution(args):
    mode = getattr(args, 'one_timer_execution', None)
    unsupported = (getattr(args, 'action_type', 'FILTERED').upper() != 'FILTERED'
                   or getattr(args, 'env', 'NHL94-Genesis-v0') != 'NHL94-Genesis-v0'
                   or getattr(args, 'selfplay', False))
    if mode is not None and (
            mode not in ONE_TIMER_EXECUTION_MODES
            or unsupported
            or any(getattr(args, name, False) for name in ONE_TIMER_EXECUTION_INCOMPATIBLE)):
        raise ValueError('One-timer execution requires full-team Classic FILTERED controls without other offensive experiments or self-play')
    return mode


def validate_rebound_recovery(args):
    enabled = getattr(args, 'rebound_recovery', False)
    supported = (getattr(args, 'action_type', 'FILTERED').upper() in ('FILTERED', 'HOCKEY_INTENT_DPAD')
                 and getattr(args, 'env', 'NHL94-Genesis-v0') == 'NHL94-Genesis-v0'
                 and not getattr(args, 'selfplay', False))
    if enabled and (not supported or any(getattr(args, name, False) for name in (
            *ONE_TIMER_EXECUTION_INCOMPATIBLE, 'one_timer_execution'))):
        raise ValueError('Rebound recovery requires full-team Classic buttons/intents without other offensive experiments or self-play')
    return enabled


@dataclass
class ShotLifecycle:
    """A normal C hold followed by a decision-counted follow-through window."""
    until_decision: int = 0
    hold_until_frame: int = 0
    shots_before: int = 0
    slot: int = -1
    side: int = 1
    extended_hold: bool = False
    rebound_recovery: bool = False
    rebound_wait_for_animation: bool = True
    release_observed: bool = False
    recoveries: int = 0

    def active(self, decision):
        return decision < self.until_decision

    def phase(self, decision, frame):
        if not self.active(decision):
            return 'idle'
        return 'holding' if frame < self.hold_until_frame else 'follow-through'

    def start(self, state, decision, frame, hold_frames):
        self.extended_hold = False
        self.release_observed = False
        self.until_decision = decision + 7
        self.hold_until_frame = frame + hold_frames
        self.shots_before = state.team1.stats.shots
        self.slot = state.engine.puck_owner

    def cancel(self):
        self.until_decision = 0
        self.extended_hold = False
        self.release_observed = False

    def same_shooter_recovery(self, state, decision):
        return (self.active(decision) and self.release_observed and not state.engine.clock_stopped
                and state.engine.puck_owner == self.slot and owns_puck(state.team1, self.slot))

    def recovery_ready(self, state):
        if not self.rebound_wait_for_animation:
            return True
        player = state.team1.get_player_by_scnum(self.slot)
        # Native doinput ignores carrier B/C while pfalock is set. Missing
        # feedback keeps the original follow-through instead of guessing.
        return (player is not None and player.selection_flags is not None
                and not player.selection_flags & 0x20)

    def observe(self, state, decision):
        if not self.active(decision):
            return False
        owner = state.engine.puck_owner
        if state.engine.clock_stopped:
            self.release_observed = False
        elif owner < 0 and self.released(state):
            self.release_observed = True
        recovered = (self.rebound_recovery and self.same_shooter_recovery(state, decision)
                     and self.recovery_ready(state))
        if recovered or owns_puck(state.team1, owner) and owner != self.slot:
            self.recoveries += bool(recovered)
            self.cancel()
            return True
        return False

    def released(self, state):
        return (state.team1.stats.shots > self.shots_before and state.engine.puck_owner != self.slot
                and state.engine.shot_player in (-1, self.slot))

    def aim(self, action):
        # Horizontal aim stays in world coordinates at either attacking end.
        if self.side:
            action[Buttons.INPUT_RIGHT if self.side > 0 else Buttons.INPUT_LEFT] = 1


@dataclass
class SetupState:
    """Legacy setup used when live passing telemetry is unavailable."""
    until_decision: int = 0
    retry_at_decision: int = 0
    slot: int | None = None


@dataclass(frozen=True)
class ActionEnd:
    reason: str
    interrupt: bool = True
    neutral_frame: bool = False


@dataclass
class ActionStep:
    buttons: np.ndarray
    intent: int
    decision: str
    target: tuple | None = None
    diagnostics: dict = field(default_factory=dict)


@dataclass
class ReceptionCorrection:
    started: int
    forecast: dict
    from_slot: int
    switch_frame: int | None = None

    def snapshot(self):
        return dict(vars(self))


@dataclass
class PassState:
    """All times are emulator frames; slot identities survive decoded-state copies."""
    passer: int
    receiver: int
    frame: int
    deadline: int
    purpose: str = 'advance-pass'
    point: tuple = (0, 0)
    passes_before: int | None = None
    direction: int = 8
    actual_receiver: int | None = None
    launched: bool = False
    flight_observed: bool = False
    reception_correction: ReceptionCorrection | None = None

    @property
    def phase(self):
        if self.reception_correction is not None:
            return 'reception-correction'
        return 'flight' if self.launched else 'release'

    def snapshot(self):
        result = dict(vars(self))
        correction = result.pop('reception_correction')
        if correction is not None:
            result['reception_correction'] = correction.snapshot()
        return result


@dataclass
class PassLifecycle:
    pending: PassState | None = None
    last_pass: dict | None = None
    last_request: dict | None = None
    stick_control: bool = False
    reception_metrics: Counter = field(default_factory=Counter)

    def start(self, state, option, frame, purpose):
        self.pending = PassState(
            passer=state.engine.puck_owner, receiver=option.slot, purpose=purpose,
            frame=frame, deadline=frame + math.ceil(option.flight_frames) + 20,
            point=option.point, passes_before=state.team1.pass_attempts, direction=option.direction)
        self.last_request = self.pending.snapshot()

    def cancel(self):
        # Existing cancellation discards the request without fabricating an outcome.
        self.pending = None

    def finish(self, state, frame, reason):
        if self.pending is None:
            return None
        if self.pending.reception_correction is not None:
            self.reception_metrics['corrected:' + reason] += 1
        self.last_pass = {**self.pending.snapshot(), 'outcome': reason,
                          'owner': state.engine.puck_owner, 'end_frame': frame}
        self.pending = None
        return ActionEnd(reason)

    def observe(self, state, frame):
        request = self.pending
        if request is None:
            return None
        owner = state.engine.puck_owner
        if state.team2.owns_scnum(owner):
            result = 'intercepted' if request.launched else 'lost-before-release'
        elif owner == request.passer and request.flight_observed:
            result = 'recovered-by-passer'
        elif frame >= request.deadline:
            # Preserve deadline-before-reception ordering during the refactor.
            result = 'flight-timeout' if request.launched else 'not-released'
        elif owner >= 0 and owner != request.passer:
            result = 'received' if owner == request.receiver else 'other-receiver'
        else:
            if owner < 0:
                request.flight_observed = True
            if (request.actual_receiver is None and state.team1.pass_attempts is not None
                    and request.passes_before is not None and state.team1.pass_attempts > request.passes_before):
                request.actual_receiver = state.engine.pass_target
                request.launched = True
            if owner < 0 and state.engine.last_puck_player == request.passer:
                request.launched = True
            return None
        return self.finish(state, frame, result)

    def receive(self, state, frame, *, b_down):
        """Correct marginal receptions with confirmed selection and no C input."""
        request = self.pending
        if request is None or not request.launched:
            return None
        if (state.engine.puck_owner >= 0 or state.engine.clock_stopped
                or controlled_slot(state.team1) < 0 or frame - request.frame < HOCKEY_PASS_PRESS_FRAMES):
            return None
        slot = request.actual_receiver
        receiver = state.team1.get_player_by_scnum(slot) if slot is not None else None
        correction = request.reception_correction
        if receiver is None or not eligible(receiver):
            return self.finish(state, frame, 'receiver-unavailable') if correction is not None else None
        actual = controlled_slot(state.team1)
        if correction is None:
            forecast = passive_reception(state, receiver, stick_only=self.stick_control)
            wait_for_native = (not self.stick_control and (receiver.decision_timer is None
                                                          or forecast is not None and forecast['frames'] <= receiver.decision_timer))
            if (forecast is None or forecast['gap'] <= -2 or receiver.assignment != 19
                    or wait_for_native):
                return None
            correction = ReceptionCorrection(frame, forecast, actual)
            request.reception_correction = correction
            self.reception_metrics['corrections'] += 1
        action = np.zeros(Buttons.INPUT_MAX, dtype=np.int8)
        intent = HOCKEY_INTENT_NOOP
        target = reception_target(state, receiver, stick_only=self.stick_control)
        if (target is None or actual != slot and (
                actual != correction.from_slot or frame - correction.started >= 16)):
            return self.finish(state, frame, 'reception-unreachable')
        mode = 'receive-pass-wait'
        if actual == slot:
            self.reception_metrics['steered-frames'] += 1
            pad, target = defensive_steering(receiver, target)
            for delta, negative, positive in ((pad[0], Buttons.INPUT_LEFT, Buttons.INPUT_RIGHT),
                                              (pad[1], Buttons.INPUT_DOWN, Buttons.INPUT_UP)):
                if delta:
                    action[positive if delta > 0 else negative] = 1
            mode = 'receive-pass'
        elif (not b_down and DefenseController._likely_switch(state) == slot
              and (correction.switch_frame is None or frame - correction.switch_frame >= 8)):
            action[Buttons.INPUT_B] = 1
            correction.switch_frame = frame
            self.reception_metrics['switches'] += 1
            mode, intent = 'receive-pass-switch', HOCKEY_INTENT_CHANGE_PLAYER
        diagnostics = {'phase': 'offense', 'mode': mode, 'decision': mode,
                       'reason': ('bring the live stick to the incoming pass' if self.stick_control
                                  else 'correct marginal reception after native receive timer'),
                       'target': target, 'destination': target, 'waypoint': target,
                       'actual_slot': actual, 'desired_slot': slot,
                       'reception_correction': correction.snapshot(), 'buttons': action.tolist()}
        return ActionStep(action, intent, mode, target, diagnostics)


@dataclass
class OneTimerState:
    passer: int
    receiver: int
    deadline: int
    started: int = 0
    passes_before: int | None = None
    attempts_before: int | None = None
    shots_before: int = 0
    actual_receiver: int | None = None
    launched: bool = False

    @property
    def phase(self):
        return 'awaiting-shot' if self.launched else 'release'


@dataclass
class OneTimerLifecycle:
    pending: OneTimerState | None = None
    retry_at_frame: int = 0
    starts: int = 0
    metrics: dict = field(default_factory=dict)
    # Candidate geometry belongs to the current tactical evaluation only.
    option: object = None
    flight_frames: float | None = None
    execution: str | None = None

    def needs_frame(self, state, frame):
        request = self.pending
        if request is None:
            return False
        return (self.execution == 'early-cue' and state.engine.puck_owner < 0
                or self.execution == 'release-retry' and state.engine.puck_owner == request.passer
                and frame - request.started >= HOCKEY_PASS_PRESS_FRAMES)

    def start(self, state, receiver, frame, duration=ONE_TIMER_TIMEOUT_FRAMES):
        self.pending = OneTimerState(
            state.engine.puck_owner, receiver, frame + duration, started=frame,
            passes_before=state.team1.pass_attempts, attempts_before=state.team1.one_timer_attempts,
            shots_before=state.team1.stats.shots)
        self.starts += 1
        self.retry_at_frame = frame + ONE_TIMER_RETRY_FRAMES

    def finish(self, reason, *, interrupt=True, neutral_frame=False):
        if self.pending is None:
            return None
        self.metrics[reason] = self.metrics.get(reason, 0) + 1
        self.pending = None
        return ActionEnd(reason, interrupt, neutral_frame)

    def observe(self, state, frame):
        request = self.pending
        if request is None:
            return None
        if state.engine.clock_stopped:
            return self.finish('play-stopped', neutral_frame=True)
        attempts = state.team1.one_timer_attempts
        released = (attempts is not None and request.attempts_before is not None
                    and attempts > request.attempts_before)
        recorded = state.team1.stats.shots > request.shots_before
        owner = state.engine.puck_owner
        if state.engine.shot_player == request.receiver and (released or recorded):
            return self.finish('shot-released')
        if owner == request.passer and request.launched:
            return self.finish('recovered-by-passer')
        if owns_puck(state.team1, owner) and owner != request.passer:
            return self.finish('possession-changed')
        if frame >= request.deadline:
            return self.finish('timeout')
        if owner < 0:
            request.launched = True
        return None

    def step(self, state, frame, *, c_down):
        request = self.pending
        receiver = state.team1.get_player_by_scnum(request.receiver)
        owner = state.engine.puck_owner
        if (request.actual_receiver is None and request.passes_before is not None
                and state.team1.pass_attempts is not None
                and state.team1.pass_attempts > request.passes_before and state.engine.pass_target is not None):
            request.actual_receiver = state.engine.pass_target
        if request.actual_receiver is not None and request.actual_receiver != request.receiver:
            return self.finish('receiver-mismatch')
        if (receiver is None or receiver.is_falling or frame >= request.deadline
                or owner >= 0 and owner != request.passer):
            return self.finish('timeout' if frame >= request.deadline else 'receiver-unavailable'
                               if receiver is None or receiver.is_falling else 'possession-changed')
        action = np.zeros(Buttons.INPUT_MAX, dtype=np.int8)
        mode, target = 'one-timer-wait', None
        elapsed = frame - request.started
        early = self.execution == 'early-cue' and owner < 0 and elapsed < HOCKEY_PASS_PRESS_FRAMES
        if elapsed < HOCKEY_PASS_PRESS_FRAMES and not early:
            aim_pass(action, state.team1.get_player_by_scnum(request.passer), receiver)
            action[Buttons.INPUT_B] = 1
        elif owner == request.passer:
            target = (receiver.x, receiver.y)
            aim_pass(action, state.team1.get_player_by_scnum(request.passer), receiver)
            if self.execution == 'release-retry' and not request.launched and elapsed >= 8:
                # Keep the original deadline and target. Four neutral B frames
                # between attempts provide a new native press after animation lock.
                action[Buttons.INPUT_B] = elapsed % 8 < 4
                mode = 'one-timer-release-retry'
        elif not c_down and not receiver.is_one_timer:
            mode = 'one-timer-shoot'
            action[Buttons.INPUT_C] = 1
        if early:
            mode = 'one-timer-early-cue'
        return ActionStep(action, HOCKEY_INTENT_NOOP, mode, target,
                          {'desired_slot': request.receiver, 'reason': 'wait for one-timer contact',
                           'actual_receiver': request.actual_receiver})
