"""One native-confirmed commitment shared by passer and selected receiver."""
from collections import deque
from dataclasses import asdict, dataclass

import numpy as np

from nhl94_ai.agents.defense import defensive_steering
from nhl94_ai.game.actors import controllable_skater
from nhl94_ai.agents.lifecycle import OneTimerLifecycle, OneTimerState
from nhl94_ai.agents.passing import shot_value
from nhl94_ai.agents.receiving import one_timer_reception, passive_reception, reachable_reception
from nhl94_ai.env.actions import HOCKEY_PASS_PRESS_FRAMES
from nhl94_ai.env.target_control import project_target
from nhl94_ai.game.constants import GameConsts as Buttons


@dataclass
class PassCommitment:
    passer: int
    receiver: int
    passer_controller: int | None
    receiver_controller: int | None
    point: tuple
    started: int
    expiry: int
    purpose: str
    passes_before: int | None
    shots_before: int
    attempts_before: int | None
    goals_before: int
    phase: str = 'release'
    released: int | None = None
    received: int | None = None
    cue: int | None = None
    shot: int | None = None

    def snapshot(self):
        return asdict(self)


class SharedReceiving:
    """Track release through a useful attack or a named unresolved ending.

    Frames are native frames. A B intention never confirms a release. Native
    counter/target feedback determines which controller, if any, receives it.
    """
    def __init__(self):
        self.pending = None
        self.last_owner = -1
        self.last_count = None
        self.events = deque(maxlen=32)
        self.frame_events = []

    def _event(self, frame, phase, **values):
        event = {'frame': frame, 'phase': phase, **values}
        if self.pending is not None:
            event['commitment'] = self.pending.snapshot()
        self.events.append(event)
        self.frame_events.append(event)

    def finish(self, frame, reason):
        if self.pending is not None:
            self._event(frame, 'ended', reason=reason)
            self.pending = None

    def propose(self, request, controller, slots, state, frame):
        if request is None:
            return
        if self.pending is not None:
            if request['frame'] <= self.pending.started:
                return
            self.finish(frame, 'next-pass-requested')
        receiver = state.team1.get_player_by_scnum(request['receiver'])
        if receiver is None:
            return
        self.pending = PassCommitment(
            request['passer'], request['receiver'], controller,
            slots.index(request['receiver']) + 1 if request['receiver'] in slots else None,
            tuple(request.get('point') or (receiver.x, receiver.y)), frame,
            frame + max(8, min(72, request['deadline'] - request['frame'])),
            request['purpose'], state.team1.pass_attempts, state.team1.stats.shots,
            state.team1.one_timer_attempts, state.team1.stats.score)
        self._event(frame, 'created')

    def observe(self, state, slots, frame):
        self.frame_events = []
        count, owner = state.team1.pass_attempts, state.engine.puck_owner
        released = count is not None and self.last_count is not None and count > self.last_count
        if state.engine.clock_stopped:
            self.finish(frame, 'stoppage')
        elif released and state.team1.owns_scnum(self.last_owner):
            actual = state.engine.pass_target
            if self.pending is not None and self.pending.phase != 'release':
                self.finish(frame, 'next-native-pass')
            receiver = state.team1.get_player_by_scnum(actual) if actual is not None else None
            if self.pending is None and receiver is not None:
                self.propose({'passer': self.last_owner, 'receiver': actual,
                              'point': (receiver.x, receiver.y), 'frame': frame,
                              'deadline': frame + 72, 'purpose': 'native-pass'},
                             slots.index(self.last_owner) + 1 if self.last_owner in slots else None,
                             slots, state, frame)
            plan = self.pending
            if plan is not None and plan.phase == 'release':
                if actual != plan.receiver:
                    self._event(frame, 'retargeted', actual_receiver=actual)
                    plan.receiver, plan.purpose = actual, 'native-pass'
                    plan.receiver_controller = slots.index(actual) + 1 if actual in slots else None
                plan.phase, plan.released = 'flight', frame
                self._event(frame, 'released')
        plan = self.pending
        if plan is not None:
            receiver = state.team1.get_player_by_scnum(plan.receiver)
            attempts = state.team1.one_timer_attempts
            shot = (state.team1.stats.shots > plan.shots_before
                    or attempts is not None and plan.attempts_before is not None and attempts > plan.attempts_before)
            if plan.shot is None and plan.released is not None and shot and state.engine.shot_player == plan.receiver:
                plan.shot, plan.phase, plan.expiry = frame, 'shot', frame + 150
                self._event(frame, 'shot')
            if plan.phase == 'shot':
                if state.team1.stats.score > plan.goals_before:
                    self.finish(frame, 'goal')
                elif owner >= 0:
                    self.finish(frame, 'shot-recovered-team' if state.team1.owns_scnum(owner) else 'shot-recovered-opponent')
            elif state.team2.owns_scnum(owner):
                self.finish(frame, 'intercepted' if plan.released is not None else 'lost-before-release')
            elif plan.released is not None and owner == plan.receiver:
                if plan.received is None:
                    plan.received, plan.phase, plan.expiry = frame, 'received', frame + 90
                    self._event(frame, 'received')
            elif plan.released is not None and owner >= 0 and owner != plan.passer:
                self.finish(frame, 'other-receiver')
            elif plan.phase == 'received' and owner != plan.receiver:
                self.finish(frame, 'reception-possession-ended')
            elif receiver is None or receiver.is_falling or not controllable_skater(receiver) and not receiver.is_one_timer:
                self.finish(frame, 'receiver-unavailable')
            elif plan.receiver_controller is not None and plan.receiver not in slots:
                self.finish(frame, 'receiver-selection-lost')
            elif plan.released is not None and owner == plan.passer:
                self.finish(frame, 'recovered-by-passer')
            if self.pending is not None and frame >= plan.expiry:
                self.finish(frame, 'unresolved-' + plan.phase)
        self.last_count, self.last_owner = count, owner

    def receiver_action(self, state, controller, frame, previous):
        plan = self.pending
        if (plan is None or plan.receiver_controller != controller
                or plan.phase not in ('release', 'flight')):
            return None
        player = state.team1.get_player_by_scnum(plan.receiver)
        if player is None or not controllable_skater(player) and not player.is_one_timer:
            return None
        action = np.zeros(12, dtype=np.int8)
        if plan.phase == 'release' or player.is_one_timer:
            return action, (player.x, player.y), 'receive'
        forecast = passive_reception(state, player)
        interception = reachable_reception(state, player)
        # A pass is led toward a coasting receiver. Preserve a useful existing
        # contact rather than moving it away to chase a distant path endpoint.
        if forecast is not None and forecast['gap'] <= -2:
            target = (player.x, player.y)
        else:
            target = project_target(interception['point'] if interception is not None else plan.point)
            pad, target = defensive_steering(player, target)
            action[Buttons.INPUT_LEFT], action[Buttons.INPUT_RIGHT] = pad[0] < 0, pad[0] > 0
            action[Buttons.INPUT_DOWN], action[Buttons.INPUT_UP] = pad[1] < 0, pad[1] > 0
        plan.point = target
        attack = 1 if state.team2.net.y > state.team1.net.y else -1
        shot_contact = one_timer_reception(state, player) if plan.purpose == 'one-timer' else None
        # Cue once, after native release and before modeled contact. Let ordinary
        # receptions keep possession when contact/shot geometry is unsuitable.
        viable = (shot_contact is not None and shot_contact['gap'] <= -2
                  and 1 < shot_contact['frames'] <= 16
                  and 175 < player.y * attack < 245 and abs(player.x) < 70
                  and shot_value(state, player) > 0)
        if viable and plan.cue is None and not previous[Buttons.INPUT_C]:
            lifecycle = OneTimerLifecycle(pending=OneTimerState(
                plan.passer, plan.receiver, plan.expiry, started=frame - HOCKEY_PASS_PRESS_FRAMES,
                passes_before=plan.passes_before, actual_receiver=plan.receiver, launched=True))
            step = lifecycle.step(state, frame, c_down=False)
            action = step.buttons
            plan.cue = frame
            self._event(frame, 'one-timer-cue')
            return action, target, 'one-timer-receiver'
        return action, target, 'receive'
