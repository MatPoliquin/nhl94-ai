"""Classic V1: protect lanes, advance by safe passes, create openings and finish.

Reactive defense chooses a target then a skater; offense ranks bounded opportunities.
There are no inherited agents or learned parameters.
"""
import math

import numpy as np

from nhl94_ai.agents.defense import DefenseController, eligible, on_ice, owns_puck
from nhl94_ai.agents.defense import controlled_slot
from nhl94_ai.agents.goalie import GoalieController
from nhl94_ai.agents.cross_crease import CrossCreaseController, crossing_entry, evaluate_cross_crease
from nhl94_ai.agents.offense import (
    GOALIE_HORIZON, GOALIE_SHOT_CLEARANCE, SHOT_RELEASE_DELAY, OffenseController, carry_clear,
    goalie_avoidance, goalie_contact_time, projected_state, shot_release_in_front,
)
from nhl94_ai.agents.passing import pass_release_frames, shot_value
from nhl94_ai.game.constants import GameConsts as Buttons
from nhl94_ai.game.geometry import aim_pass
from nhl94_ai.env.intents import (
    HOCKEY_INTENT_DPAD_ACTION_SPACE, HOCKEY_INTENT_NOOP,
    HOCKEY_INTENT_NORMAL_SHOOT, HOCKEY_INTENT_CHANGE_PLAYER,
    HOCKEY_INTENT_PASS_START, HOCKEY_INTENT_POKE_CHECK,
)
from nhl94_ai.env.actions import HOCKEY_PASS_PRESS_FRAMES


ONE_TIMER_TIMEOUT_FRAMES = 72
ONE_TIMER_CONTACT_GRACE = 20
ONE_TIMER_RETRY_FRAMES = 96


class ClassicAIV1Model:
    """Deterministic controller; shot decisions and frame-timed tactics stay distinct."""

    def __init__(self, args=None, env=None):
        self.args, self.env = args, env
        self._tick = 0
        self._switch_at = 0
        self._shot_until = 0
        self._shots_before = 0
        self._shot_slot = -1
        self._shot_side = 1
        self._c_down = False
        self._b_down = False
        self._one_timer = None
        self._pass_at = 0
        self._setup_until = 0
        self._setup_at = 0
        self._setup_slot = None
        self._one_timers = getattr(args, 'one_timers', True)
        self._last_decision = 'init'
        self._last_target = (0, 0)
        self._intents = getattr(args, 'action_type', 'FILTERED').upper() == 'HOCKEY_INTENT_DPAD'
        self._size = len(HOCKEY_INTENT_DPAD_ACTION_SPACE) if self._intents else Buttons.INPUT_MAX
        self._last_action_preferences = np.zeros((1, self._size), dtype=np.float32)
        self.defense = DefenseController()
        self.defense_diagnostics = {}
        self.offense = OffenseController(one_timers=self._one_timers)
        self.offense_diagnostics = {}
        cross_crease = getattr(args, 'cross_crease', False)
        if cross_crease and (getattr(args, 'action_type', 'FILTERED').upper() not in (
                'FILTERED', 'HOCKEY_INTENT_DPAD') or getattr(args, 'env', 'NHL94-Genesis-v0') != 'NHL94-Genesis-v0'):
            raise ValueError('Cross-crease AI requires full-team FILTERED or HOCKEY_INTENT_DPAD controls')
        self.cross_crease = CrossCreaseController() if cross_crease else None
        self._last_pass_request = None
        self._one_timer_passes_before = None
        self._one_timer_actual = None
        self._one_timer_launched = False
        self._one_timer_attempts_before = None
        self._one_timer_shots_before = 0
        self._one_timer_started = 0
        self._one_timer_flight = None
        self._decision_interval = 4
        self.one_timer_metrics = {}
        self.one_timer_starts = 0
        self.carry_metrics = {}
        self._frame_remaining = 0
        self._frame_action = np.zeros((1, self._size), dtype=np.int8)
        self._defense_elapsed = 4
        self._was_defending = False
        goalie_policy = getattr(args, 'goalie_policy', 'off')
        if goalie_policy not in ('off', 'selective', 'always'):
            raise ValueError('goalie_policy must be off, selective or always')
        if goalie_policy != 'off' and getattr(args, 'action_type', 'FILTERED').upper() != 'FILTERED':
            raise ValueError('Manual goalie AI requires FILTERED buttons')
        self.goalie = None if goalie_policy == 'off' else GoalieController(goalie_policy)
        self.goalie_diagnostics = {}

    def predict(self, _observation, deterministic=True):
        return np.zeros((1, self._size), dtype=np.int8), None

    def get_action_preferences(self, _observation=None):
        return self._last_action_preferences

    def learn(self, *args, **kwargs):
        raise NotImplementedError('ClassicAIV1 is inference-only; use nhl94 play --agent classic-v1.')

    def save(self, *args, **kwargs):
        raise NotImplementedError('ClassicAIV1 does not produce a trainable checkpoint.')

    def predict_game_state(self, state, deterministic=True):
        self._observe_follow_through(state)
        if self._observe_one_timer(state, self.defense.frames + self._defense_elapsed):
            self.defense.idle(self._defense_elapsed)
            return self._encode(np.zeros(Buttons.INPUT_MAX, dtype=np.int8), HOCKEY_INTENT_NOOP)
        if self._observe_cross_crease(state):
            return self._encode(np.zeros(Buttons.INPUT_MAX, dtype=np.int8), HOCKEY_INTENT_NOOP)
        goalie_action = self._goalie_frame(state)
        if goalie_action is not None:
            return goalie_action
        return self._predict_decision(state, deterministic)

    def _goalie_frame(self, state):
        if self.goalie is None:
            return None
        blocked = (self._one_timer is not None or self.offense.pending is not None
                   or self.cross_crease is not None and self.cross_crease.plan is not None
                   or self._tick < self._shot_until or self.defense.pending_check is not None
                   or self.defense.pending_switch is not None)
        action = self.goalie.step(state, blocked=blocked)
        self.goalie_diagnostics = self.goalie.diagnostics
        if action is None:
            return None
        self.defense.idle(1)
        self.defense_diagnostics = self.offense_diagnostics = {}
        self._end_one_timer('goalie-control')
        self._shot_until = self._setup_until = 0
        self._setup_slot = None
        self.offense.cancel()
        self._frame_remaining = 0
        self._was_defending = False
        self._last_decision = 'goalie-' + self.goalie.phase
        self._last_target = self.goalie_diagnostics['target']
        return self._encode(action, HOCKEY_INTENT_NOOP)

    def _predict_decision(self, state, deterministic=True):
        self._tick += 1
        action, intent = self._decide(state)
        if not self.defense_diagnostics:
            actual = controlled_slot(state.team1)
            self.offense_diagnostics = {
                **self.offense_diagnostics, 'phase': 'offense', 'mode': self._last_decision,
                'decision': self._last_decision, 'target': self._last_target,
                'destination': self._last_target, 'waypoint': self._last_target, 'actual_slot': actual,
                'desired_slot': self.offense_diagnostics.get('desired_slot', actual),
                'reason': self.offense_diagnostics.get('reason', self.offense_diagnostics.get('status', self._last_decision)),
                'buttons': action.tolist(), 'last_pass': self.offense.last_pass,
            }
        return self._encode(action, intent)

    def _encode(self, action, intent):
        self._c_down = bool(action[Buttons.INPUT_C])
        self._b_down = bool(action[Buttons.INPUT_B])
        if self._intents:
            action = np.asarray([intent, *action[4:8], int(action[8] and intent == HOCKEY_INTENT_NOOP)], dtype=np.int8)
        self._last_action_preferences = action[None].astype(np.float32)
        return action[None]

    def _defending(self, state):
        owner = state.engine.puck_owner
        if owns_puck(state.team1, owner):
            return False
        return owner >= 0 or (self._one_timer is None and self.offense.pending is None
                              and (self.cross_crease is None or self.cross_crease.plan is None)
                              and self._tick >= self._shot_until)

    def _observe_cross_crease(self, state):
        if self.cross_crease is None:
            return False
        if state.numPlayers != 5:
            raise ValueError('Cross-crease AI requires full-team NHL94')
        plan = self.cross_crease.plan
        self.cross_crease.observe(state, self.defense.frames)
        if plan is not None and self.cross_crease.plan is None:
            self._frame_remaining = 0
            self.defense_diagnostics = {}
            self._last_decision = 'cross-crease-ended'
            self._last_target = self.cross_crease.diagnostics.get('waypoint', plan.target)
            self.offense_diagnostics = {
                'mode': self._last_decision, 'decision': self._last_decision,
                'reason': self.cross_crease.diagnostics['outcome'],
                'target': self._last_target, 'destination': plan.target, 'waypoint': self._last_target,
                'actual_slot': controlled_slot(state.team1), 'desired_slot': plan.slot,
                'receiver': None, 'cross_crease': dict(self.cross_crease.diagnostics),
                'last_cross_crease': self.cross_crease.events[-1],
                'buttons': [0] * Buttons.INPUT_MAX,
            }
            self.defense.idle(1)
            return True
        return False

    def _cross_crease_action(self, state):
        plan = self.cross_crease.plan
        action = self.cross_crease.step(state, self.defense.frames, c_down=self._c_down)
        self.defense_diagnostics = {}
        self._last_decision = 'cross-crease-' + self.cross_crease.phase
        self._last_target = self.cross_crease.diagnostics.get('waypoint', plan.target)
        self.offense_diagnostics = {
            'mode': self._last_decision, 'decision': self._last_decision,
            'reason': (self.cross_crease.diagnostics.get('outcome')
                       or self.cross_crease.diagnostics.get('reason')
                       or self.cross_crease.diagnostics.get('release_reason', 'execute validated held-C crossing')),
            'target': self._last_target, 'destination': plan.target, 'waypoint': self._last_target,
            'actual_slot': controlled_slot(state.team1), 'desired_slot': plan.slot,
            'receiver': None, 'cross_crease': dict(self.cross_crease.diagnostics),
            'last_cross_crease': self.cross_crease.events[-1] if self.cross_crease.events else None,
            'buttons': action.tolist(),
        }
        return action, HOCKEY_INTENT_NORMAL_SHOOT if action[Buttons.INPUT_C] else HOCKEY_INTENT_NOOP

    def _choose_cross_crease(self, state, plan, one_timer):
        controller = self.cross_crease
        if controller is None or self.defense.frames < controller.retry_at:
            return False
        controller.metrics['evaluations'] += 1
        player = state.team1.get_player_by_scnum(state.engine.puck_owner)
        if not crossing_entry(state, player):
            self.offense_diagnostics['cross_crease'] = {'status': 'outside-crossing-entry'}
            controller.metrics['rejected-outside-crossing-entry'] += 1
            return False
        alternatives = {'shoot': shot_value(state, player)}
        target = self.offense.carry_target(state, player)
        if carry_clear(state, player, target):
            future = projected_state(state, target)
            if future is not None:
                alternatives['carry'] = shot_value(
                    future, future.team1.get_player_by_scnum(state.engine.puck_owner))
        if one_timer is not None:
            choices, _ = self.offense.passes(state, 'one-timer')
            if choices:
                alternatives['one-timer'] = choices[0].shot_value + min(choices[0].margin, 12)
        if plan is not None:
            mode, point, option = plan
            if option is not None:
                alternatives[mode] = option.shot_value + min(option.margin, 12)
            else:
                future = projected_state(state, point)
                if future is not None:
                    alternatives[mode] = shot_value(
                        future, future.team1.get_player_by_scnum(state.engine.puck_owner))
        crossing, diagnostics = evaluate_cross_crease(state, alternatives)
        self.offense_diagnostics['cross_crease'] = diagnostics
        for candidate in diagnostics['candidates']:
            if candidate['status'] != 'feasible':
                controller.metrics['candidate-rejected-' + candidate['status']] += 1
        if crossing is None:
            controller.metrics['rejected-' + diagnostics['status']] += 1
            return False
        self.offense.cancel()
        controller.diagnostics = diagnostics
        controller.start(crossing, state, self.defense.frames)
        return True

    def _defend(self, state):
        self.offense.cancel()
        self.offense_diagnostics = {}
        if state.engine.puck_owner >= 0:
            self._end_one_timer('possession-changed', interrupt=False)
            self._shot_until = self._setup_until = 0
        self.defense.b_down, self.defense.c_down = self._b_down, self._c_down
        action = self.defense.step(state, self._defense_elapsed)
        actual = controlled_slot(state.team1)
        player = state.team1.get_player_by_scnum(actual)
        escape = goalie_avoidance(state, player, self.defense.diagnostics['waypoint'])
        if escape is not None and not action[Buttons.INPUT_B] and self.defense.pending_check is None:
            point, details = escape
            action[4:9] = 0
            self._steer(action, player, *point)
            self.defense.c_down = False
            self.defense.diagnostics.update(
                **details, mode='goalie-avoid', decision='goalie-avoid',
                target=point, destination=point, waypoint=point, buttons=action.tolist())
        self.defense_diagnostics = self.defense.diagnostics
        self._last_target = self.defense_diagnostics['target']
        self._last_decision = self.defense_diagnostics['decision']
        mode = self.defense.diagnostics['mode']
        intent = (HOCKEY_INTENT_CHANGE_PLAYER if mode == 'switch-request'
                  else HOCKEY_INTENT_POKE_CHECK if mode == 'poke-request' else HOCKEY_INTENT_NOOP)
        return action, intent

    def predict_frame(self, state, frame_skip=4, deterministic=True):
        """Reactive defense each frame, unchanged offensive decision interval."""
        if frame_skip < 1:
            raise ValueError('Scripted decision interval must be positive')
        self._decision_interval = self.offense.decision_interval = frame_skip
        self._observe_follow_through(state)
        if self._observe_one_timer(state, self.defense.frames + 1):
            self.defense.idle(1)
            return self._encode(np.zeros(Buttons.INPUT_MAX, dtype=np.int8), HOCKEY_INTENT_NOOP)
        if self._observe_cross_crease(state):
            self._was_defending = False
            return self._encode(np.zeros(Buttons.INPUT_MAX, dtype=np.int8), HOCKEY_INTENT_NOOP)
        self._defense_elapsed = 1
        goalie_action = self._goalie_frame(state)
        if goalie_action is not None:
            self._defense_elapsed = frame_skip
            return goalie_action
        if self.cross_crease is not None and self.cross_crease.plan is not None:
            self.defense.idle(1)
            self._frame_remaining = 0
            self._was_defending = False
            self._defense_elapsed = frame_skip
            return self._encode(*self._cross_crease_action(state))
        self._observe_ordinary_pass(state)
        defending = self._defending(state)
        if self._frame_remaining == 0:
            self._frame_action = self._predict_decision(state, deterministic)
            defending = bool(self.defense_diagnostics)
            self._frame_remaining = frame_skip
        elif defending:
            self._frame_action = self._encode(*self._defend(state))
        elif self._was_defending:
            # A defensive B/C request must not become a pass/shot on recovery.
            self._frame_action = self._encode(np.zeros(Buttons.INPUT_MAX, dtype=np.int8), HOCKEY_INTENT_NOOP)
            self.defense.idle(1)
        else:
            self.defense.idle(1)
        if not defending:
            self.defense_diagnostics = {}
        self._was_defending = defending
        self._frame_remaining -= 1
        self._defense_elapsed = frame_skip
        return self._frame_action.copy()

    def _steer(self, action, player, x, y, state=None):
        escape = goalie_avoidance(state, player, (x, y)) if state is not None else None
        if escape is not None:
            (x, y), details = escape
            self._last_decision = 'goalie-avoid'
            self.offense_diagnostics.update(details)
        self._last_target = (x, y)
        dx, dy = x - player.x, y - player.y
        if abs(dx) > 3:
            action[Buttons.INPUT_RIGHT if dx > 0 else Buttons.INPUT_LEFT] = 1
        if abs(dy) > 3:
            action[Buttons.INPUT_UP if dy > 0 else Buttons.INPUT_DOWN] = 1

    def _aim(self, action):
        # Horizontal shot aim selects a goal-mouth corner at middle height.
        # Left/right stay in world coordinates for either attacking direction.
        action[Buttons.INPUT_RIGHT if self._shot_side > 0 else Buttons.INPUT_LEFT] = 1

    def _carry(self, state, player, action, *, breakaway=False):
        point, details = self.offense.fallback_carry(state, player)
        self.offense_diagnostics.update(details)
        self._last_decision = 'carry-breakaway' if breakaway and details['mode'] == 'carry' else details['mode']
        outcome = ('missing-feedback' if details['carry_safe'] is None else 'least-risk'
                   if not details['carry_safe'] else 'safe-default'
                   if details['mode'] == 'carry' else 'safe-escape')
        self.carry_metrics[outcome] = self.carry_metrics.get(outcome, 0) + 1
        self._steer(action, player, *point, state if details['carry_safe'] is None else None)
        return action, HOCKEY_INTENT_NOOP

    @staticmethod
    def _accuracy(player):
        accuracy = getattr(player, 'shot_accuracy', None)
        return 15 if accuracy is None else max(0, min(30, accuracy))

    def _one_timer_target(self, team, opponents, attack, prepare=False, state=None):
        """Rank live one-timer shots; retain the static scan without telemetry."""
        self._one_timer_flight = None
        passer = team.get_player_by_scnum(state.engine.puck_owner) if state is not None else team.get_controlled_player()
        if not self._one_timers or self.defense.frames < self._pass_at or passer.y * attack < 100:
            return None
        if state is not None and passer.passing is not None and not prepare:
            choices, details = self.offense.passes(state, 'one-timer')
            self.offense.diagnostics['one_timer_candidates'] = details
            if not choices:
                return None
            best = choices[0]
            self._one_timer_flight = best.flight_frames + pass_release_frames(team.players[best.index])
            indices = [i for i in range(len(team.players)) if team.players[i] is not passer]
            return indices.index(best.index), best.slot
        teammates = [(team.skater_scnum_base() + i, p) for i, p in enumerate(team.players)
                     if i != team.control - 1]
        for index, (slot, receiver) in enumerate(teammates):
            if (not eligible(receiver) or abs(receiver.x) > 70
                    or not 175 + max(0, 15 - self._accuracy(receiver)) < receiver.y * attack < 245):
                continue
            dx, dy = receiver.x - passer.x, receiver.y - passer.y
            length2 = dx * dx + dy * dy
            if prepare:
                if abs(dy) < 75 and receiver.y * attack >= passer.y * attack - 25:
                    return index, slot
                continue
            if not (30**2 < length2 < 150**2 and abs(dy) < 80
                    and receiver.x * passer.x <= 0 and abs(dx) > 30):
                continue
            clear = True
            for defender in opponents.players:
                if not on_ice(defender):
                    continue
                along = max(0, min(1, ((defender.x - passer.x) * dx + (defender.y - passer.y) * dy) / length2))
                if math.hypot(defender.x - passer.x - along * dx, defender.y - passer.y - along * dy) < 14:
                    clear = False
                    break
            if clear:
                return index, slot
        return None

    def _end_one_timer(self, reason, *, interrupt=True):
        if self._one_timer is None:
            return
        self.one_timer_metrics[reason] = self.one_timer_metrics.get(reason, 0) + 1
        self._one_timer = None
        self._one_timer_launched = False
        if interrupt:
            self._frame_remaining = 0
        self._last_decision = 'one-timer-ended'
        self.offense_diagnostics = {'reason': reason}

    def _observe_follow_through(self, state):
        owner = state.engine.puck_owner
        if (self._tick < self._shot_until and owns_puck(state.team1, owner)
                and owner != self._shot_slot):
            self._shot_until = 0
            self._frame_remaining = 0

    def _observe_ordinary_pass(self, state):
        request = self.offense.pending
        self.offense.observe(state, self.defense.frames)
        if (request is not None and self.offense.pending is None
                and self.offense.last_pass['outcome'] == 'recovered-by-passer'):
            self._frame_remaining = 0

    def _observe_one_timer(self, state, frame):
        if self._one_timer is None:
            return False
        _, slot, _ = self._one_timer
        if state.engine.clock_stopped:
            self._end_one_timer('play-stopped')
            return True
        attempts = state.team1.one_timer_attempts
        released = (attempts is not None and self._one_timer_attempts_before is not None
                    and attempts > self._one_timer_attempts_before)
        recorded = state.team1.stats.shots > self._one_timer_shots_before
        if state.engine.shot_player == slot and (released or recorded):
            self._end_one_timer('shot-released')
        elif state.engine.puck_owner == self._one_timer[0] and self._one_timer_launched:
            self._end_one_timer('recovered-by-passer')
        elif (owns_puck(state.team1, state.engine.puck_owner)
              and state.engine.puck_owner != self._one_timer[0]):
            self._end_one_timer('possession-changed')
        elif frame >= self._one_timer[2]:
            self._end_one_timer('timeout')
        elif state.engine.puck_owner < 0:
            self._one_timer_launched = True
        return False

    def _continue_one_timer(self, state, action):
        passer, slot, deadline = self._one_timer
        receiver = state.team1.get_player_by_scnum(slot)
        owner = state.engine.puck_owner
        if (self._one_timer_actual is None and self._one_timer_passes_before is not None
                and state.team1.pass_attempts is not None
                and state.team1.pass_attempts > self._one_timer_passes_before
                and state.engine.pass_target is not None):
            self._one_timer_actual = state.engine.pass_target
        if self._one_timer_actual is not None and self._one_timer_actual != slot:
            self._end_one_timer('receiver-mismatch')
            return None
        if (receiver is None or receiver.is_falling or self.defense.frames >= deadline
                or owner >= 0 and owner != passer):
            self._end_one_timer(
                'timeout' if self.defense.frames >= deadline else 'receiver-unavailable'
                if receiver is None or receiver.is_falling else 'possession-changed')
            return None
        self._last_decision = 'one-timer-wait'
        self.offense_diagnostics = {'desired_slot': slot, 'reason': 'wait for one-timer contact',
                                    'actual_receiver': self._one_timer_actual}
        if self.defense.frames - self._one_timer_started < HOCKEY_PASS_PRESS_FRAMES:
            aim_pass(action, state.team1.get_player_by_scnum(passer), receiver)
            action[Buttons.INPUT_B] = 1
        elif owner == passer:
            self._last_target = (receiver.x, receiver.y)
            aim_pass(action, state.team1.get_player_by_scnum(passer), receiver)
        elif not self._c_down and not receiver.is_one_timer:
            # A new C edge while the pass is free activates the ROM's automatic
            # receiver/aiming routine. NOOP + boost emits that same C in intents.
            self._last_decision = 'one-timer-shoot'
            action[Buttons.INPUT_C] = 1
        return action, HOCKEY_INTENT_NOOP

    def _decide(self, state):
        self.defense_diagnostics = {}
        self.offense_diagnostics = {}
        self._observe_ordinary_pass(state)
        if self.cross_crease is not None and self.cross_crease.plan is not None:
            self.defense.idle(self._defense_elapsed)
            return self._cross_crease_action(state)
        if self._defending(state):
            return self._defend(state)
        self.defense.idle(self._defense_elapsed)
        action = np.zeros(Buttons.INPUT_MAX, dtype=np.int8)
        team, opponents, puck = state.team1, state.team2, state.puck
        actual = controlled_slot(team)
        player = team.get_player_by_scnum(actual)
        attack = 1 if opponents.net.y > team.net.y else -1
        owner = state.engine.puck_owner
        owned = team.owns_scnum(owner)

        if self.offense.pending is not None:
            request = self.offense.pending
            self._last_decision = 'pass-flight' if request['launched'] else 'pass-release'
            self.offense_diagnostics = {
                **self.offense.diagnostics, 'mode': self._last_decision, 'decision': self._last_decision,
                'reason': 'wait for observed reception; do not boost or switch through a pass',
                'target': request['point'], 'destination': request['point'], 'waypoint': request['point'],
                'actual_slot': actual, 'desired_slot': request['receiver'], 'pending_pass': dict(request),
            }
            if request['launched'] and owner < 0 and player is not None:
                escape = goalie_avoidance(state, player, (player.x, player.y))
                if escape is not None:
                    self._steer(action, player, *escape[0])
                    self._last_decision = 'goalie-avoid'
                    self.offense_diagnostics.update(escape[1])
            return action, HOCKEY_INTENT_NOOP

        if self._one_timer is not None:
            pending = self._continue_one_timer(state, action)
            if pending is not None:
                return pending

        goalie_slot = team.goalie_scnum() if team.defense_goalie is None else team.defense_goalie
        if owner == goalie_slot:
            self._last_decision = 'goalie-outlet'
            self._shot_until = 0
            receivers = team.players[:4] if self._intents else team.players
            index = min(range(len(receivers)), key=lambda i: math.hypot(receivers[i].x - team.goalie.x, receivers[i].y - team.goalie.y))
            receiver = receivers[index]
            self._last_target = (receiver.x, receiver.y)
            aim_pass(action, team.goalie, receiver)
            action[Buttons.INPUT_B] = int(self._tick % 2 == 0)
            return action, HOCKEY_INTENT_PASS_START + index if action[0] else HOCKEY_INTENT_NOOP

        if owner >= 0 and not owned:
            self._shot_until = 0
            self._setup_until = 0
        if self._tick < self._shot_until:
            self._last_decision = 'shot-follow-through'
            released = (team.stats.shots > self._shots_before and owner != self._shot_slot
                        and state.engine.shot_player in (-1, self._shot_slot))
            if not released or player is None:
                self._aim(action)
            else:
                self._steer(action, player, player.x + self._shot_side * 40, player.y, state)
            return action, HOCKEY_INTENT_NOOP

        if owner == actual and player is not None and player is not team.goalie:
            progress = player.y * attack
            shot_y = 218 + max(0, 15 - self._accuracy(player)) * 0.4
            plan = self.offense.choose(state, self.defense.frames)
            self.offense_diagnostics = self.offense.diagnostics
            escape = goalie_avoidance(state, player, self.offense.carry_target(state, player))
            release_in_front = shot_release_in_front(
                state, player, self._decision_interval)
            self.offense_diagnostics['shot_release_model'] = (
                'native-animation-envelope' if player.shot_offsets_y is not None else 'conservative-animation-envelope')
            shot_safe = release_in_front and goalie_contact_time(
                player, opponents.goalie, clearance=GOALIE_SHOT_CLEARANCE) > GOALIE_HORIZON
            normal_finish = progress > shot_y and abs(player.x) < 48 and shot_safe
            target = self._one_timer_target(team, opponents, attack, state=state)
            if self._choose_cross_crease(state, plan, target):
                return self._cross_crease_action(state)
            if target is not None and not normal_finish:
                if self._b_down:
                    self._last_decision = 'pass-button-release'
                    return action, HOCKEY_INTENT_NOOP
                index, slot = target
                receiver = team.get_player_by_scnum(slot)
                duration = (math.ceil(self._one_timer_flight) + ONE_TIMER_CONTACT_GRACE
                            if self._one_timer_flight is not None else ONE_TIMER_TIMEOUT_FRAMES)
                self._one_timer_started = self.defense.frames
                self._one_timer = (owner, slot, self.defense.frames + duration)
                self._one_timer_launched = False
                self.one_timer_starts += 1
                self._one_timer_passes_before = team.pass_attempts
                self._one_timer_actual = None
                self._one_timer_attempts_before = team.one_timer_attempts
                self._one_timer_shots_before = team.stats.shots
                self._pass_at = self.defense.frames + ONE_TIMER_RETRY_FRAMES
                self._setup_until = 0
                self._last_decision = 'one-timer-pass'
                self._last_target = (receiver.x, receiver.y)
                self.offense_diagnostics.update(
                    desired_slot=slot, receiver=None,
                    reason='moving one-timer pass with reception margin' if player.passing is not None
                    else 'legacy one-timer geometry; motion safety unverified')
                self._last_pass_request = {
                    'frame': self.defense.frames, 'passer': owner, 'receiver': slot, 'purpose': 'one-timer',
                }
                aim_pass(action, player, receiver)
                action[Buttons.INPUT_B] = 1
                return action, HOCKEY_INTENT_PASS_START + index

            goalie_danger = progress > 175 and escape is not None
            preparing_one_timer = plan is not None and plan[0] == 'one-timer-setup'
            if normal_finish or (goalie_danger and abs(player.x) < 48 and shot_safe
                                 and not preparing_one_timer):
                self._last_decision = 'shoot'
                self._shot_side = -1 if opponents.goalie.x > 0 else 1
                self._last_target = self._shot_side * 13, opponents.net.y
                self.offense_diagnostics.update(desired_slot=actual, receiver=None,
                                                reason='finish before projected goalie contact' if goalie_danger
                                                else 'close-range finishing opportunity')
                self._aim(action)
                if self._c_down:  # A shot needs a new press, even after a checking burst.
                    return action, HOCKEY_INTENT_NOOP
                self._shot_until = self._tick + 7
                self._shots_before = team.stats.shots
                self._shot_slot = owner
                action[Buttons.INPUT_C] = 1
                return action, HOCKEY_INTENT_NORMAL_SHOOT

            if plan is not None:
                mode, point, option = plan
                self._last_decision, self._last_target = mode, point
                if option is not None:
                    if self._b_down:
                        self._last_decision = 'pass-button-release'
                        return action, HOCKEY_INTENT_NOOP
                    self.offense.start_pass(state, option, self.defense.frames, mode)
                    self._last_pass_request = dict(self.offense.last_request)
                    receiver = team.get_player_by_scnum(option.slot)
                    aim_pass(action, player, receiver)
                    action[Buttons.INPUT_B] = 1
                    indices = [i for i in range(len(team.players))
                               if team.skater_scnum_base() + i != actual]
                    return action, HOCKEY_INTENT_PASS_START + indices.index(option.index)
                # Live cuts already evaluated this exact goalie-safe route.
                if mode == 'carry-breakaway':
                    return self._carry(state, player, action, breakaway=True)
                self._steer(action, player, *point, None if mode in ('feint', 'one-timer-setup') else state)
                return action, HOCKEY_INTENT_NOOP

            # Missing live motion retains the historical setup, explicitly unverified.
            if player.passing is None and self._tick >= self._setup_at and 140 < progress < 215:
                target = self._one_timer_target(team, opponents, attack, prepare=True)
                if target is not None:
                    self._setup_slot = target[1]
                    self._setup_until = self._tick + 8
                    self._setup_at = self._tick + 48
            if self._tick < self._setup_until:
                receiver = team.get_player_by_scnum(self._setup_slot)
                if receiver is not None and not receiver.is_falling:
                    self._last_decision = 'one-timer-setup'
                    self._steer(action, player, -65 if receiver.x >= 0 else 65,
                                attack * min(215, max(progress, receiver.y * attack - 15)), state)
                    return action, HOCKEY_INTENT_NOOP
                self._setup_until = 0

            return self._carry(state, player, action)

        self._last_decision = 'recover'
        if player is None:
            return action, HOCKEY_INTENT_NOOP
        distance = math.hypot(puck.x - player.x, puck.y - player.y)
        closest = min(math.hypot(puck.x - p.x, puck.y - p.y) for p in team.players)
        if self._tick >= self._switch_at and (not team.control or closest + 20 < distance):
            self._last_decision = 'switch'
            self._switch_at = self._tick + 6
            action[Buttons.INPUT_B] = 1
            return action, HOCKEY_INTENT_CHANGE_PLAYER

        lead = min(distance / 12, 6) * 0.45
        x = max(-112, min(112, puck.x + puck.vx * lead))
        y = max(-246, min(246, puck.y + puck.vy * lead))
        self._steer(action, player, x, y, state)
        action[Buttons.INPUT_C] = int(self._tick % 2 == 0 and self._last_decision != 'goalie-avoid')
        return action, HOCKEY_INTENT_NOOP
