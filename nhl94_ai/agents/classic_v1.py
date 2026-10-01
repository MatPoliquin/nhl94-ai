"""Classic V1: protect lanes, advance by safe passes, create openings and finish.

Reactive defense chooses a target then a skater; offense ranks bounded opportunities.
There are no inherited agents or learned parameters.
"""
import math

import numpy as np

from nhl94_ai.agents.defense import DefenseController, owns_puck
from nhl94_ai.agents.defense import controlled_slot
from nhl94_ai.agents.offense import (
    GOALIE_HORIZON, GOALIE_SHOT_CLEARANCE, OffenseController, goalie_avoidance, goalie_contact_time,
)
from nhl94_ai.game.constants import GameConsts as Buttons
from nhl94_ai.game.geometry import aim_pass
from nhl94_ai.env.intents import (
    HOCKEY_INTENT_DPAD_ACTION_SPACE, HOCKEY_INTENT_NOOP,
    HOCKEY_INTENT_NORMAL_SHOOT, HOCKEY_INTENT_CHANGE_PLAYER,
    HOCKEY_INTENT_PASS_START, HOCKEY_INTENT_POKE_CHECK,
)


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
        self._last_pass_request = None
        self._one_timer_passes_before = None
        self._one_timer_actual = None
        self._frame_remaining = 0
        self._frame_action = np.zeros((1, self._size), dtype=np.int8)
        self._defense_elapsed = 4
        self._was_defending = False

    def predict(self, _observation, deterministic=True):
        return np.zeros((1, self._size), dtype=np.int8), None

    def get_action_preferences(self, _observation=None):
        return self._last_action_preferences

    def learn(self, *args, **kwargs):
        raise NotImplementedError('ClassicAIV1 is inference-only; use nhl94 play --agent classic-v1.')

    def save(self, *args, **kwargs):
        raise NotImplementedError('ClassicAIV1 does not produce a trainable checkpoint.')

    def predict_game_state(self, state, deterministic=True):
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
                              and self._tick >= self._shot_until)

    def _defend(self, state):
        self.offense.cancel()
        self.offense_diagnostics = {}
        if state.engine.puck_owner >= 0:
            self._one_timer = None
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
        self._defense_elapsed = 1
        self.offense.observe(state, self.defense.frames)
        defending = self._defending(state)
        if self._frame_remaining == 0:
            self._frame_action = self.predict_game_state(state, deterministic)
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

    @staticmethod
    def _accuracy(player):
        accuracy = getattr(player, 'shot_accuracy', None)
        return 15 if accuracy is None else max(0, min(30, accuracy))

    def _one_timer_target(self, team, opponents, attack, prepare=False, state=None):
        """Rank live one-timer shots; retain the static scan without telemetry."""
        passer = team.get_player_by_scnum(state.engine.puck_owner) if state is not None else team.get_controlled_player()
        if not self._one_timers or self._tick < self._pass_at or passer.y * attack < 100:
            return None
        if state is not None and passer.passing is not None and not prepare:
            choices, details = self.offense.passes(state, 'one-timer')
            self.offense.diagnostics['one_timer_candidates'] = details
            if not choices:
                return None
            best = choices[0]
            indices = [i for i in range(len(team.players)) if team.players[i] is not passer]
            return indices.index(best.index), best.slot
        teammates = [(team.skater_scnum_base() + i, p) for i, p in enumerate(team.players)
                     if i != team.control - 1]
        for index, (slot, receiver) in enumerate(teammates):
            if (receiver.is_falling or abs(receiver.x) > 70
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
                along = max(0, min(1, ((defender.x - passer.x) * dx + (defender.y - passer.y) * dy) / length2))
                if math.hypot(defender.x - passer.x - along * dx, defender.y - passer.y - along * dy) < 14:
                    clear = False
                    break
            if clear:
                return index, slot
        return None

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
            self._one_timer = None
            return None
        if (receiver is None or receiver.is_falling or self._tick >= deadline
                or owner >= 0 and owner != passer):
            self._one_timer = None
            return None
        self._last_decision = 'one-timer-wait'
        self.offense_diagnostics = {'desired_slot': slot, 'reason': 'wait for one-timer contact',
                                    'actual_receiver': self._one_timer_actual}
        if owner == passer:
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
        self.offense.observe(state, self.defense.frames)
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
            shot_safe = goalie_contact_time(
                player, opponents.goalie, clearance=GOALIE_SHOT_CLEARANCE) > GOALIE_HORIZON
            normal_finish = progress > shot_y and abs(player.x) < 48 and shot_safe
            target = self._one_timer_target(team, opponents, attack, state=state)
            if target is not None and not normal_finish:
                if self._b_down:
                    self._last_decision = 'pass-button-release'
                    return action, HOCKEY_INTENT_NOOP
                index, slot = target
                receiver = team.get_player_by_scnum(slot)
                self._one_timer = (owner, slot, self._tick + 18)
                self._one_timer_passes_before = team.pass_attempts
                self._one_timer_actual = None
                self._pass_at = self._tick + 24
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
            if normal_finish or (goalie_danger and abs(player.x) < 48 and shot_safe and not preparing_one_timer):
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

            self._last_decision = 'carry'
            self._steer(action, player, *self.offense.carry_target(state, player), state)
            if self._last_decision != 'goalie-avoid':
                self.offense_diagnostics['reason'] = (
                    'missing feedback; retain legacy carry/setup' if player.passing is None
                    else 'close to shot range; retain puck for finishing'
                    if self.offense_diagnostics.get('status') == 'close-to-finish'
                    else 'no safer valuable pass or cut; carry toward the slot')
            return action, HOCKEY_INTENT_NOOP

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
