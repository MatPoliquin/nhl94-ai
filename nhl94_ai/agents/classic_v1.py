"""Classic V1: protect lanes, advance by safe passes, create openings and finish.

Reactive defense chooses a target then a skater; offense ranks bounded opportunities.
There are no inherited agents or learned parameters.
"""
import math
from dataclasses import asdict

import numpy as np

from nhl94_ai.agents.scheduling import ActionScheduler, ButtonState
from nhl94_ai.agents.lifecycle import (
    ActionEnd, OneTimerLifecycle, ShotLifecycle, SetupState,
    ONE_TIMER_TIMEOUT_FRAMES, ONE_TIMER_CONTACT_GRACE,
)
from nhl94_ai.agents.defense import DefenseController, eligible, on_ice, owns_puck
from nhl94_ai.agents.defense import controlled_slot
from nhl94_ai.agents.carry import carry_pad
from nhl94_ai.agents.goalie import GoalieController, outlet_options
from nhl94_ai.agents.cross_crease import CrossCreaseController, evaluate_cross_crease
from nhl94_ai.agents.deke import DekeController, evaluate_deke
from nhl94_ai.agents.offense import (
    OffenseController, carry_clear,
    goalie_avoidance, ordinary_shot_conditions, projected_state,
)
from nhl94_ai.agents.passing import pass_release_frames, shot_value
from nhl94_ai.agents.possession import PossessionOffenseController
from nhl94_ai.agents.finishing import normal_finish as evaluate_finish, one_timer_finish
from nhl94_ai.game.constants import GameConsts as Buttons
from nhl94_ai.game.geometry import aim_pass
from nhl94_ai.env.intents import (
    HOCKEY_INTENT_DPAD_ACTION_SPACE, HOCKEY_INTENT_NOOP,
    HOCKEY_INTENT_NORMAL_SHOOT, HOCKEY_INTENT_CHANGE_PLAYER,
    HOCKEY_INTENT_PASS_START, HOCKEY_INTENT_POKE_CHECK,
)
from nhl94_ai.env.actions import HOCKEY_PASS_PRESS_FRAMES



class ClassicAIV1Model:
    """Deterministic controller; shot decisions and frame-timed tactics stay distinct."""

    def __init__(self, args=None, env=None):
        self.args, self.env = args, env
        self.refinements = frozenset(getattr(args, 'classic_refinements', ()))
        self.scheduler = ActionScheduler()
        self.buttons = ButtonState()
        self.recovery_switch_at_decision = 0
        self.shot = ShotLifecycle()
        self.one_timer = OneTimerLifecycle()
        self.setup = SetupState()
        self._one_timers = getattr(args, 'one_timers', True)
        self.input_owner = 'idle'
        self._last_decision = 'init'
        self._last_target = (0, 0)
        self._intents = getattr(args, 'action_type', 'FILTERED').upper() == 'HOCKEY_INTENT_DPAD'
        self._size = len(HOCKEY_INTENT_DPAD_ACTION_SPACE) if self._intents else Buttons.INPUT_MAX
        self._last_action_preferences = np.zeros((1, self._size), dtype=np.float32)
        self.defense = DefenseController(verify_interceptions='interceptions' in self.refinements)
        self.defense_diagnostics = {}
        offense_type = OffenseController if getattr(args, 'offense_lookahead', False) else PossessionOffenseController
        self.offense = offense_type(one_timers=self._one_timers,
                                    allow_uncertified=getattr(args, 'uncertain_carry', False),
                                    chance_creation=getattr(args, 'chance_creation', False))
        self.offense.pass_timing = 'pass-timing' in self.refinements
        self.offense.carry_motion = 'carry-motion' in self.refinements
        if getattr(args, 'chance_creation', False) and (
                getattr(args, 'action_type', 'FILTERED').upper() not in ('FILTERED', 'HOCKEY_INTENT_DPAD')
                or getattr(args, 'env', 'NHL94-Genesis-v0') != 'NHL94-Genesis-v0'):
            raise ValueError('Chance-creation AI requires full-team FILTERED or HOCKEY_INTENT_DPAD controls')
        self.offense_diagnostics = {}
        cross_crease = getattr(args, 'cross_crease', False)
        if cross_crease and (getattr(args, 'action_type', 'FILTERED').upper() not in (
                'FILTERED', 'HOCKEY_INTENT_DPAD') or getattr(args, 'env', 'NHL94-Genesis-v0') != 'NHL94-Genesis-v0'):
            raise ValueError('Cross-crease AI requires full-team FILTERED or HOCKEY_INTENT_DPAD controls')
        self.cross_crease = CrossCreaseController() if cross_crease else None
        deke = getattr(args, 'deke', False)
        if deke and (getattr(args, 'action_type', 'FILTERED').upper() not in (
                'FILTERED', 'HOCKEY_INTENT_DPAD') or getattr(args, 'env', 'NHL94-Genesis-v0') != 'NHL94-Genesis-v0'):
            raise ValueError('Deke AI requires full-team FILTERED or HOCKEY_INTENT_DPAD controls')
        self.deke = DekeController() if deke else None
        self._last_pass_request = None
        self.carry_metrics = {}
        self.scheduler.action = np.zeros((1, self._size), dtype=np.int8)
        goalie_policy = getattr(args, 'goalie_policy', 'off')
        if goalie_policy not in ('off', 'selective', 'always'):
            raise ValueError('goalie_policy must be off, selective or always')
        if goalie_policy != 'off' and getattr(args, 'action_type', 'FILTERED').upper() != 'FILTERED':
            raise ValueError('Manual goalie AI requires FILTERED buttons')
        self.goalie = None if goalie_policy == 'off' else GoalieController(
            goalie_policy, pass_timing=self.offense.pass_timing)
        self.goalie_diagnostics = {}

    def predict(self, _observation, deterministic=True):
        return np.zeros((1, self._size), dtype=np.int8), None

    def get_action_preferences(self, _observation=None):
        return self._last_action_preferences

    def learn(self, *args, **kwargs):
        raise NotImplementedError('ClassicAIV1 is inference-only; use nhl94 play --agent classic-v1.')

    def save(self, *args, **kwargs):
        raise NotImplementedError('ClassicAIV1 does not produce a trainable checkpoint.')

    def _observe_actions(self, state, elapsed, *, native=False):
        """Observe commitments in the established order before selecting input."""
        self._observe_follow_through(state)
        if self._observe_one_timer(state, self.scheduler.frames + elapsed):
            self._idle(elapsed)
            return True
        if self._observe_cross_crease(state) or self._observe_deke(state):
            if native:
                self.scheduler.was_defending = False
            return True
        return False

    def _neutral(self, owner):
        self.input_owner = owner
        return self._encode(np.zeros(Buttons.INPUT_MAX, dtype=np.int8), HOCKEY_INTENT_NOOP)

    def predict_game_state(self, state, deterministic=True):
        if self._observe_actions(state, self.scheduler.elapsed):
            return self._neutral('transition')
        goalie_action = self._goalie_frame(state)
        if goalie_action is not None:
            return goalie_action
        return self._predict_decision(state, deterministic)

    def _goalie_frame(self, state):
        if self.goalie is None:
            return None
        blocked = (self.one_timer.pending is not None or self.offense.pass_action.pending is not None
                   or self.cross_crease is not None and self.cross_crease.plan is not None
                   or self.deke is not None and self.deke.plan is not None
                   or self.shot.active(self.scheduler.decisions) or self.defense.pending_check is not None
                   or self.defense.pending_switch is not None)
        action = self.goalie.step(state, blocked=blocked)
        self.goalie_diagnostics = self.goalie.diagnostics
        if action is None:
            return None
        self.input_owner = 'goalie'
        self._idle(1)
        self.defense_diagnostics = self.offense_diagnostics = {}
        self.end_one_timer('goalie-control')
        self.shot.cancel()
        self.setup.until_decision = 0
        self.setup.slot = None
        self.offense.cancel()
        self.scheduler.interrupt()
        self.scheduler.was_defending = False
        self._last_decision = 'goalie-' + self.goalie.phase
        self._last_target = self.goalie_diagnostics['target']
        return self._encode(action, HOCKEY_INTENT_NOOP)

    def _predict_decision(self, state, deterministic=True):
        self.scheduler.decisions += 1
        previous_decision = self._last_decision
        action, intent = self._decide(state)
        if self.offense.chance_owner is not None and self._last_decision != 'create-chance':
            self.offense.chance_owner = None
            self.offense.chance_until = 0
            if previous_decision == 'create-chance':
                self.offense.chance_at = self.scheduler.frames + 72
            self.offense_diagnostics['chance_cancelled'] = 'live action replaced the predicted setup'
        if not self.defense_diagnostics:
            actual = controlled_slot(state.team1)
            self.offense_diagnostics = {
                **self.offense_diagnostics, 'phase': 'offense', 'mode': self._last_decision,
                'decision': self._last_decision, 'target': self._last_target,
                'destination': self._last_target, 'waypoint': self._last_target, 'actual_slot': actual,
                'desired_slot': self.offense_diagnostics.get('desired_slot', actual),
                'reason': self.offense_diagnostics.get('reason', self.offense_diagnostics.get('status', self._last_decision)),
                'buttons': action.tolist(), 'last_pass': self.offense.pass_action.last_pass,
            }
            self.offense_diagnostics['teammate_scores'] = self._teammate_scores(state)
        return self._encode(action, intent)

    def _teammate_scores(self, state):
        ordinary = {item['slot']: item for item in self.offense_diagnostics.get('candidates', ())}
        one_timers = {item['slot']: item for item in self.offense_diagnostics.get('one_timer_candidates', ())}
        selected = self.offense_diagnostics.get('desired_slot')
        return [
            {'slot': slot, 'selected': slot == selected,
             'pass': ordinary.get(slot, {'status': 'not-evaluated'}),
             'one_timer': one_timers.get(slot)}
            for index, _ in enumerate(state.team1.players)
            if (slot := state.team1.skater_scnum_base() + index) != state.engine.puck_owner
        ]

    def _encode(self, action, intent):
        self.buttons.c_down = bool(action[Buttons.INPUT_C])
        self.buttons.b_down = bool(action[Buttons.INPUT_B])
        if self._intents:
            action = np.asarray([intent, *action[4:8], int(action[8] and intent == HOCKEY_INTENT_NOOP)], dtype=np.int8)
        self._last_action_preferences = action[None].astype(np.float32)
        return action[None]

    def _defending(self, state):
        owner = state.engine.puck_owner
        if owns_puck(state.team1, owner):
            return False
        return owner >= 0 or (self.one_timer.pending is None and self.offense.pass_action.pending is None
                              and (self.cross_crease is None or self.cross_crease.plan is None)
                              and (self.deke is None or self.deke.plan is None)
                              and not self.shot.active(self.scheduler.decisions))

    def _observe_cross_crease(self, state):
        if self.cross_crease is None:
            return False
        if state.numPlayers != 5:
            raise ValueError('Cross-crease AI requires full-team NHL94')
        plan = self.cross_crease.plan
        self.cross_crease.observe(state, self.scheduler.frames)
        if plan is not None and self.cross_crease.plan is None:
            self.scheduler.interrupt()
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
            self._idle(1)
            return True
        return False

    def _cross_crease_action(self, state):
        self.input_owner = 'cross-crease'
        plan = self.cross_crease.plan
        action = self.cross_crease.step(state, self.scheduler.frames, c_down=self.buttons.c_down)
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

    def _finishing_alternatives(self, state, plan, one_timer):
        player = state.team1.get_player_by_scnum(state.engine.puck_owner)
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
        return alternatives

    def _choose_finisher(self, state, plan, one_timer):
        controllers = [('cross_crease', self.cross_crease, evaluate_cross_crease),
                       ('deke', self.deke, evaluate_deke)]
        active = [(name, controller, evaluate) for name, controller, evaluate in controllers
                  if controller is not None and self.scheduler.frames >= controller.retry_at]
        if not active:
            return None
        alternatives = self._finishing_alternatives(state, plan, one_timer)
        choices = []
        for name, controller, evaluate in active:
            controller.metrics['evaluations'] += 1
            finish, diagnostics = evaluate(state, alternatives)
            self.offense_diagnostics[name] = diagnostics
            for candidate in diagnostics['candidates']:
                if candidate['status'] != 'feasible':
                    controller.metrics['candidate-rejected-' + candidate['status']] += 1
            if finish is None:
                controller.metrics['rejected-' + diagnostics['status']] += 1
            else:
                choices.append((name, controller, finish))
        if not choices:
            return None
        name, controller, finish = max(choices, key=lambda row: row[2].value)
        self.offense.cancel()
        controller.diagnostics = self.offense_diagnostics[name]
        controller.start(finish, state, self.scheduler.frames)
        return name

    def _observe_deke(self, state):
        if self.deke is None:
            return False
        if state.numPlayers != 5:
            raise ValueError('Deke AI requires full-team NHL94')
        plan = self.deke.plan
        self.deke.observe(state, self.scheduler.frames)
        if plan is None or self.deke.plan is not None:
            return False
        self.scheduler.interrupt()
        self.defense_diagnostics = {}
        self._last_decision = 'deke-ended'
        self._last_target = plan.target
        self.offense_diagnostics = {
            'mode': self._last_decision, 'decision': self._last_decision,
            'reason': self.deke.diagnostics['outcome'], 'target': plan.target,
            'destination': plan.target, 'waypoint': plan.target,
            'actual_slot': controlled_slot(state.team1), 'desired_slot': plan.slot,
            'receiver': None, 'deke': dict(self.deke.diagnostics), 'last_deke': self.deke.events[-1],
            'buttons': [0] * Buttons.INPUT_MAX,
        }
        self._idle(1)
        return True

    def _deke_action(self, state):
        self.input_owner = 'deke'
        plan = self.deke.plan
        action = self.deke.step(state, self.scheduler.frames, c_down=self.buttons.c_down)
        self.defense_diagnostics = {}
        self._last_decision = 'deke-' + self.deke.phase
        self._last_target = self.deke.diagnostics.get('waypoint', plan.bait)
        self.offense_diagnostics = {
            'mode': self._last_decision, 'decision': self._last_decision,
            'reason': self.deke.diagnostics.get('outcome', 'execute observed goalie-bait maneuver'),
            'target': self._last_target, 'destination': plan.target, 'waypoint': self._last_target,
            'actual_slot': controlled_slot(state.team1), 'desired_slot': plan.slot,
            'receiver': None, 'deke': dict(self.deke.diagnostics),
            'last_deke': self.deke.events[-1] if self.deke.events else None, 'buttons': action.tolist(),
        }
        return action, HOCKEY_INTENT_NORMAL_SHOOT if action[Buttons.INPUT_C] else HOCKEY_INTENT_NOOP

    def _idle(self, elapsed):
        """Advance the shared clock while releasing defensive action state."""
        self.defense.idle(frame=self.scheduler.advance(elapsed))

    def _defend(self, state):
        self.input_owner = 'defense'
        self.offense.cancel()
        self.offense_diagnostics = {}
        if state.engine.puck_owner >= 0:
            self.end_one_timer('possession-changed', interrupt=False)
            self.shot.cancel()
            self.setup.until_decision = 0
        self.defense.b_down, self.defense.c_down = self.buttons.b_down, self.buttons.c_down
        frame = self.scheduler.advance(self.scheduler.elapsed)
        action = self.defense.step(state, frame=frame)
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

    def _prepare_exclusive_frame(self):
        self._idle(1)
        self.scheduler.interrupt()
        self.scheduler.was_defending = False
        self.scheduler.elapsed = self.scheduler.interval

    def predict_frame(self, state, frame_skip=4, deterministic=True):
        """Observe, hand off exclusive actions, then run/repeat the tactical input.

        Goalies and finishers retain their existing precedence. Ordinary pass
        correction observes the old frame before advancing; finisher execution
        observes the advanced frame. Those boundaries are intentionally distinct.
        """
        self.scheduler.configure(frame_skip)
        self.offense.decision_interval = frame_skip
        if self._observe_actions(state, 1, native=True):
            return self._neutral('transition')
        self.scheduler.elapsed = 1
        goalie_action = self._goalie_frame(state)
        if goalie_action is not None:
            self.scheduler.elapsed = frame_skip
            return goalie_action
        for controller, execute in ((self.cross_crease, self._cross_crease_action),
                                    (self.deke, self._deke_action)):
            if controller is not None and controller.plan is not None:
                self._prepare_exclusive_frame()
                return self._encode(*execute(state))
        if self.offense.chance_owner is not None and (
                state.engine.puck_owner != self.offense.chance_owner
                or self.scheduler.frames + 1 >= self.offense.chance_until):
            self.scheduler.interrupt()
        self._observe_ordinary_pass(state)
        reception = self._receive_pass(state)
        if reception is not None:
            self._prepare_exclusive_frame()
            return self._encode(*reception)
        return self._scheduled_frame(state, deterministic)

    def _scheduled_frame(self, state, deterministic):
        defending = self._defending(state)
        if self.scheduler.remaining == 0:
            self.scheduler.cache(self._predict_decision(state, deterministic))
            defending = bool(self.defense_diagnostics)
        elif defending:
            self.scheduler.action = self._encode(*self._defend(state))
        elif self.scheduler.was_defending:
            # Release defensive B/C before their meaning changes on recovery.
            self.scheduler.action = self._neutral('handoff')
            self._idle(1)
        else:
            self._idle(1)
        if not defending:
            self.defense_diagnostics = {}
        self.scheduler.was_defending = defending
        self.scheduler.remaining -= 1
        self.scheduler.elapsed = self.scheduler.interval
        if self._last_decision == 'shoot' and self.scheduler.frames >= self.shot.hold_until_frame:
            self.scheduler.action = self.scheduler.action.copy()
            if self._intents:
                self.scheduler.action[0, 0] = HOCKEY_INTENT_NOOP
            else:
                self.scheduler.action[0, Buttons.INPUT_C] = 0
            self.buttons.c_down = False
        return self.scheduler.action.copy()

    def _goalie_escape(self, state, player, target):
        return goalie_avoidance(state, player, target, decision_interval=(
            self.scheduler.interval if self.offense.uses_lookahead else None))

    def _steer(self, action, player, x, y, state=None):
        escape = self._goalie_escape(state, player, (x, y)) if state is not None else None
        if escape is not None:
            (x, y), details = escape
            self._last_decision = 'goalie-avoid'
            self.offense_diagnostics.update(details)
        self._last_target = (x, y)
        dx, dy = carry_pad(player, (x, y))
        if dx:
            action[Buttons.INPUT_RIGHT if dx > 0 else Buttons.INPUT_LEFT] = 1
        if dy:
            action[Buttons.INPUT_UP if dy > 0 else Buttons.INPUT_DOWN] = 1

    def _carry(self, state, player, action, *, breakaway=False):
        point, details = self.offense.fallback_carry(state, player)
        self.offense_diagnostics.update(details)
        self._last_decision = 'carry-breakaway' if breakaway and details['mode'] == 'carry' else details['mode']
        estimated = details.get('carry_pressure_model') == 'arrival-estimate'
        outcome = ('estimated-clear' if details.get('carry_estimated_clear') else 'estimated-fallback') if estimated else (
                   'missing-feedback' if details['carry_safe'] is None else
                   'uncertified-viable' if not details['carry_safe'] and details.get('carry_viable') else
                   'least-risk' if not details['carry_safe'] else 'safe-default'
                   if details['mode'] == 'carry' else 'safe-escape')
        self.carry_metrics[outcome] = self.carry_metrics.get(outcome, 0) + 1
        self._steer(action, player, *point, state if details['carry_safe'] is None and not estimated else None)
        return action, HOCKEY_INTENT_NOOP

    @staticmethod
    def _accuracy(player):
        accuracy = getattr(player, 'shot_accuracy', None)
        return 15 if accuracy is None else max(0, min(30, accuracy))

    def _one_timer_target(self, team, opponents, attack, prepare=False, state=None):
        """Rank live one-timer shots; retain the static scan without telemetry."""
        self.one_timer.flight_frames = None
        self.one_timer.option = None
        passer = team.get_player_by_scnum(state.engine.puck_owner) if state is not None else team.get_controlled_player()
        if not self._one_timers or self.scheduler.frames < self.one_timer.retry_at_frame or passer.y * attack < 100:
            return None
        if state is not None and passer.passing is not None and not prepare:
            choices, details = self.offense.passes(state, 'one-timer')
            self.offense.diagnostics['one_timer_candidates'] = details
            if not choices:
                return None
            best = choices[0]
            self.one_timer.option = best
            self.one_timer.flight_frames = best.flight_frames + pass_release_frames(team.players[best.index])
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

    def _apply_one_timer_end(self, event):
        if event is None:
            return False
        if event.interrupt:
            self.scheduler.interrupt()
        self._last_decision = 'one-timer-ended'
        self.offense_diagnostics = {'reason': event.reason}
        return event.neutral_frame

    def end_one_timer(self, reason, *, interrupt=True):
        """Finish once, including explicit benchmark/period boundaries."""
        self._apply_one_timer_end(self.one_timer.finish(reason, interrupt=interrupt))

    def _observe_follow_through(self, state):
        if self.shot.observe(state, self.scheduler.decisions):
            self.scheduler.interrupt()

    def _observe_ordinary_pass(self, state):
        event = self.offense.observe(state, self.scheduler.frames)
        if event is not None and (event.reason == 'recovered-by-passer' or self.offense.uses_lookahead
                                  and event.reason in ('received', 'other-receiver')):
            self.scheduler.interrupt()

    def _receive_pass(self, state):
        step = self.offense.pass_action.receive(state, self.scheduler.frames, b_down=self.buttons.b_down)
        if isinstance(step, ActionEnd):
            self.scheduler.interrupt()
            return None
        if step is None:
            return None
        self.input_owner = 'pass'
        self._last_decision, self._last_target = step.decision, step.target
        self.defense_diagnostics = {}
        self.offense_diagnostics = step.diagnostics
        return step.buttons, step.intent

    def _observe_one_timer(self, state, frame):
        return self._apply_one_timer_end(self.one_timer.observe(state, frame))

    def _continue_one_timer(self, state):
        self.input_owner = 'one-timer'
        step = self.one_timer.step(state, self.scheduler.frames, c_down=self.buttons.c_down)
        if isinstance(step, ActionEnd):
            self._apply_one_timer_end(step)
            return None
        self._last_decision = step.decision
        if step.target is not None:
            self._last_target = step.target
        self.offense_diagnostics = step.diagnostics
        return step.buttons, step.intent

    def _decide(self, state):
        self.defense_diagnostics = {}
        self.offense_diagnostics = {}
        self._observe_ordinary_pass(state)
        if self.cross_crease is not None and self.cross_crease.plan is not None:
            self._idle(self.scheduler.elapsed)
            return self._cross_crease_action(state)
        if self.deke is not None and self.deke.plan is not None:
            self._idle(self.scheduler.elapsed)
            return self._deke_action(state)
        if self._defending(state):
            return self._defend(state)
        self._idle(self.scheduler.elapsed)
        action = np.zeros(Buttons.INPUT_MAX, dtype=np.int8)
        team, opponents = state.team1, state.team2
        actual = controlled_slot(team)
        player = team.get_player_by_scnum(actual)
        attack = 1 if opponents.net.y > team.net.y else -1
        owner = state.engine.puck_owner
        owned = team.owns_scnum(owner)

        if self.offense.pass_action.pending is not None:
            return self._continue_pass(state, action, player, actual)

        if self.one_timer.pending is not None:
            pending = self._continue_one_timer(state)
            if pending is not None:
                return pending

        goalie_slot = team.goalie_scnum() if team.defense_goalie is None else team.defense_goalie
        if owner == goalie_slot:
            return self._goalie_outlet(state, action)

        if owner >= 0 and not owned:
            self.shot.cancel()
            self.setup.until_decision = 0
        if self.shot.active(self.scheduler.decisions):
            return self._continue_shot(state, action, player)

        if owner == actual and player is not None and player is not team.goalie:
            return self._attack(state, action, player, actual, attack)
        return self._recover(state, action, player)

    def _continue_pass(self, state, action, player, actual):
        self.input_owner = 'pass'
        team, owner = state.team1, state.engine.puck_owner
        request = self.offense.pass_action.pending
        self._last_decision = 'pass-flight' if request.launched else 'pass-release'
        self.offense_diagnostics = {
            **self.offense.diagnostics, 'mode': self._last_decision, 'decision': self._last_decision,
            'reason': 'wait for observed reception; do not boost or switch through a pass',
            'target': request.point, 'destination': request.point, 'waypoint': request.point,
            'actual_slot': actual, 'desired_slot': request.receiver, 'pending_pass': request.snapshot(),
        }
        if (owner == request.passer and not request.launched
                and self.scheduler.frames - request.frame < HOCKEY_PASS_PRESS_FRAMES):
            receiver = team.get_player_by_scnum(request.receiver)
            if receiver is not None:
                aim_pass(action, team.get_player_by_scnum(owner), receiver)
                action[Buttons.INPUT_B] = 1
                index = team.players.index(receiver)
                if owner != team.goalie_scnum():
                    index = [i for i in range(len(team.players))
                             if team.skater_scnum_base() + i != owner].index(index)
                return action, HOCKEY_INTENT_PASS_START + index
        if request.launched and owner < 0 and player is not None:
            escape = self._goalie_escape(state, player, (player.x, player.y))
            if escape is not None:
                self._steer(action, player, *escape[0])
                self._last_decision = 'goalie-avoid'
                self.offense_diagnostics.update(escape[1])
        return action, HOCKEY_INTENT_NOOP

    def _goalie_outlet(self, state, action):
        self.input_owner = 'goalie-outlet'
        team = state.team1
        self.shot.cancel()
        options, candidates = outlet_options(
            state, decision_interval=self.scheduler.interval, limit=4 if self._intents else None,
            release_prediction=self.offense.pass_timing)
        self.offense_diagnostics = {'outlet_candidates': candidates}
        if self.buttons.b_down or not options:
            self._last_decision = 'goalie-hold'
            self._last_target = (team.goalie.x, team.goalie.y)
            self.offense_diagnostics['reason'] = (
                'release B before requesting an outlet' if self.buttons.b_down else
                'no evaluated outlet; allow the automatic goalie to cover')
            return action, HOCKEY_INTENT_NOOP
        option = options[0]
        receiver = team.players[option.index]
        self._last_decision = 'goalie-outlet'
        self._last_target = (receiver.x, receiver.y)
        self.offense.start_pass(state, option, self.scheduler.frames, 'goalie-outlet')
        self._last_pass_request = dict(self.offense.pass_action.last_request)
        self.offense_diagnostics.update(desired_slot=option.slot, reason='evaluated safe goalie outlet')
        aim_pass(action, team.goalie, receiver)
        action[Buttons.INPUT_B] = 1
        return action, HOCKEY_INTENT_PASS_START + option.index

    def _continue_shot(self, state, action, player):
        self.input_owner = 'shot'
        self._last_decision = 'shot-follow-through'
        released = self.shot.released(state)
        if not released or player is None:
            self.shot.aim(action)
        else:
            self._steer(action, player, player.x + self.shot.side * 40, player.y, state)
        return action, HOCKEY_INTENT_NOOP

    def _attack(self, state, action, player, actual, attack):
        self.input_owner = 'offense'
        team, opponents = state.team1, state.team2
        owner = state.engine.puck_owner
        progress = player.y * attack
        self.offense.one_timer_at = self.one_timer.retry_at_frame
        plan = self.offense.choose(state, self.scheduler.frames)
        self.offense_diagnostics = self.offense.diagnostics
        escape = self._goalie_escape(state, player, self.offense.carry_target(state, player))
        self.offense_diagnostics['shot_release_model'] = (
            'native-animation-envelope' if player.shot_offsets_y is not None else 'conservative-animation-envelope')
        normal_finish, early_finish = ordinary_shot_conditions(
            state, player, self.scheduler.interval, escape)
        target = self._one_timer_target(team, opponents, attack, state=state)
        finish = evaluate_finish(state, player, self.scheduler.interval) if (
            'finishing' in self.refinements and (normal_finish or early_finish)) else None
        one_timer_finish_option = (one_timer_finish(state, self.one_timer.option)
                                   if 'finishing' in self.refinements else None)
        if finish is not None:
            self.offense_diagnostics['normal_finish'] = asdict(finish)
        if one_timer_finish_option is not None:
            self.offense_diagnostics['one_timer_finish'] = asdict(one_timer_finish_option)
        if 'finishing' in self.refinements:
            self.offense_diagnostics['finish_value_model'] = 'release-geometry-not-scoring-probability'
        prefer_one_timer = target is not None and (not normal_finish or (
            finish is not None and one_timer_finish_option is not None
            and one_timer_finish_option.value > finish.value + 4))
        finisher = self._choose_finisher(state, plan, target)
        if finisher == 'cross_crease':
            return self._cross_crease_action(state)
        if finisher == 'deke':
            return self._deke_action(state)
        if prefer_one_timer:
            self.input_owner = 'one-timer'
            if self.buttons.b_down:
                self._last_decision = 'pass-button-release'
                return action, HOCKEY_INTENT_NOOP
            index, slot = target
            receiver = team.get_player_by_scnum(slot)
            duration = (math.ceil(self.one_timer.flight_frames) + ONE_TIMER_CONTACT_GRACE
                        if self.one_timer.flight_frames is not None else ONE_TIMER_TIMEOUT_FRAMES)
            self.one_timer.start(state, slot, self.scheduler.frames, duration)
            self.setup.until_decision = 0
            self._last_decision = 'one-timer-pass'
            self._last_target = (receiver.x, receiver.y)
            self.offense_diagnostics.update(
                desired_slot=slot, receiver=None,
                reason='moving one-timer pass with reception margin' if player.passing is not None
                else 'legacy one-timer geometry; motion safety unverified')
            self._last_pass_request = {
                'frame': self.scheduler.frames, 'passer': owner, 'receiver': slot, 'purpose': 'one-timer',
            }
            aim_pass(action, player, receiver)
            action[Buttons.INPUT_B] = 1
            return action, HOCKEY_INTENT_PASS_START + index

        goalie_danger = progress > 175 and escape is not None
        preparing_one_timer = plan is not None and plan[0] == 'one-timer-setup'
        if normal_finish or (early_finish and not preparing_one_timer):
            self.input_owner = 'shot'
            self._last_decision = 'shoot'
            self.shot.side = finish.side if finish is not None else (-1 if opponents.goalie.x > 0 else 1)
            self._last_target = self.shot.side * 13, opponents.net.y
            self.offense_diagnostics.update(desired_slot=actual, receiver=None,
                                            reason='finish before projected goalie contact' if goalie_danger
                                            else 'close-range finishing opportunity')
            self.shot.aim(action)
            if self.buttons.c_down:  # A shot needs a new press, even after a checking burst.
                return action, HOCKEY_INTENT_NOOP
            self.shot.start(state, self.scheduler.decisions, self.scheduler.frames,
                            finish.hold_frames if finish is not None else self.scheduler.interval)
            action[Buttons.INPUT_C] = 1
            return action, HOCKEY_INTENT_NORMAL_SHOOT

        if plan is not None:
            mode, point, option = plan
            self._last_decision, self._last_target = mode, point
            if option is not None:
                self.input_owner = 'pass'
                if self.buttons.b_down:
                    self._last_decision = 'pass-button-release'
                    return action, HOCKEY_INTENT_NOOP
                self.offense.start_pass(state, option, self.scheduler.frames, mode)
                self._last_pass_request = dict(self.offense.pass_action.last_request)
                receiver = team.get_player_by_scnum(option.slot)
                aim_pass(action, player, receiver)
                action[Buttons.INPUT_B] = 1
                indices = [i for i in range(len(team.players))
                           if team.skater_scnum_base() + i != actual]
                return action, HOCKEY_INTENT_PASS_START + indices.index(option.index)
            # Live cuts already evaluated this exact goalie-safe route.
            if mode == 'carry-breakaway':
                return self._carry(state, player, action, breakaway=True)
            self._steer(action, player, *point, None if mode in (
                'feint', 'one-timer-setup', 'carry-opportunity', 'create-chance') else state)
            return action, HOCKEY_INTENT_NOOP

        # Missing live motion retains the historical setup, explicitly unverified.
        if player.passing is None and self.scheduler.decisions >= self.setup.retry_at_decision and 140 < progress < 215:
            target = self._one_timer_target(team, opponents, attack, prepare=True)
            if target is not None:
                self.setup.slot = target[1]
                self.setup.until_decision = self.scheduler.decisions + 8
                self.setup.retry_at_decision = self.scheduler.decisions + 48
        if self.scheduler.decisions < self.setup.until_decision:
            receiver = team.get_player_by_scnum(self.setup.slot)
            if receiver is not None and not receiver.is_falling:
                self._last_decision = 'one-timer-setup'
                self._steer(action, player, -65 if receiver.x >= 0 else 65,
                            attack * min(215, max(progress, receiver.y * attack - 15)), state)
                return action, HOCKEY_INTENT_NOOP
            self.setup.until_decision = 0

        return self._carry(state, player, action)

    def _recover(self, state, action, player):
        self.input_owner = 'recovery'
        team, puck = state.team1, state.puck
        self._last_decision = 'recover'
        if player is None:
            return action, HOCKEY_INTENT_NOOP
        distance = math.hypot(puck.x - player.x, puck.y - player.y)
        closest = min(math.hypot(puck.x - p.x, puck.y - p.y) for p in team.players)
        if self.scheduler.decisions >= self.recovery_switch_at_decision and (not team.control or closest + 20 < distance):
            self._last_decision = 'switch'
            self.recovery_switch_at_decision = self.scheduler.decisions + 6
            action[Buttons.INPUT_B] = 1
            return action, HOCKEY_INTENT_CHANGE_PLAYER

        lead = min(distance / 12, 6) * 0.45
        x = max(-112, min(112, puck.x + puck.vx * lead))
        y = max(-246, min(246, puck.y + puck.vy * lead))
        self._steer(action, player, x, y, state)
        action[Buttons.INPUT_C] = int(self.scheduler.decisions % 2 == 0 and self._last_decision != 'goalie-avoid')
        return action, HOCKEY_INTENT_NOOP
