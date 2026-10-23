"""Native teammate controllers, one shared tactical plan, independent execution state."""
from copy import copy
import math

import numpy as np

from nhl94_ai.agents.base import AgentOutput
from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.agents.classic_config import decision_state
from nhl94_ai.agents.defense import defensive_steering
from nhl94_ai.game.actors import controllable_skater
from nhl94_ai.game.motion import arrival_time
from nhl94_ai.agents.teammate_receiving import SharedReceiving
from nhl94_ai.agents.teammate_support import SupportPlanner
from nhl94_ai.agents.teammate_faceoff import FaceoffInput
from nhl94_ai.agents.teammate_context import TeamControlContext
from nhl94_ai.agents.teammate_offpuck import movement_action, SharedRoles
from nhl94_ai.agents.teammate_arbitration import BArbitrator
from nhl94_ai.agents.teammate_pressure import SharedPressure, TRAP_ROLES
from nhl94_ai.env.target_control import project_target
from nhl94_ai.game.constants import GameConsts as Buttons
from nhl94_ai.game.ram import teammate_controller_info
from nhl94_ai.game.state import NHL94GameState


class TeammateCoordinator:
    """Classic handles the carrier; a shared plan positions the other skater.

    Selection changes are requests through B, never gameplay RAM writes. Roles
    follow actual selections and possession, with hysteresis for loose-puck races.
    """
    input_schema = 'teammate-game-state-v1'
    action_schema = 'two-genesis-controllers-v1'

    def __init__(self, args=None, *, strategy='coordinated'):
        if strategy not in ('coordinated', 'independent'):
            raise ValueError('Team strategy must be coordinated or independent')
        self.args = copy(args)
        self.side = 2 if getattr(args, 'side', 'home') == 'away' else 1
        self.team_size = getattr(args, 'team_size', None) or 2
        if isinstance(self.team_size, bool) or self.team_size not in (2, 3):
            raise ValueError('Team size must be 2 or 3')
        self.human = getattr(args, 'mode', None) == 'team_vs_player'
        self.strategy = strategy
        self.support_options = bool(getattr(args, 'teammate_support_options', False))
        self.offpuck = bool(getattr(args, 'teammate_offpuck', False))
        self.intent_arbitration = bool(getattr(args, 'teammate_intent_arbitration', False))
        self.defensive_support = bool(getattr(args, 'teammate_defensive_support', False))
        self.primary_policy = getattr(args, 'teammate_primary_policy', 'baseline')
        if self.primary_policy not in ('baseline', 'short-lease', 'lost-race'):
            raise ValueError('Unknown teammate primary policy')
        self.primary_lease = 12 if self.primary_policy == 'short-lease' else 24
        self.role_balance = bool(getattr(args, 'teammate_role_balance', False))
        transient_receiving = getattr(args, 'teammate_transient_receiving', None)
        self.transient_receiving = (strategy == 'coordinated' if transient_receiving is None
                                    else bool(transient_receiving))
        self.anticipate_cue = bool(getattr(args, 'teammate_cue_anticipation', False))
        if self.strategy != 'coordinated' and any((self.offpuck, self.intent_arbitration,
                self.defensive_support, self.role_balance, self.transient_receiving, self.anticipate_cue,
                self.primary_policy != 'baseline')):
            raise ValueError('Teammate variants require coordinated strategy')
        if self.offpuck and not self.support_options:
            raise ValueError('Off-puck movement requires scored support positioning')
        self.interval = int(getattr(args, 'hyperparams_dict', {}).get('frame_skip', 4))
        if self.interval < 1:
            raise ValueError('Teammate decision interval must be positive')
        self.reset()

    def reset(self):
        self.states = [NHL94GameState(5) for _ in range(self.team_size)]
        if self.side == 2:
            for state in self.states:
                state.Flip()
        self.controllers = [ClassicAIV1Model(self.args) for _ in range(self.team_size)]
        self.previous = np.zeros((self.team_size, 12), dtype=np.int8)
        self.frames = 0
        self.transient_frames = 0
        self.primary = None
        self.primary_until = 0
        self.primary_key = None
        self.role_planner = SharedRoles()
        self.role_targets = {}
        self.switch_at = 0
        self.arbitrator = BArbitrator(self.team_size)
        self.last_roles = [None] * self.team_size
        self.last_slots = [None] * self.team_size
        self.last_menu = -120
        self.pending_pass = None
        self.receiving = SharedReceiving(preserve_transient=self.transient_receiving,
                                         anticipate_cue=self.anticipate_cue)
        self.pressure = SharedPressure()
        self.support = [SupportPlanner() for _ in range(self.team_size)]
        self.faceoff = FaceoffInput()
        self.context_owner = None
        self.last_diagnostics = {}
        self.display_probs = np.zeros(12)
        self.model_in_use = 1

    @staticmethod
    def _game_action(buttons):
        return [int(buttons[i]) for i in (4, 5, 6, 7, 0, 8)]

    @staticmethod
    def _steer(player, target):
        pad, waypoint = defensive_steering(player, target)
        action = np.zeros(12, dtype=np.int8)
        action[Buttons.INPUT_UP], action[Buttons.INPUT_DOWN] = pad[1] > 0, pad[1] < 0
        action[Buttons.INPUT_LEFT], action[Buttons.INPUT_RIGHT] = pad[0] < 0, pad[0] > 0
        return action, waypoint

    def _choose_primary(self, state, slots):
        owner = state.engine.puck_owner
        plan = self.receiving.pending
        availability = tuple(slot for slot in slots
                             if (player := state.team1.get_player_by_scnum(slot)) is not None
                             and slot != state.team1.goalie_scnum() and controllable_skater(player))
        key = (tuple(slots), availability, owner, plan.receiver if plan is not None else None)
        if self.primary_policy != 'baseline' and self.primary_key != key:
            self.primary = None
        self.primary_key = key
        if owner >= 0 and owner in slots:
            self.primary = slots.index(owner)
            self.primary_until = self.frames + self.primary_lease
            return self.primary
        plan = self.receiving.pending
        if plan is not None and plan.phase == 'flight' and plan.receiver in slots:
            self.primary = slots.index(plan.receiver)
            self.primary_until = self.frames + self.primary_lease
            return self.primary
        choices = [(arrival_time(player, (state.puck.x, state.puck.y)), i)
                   for i, slot in enumerate(slots)
                   if (player := state.team1.get_player_by_scnum(slot)) is not None
                   and slot != state.team1.goalie_scnum() and controllable_skater(player)]
        if not choices:
            self.primary = None
            return None
        best_time, best = min(choices)
        current = next((time for time, i in choices if i == self.primary), math.inf)
        lost_race = self.primary_policy == 'lost-race' and best_time + 18 < current
        if (self.primary is None or current == math.inf or lost_race
                or self.frames >= self.primary_until and best_time + 6 < current):
            self.primary = best
            self.primary_until = self.frames + self.primary_lease
        return self.primary

    def _support_target(self, state, controller=0, slots=None):
        if controller in self.role_targets:
            self.support[controller].coast = False
            return self.role_targets[controller]
        supporters = [i for i in range(self.team_size) if i != self.primary]
        if not self.role_balance and self.team_size == 3 and len(supporters) > 1 and controller == supporters[-1]:
            team, puck = state.team1, state.puck
            y = puck.y + (team.net.y - puck.y) * 0.65
            return project_target((max(-30, min(30, puck.x * 0.25)), y)), 'deep-coverage'
        if self.defensive_support and not state.team1.owns_scnum(state.engine.puck_owner):
            slots = slots or [state.team1.controlled_scnum()] * self.team_size
            player = state.team1.get_player_by_scnum(slots[controller])
            primary_slot = slots[self.primary] if self.primary is not None else None
            if player is not None and controllable_skater(player):
                self.support[controller].coast = False
                target = SupportPlanner._coverage(state, player, primary_slot)
                return target, 'screen-cover'
        if not self.support_options:
            target, role = self._classic_support_target(state)
            for other, (reserved, _) in self.role_targets.items():
                if other != controller and math.dist(target, reserved) < 24:
                    target = project_target((reserved[0] + (28 if target[0] >= reserved[0] else -28), target[1]))
            return target, role
        slots = slots or [state.team1.controlled_scnum()] * self.team_size
        primary_slot = slots[self.primary] if self.primary is not None else None
        return self.support[controller].choose(state, slots[controller], self.frames, primary_slot,
                                               movement=self.offpuck,
                                               reserved_targets=tuple(point for other, (point, _) in self.role_targets.items()
                                                                      if other != controller))

    @staticmethod
    def _classic_support_target(state):
        team, puck = state.team1, state.puck
        attack = 1 if state.team2.net.y > team.net.y else -1
        carrier = team.get_player_by_scnum(state.engine.puck_owner)
        if carrier is not None and carrier is not team.goalie:
            forward = carrier.y * attack
            y = min(225, forward + 45)
            if state.engine.offsides_enabled is not False and forward < 88:
                y = min(y, 76)
            return project_target((-45 if carrier.x >= 0 else 45, y * attack)), 'passing-option'
        if team.goalie_haspuck:
            return project_target((45 if puck.x < 0 else -45, team.net.y * 0.55)), 'outlet-option'
        y = puck.y + (team.net.y - puck.y) * 0.45
        return project_target((max(-48, min(48, puck.x * 0.45)), y)), 'coverage'

    def predict(self, _observation=None, info=None, deterministic=True):
        """Playback facade. Every invocation is exactly one native frame."""
        if not info:
            raise ValueError('Teammate coordination requires fresh native RAM feedback')
        return [self.act(info[0], deterministic).action]

    def act(self, info, deterministic=True):
        views = [teammate_controller_info(info, i + 1, self.side, team_size=self.team_size, human=self.human) for i in range(self.team_size)]
        # Highlight restoration can leave aliased selections throughout a menu.
        # Only live play must settle within the bounded ownership deadline.
        live = not views[0]['teammate_highlight'] and views[0]['time'] > 0 and not views[0]['sflags'] & 0x0100
        self.transient_frames = self.transient_frames + 1 if live and views[0].get('teammate_transient') else 0
        if self.transient_frames > 120:
            raise RuntimeError('Native controller ownership failed to settle for 120 frames')
        if views[0]['teammate_highlight']:
            self.pressure.reset('highlight', self.frames)
            self.frames += 1
            actions = np.zeros((self.team_size, 12), dtype=np.int8)
            if self.frames - self.last_menu >= 120:
                actions[0, Buttons.INPUT_START] = 1
                self.last_menu = self.frames
            self.pending_pass = self.primary = None
            self.receiving.frame_events = []
            self.receiving.finish(self.frames, 'highlight')
            self.receiving.last_owner, self.receiving.last_count = -1, None
            self.last_roles, self.last_slots = ['highlight'] * self.team_size, [-1] * self.team_size
            self.previous = actions.copy()
            for i, controller in enumerate(self.controllers):
                controller.cancel_actions('highlight')
                controller.advance_held_frame()
                controller.accept_applied_action(actions[i])
            self.last_diagnostics = {'teammates': {'frame': self.frames, 'strategy': self.strategy,
                'primary_controller': None, 'controllers': [
                    {'controller': i + 1, 'slot': -1, 'role': 'highlight',
                     'target': [0, 0], 'buttons': actions[i].tolist()} for i in range(self.team_size)],
                'pending_pass': None}}
            return AgentOutput(actions.reshape(12 * self.team_size).copy(), self.last_diagnostics)
        for i, state in enumerate(self.states):
            state.BeginFrame(views[i], self._game_action(self.previous[i]))
            self.states[i] = decision_state(state, getattr(self.args, 'player_ratings_consideration', True))
        state = self.states[0]
        self.frames += 1
        slots = [view[f'p{self.side}_control_slot'] for view in views]
        draw_controller, draw_action, draw_center = self.faceoff.choose(
            state, slots, self.frames, transient=bool(views[0].get('teammate_transient')), previous=self.previous)
        owner = state.engine.puck_owner
        known_slots = not views[0].get('teammate_transient') and all(slot >= 0 for slot in slots)
        for i, controller in enumerate(self.controllers):
            context = TeamControlContext(tuple(slots), known_slots, self._classic_support_target(self.states[i])[0],
                                         goalie_ownership=True)
            previous_context = controller.team_context
            # Interrupt only CPU-carrier transitions, preserving other tactics' cadence.
            was_cpu = (previous_context is not None and previous_context.known
                       and self.context_owner is not None and state.team1.owns_scnum(self.context_owner)
                       and self.context_owner != state.team1.goalie_scnum()
                       and self.context_owner not in previous_context.slots)
            if was_cpu != context.cpu_carrier(state):
                controller.scheduler.interrupt()
            controller.team_context = context
        self.context_owner = owner
        if self.strategy == 'coordinated':
            self.receiving.observe(state, slots, self.frames)
            for event in self.receiving.frame_events:
                sender = event.get('commitment', {}).get('passer_controller')
                if event['phase'] == 'ended' and sender is not None:
                    self.controllers[sender - 1].delegate_reception(self.states[sender - 1], event['reason'])
        primary = self._choose_primary(state, slots)
        pending_contact = next((i for i, controller in enumerate(self.controllers)
                                if controller.defense.pending_check is not None), None)
        assignments = (self.pressure.choose(
            state, slots, self.frames, pending_contact=pending_contact,
            busy=bool(views[0].get('teammate_transient')))
            if self.strategy == 'coordinated' and getattr(self.args, 'teammate_pressure', True) else None)
        if assignments is None and any(role in TRAP_ROLES for role in self.last_roles):
            for controller in self.controllers:
                controller.cancel_actions('trap-ended')
            # A loose puck or new receiver needs a fresh race, not the old primary's lease.
            self.primary = None
            primary = self._choose_primary(state, slots)
        actions = np.zeros((self.team_size, 12), dtype=np.int8)
        rows = [None] * self.team_size
        intermission = state.time == 0 or bool(state.engine.sflags & 0x0100)
        if intermission:
            self.receiving.finish(self.frames, 'intermission')
        self.role_targets = {}
        if self.role_balance and self.team_size == 3 and self.strategy == 'coordinated':
            plan = self.receiving.pending
            reserved = tuple(i for i, controller in enumerate(self.controllers)
                             if assignments is not None or intermission or state.engine.clock_stopped
                             or controller.has_committed_action()
                             or plan is not None and plan.phase in ('release', 'flight') and plan.receiver_controller == i + 1)
            self.role_targets = self.role_planner.choose(state, slots, primary, self.frames, reserved)
        order = tuple(range(self.team_size)) if primary is None else (primary, *(i for i in range(self.team_size) if i != primary))
        for i in order:
            view = self.states[i]
            slot = slots[i]
            input_source, requested, b_intent = 'external', None, 'none'
            decisions_before = self.controllers[i].scheduler.decisions
            executor = self.controllers[i]
            b_history = (executor.defense.switch_at, executor.defense.poke_at,
                         executor.defense.switch_attempts, copy(executor.defense.pending_switch),
                         executor.recovery_switch_at_decision)
            player = view.team1.get_player_by_scnum(slot)
            target, role = self._support_target(view, i, slots)
            plan = self.receiving.pending
            if (plan is not None and plan.phase == 'flight' and plan.passer_controller == i + 1
                    and plan.receiver_controller is not None and plan.receiver_controller != i + 1):
                self.controllers[i].delegate_reception()
            reception = self.receiving.receiver_action(view, i + 1, self.frames, self.previous[i])
            unavailable = player is None or slot == view.team1.goalie_scnum()
            if not unavailable:
                unavailable = player.is_falling or player.unavailable & 4 or player.role is not None and player.role <= 0
                unavailable = unavailable or (not controllable_skater(player) and not player.is_one_timer
                                              and not self.controllers[i].has_committed_action())
            if intermission:
                role = 'intermission'
                if info['period'] in (1, 2) and state.time > 0 and self.frames - self.last_menu >= 120:
                    if i == 0:
                        actions[i, Buttons.INPUT_START] = 1
                        self.last_menu = self.frames
            elif slot < 0:
                role = 'unavailable'
            elif state.engine.clock_stopped:
                role = 'stoppage' if draw_center is None else 'faceoff' if i == draw_controller else 'faceoff-support'
                if i == draw_controller:
                    actions[i] = draw_action
            elif slot == owner == view.team1.goalie_scnum():
                # Automatic goalies still need the selected controller to release.
                if self.last_slots[i] != slot:
                    self.controllers[i].scheduler.interrupt()
                actions[i] = self.controllers[i].predict_frame(view, frame_skip=self.interval,
                                                              deterministic=deterministic)[0]
                role, target = 'goalie-outlet', tuple(self.controllers[i]._last_target)
                input_source = 'classic'
            elif unavailable:
                role = 'unavailable'
            elif assignments is not None:
                assignment = assignments[i]
                if self.last_roles[i] not in TRAP_ROLES:
                    self.controllers[i].cancel_actions('trap-started')
                actions[i] = self.controllers[i].predict_team_defense(view, assignment, frame_skip=self.interval)
                role, target = assignment.plan.mode, assignment.plan.target
                input_source = 'pressure'
            elif reception is not None:
                actions[i], target, role = reception
                input_source = 'receiving'
            elif (self.strategy == 'independent' or i == primary
                  or self.controllers[i].has_committed_action()):
                role = 'carrier' if owner == slot else 'pressure' if view.team2.owns_scnum(owner) else 'retrieve'
                if self.last_slots[i] != slot:
                    self.controllers[i].scheduler.interrupt()
                actions[i] = self.controllers[i].predict_frame(view, frame_skip=self.interval,
                                                              deterministic=deterministic)[0]
                input_source = 'classic'
                requested = actions[i].copy()
                if self.strategy == 'coordinated':
                    self.receiving.propose(self.controllers[i].pass_intention(), i + 1, slots, state, self.frames)
                target = tuple(self.controllers[i]._last_target)
                # One controller requests a switch at a time. Never release a
                # carrier with B interpreted as a pass by the teammate's view.
                if (not self.intent_arbitration and self.strategy == 'coordinated'
                        and owner != slot and actions[i, Buttons.INPUT_B]):
                    if self.frames < self.switch_at or plan is not None and plan.phase in ('release', 'flight'):
                        actions[i, Buttons.INPUT_B] = 0
                    elif not self.previous[i, Buttons.INPUT_B]:
                        self.switch_at = self.frames + 8
            else:
                if self.offpuck:
                    actions[i], target = movement_action(player, target,
                        coast=role != 'deep-coverage' and self.support[i].coast)
                    input_source = 'offpuck'
                elif self.support_options and role != 'deep-coverage' and self.support[i].coast:
                    actions[i] = np.zeros(12, dtype=np.int8)
                else:
                    actions[i], target = self._steer(player, target)
            controller = self.controllers[i]
            if requested is None:
                requested = actions[i].copy()
            if requested[Buttons.INPUT_B]:
                mode = controller.defense_diagnostics.get('mode') if input_source in ('classic', 'pressure') else None
                b_intent = ('faceoff' if role == 'faceoff' else 'poke' if mode == 'poke-request'
                            else 'switch' if mode == 'switch-request' or input_source == 'classic' and controller._last_decision == 'switch'
                            else 'goalie-outlet' if slot == owner == view.team1.goalie_scnum()
                            else 'pass' if input_source == 'classic' and controller.input_owner == 'pass'
                            or owner == slot else 'unknown')
            arbitration = 'baseline'
            if self.intent_arbitration and self.strategy == 'coordinated' and requested[Buttons.INPUT_B]:
                active_pass = self.receiving.pending
                allowed, arbitration = self.arbitrator.allow(
                    intent=b_intent, frame=self.frames, controller=i,
                    previous=bool(self.previous[i, Buttons.INPUT_B]), owns_puck=owner == slot,
                    known=slot >= 0 and not views[0].get('teammate_transient'),
                    protected=active_pass is not None and active_pass.phase in ('release', 'flight'),
                    likely_keeps_selection=controller.defense_diagnostics.get('likely_switch_slot') == slot)
                if not allowed:
                    actions[i, Buttons.INPUT_B] = 0
                    # Rejected proposals cannot create a pending native switch.
                    (controller.defense.switch_at, controller.defense.poke_at,
                     controller.defense.switch_attempts, controller.defense.pending_switch,
                     controller.recovery_switch_at_decision) = b_history
                    controller.scheduler.interrupt()
            if controller.scheduler.frames < self.frames:
                if role in ('intermission', 'unavailable', 'faceoff', 'faceoff-support', 'stoppage'):
                    controller.cancel_actions(role)
                controller.observe_external_frame(view, actions[i], frame_skip=self.interval)
            controller.accept_applied_action(actions[i])
            if role == 'one-timer-receiver':
                self.receiving.accept_applied_action(i + 1, slot, self.frames, actions[i], self.previous[i])
            self.last_roles[i], self.last_slots[i] = role, slot
            rows[i] = {'controller': i + 1, 'slot': slot, 'role': role,
                         'target': list(target), 'buttons': actions[i].tolist(),
                         'executor_frame': controller.scheduler.frames,
                         'decision_tick': controller.scheduler.decisions,
                         'decision_tick_fired': controller.scheduler.decisions > decisions_before,
                         'proposal_frame': self.frames if input_source == 'pressure' or
                             input_source == 'classic' and (controller.scheduler.decisions > decisions_before
                             or controller.input_owner == 'defense') else None,
                         'decision': controller._last_decision if input_source in ('classic', 'pressure') else role,
                         'input_source': input_source, 'b_intent': b_intent,
                         'b_arbitration': arbitration,
                         'requested_buttons': requested.tolist(),
                         'applied_b_edge': bool(actions[i, Buttons.INPUT_B] and not self.previous[i, Buttons.INPUT_B]),
                         'applied_c_edge': bool(actions[i, Buttons.INPUT_C] and not self.previous[i, Buttons.INPUT_C]),
                         'support': self.support[i].diagnostics if role != 'deep-coverage' else {},
                         'pass': (controller.offense.pass_action.pending.snapshot()
                                  if controller.offense.pass_action.pending is not None else None),
                         'one_timer': (dict(vars(controller.one_timer.pending))
                                       if controller.one_timer.pending is not None else None),
                         'shot_phase': controller.shot.phase(controller.scheduler.decisions,
                                                              controller.scheduler.frames)}
        if self.strategy == 'coordinated' and not intermission and not state.engine.clock_stopped:
            for i, controller in enumerate(self.controllers):
                self.receiving.propose(controller.pass_intention(), i + 1, slots, state, self.frames)
        self.pending_pass = self.receiving.pending.snapshot() if self.receiving.pending is not None else None
        self.previous = actions.copy()
        self.display_probs = actions[0].copy()
        self.last_diagnostics = {'teammates': {'frame': self.frames, 'strategy': self.strategy,
                                             'transient': views[0].get('teammate_transient', False),
                                             'native_teams': views[0].get('native_controller_teams'),
                                             'native_slots': views[0].get('native_controller_slots'),
                                             'primary_controller': None if primary is None else primary + 1,
                                             'controllers': rows, 'pending_pass': copy(self.pending_pass),
                                             'pressure': dict(self.pressure.diagnostics),
                                             'roles': dict(self.role_planner.diagnostics),
                                             'events': self.receiving.frame_events.copy()}}
        for view in self.states:
            view.EndFrame()
        return AgentOutput(actions.reshape(12 * self.team_size).copy(), self.last_diagnostics)
