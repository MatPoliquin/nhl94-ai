"""Native teammate controllers, one shared tactical plan, independent execution state."""
from copy import copy
import math

import numpy as np

from nhl94_ai.agents.base import AgentOutput
from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.agents.classic_config import decision_state
from nhl94_ai.agents.defense import defensive_steering, eligible
from nhl94_ai.agents.motion import arrival_time, velocity
from nhl94_ai.agents.teammate_receiving import SharedReceiving
from nhl94_ai.agents.teammate_support import SupportPlanner
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
        self.switch_at = 0
        self.last_roles = [None] * self.team_size
        self.last_slots = [None] * self.team_size
        self.last_menu = -120
        self.pending_pass = None
        self.receiving = SharedReceiving()
        self.support = [SupportPlanner() for _ in range(self.team_size)]
        self.last_owner = -1
        self.last_pass_count = None
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
        if owner in slots:
            self.primary = slots.index(owner)
            self.primary_until = self.frames + 24
            return self.primary
        plan = self.receiving.pending
        if plan is not None and plan.phase == 'flight' and plan.receiver in slots:
            self.primary = slots.index(plan.receiver)
            self.primary_until = self.frames + 24
            return self.primary
        choices = [(arrival_time(player, (state.puck.x, state.puck.y)), i)
                   for i, slot in enumerate(slots)
                   if (player := state.team1.get_player_by_scnum(slot)) is not None
                   and slot != state.team1.goalie_scnum() and eligible(player)]
        if not choices:
            self.primary = None
            return None
        best_time, best = min(choices)
        current = next((time for time, i in choices if i == self.primary), math.inf)
        if (self.primary is None or current == math.inf
                or self.frames >= self.primary_until and best_time + 6 < current):
            self.primary = best
            self.primary_until = self.frames + 24
        return self.primary

    def _support_target(self, state, controller=0, slots=None):
        supporters = [i for i in range(self.team_size) if i != self.primary]
        if self.team_size == 3 and len(supporters) > 1 and controller == supporters[-1]:
            team, puck = state.team1, state.puck
            y = puck.y + (team.net.y - puck.y) * 0.65
            return project_target((max(-30, min(30, puck.x * 0.25)), y)), 'deep-coverage'
        if not self.support_options:
            return self._classic_support_target(state)
        slots = slots or [state.team1.controlled_scnum()] * self.team_size
        primary_slot = slots[self.primary] if self.primary is not None else None
        return self.support[controller].choose(state, slots[controller], self.frames, primary_slot)

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
        self.transient_frames = self.transient_frames + 1 if views[0].get('teammate_transient') else 0
        if self.transient_frames > 120:
            raise RuntimeError('Native controller ownership failed to settle for 120 frames')
        if views[0]['teammate_highlight']:
            self.frames += 1
            actions = np.zeros((self.team_size, 12), dtype=np.int8)
            if self.frames - self.last_menu >= 120:
                actions[0, Buttons.INPUT_START] = 1
                self.last_menu = self.frames
            self.pending_pass = self.primary = None
            self.receiving.frame_events = []
            self.receiving.finish(self.frames, 'highlight')
            self.receiving.last_owner, self.receiving.last_count = -1, None
            self.last_owner, self.last_pass_count = -1, None
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
        owner = state.engine.puck_owner
        count = state.team1.pass_attempts
        if self.strategy == 'coordinated':
            self.receiving.observe(state, slots, self.frames)
            for event in self.receiving.frame_events:
                sender = event.get('commitment', {}).get('passer_controller')
                if event['phase'] == 'ended' and sender is not None:
                    self.controllers[sender - 1].delegate_reception(self.states[sender - 1], event['reason'])
        primary = self._choose_primary(state, slots)
        actions = np.zeros((self.team_size, 12), dtype=np.int8)
        rows = [None] * self.team_size
        intermission = state.time == 0 or bool(state.engine.sflags & 0x0100)
        if intermission:
            self.receiving.finish(self.frames, 'intermission')
        order = tuple(range(self.team_size)) if primary is None else (primary, *(i for i in range(self.team_size) if i != primary))
        for i in order:
            view = self.states[i]
            slot = slots[i]
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
                unavailable = unavailable or (not eligible(player) and not player.is_one_timer
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
                role = 'faceoff' if i == 0 else 'faceoff-support'
                if i == 0:
                    actions[i, Buttons.INPUT_B] = self.frames % 8 < 4
            elif unavailable:
                role = 'unavailable'
            elif reception is not None:
                actions[i], target, role = reception
            elif (self.strategy == 'independent' or i == primary
                  or self.controllers[i].has_committed_action()):
                role = 'carrier' if owner == slot else 'pressure' if view.team2.owns_scnum(owner) else 'retrieve'
                if self.last_slots[i] != slot:
                    self.controllers[i].scheduler.interrupt()
                actions[i] = self.controllers[i].predict_frame(view, frame_skip=self.interval,
                                                              deterministic=deterministic)[0]
                if self.strategy == 'coordinated':
                    self.receiving.propose(self.controllers[i].pass_intention(), i + 1, slots, state, self.frames)
                target = tuple(self.controllers[i]._last_target)
                # One controller requests a switch at a time. Never release a
                # carrier with B interpreted as a pass by the teammate's view.
                if self.strategy == 'coordinated' and owner != slot and actions[i, Buttons.INPUT_B]:
                    if self.frames < self.switch_at or plan is not None and plan.phase in ('release', 'flight'):
                        actions[i, Buttons.INPUT_B] = 0
                    elif not self.previous[i, Buttons.INPUT_B]:
                        self.switch_at = self.frames + 8
            else:
                if self.support_options and role != 'deep-coverage' and self.support[i].coast:
                    actions[i] = np.zeros(12, dtype=np.int8)
                else:
                    actions[i], target = self._steer(player, target)
            controller = self.controllers[i]
            if controller.scheduler.frames < self.frames:
                if role in ('intermission', 'unavailable', 'faceoff', 'faceoff-support'):
                    controller.cancel_actions(role)
                controller.observe_external_frame(view, actions[i], frame_skip=self.interval)
            controller.accept_applied_action(actions[i])
            self.last_roles[i], self.last_slots[i] = role, slot
            rows[i] = {'controller': i + 1, 'slot': slot, 'role': role,
                         'target': list(target), 'buttons': actions[i].tolist(),
                         'executor_frame': controller.scheduler.frames,
                         'decision_tick': controller.scheduler.decisions,
                         'decision': controller._last_decision,
                         'support': self.support[i].diagnostics if role != 'deep-coverage' else {},
                         'pass': (controller.offense.pass_action.pending.snapshot()
                                  if controller.offense.pass_action.pending is not None else None),
                         'one_timer': (dict(vars(controller.one_timer.pending))
                                       if controller.one_timer.pending is not None else None),
                         'shot_phase': controller.shot.phase(controller.scheduler.decisions,
                                                              controller.scheduler.frames)}
        self.last_owner, self.last_pass_count = owner, count
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
                                             'events': self.receiving.frame_events.copy()}}
        for view in self.states:
            view.EndFrame()
        return AgentOutput(actions.reshape(12 * self.team_size).copy(), self.last_diagnostics)
