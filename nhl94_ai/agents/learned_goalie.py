"""Goalie-only commands and observations; Classic owns handoffs and outlets."""
from copy import deepcopy
import math

import numpy as np

from nhl94_ai.agents.defense import controlled_slot
from nhl94_ai.agents.goalie import DIVE_ANIMATION, SAVE_ANIMATIONS, GoalieController, goalie_target
from nhl94_ai.agents.motion import velocity
from nhl94_ai.game.constants import GameConsts as Buttons


PADS = ((0, 0), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1))
ACTION_NAMES = ('neutral', 'up', 'up-right', 'right', 'down-right', 'down',
                'down-left', 'left', 'up-left', 'save', 'dive-left', 'dive-right')
N_ACTIONS = len(ACTION_NAMES)
TOKEN_COUNT, TOKEN_WIDTH = 14, 20
SCHEMA = {'version': 2, 'tokens': TOKEN_COUNT, 'width': TOKEN_WIDTH,
          'actions': list(ACTION_NAMES), 'defended_end': 'negative-y',
          'context_tail': ['native_one_timer_shot', 'reserved', 'reserved', 'reserved'],
          'control': 'positioning-only; native handoff/lock/outlet interrupts'}


def decision_available(state, manager):
    goalie = state.team1.goalie
    available = bool(controlled_slot(state.team1) == state.team1.defense_goalie
                and goalie.selection_flags & 8 and not goalie.live_state_flags & 4
                and not goalie.unavailable & 6 and not goalie.selection_flags & 0x20
                and goalie.role == 0 and not state.engine.clock_stopped and not state.engine.input_block
                and state.engine.puck_owner != state.team1.defense_goalie
                and not state.team1.owns_scnum(state.engine.puck_owner)
                and manager.pending_save is None and manager.outlet is None
                and manager.phase not in ('prepare-goalie', 'request-goalie', 'release-goalie',
                                          'request-skater', 'neutral-handoff', 'backoff'))
    if not available:
        return False
    sign = 1 if state.team1.net.y > 0 else -1
    # _step returns to a skater here, so no learned command can be consumed.
    return state.puck.y * sign >= 70 or goalie_target(state, manager.pass_windup).deadline is not None


def action_mask(state, manager):
    mask = np.zeros(N_ACTIONS, dtype=np.float32)
    mask[0] = 1
    if decision_available(state, manager):
        mask[:9] = 1
        if manager.frames >= manager.save_at:
            mask[9:] = 1
    return mask


def observation(state, manager):
    """Exact current physical state, with the same physics features for all models."""
    rotate = -1 if state.team1.net.y > 0 else 1
    mask = action_mask(state, manager)
    values = np.zeros((TOKEN_COUNT, TOKEN_WIDTH), dtype=np.float32)
    players = [(state.team1.goalie, state.team1.defense_goalie, 1),
               (state.team2.goalie, state.team2.defense_goalie, -1)]
    for team, affiliation in ((state.team1, 1), (state.team2, -1)):
        players.extend((player, team.skater_scnum_base() + index, affiliation)
                       for index, player in enumerate(team.players))
    for index, (player, slot, affiliation) in enumerate(players):
        vx, vy = velocity(player)
        angle = (player.facing if player.facing is not None else player.orientation) * math.pi / 4
        values[index] = (
            rotate * player.x / 128, rotate * player.y / 288, rotate * vx / 16, rotate * vy / 16,
            rotate * math.sin(angle), rotate * math.cos(angle), (player.speed or 0) / 30,
            (player.agility or 0) / 30, (player.weight or 0) / 128, (player.energy or 0) / 4096,
            (player.role or 0) / 6, affiliation, state.engine.puck_owner == slot,
            controlled_slot(state.team1) == slot, player.live_anim in SAVE_ANIMATIONS,
            player.live_anim == DIVE_ANIMATION, bool((player.unavailable or 0) & 6),
            bool((player.live_state_flags or 0) & 4), rotate * (player.stick_x or 0) / 32,
            rotate * (player.stick_y or 0) / 32)
    puck = state.puck
    vx, vy = velocity(puck)
    values[12, :7] = (rotate * puck.x / 128, rotate * puck.y / 288,
                      rotate * vx / 16, rotate * vy / 16,
                      (puck.height or 0) / 32, (puck.motion_z or 0) / 16, puck.friction)
    plan = goalie_target(state, manager.pass_windup)
    receiver = plan.receiver or (0, 0)
    values[13, :16] = (
        rotate * plan.target[0] / 128, rotate * plan.target[1] / 288,
        plan.crossing_frames is not None, min(120, plan.crossing_frames or 0) / 120,
        rotate * (plan.crossing_x or 0) / 128, (plan.height or 0) / 32,
        min(120, plan.deadline or 0) / 120, plan.receiver is not None,
        rotate * receiver[0] / 128, rotate * receiver[1] / 288,
        min(120, plan.reception_frames or 0) / 120,
        min(40, max(0, manager.save_at - manager.frames)) / 40,
        state.engine.puck_owner < 0, state.team2.owns_scnum(state.engine.puck_owner),
        bool(state.is_shooting), mask[1])
    values[13, 16] = state.engine.native_one_timer_shot
    return np.concatenate((np.clip(values.ravel(), -1, 1), mask))


def command_buttons(command, state):
    if not 0 <= command < N_ACTIONS:
        raise ValueError('Invalid goalie command')
    action = np.zeros(Buttons.INPUT_MAX, dtype=np.int8)
    rotate = -1 if state.team1.net.y > 0 else 1
    dx, dy = PADS[command] if command < 9 else ((-1 if command == 10 else 1), 0)
    if command == 9:
        dx = dy = 0
        action[Buttons.INPUT_C] = 1
    elif command >= 10:
        action[Buttons.INPUT_A] = 1
    dx, dy = dx * rotate, dy * rotate
    if dx:
        action[Buttons.INPUT_RIGHT if dx > 0 else Buttons.INPUT_LEFT] = 1
    if dy:
        action[Buttons.INPUT_UP if dy > 0 else Buttons.INPUT_DOWN] = 1
    return action


class CommandGoalie(GoalieController):
    """External movement/save choices with the existing native safety lifecycle."""

    def __init__(self, policy='selective', *, command=0, **kwargs):
        super().__init__(policy, **kwargs)
        self.command = command
        self.commands_used = 0

    def _position(self, state, plan):
        command = int(self.command)
        mask = action_mask(state, self)
        if not mask[command]:
            command = 0
        action = command_buttons(command, state)
        if command >= 9:
            self._request_save('save' if command == 9 else 'dive')
        self.commands_used += 1
        self.diagnostics.update(learned_command=ACTION_NAMES[command])
        return action


class SampledGoalie(GoalieController):
    """Match scripted positioning decisions to a neural policy's decision rate."""

    def __init__(self, policy='selective', *, interval=4, **kwargs):
        super().__init__(policy, **kwargs)
        if interval < 1:
            raise ValueError('Goalie interval must be positive')
        self.interval, self.next_choice = interval, 0
        self.cached = np.zeros(Buttons.INPUT_MAX, dtype=np.int8)

    def _position(self, state, plan):
        if not self.next_choice:
            self.next_choice = self.frames
        if self.frames > self.next_choice:
            self.next_choice += math.ceil((self.frames - self.next_choice) / self.interval) * self.interval
            self.cached[:] = 0
        if self.frames == self.next_choice:
            self.cached = super()._position(state, plan)
            self.next_choice = self.frames + self.interval
        return self.cached.copy()


def replace_goalie(previous, replacement):
    """Copy live handoff/save history without overwriting the new policy options."""
    configured = {'save_timing', 'command', 'commands_used', 'interval', 'next_choice', 'cached'}
    for name, value in vars(previous).items():
        if name not in configured and hasattr(replacement, name):
            setattr(replacement, name, deepcopy(value))
    return replacement
