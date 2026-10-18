"""Opt-in bounded receding-horizon search over Classic's goalie physics.

The search handles already released, unobstructed shots. It retains Classic's
handoffs, pass anticipation, outlets, dives and unsupported-state behavior.
Its value is contact coverage under one-update timing uncertainty, not xG.
Native replay and complete matches must validate it before default promotion.
"""
from copy import copy
from dataclasses import dataclass
import math
from time import perf_counter_ns

import numpy as np

from nhl94_ai.agents.goalie import GoalieController
from nhl94_ai.agents.goalie_physics import KeeperMotion, PuckFlight, PADS, contact_margin, free_trajectory
from nhl94_ai.agents.motion import velocity
from nhl94_ai.game.constants import GameConsts as Buttons


MODES = ('physics', 'search-12', 'search-24', 'search-48')
SAVE = len(PADS)
BLOCK = 4
BEAM = 6
MAX_STEPS = 4096


def encode(command):
    action = np.zeros(Buttons.INPUT_MAX, dtype=np.int8)
    if command == SAVE:
        action[Buttons.INPUT_C] = 1
        return action
    dx, dy = PADS[command]
    if dx:
        action[Buttons.INPUT_RIGHT if dx > 0 else Buttons.INPUT_LEFT] = 1
    if dy:
        action[Buttons.INPUT_UP if dy > 0 else Buttons.INPUT_DOWN] = 1
    return action


def decode(action):
    if action[Buttons.INPUT_A] or action[Buttons.INPUT_B]:
        return None
    if action[Buttons.INPUT_C]:
        return SAVE
    return PADS.index((int(action[Buttons.INPUT_RIGHT])-int(action[Buttons.INPUT_LEFT]),
                       int(action[Buttons.INPUT_UP])-int(action[Buttons.INPUT_DOWN])))


def supported(keeper):
    return abs(keeper.x >> 16) <= 34 and 234 <= abs(keeper.y >> 16) <= 254


@dataclass(frozen=True, slots=True)
class Branch:
    keeper: KeeperMotion
    first: int
    commands: tuple
    contacts: int = 0
    margin: float = -1000.0
    contact_time: int = 1000


def extend(branch, command, start, stop, flight, one_timer):
    keeper, hits, margin, contact_time = branch.keeper, branch.contacts, branch.margin, branch.contact_time
    for tick in range(start, stop):
        keeper = keeper.step((0, 0) if command == SAVE else PADS[command], save=command == SAVE)
        if not supported(keeper):
            return None
        # The emulator may observe objects on either side of an in-progress
        # update. Evaluate one tick early/on time/late; do not access ROM RNG.
        for scenario, offset in enumerate((-1, 0, 1)):
            puck = flight[max(0, min(len(flight)-1, tick+1+offset))]
            gap = contact_margin(keeper, puck, one_timer=one_timer)
            if scenario == 1:
                margin = max(margin, gap)
            if gap >= 0:
                hits |= 1 << scenario
                contact_time = min(contact_time, tick+1)
    return Branch(keeper, branch.first if branch.first >= 0 else command,
                  branch.commands+(command,), hits, margin, contact_time)


def rank(branch, target, baseline):
    x, y = branch.keeper.point
    # Contact dominates geometry. Favor central coverage over grazing contact;
    # stable fallback wins exact ties. This score is deliberately not called xG.
    return (branch.contacts.bit_count(), min(6.0, branch.margin),
            -math.hypot(x-target[0], y-target[1]), -branch.contact_time,
            branch.first == baseline, -len(set(branch.commands)))


def search(keeper, puck, one_timer, sign, baseline, *, horizon=24, beam=BEAM, fixed=False,
           can_save=True):
    trajectory = [puck, *free_trajectory(puck, 49, sign)]
    end = trajectory[-1]
    if ((end.y >> 16)*sign < 261 or abs(end.x >> 16) > 23 or end.z >> 16 > 15):
        return None, {'status': 'no-modeled-goal-threat'}
    horizon = min(horizon, len(trajectory)-1)
    target = min(trajectory[1:], key=lambda p: abs(p.y-keeper.y)).point[:2]
    commands = tuple(range(SAVE + int(can_save)))
    root = Branch(keeper, -1, ())
    # Explicitly retain the incumbent input throughout selection/pruning.
    incumbent = extend(root, baseline, 0, horizon, trajectory, one_timer)
    evaluations = horizon
    if fixed:
        candidates = [extend(root, command, 0, horizon, trajectory, one_timer) for command in commands]
        evaluations += horizon*len(commands)
    else:
        candidates = [root]
        for start in range(0, horizon, BLOCK):
            stop = min(horizon, start+BLOCK)
            expanded = []
            for branch in candidates:
                for command in ((SAVE,) if branch.keeper.saving else commands):
                    if evaluations+stop-start > MAX_STEPS:
                        break
                    child = extend(branch, command, start, stop, trajectory, one_timer)
                    evaluations += stop-start
                    if child is not None:
                        expanded.append(child)
            if not expanded:
                candidates = []
                break
            # Maintain distinct initial actions so a single early attractive
            # action cannot consume the entire beam before contact is possible.
            expanded.sort(key=lambda branch: rank(branch, target, baseline), reverse=True)
            candidates, seen = [], set()
            for branch in expanded:
                key = branch.first
                if key not in seen:
                    candidates.append(branch)
                    seen.add(key)
                if len(candidates) == beam:
                    break
    candidates = [branch for branch in candidates if branch is not None]
    if incumbent is not None:
        candidates.append(incumbent)
    if not candidates:
        return None, {'status': 'unsupported-forecast', 'steps': evaluations}
    chosen = max(candidates, key=lambda branch: rank(branch, target, baseline))
    # A candidate must predict contact, and improve coverage or margin over the
    # incumbent. Do not replace established behavior solely on terminal distance.
    improved = (chosen.contacts and (incumbent is None or
                (chosen.contacts.bit_count(), chosen.margin) >
                (incumbent.contacts.bit_count(), incumbent.margin+1)))
    command = chosen.first if improved else baseline
    return command, {'status': 'searched', 'horizon': horizon, 'steps': evaluations,
                     'coverage': chosen.contacts.bit_count(), 'margin': chosen.margin,
                     'baseline_coverage': incumbent.contacts.bit_count() if incumbent else 0,
                     'baseline_margin': incumbent.margin if incumbent else None,
                     'sequence': chosen.commands, 'changed': command != baseline}


class PlanningGoalie(GoalieController):
    """Same native lifecycle, with an optional released-shot decision search."""

    def __init__(self, policy='selective', *, planning='search-24', **kwargs):
        super().__init__(policy, **kwargs)
        if planning not in MODES:
            raise ValueError('Unknown goalie planner')
        self.planning = planning
        self.planner_times_ns = []

    def _search(self, state, keeper, puck, baseline):
        return search(keeper, puck, state.engine.native_one_timer_shot,
                      1 if state.team1.net.y > 0 else -1, baseline,
                      horizon=24 if self.planning == 'physics' else int(self.planning.split('-')[1]),
                      fixed=self.planning == 'physics', can_save=self.frames >= self.save_at)

    def _position(self, state, plan):
        keeper, puck = KeeperMotion.observe(state.team1.goalie), PuckFlight.observe(state.puck)
        reason = None
        if keeper is None or puck is None or not supported(keeper):
            reason = 'unsupported-state'
        elif state.engine.puck_owner >= 0 or plan.receiver is not None:
            reason = 'possession-or-pass'
        elif (state.puck.y * (1 if state.team1.net.y > 0 else -1) < 150
              or abs(state.puck.x) > 100):
            reason = 'outside-shot-region'
        if reason:
            self.metrics['planner_fallback:'+reason] += 1
            return super()._position(state, plan)
        # A body/stick event changes free flight. Leave such situations to the
        # existing controller rather than silently forecast through a skater.
        path = free_trajectory(puck, 24, 1 if state.team1.net.y > 0 else -1)
        for team in (state.team1, state.team2):
            for player in team.players:
                if player.role is None or player.role <= 0:
                    continue
                vx, vy = velocity(player)
                if any(math.hypot(p.x/65536-player.x-vx*(tick+1),
                                  p.y/65536-player.y-vy*(tick+1)) < 18
                       for tick, p in enumerate(path)):
                    self.metrics['planner_fallback:possible-skater-contact'] += 1
                    return super()._position(state, plan)
        # Preview without consuming a C edge or recording a phantom save request.
        preview = copy(self)
        preview.metrics, preview.diagnostics = self.metrics.copy(), self.diagnostics.copy()
        action = GoalieController._position(preview, state, plan)
        baseline = decode(action)
        if baseline is None:
            self.metrics['planner_fallback:baseline-dive'] += 1
            return super()._position(state, plan)
        started = perf_counter_ns()
        command, details = self._search(state, keeper, puck, baseline)
        self.planner_times_ns.append(perf_counter_ns()-started)
        self.metrics['planner_calls'] += 1
        self.metrics['planner_steps'] += details.get('steps', 0)
        self.diagnostics['planner'] = details
        if command is None or command == baseline:
            return super()._position(state, plan)
        self.metrics['planner_changes'] += 1
        if command == SAVE:
            self._request_save('save')
        return encode(command)
