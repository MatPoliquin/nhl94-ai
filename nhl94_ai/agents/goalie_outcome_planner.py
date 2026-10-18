"""Experimental first-contact search with conditional catch/rebound outcomes.

Uses the same 24-update, four-update-block, six-plan budget as v1. Each timing
scenario stops at its first contact. An adaptive rollout of default Classic is
the incumbent. Low-puck stick events and save motion beyond twelve updates are
unsupported, so the existing controller handles them.
"""
from copy import copy
from dataclasses import dataclass
import math

from nhl94_ai.agents.goalie import GoalieController, goalie_target
from nhl94_ai.agents.goalie_outcomes import body_rebound, catch_probability
from nhl94_ai.agents.goalie_physics import KeeperMotion, PADS, PuckFlight, contact_margin, free_trajectory
from nhl94_ai.agents.goalie_planner import BEAM, BLOCK, MAX_STEPS, SAVE, PlanningGoalie, decode, supported
from nhl94_ai.agents.motion import arrival_time


@dataclass(frozen=True, slots=True)
class Contact:
    tick: int
    catch: float
    goal_risk: float
    rebound: PuckFlight


@dataclass(frozen=True, slots=True)
class OutcomeBranch:
    keeper: KeeperMotion
    first: int = -1
    commands: tuple = ()
    contacts: tuple = (None, None, None)
    save_tick: int | None = None


def collision(keeper, puck, tick, control, sign):
    rebound = body_rebound(keeper, puck)
    if rebound is None:
        return None
    chance = catch_probability(control, puck.vx, puck.vy, animated=keeper.saving,
                               close=abs(puck.x >> 16) <= 44 and 210 <= abs(puck.y >> 16) <= 270)
    trajectory = free_trajectory(rebound, 24, sign)
    # This is the risk of a goal on an unopposed rebound path, not calibrated xG.
    at_goal = next((p for p in trajectory if (p.y >> 16)*sign >= 261), None)
    goal = bool(at_goal and abs(at_goal.x >> 16) <= 23 and at_goal.z >> 16 <= 24)
    return Contact(tick, chance, (1-chance)*goal, rebound)


def extend(branch, commands, start, flight, one_timer, control, sign):
    keeper, contacts, save_tick = branch.keeper, list(branch.contacts), branch.save_tick
    used = []
    for tick, command in enumerate(commands, start):
        if all(event is not None for event in contacts):
            break
        if command == SAVE and save_tick is None:
            save_tick = tick
        keeper = keeper.step((0, 0) if command == SAVE else PADS[command], save=command == SAVE)
        if not supported(keeper) or save_tick is not None and tick-save_tick >= 12:
            return None
        used.append(command)
        for scenario, offset in enumerate((-1, 0, 1)):
            if contacts[scenario] is not None:
                continue
            puck = flight[max(0, min(len(flight)-1, tick+1+offset))]
            # Stick collision has priority over body contact in checkpuckcoll.
            # Its future animation hotspot is not certified by this model.
            if puck.z >> 16 <= 5 and puck.vz <= 512:
                return None
            if contact_margin(keeper, puck, one_timer=one_timer) >= 0:
                contacts[scenario] = collision(keeper, puck, tick+1, control, sign)
                if contacts[scenario] is None:
                    return None
    return OutcomeBranch(keeper, branch.first if branch.first >= 0 else commands[0],
                         branch.commands+tuple(used), tuple(contacts), save_tick)


def objective(branch):
    events = branch.contacts
    return (sum(e is not None for e in events),
            -max(e.goal_risk if e else 1.0 for e in events),
            min(e.catch if e else 0.0 for e in events),
            sum(e.catch if e else 0.0 for e in events)/3)


def rank(branch, target, baseline):
    return (*objective(branch), -math.dist(branch.keeper.point, target),
            branch.first == baseline, -len(set(branch.commands)))


def adaptive_incumbent(state, manager, keeper, flight, baseline, horizon):
    """Project default positioning each update, including its later C request."""
    projected, preview = copy(state), copy(manager)
    projected.team1, projected.puck = copy(state.team1), copy(state.puck)
    projected.team1.goalie = copy(state.team1.goalie)
    preview.metrics, preview.diagnostics = manager.metrics.copy(), manager.diagnostics.copy()
    commands = []
    for tick in range(horizon):
        goalie, puck = projected.team1.goalie, projected.puck
        for actor, value in ((goalie, keeper), (puck, flight[tick])):
            actor.x, actor.y = value.x >> 16, value.y >> 16
            actor.precise_x, actor.precise_y = value.x/65536, value.y/65536
            actor.motion_x, actor.motion_y = value.vx*17/65536, value.vy*17/65536
        puck.height, puck.motion_z = flight[tick].z/65536, flight[tick].vz*17/65536
        preview.frames = manager.frames+tick
        command = (baseline if tick == 0 else SAVE if keeper.saving else
                   decode(GoalieController._position(preview, projected,
                                                     goalie_target(projected, preview.pass_windup))))
        if command is None:
            return None
        commands.append(command)
        keeper = keeper.step((0, 0) if command == SAVE else PADS[command], save=command == SAVE)
    return commands


def recovery_estimate(state, event, sign):
    """Diagnostic earliest skater reach; an estimate, never a save certificate.

    Existing rating-aware arrival_time includes acceleration/turning. Opposing
    skater decisions are unknown, so this estimate is not used to change inputs.
    """
    arrivals = [None, None]
    for tick, puck in enumerate(free_trajectory(event.rebound, 48, sign), 1):
        if puck.z >> 16 > 5 or puck.vz > 512:
            continue
        for side, team in enumerate((state.team1, state.team2)):
            if arrivals[side] is None and any(player.role and not player.unavailable & 6
                    and arrival_time(player, puck.point[:2]) <= event.tick+tick
                    for player in team.players):
                arrivals[side] = tick
        if all(value is not None for value in arrivals):
            break
    return {'own_skater': arrivals[0], 'opponent_skater': arrivals[1],
            'scope': 'earliest feasible reach, not predicted opponent intent'}


def outcome_search(state, manager, keeper, puck, baseline):
    sign = 1 if state.team1.net.y > 0 else -1
    control = state.team1.goalie.puck_control
    if control is None or not 0 <= control <= 30:
        return None, {'status': 'unsupported-live-puck-control'}
    flight = [puck, *free_trajectory(puck, 49, sign)]
    end = flight[-1]
    if (end.y >> 16)*sign < 261 or abs(end.x >> 16) > 23 or end.z >> 16 > 15:
        return None, {'status': 'no-modeled-goal-threat'}
    horizon = min(24, len(flight)-1)
    one_timer = state.engine.native_one_timer_shot
    target = min(flight[1:], key=lambda p: abs(p.y-keeper.y)).point[:2]
    root = OutcomeBranch(keeper)
    incumbent_commands = adaptive_incumbent(state, manager, keeper, flight, baseline, horizon)
    incumbent = (extend(root, incumbent_commands, 0, flight, one_timer, control, sign)
                 if incumbent_commands else None)
    if incumbent is None:
        return None, {'status': 'unsupported-incumbent'}
    candidates, steps = [root], horizon
    actions = tuple(range(SAVE+int(manager.frames >= manager.save_at)))
    for start in range(0, horizon, BLOCK):
        length = min(BLOCK, horizon-start)
        expanded = []
        for branch in candidates:
            if all(event is not None for event in branch.contacts):
                expanded.append(branch)
                continue
            for command in ((SAVE,) if branch.keeper.saving else actions):
                if steps+length > MAX_STEPS:
                    break
                child = extend(branch, [command]*length, start, flight, one_timer, control, sign)
                steps += length
                if child is not None:
                    expanded.append(child)
        expanded.sort(key=lambda branch: rank(branch, target, baseline), reverse=True)
        candidates, seen = [], set()
        for branch in expanded:
            if branch.first not in seen:
                candidates.append(branch)
                seen.add(branch.first)
            if len(candidates) == BEAM:
                break
        if not candidates:
            break
    chosen = max([incumbent, *candidates], key=lambda branch: rank(branch, target, baseline))
    # Require an outcome gain, never just a nicer post-contact position/margin.
    if objective(chosen) <= objective(incumbent):
        chosen = incumbent
    event = chosen.contacts[1]
    return chosen.first, {'status': 'searched', 'horizon': horizon, 'steps': steps,
        'changed': chosen.first != baseline, 'sequence': chosen.commands,
        'objective': objective(chosen), 'baseline_objective': objective(incumbent),
        'contact_tick': event.tick if event else None,
        'catch_probability': event.catch if event else 0,
        'rebound_goal_risk': event.goal_risk if event else 1,
        'recovery': recovery_estimate(state, event, sign) if event else None}


class OutcomeGoalie(PlanningGoalie):
    """Opt-in outcome evaluator; default Classic remains its fallback."""

    def __init__(self, policy='selective', **kwargs):
        super().__init__(policy, planning='search-24', **kwargs)
        self.planning = 'outcome-24'

    def _search(self, state, keeper, puck, baseline):
        return outcome_search(state, self, keeper, puck, baseline)
