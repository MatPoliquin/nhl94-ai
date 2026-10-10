"""A common, motion-aware utility for executable offensive alternatives.

Units are heuristic possession points, NOT goal probabilities or learned expected
returns. All alternatives use the same terminal value, time cost and risk costs.
Only the first action is committed. A projected continuation is rechecked live.
"""
from collections import Counter
from copy import copy
from dataclasses import asdict, dataclass
import math

from nhl94_ai.agents.carry import body_clearance, carry_path
from nhl94_ai.agents.defense import controlled_slot, eligible, on_ice
from nhl94_ai.agents.finishing import automatic_aim, normal_finish, shot_speed
from nhl94_ai.agents.motion import bounded_projection, facing, velocity
from nhl94_ai.agents.offense import (
    carry_future, goalie_avoidance, normal_shot_release_frames,
    ordinary_shot_conditions, reception_frames, reception_state,
)
from nhl94_ai.agents.passing import (
    evaluate_pass, one_timer_contact_frame, option_receiver, pass_point_at,
    pass_release_frames, pressure_margin,
)
from nhl94_ai.env.target_control import project_target, route_waypoint


VALUE_MODEL = 'possession-utility-v1-not-probability'
CARRY_HORIZON = 18
TIME_COST = 0.12
UNCERTAINTY_COST = 0.06
SETTLE_FRAMES = 4


def _unit(value):
    return max(0.0, min(1.0, value))


@dataclass(frozen=True)
class PossessionScore:
    terminal: float
    time_cost: float = 0.0
    pressure_cost: float = 0.0
    turnover_cost: float = 0.0
    uncertainty_cost: float = 0.0
    continuation: str = 'retain'
    continuation_slot: int | None = None
    elapsed_frames: float = 0.0

    @property
    def value(self):
        return self.terminal - self.time_cost - self.pressure_cost - self.turnover_cost - self.uncertainty_cost

    def snapshot(self):
        return {**asdict(self), 'value': self.value, 'value_model': VALUE_MODEL}


@dataclass(frozen=True)
class PossessionChoice:
    kind: str
    target: tuple
    score: PossessionScore
    option: object = None
    finish: object = None
    maneuver: object = None


def position_value(state, player):
    """Territory and usable central space, without counting remote bypassed bodies."""
    sign = 1 if state.team2.net.y > state.team1.net.y else -1
    depth = max(-264, min(244, player.y * sign))
    return (depth + 264) * 0.045 + 6 * _unit(1 - abs(player.x) / 100)


def turnover_danger(state, point):
    """Same risk is more expensive near our slot; units match possession utility."""
    sign = 1 if state.team2.net.y > state.team1.net.y else -1
    own_depth = _unit((88 - point[1] * sign) / 300)
    central = _unit(1 - abs(point[0]) / 120)
    return 10 + 24 * own_depth * (0.5 + 0.5 * central)


def transition_score(state, terminal, point, elapsed, margin, *, robustness=1.0,
                     uncertainty=0.0, continuation='retain', continuation_slot=None):
    """Soft race/scenario costs; neither scenario frequency nor risk is a probability."""
    pressure = _unit((8 - margin) / 16)
    risk = max(pressure, 1 - robustness)
    return PossessionScore(
        terminal, TIME_COST * elapsed, 8 * pressure, risk * turnover_danger(state, point),
        UNCERTAINTY_COST * elapsed + uncertainty, continuation, continuation_slot, elapsed)


def _segment_distance(start, end):
    dx, dy = end[0] - start[0], end[1] - start[1]
    t = _unit(-(start[0] * dx + start[1] * dy) / max(0.01, dx * dx + dy * dy))
    return math.hypot(start[0] + dx * t, start[1] + dy * t)


def shot_quality(state, shooter, point, side, delay, speed):
    """Continuous two-dimensional shot utility, with both teams' moving blockers.

    This is a shared estimate, not a complete ROM shot/goalie/rebound simulator.
    Goalie velocity is evidence over 12 frames; longer forecasts pay uncertainty.
    """
    sign = 1 if state.team2.net.y > state.team1.net.y else -1
    if not 150 < point[1] * sign < abs(state.team2.net.y) or abs(point[0]) > 80:
        return 0.0
    goal = side * 16, state.team2.net.y
    travel = math.dist(point, goal) / speed
    clearance = 24.0
    for team in (state.team1, state.team2):
        for other in (*team.players, team.goalie):
            if other is shooter or not on_ice(other):
                continue
            goalie = other is state.team1.goalie or other is state.team2.goalie
            vx, vy = velocity(other)
            start_time, end_time = ((min(delay, 12), min(delay + travel, 12)) if goalie
                                    else (delay, delay + travel))
            start = (point[0] - other.x - vx * start_time, point[1] - other.y - vy * start_time)
            end = (goal[0] - other.x - vx * end_time, goal[1] - other.y - vy * end_time)
            radius = 12 if goalie else 8
            clearance = min(clearance, _segment_distance(start, end) - radius - other.projection_uncertainty)
            if not goalie and other.stick_x is not None and other.stick_y is not None:
                clearance = min(clearance, _segment_distance(
                    (start[0] - other.stick_x, start[1] - other.stick_y),
                    (end[0] - other.stick_x, end[1] - other.stick_y)) - 8 - other.projection_uncertainty)
    accuracy = 15 if shooter.shot_accuracy is None else shooter.shot_accuracy
    distance = math.dist(point, goal)
    geometry = _unit(1 - distance / 220) * (0.7 + 0.3 * _unit(accuracy / 30))
    return 100 * geometry * (0.15 + 0.85 * _unit((clearance + 10) / 24))


def coast_scene(state, frames):
    """Advance a settled carrier and all actors once; keep puck and velocity coherent."""
    future = copy(state)
    for name in ('team1', 'team2'):
        source = getattr(state, name)
        team = copy(source)
        actors = []
        for other in (*source.players, source.goalie):
            actor = copy(other)
            if on_ice(actor):
                (actor.x, actor.y), actor.projection_uncertainty = bounded_projection(
                    other, frames, boards=other is not source.goalie)
                actor.precise_x, actor.precise_y = float(actor.x), float(actor.y)
            actors.append(actor)
        team.players, team.goalie = actors[:-1], actors[-1]
        setattr(future, name, team)
    player = state.team1.get_player_by_scnum(state.engine.puck_owner)
    carrier = future.team1.get_player_by_scnum(state.engine.puck_owner)
    future.puck = copy(state.puck)
    future.puck.x += carrier.x - player.x
    future.puck.y += carrier.y - player.y
    future.puck.motion_x, future.puck.motion_y = velocity(carrier)
    future.puck.precise_x, future.puck.precise_y = float(future.puck.x), float(future.puck.y)
    return future


class PossessionValuePlanner:
    """Bounded candidate search; at most one additional pass after reception."""

    def __init__(self):
        self.diagnostics = {}
        self.metrics = Counter()
        self.interval = 4
        self.one_timers = True
        self.one_timer_at = 0
        self.pass_timing = False
        self.refine_finishing = False

    @staticmethod
    def available(state):
        player = state.team1.get_player_by_scnum(state.engine.puck_owner)
        return (state.numPlayers == 5 and state.engine.puck_owner_known and player is not None
                and player is not state.team1.goalie and eligible(player)
                and controlled_slot(state.team1) == state.engine.puck_owner
                and all(getattr(player, key) is not None for key in (
                    'motion_x', 'motion_y', 'speed', 'agility', 'weight', 'energy')))

    def _passes(self, state, purpose):
        player = state.team1.get_player_by_scnum(state.engine.puck_owner)
        options, rows = [], []
        for index, receiver in enumerate(state.team1.players):
            if receiver is player:
                continue
            option, details = evaluate_pass(
                state, player, index, receiver, purpose, decision_interval=self.interval,
                release_prediction=self.pass_timing, rank_risk=True)
            details = {**details, 'purpose': purpose}
            # "safe" is the legacy evaluator's label. A ranked option may be contested.
            if option is not None:
                details.update(status='eligible', risk_model='soft-margin-and-scenario-cost')
                options.append(option)
            rows.append(details)
        return options, rows

    def _shot(self, state):
        player = state.team1.get_player_by_scnum(state.engine.puck_owner)
        escape = goalie_avoidance(state, player, (player.x, state.team2.net.y))
        if not any(ordinary_shot_conditions(state, player, self.interval, escape)):
            return None
        finish = normal_finish(state, player, self.interval) if self.refine_finishing else None
        side = finish.side if finish is not None else (-1 if state.team2.goalie.x > 0 else 1)
        hold = finish.hold_frames if finish is not None else self.interval
        release = normal_shot_release_frames(player, hold)
        path = carry_path(player, None, release)
        if path is None:
            return None
        point = state.puck.x + path[-1].x - player.x, state.puck.y + path[-1].y - player.y
        value = shot_quality(state, player, point, side, release, shot_speed(player, hold))
        margin = pressure_margin(state.team2, point, release)
        return PossessionChoice('shoot', (side * 13, state.team2.net.y),
                                transition_score(state, value, point, release, margin,
                                                 continuation='shot'), finish=finish)

    def _one_timer(self, state, option):
        shooter = state.team1.players[option.index]
        receiver = option_receiver(shooter, option)
        release = option.release_frames if option.release_frames is not None else pass_release_frames(receiver)
        contact = one_timer_contact_frame(state.puck, receiver, option.contact, release)
        if contact is None:
            return None
        point = pass_point_at(state.puck, receiver, option.contact, contact - release)
        side = automatic_aim(state, point, contact)
        value = shot_quality(state, shooter, point, side, contact, shot_speed(shooter, 0, one_timer=True))
        score = transition_score(state, value, point, contact, option.margin,
                                 robustness=option.robustness, continuation='one-timer',
                                 continuation_slot=option.slot)
        return PossessionChoice('one-timer', option.point, score, option=option)

    def _terminal(self, state, frame):
        """Retain, actually executable shot, or one more pass; never recurse."""
        player = state.team1.get_player_by_scnum(state.engine.puck_owner)
        best = position_value(state, player), 'retain', None
        shot = self._shot(state)
        if shot is not None and shot.score.value > best[0]:
            best = shot.score.value, 'shoot', None
        if self.one_timers and frame >= self.one_timer_at:
            options, _ = self._passes(state, 'one-timer')
            for option in options:
                candidate = self._one_timer(state, option)
                if candidate is not None and candidate.score.value > best[0]:
                    best = candidate.score.value, 'one-timer', option.slot
        return best

    def score_pass(self, state, option, frame):
        elapsed = reception_frames(state, option)
        if elapsed is None:
            return None
        future = reception_state(state, option, elapsed)
        # The default executor next acts on its tactical cadence after reception.
        settle = max(SETTLE_FRAMES, math.ceil(elapsed / self.interval) * self.interval - elapsed)
        future = coast_scene(future, settle)
        elapsed += settle
        receiver = future.team1.players[option.index]
        if receiver.projection_uncertainty:
            return None
        terminal, continuation, slot = self._terminal(future, frame + elapsed)
        point = receiver.x, receiver.y
        margin = min(option.margin, pressure_margin(state.team2, point, elapsed))
        return transition_score(state, terminal, point, elapsed, margin,
                                robustness=option.robustness, continuation=continuation,
                                continuation_slot=slot)

    def _carries(self, state, offense, frame):
        player = state.team1.get_player_by_scnum(state.engine.puck_owner)
        horizon = max(CARRY_HORIZON, self.interval)
        sign = 1 if state.team2.net.y > state.team1.net.y else -1
        fx, fy = facing(player)
        fallback, _ = offense.fallback_carry(state, player)
        targets = (fallback, offense.carry_target(state, player),
                   (player.x, player.y + sign * 24),
                   (player.x - 26, player.y + sign * 8), (player.x + 26, player.y + sign * 8),
                   (player.x - fx * 48, player.y - fy * 48))
        choices, rows, seen = [], [], set()
        for target in targets:
            target = route_waypoint((player.x, player.y), project_target(target))
            if target in seen:
                continue
            seen.add(target)
            row = {'kind': 'carry', 'target': target, 'status': 'unsafe-route'}
            rows.append(row)
            path = carry_path(player, target, horizon, decision_interval=self.interval)
            if path is None or body_clearance(state, player, path[:self.interval + 1]) <= 0:
                continue
            # Preserve goalie avoidance for the complete evaluated route.
            escape = goalie_avoidance(state, player, target, decision_interval=self.interval)
            if escape is not None and not escape[1]['goalie_avoidance_safe']:
                continue
            if escape is not None and math.dist(escape[0], target) > 1:
                continue
            future = carry_future(state, player, path[-1], frames=horizon)
            future.puck.motion_x, future.puck.motion_y = velocity(path[-1])
            terminal, continuation, slot = self._terminal(future, frame + horizon)
            margin = min(pressure_margin(state.team2, (path[t].x, path[t].y), t)
                         for t in (4, 8, 12, horizon))
            clearance = body_clearance(state, player, path)
            score = transition_score(
                state, terminal, (path[-1].x, path[-1].y), horizon, margin,
                uncertainty=max(0, -clearance), continuation=continuation, continuation_slot=slot)
            choices.append(PossessionChoice('carry', target, score))
            row.update(status='eligible', **score.snapshot())
        return choices, rows

    def _maneuvers(self, state, frame, finishers):
        choices, rows = [], []
        player = state.team1.get_player_by_scnum(state.engine.puck_owner)
        for kind, controller, evaluate in finishers:
            if controller is None or frame < controller.retry_at:
                continue
            controller.metrics['evaluations'] += 1
            plan, details = evaluate(state, {})
            controller.diagnostics = details
            row = {'kind': kind, 'status': details['status']}
            rows.append(row)
            if plan is None:
                controller.metrics['rejected-' + details['status']] += 1
                continue
            if kind == 'cross_crease':
                point, elapsed, hold = plan.release_point, plan.approach_frames + plan.shot_frames, 24
            else:
                preview = next(item for item in details['candidates'] if item.get('status') == 'feasible'
                               and item['direction'] == plan.direction and item['target'] == plan.target)
                point, hold = preview['release_point'], preview['hold_frames']
                elapsed = plan.frames + normal_shot_release_frames(player, hold)
            value = shot_quality(state, player, point, plan.aim_side, elapsed, shot_speed(player, hold))
            score = transition_score(state, value, point, elapsed, plan.pressure, continuation='shot')
            choices.append(PossessionChoice(kind, plan.target, score, maneuver=plan))
            row.update(status='eligible', target=plan.target, **score.snapshot())
        return choices, rows

    def choose(self, state, offense, frame, *, one_timer_at=0, refine_finishing=False, finishers=()):
        self.interval, self.one_timers = offense.decision_interval, offense.one_timers
        self.one_timer_at, self.pass_timing = one_timer_at, offense.pass_timing
        self.refine_finishing = refine_finishing
        choices, rows = self._carries(state, offense, frame)
        maneuvers, maneuver_rows = self._maneuvers(state, frame, finishers)
        choices.extend(maneuvers)
        rows.extend(maneuver_rows)
        shot = self._shot(state)
        if shot is not None:
            choices.append(shot)
            rows.append({'kind': 'shoot', 'target': shot.target, 'status': 'eligible', **shot.score.snapshot()})
        pass_rows, timer_rows = [], []
        for purpose in ('position', 'one-timer'):
            if purpose == 'one-timer' and (not self.one_timers or frame < one_timer_at):
                continue
            options, details = self._passes(state, purpose)
            for option in options:
                candidate = self._one_timer(state, option) if purpose == 'one-timer' else None
                if purpose != 'one-timer':
                    score = self.score_pass(state, option, frame)
                    if score is not None:
                        candidate = PossessionChoice('pass', option.point, score, option=option)
                row = next(item for item in details if item['slot'] == option.slot)
                if candidate is None:
                    row.update(status='unforecastable-continuation')
                    continue
                row.update(candidate.score.snapshot())
                if purpose != 'one-timer' and frame < offense.pass_at:
                    row.update(status='pass-cooldown')
                    continue
                choices.append(candidate)
                rows.append({'kind': candidate.kind, 'slot': option.slot, 'target': option.point,
                             'status': 'eligible', **candidate.score.snapshot()})
            if purpose == 'one-timer':
                timer_rows = details
            else:
                pass_rows = details
        # Prefer less commitment on exact ties. No tactical category has an override.
        best = max(choices, key=lambda choice: (choice.score.value, -choice.score.elapsed_frames), default=None)
        self.diagnostics = {
            'evaluation_frame': frame, 'evaluation_carrier': state.engine.puck_owner,
            'value_model': VALUE_MODEL, 'possession_candidates': rows,
            'candidates': pass_rows, 'one_timer_candidates': timer_rows,
            'status': 'selected' if best is not None else 'no-executable-candidate',
            'reason': 'highest shared possession utility; continuations rechecked live',
        }
        if best is not None:
            self.diagnostics.update(possession_score=best.score.snapshot(), possession_action=best.kind,
                                    target=best.target, desired_slot=best.option.slot if best.option else state.engine.puck_owner)
            self.metrics['selected:' + best.kind] += 1
            self.metrics[best.kind + ':next:' + best.score.continuation] += 1
        else:
            self.metrics['no-executable-candidate'] += 1
        return best
