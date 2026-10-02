"""Target first, defender second: bounded lane protection and puck races."""
from dataclasses import asdict, dataclass, replace
import math

import numpy as np

from nhl94_ai.env.target_control import project_target, route_waypoint
from nhl94_ai.game.constants import GameConsts as Buttons
from nhl94_ai.agents.motion import (
    LANE_HORIZON, SHOT_SPEED_RANGE, arrival_time, blocks_shot_lane,
    boost_safe, burst_velocity, check_approach, facing, puck_path, skating, velocity,
)


SWITCH_DELAY = 6
SWITCH_INTERVAL = 12
SWITCH_TIMEOUT = 8
SWITCH_MARGIN = 4


def controlled_slot(team):
    return team.controlled_scnum() if team.defense_control is None else team.defense_control


def owns_puck(team, owner):
    goalie = team.goalie_scnum() if team.defense_goalie is None else team.defense_goalie
    return team.skater_scnum_base() <= owner < team.skater_scnum_base() + len(team.players) or owner == goalie


def eligible(player):
    locked = player.is_falling if player.selection_flags is None else player.selection_flags & 0x20
    return not locked and (player.role is None or player.role > 0) and not player.unavailable & 4


def skaters(team):
    return [(team.skater_scnum_base() + i, player)
            for i, player in enumerate(team.players) if eligible(player)]


@dataclass(frozen=True)
class ShotLane:
    origin: tuple
    goal: tuple
    threat_slot: int | None
    blockers: tuple
    release_delay: float


@dataclass(frozen=True)
class DefensePlan:
    target: tuple
    mode: str
    reason: str
    threat_slot: int | None = None
    puck_arrival: float | None = None
    opponent_arrival: float | None = None
    receiver: tuple | None = None
    shot_goal: tuple | None = None
    lanes: tuple = ()
    reserved_slots: tuple = ()


class DefenseController:
    def __init__(self):
        self.frames = 0
        self.plan = None
        self.desired_slot = None
        self.switch_at = self.poke_at = self.boost_at = 0
        self.switch_attempts = 0
        self.pending_switch = None
        self.pending_check = None
        self.last_switch_result = None
        self.last_selection = None
        self.b_down = self.c_down = False
        self.last_owner = -1
        self.diagnostics = {}

    def idle(self, elapsed):
        self.frames += elapsed
        self.plan = self.desired_slot = None
        self.switch_attempts = 0
        self.pending_switch = self.last_switch_result = self.last_selection = None
        self.pending_check = None
        self.b_down = self.c_down = False
        self.diagnostics = {}

    @staticmethod
    def _goal_side(team, point, threat, slack=0):
        sign = 1 if team.net.y > 0 else -1
        return point[1] * sign >= threat[1] * sign - slack

    def _shot_lanes(self, state, threat, threat_slot=None, excluding=None, delay=0):
        sign = 1 if state.team1.net.y > 0 else -1
        in_front = threat[1] * sign < abs(state.team1.net.y)
        friends = skaters(state.team1)
        return tuple(
            ShotLane(threat, goal, threat_slot, tuple(
                slot for slot, player in friends if slot != excluding and in_front
                and (self.pending_switch is None or slot != self.pending_switch['to_slot'])
                and blocks_shot_lane(player, threat, goal, delay)), delay)
            for goal in state._shot_target_points_for_net(state.team1.net)
        )

    def _covered(self, state, target, excluding=None, delay=0):
        return all(lane.blockers for lane in self._shot_lanes(state, target, excluding=excluding, delay=delay))

    def _open_receiver(self, state, threat, carrier_slot, primary):
        actual = controlled_slot(state.team1)
        sign = 1 if state.team1.net.y > 0 else -1
        carrier = state.team2.get_player_by_scnum(carrier_slot)
        carrier_vx, carrier_vy = velocity(carrier)
        immediate = self._shot_lanes(state, (carrier.x, carrier.y), carrier_slot, actual)
        choices = []
        for slot, player in skaters(state.team2):
            if slot == carrier_slot:
                continue
            vx, vy = velocity(player)
            delay = math.dist(threat, (player.x, player.y)) / SHOT_SPEED_RANGE[0]
            point = player.x + vx * delay, player.y + vy * delay
            delay = math.dist(threat, point) / SHOT_SPEED_RANGE[0]
            if (delay > LANE_HORIZON or abs(point[0]) > Buttons.SLOT_BAND_MAX_X
                    or not Buttons.SLOT_BAND_MIN_Y < point[1] * sign < 244):
                continue
            lanes = self._shot_lanes(state, point, slot, actual, delay)
            open_lanes = [lane for lane in lanes if not lane.blockers]
            if not open_lanes:
                continue
            # Do not leave a covered carrier if that cover will drift away
            # before the possible pass reaches its receiver.
            future_origin = carrier.x + carrier_vx * delay, carrier.y + carrier_vy * delay
            if math.dist(future_origin, project_target(future_origin)) > 1e-6:
                continue
            future = self._shot_lanes(state, future_origin, carrier_slot, actual, delay)
            retained = tuple(replace(lane, blockers=tuple(
                blocker for blocker in lane.blockers if blocker in now.blockers and blocker in later.blockers))
                for lane, now, later in zip(primary, immediate, future))
            if not all(lane.blockers for lane in retained):
                continue
            danger_time = delay + min(math.dist(point, lane.goal) for lane in open_lanes) / SHOT_SPEED_RANGE[-1]
            choices.append((danger_time, slot, point, lanes, retained))
        return min(choices, key=lambda row: (row[0], row[1])) if choices else None

    def _lane(self, state, threat, slot=None, reason='protect shooting lane', receiver=None,
              delay=0, check_receivers=False):
        lanes = self._shot_lanes(state, threat, slot, controlled_slot(state.team1), delay)
        open_lanes = [lane for lane in lanes if not lane.blockers]
        primary = ()
        if not open_lanes and check_receivers:
            alternative = self._open_receiver(state, threat, slot, lanes)
            if alternative:
                _, slot, threat, lanes, primary = alternative
                open_lanes = [lane for lane in lanes if not lane.blockers]
                receiver, reason = threat, 'teammate covers carrier; deny uncovered receiver lane'
            else:
                reason = 'teammate covers shot lanes; retain goal-side support'
        sign = 1 if state.team1.net.y > 0 else -1
        depth = threat[1] * sign
        goal_x = max(-12, min(12, -state.team1.goalie.x * 0.5))
        reserved = ()
        if open_lanes and (primary or len(open_lanes) < len(lanes)):
            goal_x = max(open_lanes, key=lambda lane: abs(lane.goal[0] - state.team1.goalie.x)).goal[0]
            reserved = tuple(sorted({lane.blockers[0] for lane in (*primary, *lanes) if lane.blockers}))
            if not primary:
                reason = 'cover goal-mouth gap beside teammate'
        delta = goal_x - threat[0], state.team1.net.y - threat[1]
        length = max(1, math.hypot(*delta))
        gap = min(34, max(14, (264 - depth) * 0.18))
        target = (threat[0] + delta[0] * gap / length,
                  sign * min(244, depth + abs(delta[1]) * gap / length))
        mode = 'protect-lane' if receiver is None else 'deny-reception'
        if receiver is None and not reserved and abs(threat[0]) > 82 and -50 < depth < 232:
            inside = [slot for slot, player in skaters(state.team1)
                      if abs(player.x) < abs(threat[0]) - 8 and math.dist((player.x, player.y), threat) < 75]
            receiver_lanes = tuple(
                lane for other_slot, player in skaters(state.team2)
                if other_slot != slot and abs(player.x) < 55 and depth + 15 < player.y * sign < 244
                for lane in self._shot_lanes(
                    state, (player.x, player.y), other_slot, controlled_slot(state.team1),
                    math.dist(threat, (player.x, player.y)) / SHOT_SPEED_RANGE[0]))
            if inside and all(lane.blockers for lane in receiver_lanes):
                target = (threat[0] - math.copysign(20, threat[0]), threat[1] + sign * 12)
                mode, reason = 'contain-boards', 'close inside escape; retain goal-side position'
                primary = (*primary, *receiver_lanes)
                reserved = tuple(sorted({lane.blockers[0] for lane in receiver_lanes}))
        return DefensePlan(project_target(target), mode, reason, slot, receiver=receiver,
                           shot_goal=(goal_x, state.team1.net.y), lanes=(*primary, *lanes),
                           reserved_slots=reserved)

    def _cost(self, state, slot, player, target):
        actual = controlled_slot(state.team1)
        switch_delay = 0
        if slot != actual:
            switch_delay = SWITCH_DELAY + max(0, self.switch_at - self.frames)
            if self.pending_switch and slot == self.pending_switch['to_slot']:
                switch_delay = max(0, self.pending_switch['frame'] + SWITCH_DELAY - self.frames)
        cost = arrival_time(player, target, boost=self.frames >= self.boost_at) + switch_delay
        if (not self._goal_side(state.team1, target, (state.puck.x, state.puck.y), 12)
                and not self._covered(state, (state.puck.x, state.puck.y), excluding=slot)):
            cost += 24
        return cost

    @staticmethod
    def _likely_switch(state):
        vx, vy = velocity(state.puck)
        # chgplayer adds the signed velocity high bytes, includes the current
        # skater (which means sweep-check), and resolves ties toward later slots.
        point = state.puck.x + math.floor(vx * 256 / 17), state.puck.y + math.floor(vy * 256 / 17)
        actual = controlled_slot(state.team1)
        candidates = [
            (slot, player) for slot, player in skaters(state.team1)
            if player.selection_flags is None or (
                not player.selection_flags & 0x20 and (slot == actual or not player.selection_flags & 8))
        ]
        return min(candidates, key=lambda row: (
            math.dist((row[1].x, row[1].y), point), -row[0]))[0] if candidates else None

    def _observe_switch(self, actual):
        if actual >= 0 and actual != self.last_selection:
            self.switch_attempts = 0
        self.last_selection = actual
        request = self.pending_switch
        if request is None:
            return
        elapsed = self.frames - request['frame']
        if actual >= 0 and actual != request['from_slot']:
            outcome = 'confirmed' if actual == request['to_slot'] else 'unexpected-selection'
        elif elapsed >= SWITCH_TIMEOUT:
            outcome = 'unconfirmed'
            self.switch_at = self.frames + min(24, 4 * self.switch_attempts)
        else:
            return
        self.last_switch_result = {
            **request, 'actual_slot': actual, 'elapsed_frames': elapsed, 'outcome': outcome,
        }
        self.pending_switch = None

    @staticmethod
    def _race_margin(player):
        missing = player.motion_x is None or player.speed is None or player.energy is None
        return 10 if missing else 5

    def choose_target(self, state):
        owner = state.engine.puck_owner
        opponents = skaters(state.team2)
        goalie = state.team2.goalie
        goalie_slot = state.team2.goalie_scnum() if state.team2.defense_goalie is None else state.team2.defense_goalie
        if owner == goalie_slot:
            if opponents:
                slot, receiver = min(opponents, key=lambda row: math.dist(
                    (row[1].x, row[1].y), (goalie.x, goalie.y)))
                return replace(self._lane(state, (receiver.x, receiver.y), slot),
                               mode='deny-goalie-outlet', reason='goalie owns puck; cover outlet instead of chasing crease')
            sign = 1 if state.team1.net.y > state.team2.net.y else -1
            return DefensePlan(project_target((goalie.x, goalie.y + sign * 48)), 'deny-goalie-outlet',
                               'goalie owns puck; retreat clear of crease', goalie_slot)
        carrier = next(((slot, player) for slot, player in opponents if slot == owner), None)
        if carrier is not None:
            slot, player = carrier
            vx, vy = velocity(player)
            threat = project_target((player.x + vx * 8, player.y + vy * 8))
            return self._lane(state, threat, slot, check_receivers=True)

        path = puck_path(state.puck)
        friends = skaters(state.team1)
        actual, likely = controlled_slot(state.team1), self._likely_switch(state)
        reachable = [(slot, player) for slot, player in friends
                     if slot == actual or actual >= 0 and slot == likely]
        reception = None
        for time, point, height in path:
            if height > 8:
                continue
            arrivals = [(arrival_time(player, point, optimistic=True), slot, player)
                        for slot, player in opponents]
            if arrivals:
                enemy_time, slot, player = min(arrivals, key=lambda row: row[0])
                if enemy_time <= time:
                    reception = (time, point, slot, player)
                    break
        previous = state.engine.last_puck_player
        if previous is None:
            previous = self.last_owner
        passing = (owns_puck(state.team2, previous)
                   and math.hypot(*velocity(state.puck)) > 0.8)
        # passplayer can remain stale after reception. Only use it when a live
        # trajectory reaches that receiver; never follow the slot in isolation.
        if passing and state.engine.pass_target is not None:
            intended = next(((slot, player) for slot, player in opponents
                             if slot == state.engine.pass_target), None)
            if intended is not None:
                slot, player = intended
                vx, vy = velocity(player)
                for time, point, height in path:
                    receiver = player.x + vx * time, player.y + vy * time
                    if height <= 8 and math.dist(point, receiver) <= 20:
                        if reception is None or time <= reception[0] + 4:
                            reception = (time, point, slot, player)
                        break
        reason = 'no safe race before opponent reception' if reception else 'uncertain or unreachable puck path'
        puck_known = all(value is not None for value in (
            state.puck.motion_x, state.puck.motion_y, state.puck.height))
        for time, point, height in path:
            if not puck_known or height > 8 or reception and time >= reception[0] - 3:
                continue
            candidates = [(self._cost(state, slot, player, point), slot, player) for slot, player in reachable]
            if not candidates:
                break
            ours, slot, player = min(candidates, key=lambda row: row[0])
            enemy = min((arrival_time(rival, point, optimistic=True) for _, rival in opponents), default=math.inf)
            margin = self._race_margin(player)
            # Recovery requires time to settle the puck, not just win a collision.
            settle = 4 + max(0, 15 - (player.stick if player.stick is not None else 5)) / 3
            if ours + margin > time or enemy <= time + settle + margin:
                continue
            if (not self._goal_side(state.team1, point, (state.puck.x, state.puck.y), 12)
                    and not self._covered(state, (state.puck.x, state.puck.y), excluding=slot, delay=time)):
                reason = 'recovery would abandon the last goal-side defender'
                continue
            relative = math.dist(velocity(player), velocity(state.puck))
            if relative > 3.5 and time < ours + 12:
                reason = 'insufficient time to control the incoming puck'
                continue
            return DefensePlan(point, 'intercept-pass' if passing else 'recover-safe',
                               'win the puck race with control and coverage margin',
                               reception[2] if reception else None, time,
                               None if math.isinf(enemy) else enemy)
        if reception is not None:
            time, point, slot, _ = reception
            plan = self._lane(state, point, slot, reason, point, delay=time)
            return replace(plan, puck_arrival=time, opponent_arrival=time)
        if opponents:
            slot, player = min(opponents, key=lambda row: arrival_time(
                row[1], (state.puck.x, state.puck.y), optimistic=True))
            vx, vy = velocity(player)
            threat = project_target((player.x + vx * 8, player.y + vy * 8))
            return self._lane(state, threat, slot, reason)
        return self._lane(state, (state.puck.x, state.puck.y), reason=reason)

    def select_player(self, state, target):
        reserved = self.plan.reserved_slots if self.plan else ()
        candidates = [(slot, player) for slot, player in skaters(state.team1) if slot not in reserved]
        costs = {slot: self._cost(state, slot, player, target) for slot, player in candidates}
        actual = controlled_slot(state.team1)
        available = (actual, self._likely_switch(state))
        reachable = {slot: cost for slot, cost in costs.items() if slot in available}
        best = min(reachable, key=reachable.get) if reachable else None
        current_can_wait = True
        if self.plan and self.plan.mode in ('recover-safe', 'intercept-pass') and actual in costs:
            player = next(player for slot, player in candidates if slot == actual)
            current_can_wait = costs[actual] + self._race_margin(player) <= self.plan.puck_arrival
        if actual in costs and costs[actual] <= costs[best] + SWITCH_MARGIN and current_can_wait:
            best = actual
        self.desired_slot = best
        return best, costs

    def _body_check(self, state, actual, player):
        owner = state.engine.puck_owner
        carrier = next((p for slot, p in skaters(state.team2) if slot == owner), None)
        details = {'status': 'no-active-carrier', 'carrier_slot': owner}
        if not state.engine.puck_owner_known or carrier is None:
            return details
        if self.pending_check:
            return {**details, 'status': 'follow-through'}
        if player is None:
            return {**details, 'status': 'unavailable-checker'}
        required = ('motion_x', 'motion_y', 'weight', 'selection_flags', 'facing')
        if any(getattr(p, name) is None for p in (player, carrier) for name in required) or player.energy is None:
            return {**details, 'status': 'missing-feedback'}
        if (not player.selection_flags & 8 or any(
                p.selection_flags & 4 or p.unavailable & 0x21 for p in (player, carrier))):
            return {**details, 'status': 'unavailable-contact'}
        if self.pending_switch or self.frames < self.switch_at:
            return {**details, 'status': 'selection-busy'}
        if self.frames < self.boost_at:
            return {**details, 'status': 'cooldown'}
        if self.b_down or self.c_down:
            return {**details, 'status': 'release-buttons'}
        if player.energy < 1024:
            return {**details, 'status': 'low-energy'}
        dx, dy = carrier.x - player.x, carrier.y - player.y
        distance = math.hypot(dx, dy)
        if distance > 30:
            return {**details, 'status': 'outside-range'}
        if not self._goal_side(state.team1, (player.x, player.y), (carrier.x, carrier.y)):
            return {**details, 'status': 'not-goal-side'}
        fx, fy = facing(player)
        if dx * fx + dy * fy < distance * 0.9:
            return {**details, 'status': 'not-facing-carrier'}
        if self.plan and self.plan.threat_slot not in (None, owner):
            return {**details, 'status': 'covering-other-threat'}
        approach = check_approach(player, carrier)
        if approach is None:
            return {**details, 'status': 'no-contact-course'}
        time, point, impact = approach
        threshold = ((240 - player.weight + carrier.weight) & 0xFF) >> 1
        details.update(contact_frames=time, contact_point=point, impact_estimate=impact,
                       weight_threshold=threshold)
        if impact < 20 or impact < threshold + 4:
            return {**details, 'status': 'unfavorable-impact'}
        duration = time + 2
        vx, vy = burst_velocity(player)
        end = player.x + vx * duration, player.y + vy * duration
        tx, ty = velocity(carrier)
        carrier_end = carrier.x + tx * duration, carrier.y + ty * duration
        if (not self._goal_side(state.team1, end, carrier_end, 4)
                or math.dist(end, project_target(end)) > 1e-6
                or route_waypoint((player.x, player.y), end) != end):
            return {**details, 'status': 'unsafe-route'}
        for team in (state.team1, state.team2):
            for other in (*team.players, team.goalie):
                if other is player or other is carrier or other.role is not None and other.role < 0:
                    continue
                ox, oy = velocity(other)
                rx, ry = vx - ox, vy - oy
                dx, dy = player.x - other.x, player.y - other.y
                closest = max(0, min(duration, -(dx * rx + dy * ry) / max(0.01, rx * rx + ry * ry)))
                if math.hypot(dx + rx * closest, dy + ry * closest) <= 16:
                    return {**details, 'status': 'obstructed-contact'}
        sign = 1 if state.team1.net.y > 0 else -1
        for slot, receiver in skaters(state.team2):
            if (slot == owner or abs(receiver.x) > Buttons.SLOT_BAND_MAX_X
                    or not Buttons.SLOT_BAND_MIN_Y < receiver.y * sign < 244):
                continue
            lanes = self._shot_lanes(state, (receiver.x, receiver.y), slot)
            if any(lane.blockers == (actual,) for lane in lanes):
                return {**details, 'status': 'leaves-receiver-lane'}
        return {**details, 'status': 'ready'}

    def step(self, state, elapsed=1):
        self.frames += elapsed
        actual = controlled_slot(state.team1)
        self._observe_switch(actual)
        if owns_puck(state.team1, state.engine.puck_owner):
            self.idle(0)
            return np.zeros(Buttons.INPUT_MAX, dtype=np.int8)
        if self.pending_check and (
                self.frames >= self.pending_check['until'] or actual != self.pending_check['from_slot']
                or state.engine.puck_owner != self.pending_check['carrier_slot']):
            self.pending_check = None
        plan = self.choose_target(state)
        if (self.plan and plan.mode in ('protect-lane', 'contain-boards')
                and (plan.mode, plan.threat_slot) == (self.plan.mode, self.plan.threat_slot)
                and (plan.shot_goal, plan.reserved_slots) == (self.plan.shot_goal, self.plan.reserved_slots)
                and math.dist(plan.target, self.plan.target) < 3):
            plan = replace(plan, target=self.plan.target)
        self.plan = plan
        best, costs = self.select_player(state, plan.target)
        action = np.zeros(Buttons.INPUT_MAX, dtype=np.int8)
        mode, waypoint = 'waiting-for-selection', plan.target
        player = next((p for slot, p in skaters(state.team1) if slot == actual), None)
        check = self._body_check(state, actual, player)
        puck_vx, puck_vy = velocity(state.puck)
        likely = self._likely_switch(state)
        if actual < 0:
            switch_status = 'no-selection'
        elif self.pending_check:
            switch_status = 'check-follow-through'
        elif self.pending_switch:
            switch_status = 'awaiting-selection'
        elif best is None:
            switch_status = 'no-eligible-skater'
        elif likely == actual:
            switch_status = 'rom-keeps-current'
        elif best == actual:
            switch_status = 'no-useful-switch'
        elif self.frames < self.switch_at:
            switch_status = 'cooldown'
        elif self.b_down:
            switch_status = 'release-b'
        else:
            switch_status = 'request'
        if switch_status == 'request':
            action[Buttons.INPUT_B] = 1
            self.switch_at, self.poke_at = self.frames + SWITCH_INTERVAL, self.frames + 16
            self.switch_attempts += 1
            self.pending_switch = {'from_slot': actual, 'to_slot': best, 'frame': self.frames}
            mode = 'switch-request'
        elif self.pending_check:
            mode = 'check-follow-through'
        elif player is not None:
            waypoint = route_waypoint((player.x, player.y), plan.target)
            vx, vy = velocity(player)
            acceleration, _, _ = skating(player)
            brake_x = vx * abs(vx) / (2 * acceleration + abs(vx) / 64)
            brake_y = vy * abs(vy) / (2 * acceleration + abs(vy) / 64)
            corrections = waypoint[0] - player.x - brake_x, waypoint[1] - player.y - brake_y
            for delta, negative, positive in (
                (corrections[0], Buttons.INPUT_LEFT, Buttons.INPUT_RIGHT),
                (corrections[1], Buttons.INPUT_DOWN, Buttons.INPUT_UP),
            ):
                if abs(delta) > 3:
                    action[positive if delta > 0 else negative] = 1
            mode = 'holding' if not action.any() else 'skating'
            direction = facing(player)
            dx, dy = state.puck.x - player.x, state.puck.y - player.y
            distance = math.hypot(dx, dy)
            rel_v = puck_vx - vx, puck_vy - vy
            closest_time = max(0, min(4, -(dx * rel_v[0] + dy * rel_v[1]) / max(0.01, rel_v[0]**2 + rel_v[1]**2)))
            contact = math.hypot(dx + rel_v[0] * closest_time, dy + rel_v[1] * closest_time)
            aligned = dx * direction[0] + dy * direction[1] > distance * 0.65
            can_poke = (self.frames >= max(self.poke_at, self.switch_at) and not self.b_down
                        and self.pending_switch is None and likely == actual
                        and state.engine.puck_owner != (
                            state.team2.goalie_scnum() if state.team2.defense_goalie is None else state.team2.defense_goalie))
            goal_side = self._goal_side(state.team1, (player.x, player.y), (state.puck.x, state.puck.y), 5)
            reachable = distance < 22 and contact < 16 and (state.puck.height or 0) < 8
            if check['status'] == 'ready':
                action[4:8] = 0
                action[Buttons.INPUT_C] = 1
                self.boost_at, self.poke_at = self.frames + 36, self.frames + 16
                self.pending_check = {'from_slot': actual, 'carrier_slot': state.engine.puck_owner,
                                      'until': self.frames + check['contact_frames'] + 2}
                mode = 'check-request'
            elif reachable and aligned and goal_side and can_poke:
                action[Buttons.INPUT_B] = 1
                self.poke_at = self.frames + 16
                mode = 'poke-request'
            elif (self.pending_switch is None and self.frames >= self.boost_at
                  and not self.c_down and boost_safe(player, waypoint)):
                action[Buttons.INPUT_C] = 1
                self.boost_at = self.frames + 36
                mode = 'boost-request'
        self.b_down, self.c_down = bool(action[Buttons.INPUT_B]), bool(action[Buttons.INPUT_C])
        if state.engine.puck_owner >= 0:
            self.last_owner = state.engine.puck_owner
        missing = sorted({name for _, p in skaters(state.team1) + skaters(state.team2)
                          for name in ('motion_x', 'motion_y', 'speed', 'agility', 'weight', 'energy',
                                       'selection_flags')
                          if getattr(p, name) is None})
        missing.extend('puck.' + name for name in ('motion_x', 'motion_y', 'height')
                       if getattr(state.puck, name) is None)
        self.diagnostics = {
            **asdict(plan), 'decision': plan.mode, 'mode': mode, 'frame': self.frames,
            'destination': plan.target, 'waypoint': waypoint, 'desired_slot': best,
            'actual_slot': actual, 'acting_slot': actual, 'switch_attempts': self.switch_attempts,
            'ideal_slot': min(costs, key=costs.get) if costs else None,
            'switch_status': switch_status, 'pending_switch': self.pending_switch,
            'check': check, 'pending_check': self.pending_check,
            'last_switch_result': self.last_switch_result,
            'likely_switch_slot': likely, 'arrival_frames': costs, 'missing_feedback': missing,
            'buttons': action.tolist(), 'puck_path': [(t, point) for t, point, _ in puck_path(state.puck, 32)],
            'selection_available': actual >= 0,
        }
        return action
