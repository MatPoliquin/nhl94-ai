"""Target first, defender second: bounded lane protection and puck races."""
from dataclasses import asdict, dataclass
import math

import numpy as np

from nhl94_ai.env.target_control import project_target, route_waypoint
from nhl94_ai.game.constants import GameConsts as Buttons
from nhl94_ai.agents.motion import arrival_time, boost_safe, facing, puck_path, skating, velocity


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
class DefensePlan:
    target: tuple
    mode: str
    reason: str
    threat_slot: int | None = None
    puck_arrival: float | None = None
    opponent_arrival: float | None = None
    receiver: tuple | None = None


class DefenseController:
    def __init__(self):
        self.frames = 0
        self.plan = None
        self.desired_slot = None
        self.switch_at = self.poke_at = self.boost_at = 0
        self.switch_attempts = 0
        self.pending_switch = None
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
        self.b_down = self.c_down = False
        self.diagnostics = {}

    @staticmethod
    def _goal_side(team, point, threat, slack=0):
        sign = 1 if team.net.y > 0 else -1
        return point[1] * sign >= threat[1] * sign - slack

    def _covered(self, state, target, excluding=None):
        sign = 1 if state.team1.net.y > 0 else -1
        return any(slot != excluding and abs(player.x) < 60 and player.y * sign > target[1] * sign - 12
                   for slot, player in skaters(state.team1))

    def _lane(self, state, threat, slot=None, reason='protect shooting lane', receiver=None):
        sign = 1 if state.team1.net.y > 0 else -1
        depth = threat[1] * sign
        goal_x = max(-12, min(12, -state.team1.goalie.x * 0.5))
        delta = goal_x - threat[0], state.team1.net.y - threat[1]
        length = max(1, math.hypot(*delta))
        gap = min(34, max(14, (264 - depth) * 0.18))
        target = (threat[0] + delta[0] * gap / length,
                  sign * min(244, depth + abs(delta[1]) * gap / length))
        mode = 'protect-lane' if receiver is None else 'deny-reception'
        if receiver is None and abs(threat[0]) > 82 and -50 < depth < 232:
            inside = [slot for slot, player in skaters(state.team1)
                      if abs(player.x) < abs(threat[0]) - 8 and math.dist((player.x, player.y), threat) < 75]
            open_receiver = any(abs(player.x) < 55 and player.y * sign > depth + 15
                                for _, player in skaters(state.team2))
            if inside and (not open_receiver or any(self._covered(state, threat, slot) for slot in inside)):
                target = (threat[0] - math.copysign(20, threat[0]), threat[1] + sign * 12)
                mode, reason = 'contain-boards', 'close inside escape; retain goal-side position'
        return DefensePlan(project_target(target), mode, reason, slot, receiver=receiver)

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
        carrier = next(((slot, player) for slot, player in opponents if slot == owner), None)
        if carrier is not None:
            slot, player = carrier
            vx, vy = velocity(player)
            threat = project_target((player.x + vx * 8, player.y + vy * 8))
            return self._lane(state, threat, slot)

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
                    and not self._covered(state, (state.puck.x, state.puck.y), excluding=slot)):
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
            plan = self._lane(state, point, slot, reason, point)
            return DefensePlan(plan.target, plan.mode, plan.reason, slot, time, time, point)
        if opponents:
            slot, player = min(opponents, key=lambda row: arrival_time(
                row[1], (state.puck.x, state.puck.y), optimistic=True))
            vx, vy = velocity(player)
            threat = project_target((player.x + vx * 8, player.y + vy * 8))
            return self._lane(state, threat, slot, reason)
        return self._lane(state, (state.puck.x, state.puck.y), reason=reason)

    def select_player(self, state, target):
        candidates = skaters(state.team1)
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

    def step(self, state, elapsed=1):
        self.frames += elapsed
        actual = controlled_slot(state.team1)
        self._observe_switch(actual)
        if owns_puck(state.team1, state.engine.puck_owner):
            self.idle(0)
            return np.zeros(Buttons.INPUT_MAX, dtype=np.int8)
        plan = self.choose_target(state)
        if (self.plan and plan.mode in ('protect-lane', 'contain-boards')
                and (plan.mode, plan.threat_slot) == (self.plan.mode, self.plan.threat_slot)
                and math.dist(plan.target, self.plan.target) < 3):
            plan = self.plan
        self.plan = plan
        best, costs = self.select_player(state, plan.target)
        action = np.zeros(Buttons.INPUT_MAX, dtype=np.int8)
        mode, waypoint = 'waiting-for-selection', plan.target
        player = next((p for slot, p in skaters(state.team1) if slot == actual), None)
        puck_vx, puck_vy = velocity(state.puck)
        likely = self._likely_switch(state)
        if actual < 0:
            switch_status = 'no-selection'
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
                        and self.pending_switch is None and likely == actual)
            goal_side = self._goal_side(state.team1, (player.x, player.y), (state.puck.x, state.puck.y), 5)
            reachable = distance < 22 and contact < 16 and (state.puck.height or 0) < 8
            if reachable and aligned and goal_side and can_poke:
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
            'last_switch_result': self.last_switch_result,
            'likely_switch_slot': likely, 'arrival_frames': costs, 'missing_feedback': missing,
            'buttons': action.tolist(), 'puck_path': [(t, point) for t, point, _ in puck_path(state.puck, 32)],
            'selection_available': actual >= 0,
        }
        return action
