"""Shared, bounded carrier traps. Coverage must survive both attackers leaving."""
from copy import copy
from dataclasses import dataclass
import math

from nhl94_ai.agents.defense import DefensePlan, defensive_steering, eligible, skaters
from nhl94_ai.agents.motion import arrival_time, blocks_shot_lane, bounded_projection
from nhl94_ai.agents.skating import grounded_step
from nhl94_ai.env.target_control import project_target


HORIZON = 24
APPROACH_LIMIT = 48
TRAP_ROLES = ('contain', 'close', 'cover')


@dataclass(frozen=True)
class PressureAssignment:
    plan: DefensePlan
    contact: bool = False
    covered_closer: bool = False


class SharedPressure:
    def __init__(self):
        self.key = None
        self.started = 0
        self.until = 0
        self.retry_at = 0
        self.diagnostics = {'phase': 'idle', 'reason': 'no-opponent-carrier'}

    def reset(self, reason, frame=0):
        if self.key is not None:
            self.retry_at = frame + 12
        self.key = None
        self.diagnostics = {'phase': 'idle', 'reason': reason}

    @staticmethod
    def _known(player):
        return eligible(player) and not player.is_falling and all(
            getattr(player, field) is not None
            for field in ('motion_x', 'motion_y', 'speed', 'agility', 'weight', 'energy', 'facing'))

    @staticmethod
    def _screen(point, threat, net):
        dx, dy = -threat[0], net - threat[1]
        length = max(1, math.hypot(dx, dy))
        px, py = point[0] - threat[0], point[1] - threat[1]
        along = (px * dx + py * dy) / length
        across = abs(px * dy - py * dx) / length
        return 24 <= along <= length - 16 and across <= 26

    @staticmethod
    def _forecast(player, target=None):
        if target is not None:
            result = [player]
            for _ in range(HORIZON):
                result.append(grounded_step(result[-1], defensive_steering(result[-1], target)[0]))
            return result
        result = []
        for time in range(HORIZON + 1):
            future = copy(player)
            future.x, future.y = bounded_projection(player, time)[0]
            result.append(future)
        return result

    def _coverage(self, state, attackers, guard, target, carrier):
        # Uncontrolled teammates get only observed-motion credit, never assumed steering.
        guards = [(slot, self._forecast(player, target if slot == guard else None))
                  for slot, player in skaters(state.team1)
                  if slot not in attackers and self._known(player)]
        sign = 1 if state.team1.net.y > 0 else -1
        threats = [carrier]
        for _, player in skaters(state.team2):
            if player is carrier:
                continue
            # Include receivers who enter a dangerous lane during the approach.
            for time in (0, HORIZON):
                origin = bounded_projection(player, time)[0]
                carrier_point = bounded_projection(carrier, time)[0]
                if abs(origin[0]) < 85 and (origin[1] - carrier_point[1]) * sign > 15:
                    threats.append(player)
                    break
        for time in (0, 8, 16, 24):
            for threat in threats:
                if not self._known(threat):
                    return False
                origin = bounded_projection(threat, time)[0]
                available = [path[time] for _, path in guards]
                if not any(self._screen((p.x, p.y), origin, state.team1.net.y) for p in available):
                    return False
                # An advanced receiver needs actual shot-lane cover, not just a screen.
                if origin[1] * sign > 130 and abs(origin[0]) < 85:
                    if not all(any(blocks_shot_lane(p, origin, goal) for p in available)
                               for goal in state._shot_target_points_for_net(state.team1.net)):
                        return False
        return bool(guards)

    def _clear_routes(self, state, targets, carrier_slot):
        paths = [(slot, self._forecast(player, targets.get(slot)))
                 for slot, player in skaters(state.team1)]
        paths += [(slot, self._forecast(player)) for slot, player in skaters(state.team2)
                  if slot != carrier_slot]
        # Check every predicted frame, including collisions between an attacker and cover.
        for i, (slot, path) in enumerate(paths):
            for other_slot, other in paths[i + 1:]:
                if slot not in targets and other_slot not in targets:
                    continue
                if any(math.dist((a.x, a.y), (b.x, b.y)) < 18 for a, b in zip(path, other)):
                    return False
        return True

    def choose(self, state, slots, frame, *, busy=False, pending_contact=None):
        owner = state.engine.puck_owner
        carrier = next((p for slot, p in skaters(state.team2) if slot == owner), None)
        players = [state.team1.get_player_by_scnum(slot) for slot in slots]
        if (busy or state.engine.clock_stopped or state.time == 0 or state.engine.sflags & 0x0100
                or not state.engine.puck_owner_known or carrier is None):
            self.reset('no-opponent-carrier-or-busy', frame)
            return None
        if (len(set(slots)) != len(slots) or any(p is None or not self._known(p) for p in players)
                or not self._known(carrier)):
            self.reset('unavailable-or-missing-feedback', frame)
            return None
        if self.key is not None and (owner, tuple(slots)) != self.key[:2]:
            self.reset('possession-or-selection-changed', frame)
            return None
        sign = 1 if state.team1.net.y > 0 else -1
        point = bounded_projection(carrier, 8)[0]
        depth = max(carrier.y * sign, bounded_projection(carrier, HORIZON)[0][1] * sign)
        boards = abs(point[0]) > 82
        if depth > (150 if boards else 88):
            self.reset('too-close-to-own-goal', frame)
            return None
        if self.key is not None and frame - self.started >= 120:
            self.reset('trap-timeout', frame)
            self.retry_at = frame + 36
        if frame < self.retry_at:
            self.diagnostics = {'phase': 'idle', 'reason': 'regroup'}
            return None
        # Evaluate the whole assignment. The nearest attacker must not be reserved
        # as cover merely because it is also nearest an arbitrary coverage point.
        guard_indices = list(range(3)) if len(slots) == 3 else [None]
        if self.key is not None and frame < self.until:
            guard_indices = [self.key[4]]
        choices = []
        covered = reachable = False
        for guard_index in guard_indices:
            attackers = [i for i in range(len(slots)) if i != guard_index]
            guard = slots[guard_index] if guard_index is not None else None
            # Hold existing depth while adjusting laterally to screen the carrier.
            # A cover bot already behind the play need not skate up to the trap.
            y = (players[guard_index].y if guard_index is not None else point[1] + sign * 72)
            y = sign * min(232, max(point[1] * sign + 40, y * sign))
            fraction = (state.team1.net.y - y) / (state.team1.net.y - point[1])
            cover_target = project_target((point[0] * fraction, y))
            if not self._coverage(state, [slots[i] for i in attackers], guard, cover_target, carrier):
                continue
            covered = True
            for contain, close in (attackers, attackers[::-1]):
                for flank in (-1, 1):
                    identity = (owner, tuple(slots), contain, close, guard_index, flank)
                    if self.key is not None and frame < self.until and identity != self.key:
                        continue
                    ready = ((players[contain].y - carrier.y) * sign >= 8
                             and math.dist((players[contain].x, players[contain].y), point) <= 42
                             and math.dist((players[close].x, players[close].y), point) <= 48)
                    gap = 12 if ready else 22
                    if boards:
                        side = 1 if point[0] > 0 else -1
                        first = project_target((point[0] - side * gap, point[1] + sign * gap))
                        second = project_target((point[0] - side * 6, point[1] - sign * gap))
                    else:
                        first = project_target((point[0] + flank * gap, point[1] + sign * gap))
                        second = project_target((point[0] - flank * gap, point[1] - sign * gap))
                    times = [arrival_time(players[i], target) for i, target in ((contain, first), (close, second))]
                    # During setup the first arrival contains while its partner closes.
                    # Actual contact is still withheld until both are near the carrier.
                    if max(times) > APPROACH_LIMIT or abs(times[0] - times[1]) > 32:
                        continue
                    reachable = True
                    targets = {slots[contain]: first, slots[close]: second}
                    if guard is not None:
                        targets[guard] = cover_target
                    if not self._clear_routes(state, targets, owner):
                        continue
                    score = sum(times)
                    if identity == self.key:
                        score -= 6
                    choices.append((score, identity, ready, targets))
        if not choices:
            self.reset('exposed-lane' if not covered else
                       'approach-unreachable' if not reachable else 'approach-obstructed', frame)
            return None
        _, identity, ready, targets = min(choices, key=lambda item: item[0])
        if identity != self.key:
            self.started = frame if self.key is None else self.started
            self.until = frame + 24
        self.key = identity
        contain, close = identity[2:4]
        guard = slots[identity[4]] if identity[4] is not None else None
        contact = (pending_contact if pending_contact in (contain, close) else
                   min((contain, close), key=lambda i: arrival_time(players[i], (carrier.x, carrier.y)))) if ready else None
        result = []
        for index, slot in enumerate(slots):
            role = 'contain' if index == contain else 'close' if index == close else 'cover'
            result.append(PressureAssignment(
                DefensePlan(targets[slot], role, 'coordinated carrier trap', threat_slot=owner),
                contact == index, role == 'close'))
        self.diagnostics = {'phase': 'close' if ready else 'contain', 'reason': 'covered-approach',
                            'carrier': owner, 'cover_slot': guard, 'contact_controller': contact,
                            'roles': [item.plan.mode for item in result], 'started': self.started}
        return result
