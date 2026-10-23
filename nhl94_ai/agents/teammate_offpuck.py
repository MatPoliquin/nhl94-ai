"""Off-puck candidate geometry and the exact steering policy used by forecasts.

No C bursts: support pass prices assume the same grounded steering/coasting
for the complete forecast window. Higher-priority receiving and committed
executors bypass this module.
"""
import numpy as np

from nhl94_ai.agents.defense import defensive_steering
from nhl94_ai.game.constants import GameConsts as Buttons


def attack_options(carrier, attack, phase, frame):
    forward = carrier.y * attack
    flank = -1 if carrier.x >= 0 else 1
    if phase == 'attack':
        # Cut timing changes candidate availability, never an already owned target.
        cut_side = flank if frame // 48 % 2 == 0 else -flank
        return [(cut_side * 24, min(225, forward + 30), 'timed-cut'),
                (flank * 32, 205, 'slot-option'),
                (-flank * 20, 230, 'net-front-option'),
                (flank * 75, min(225, forward + 10), 'flank-option')]
    return [(flank * 60, min(76, forward + 35), 'entry-outlet'),
            (-flank * 40, forward - 30, 'trailing-outlet')]


def movement_action(player, target, *, coast=False):
    """Execution and support_forecast share defensive_steering, with no burst."""
    action = np.zeros(12, dtype=np.int8)
    if coast:
        return action, target
    pad, waypoint = defensive_steering(player, target)
    action[Buttons.INPUT_UP], action[Buttons.INPUT_DOWN] = pad[1] > 0, pad[1] < 0
    action[Buttons.INPUT_LEFT], action[Buttons.INPUT_RIGHT] = pad[0] < 0, pad[0] > 0
    return action, waypoint


class SharedRoles:
    """One safety assignment chosen by reachability, with bounded hysteresis.

    Reservations from pressure, reception and committed actions are authoritative.
    Only unreserved fallback movement is assigned here.
    """
    def __init__(self):
        self.guard = None
        self.key = None
        self.until = 0
        self.diagnostics = {}

    def choose(self, state, slots, primary, frame, reserved=()):
        from nhl94_ai.agents.teammate_geometry import coverage_target
        from nhl94_ai.game.actors import controllable_skater
        from nhl94_ai.game.motion import arrival_time
        from nhl94_ai.env.target_control import project_target
        team, owner = state.team1, state.engine.puck_owner
        attack = 1 if state.team2.net.y > team.net.y else -1
        carrier = team.get_player_by_scnum(owner)
        attacking = carrier is not None and carrier is not team.goalie
        # An opponent behind the carrier is a turnover threat even during our possession.
        exposed = attacking and any(controllable_skater(p) and abs(p.x) < 85
                                    and (carrier.y - p.y) * attack > 75 for p in state.team2.players)
        phase = 'attack' if attacking and not exposed else 'safety'
        eligible = [(i, team.get_player_by_scnum(slot)) for i, slot in enumerate(slots)
                    if i != primary and i not in reserved and slot != team.goalie_scnum()]
        eligible = [(i, p) for i, p in eligible if p is not None and controllable_skater(p)]
        key = (tuple(slots), primary, tuple(reserved), phase, attack, tuple(i for i, _ in eligible))
        if key != self.key:
            self.guard, self.until, self.key = None, 0, key
        choices = []
        for i, player in eligible:
            if phase == 'attack':
                target = (max(-35, min(35, -carrier.x * 0.4)), min(150, carrier.y * attack - 45) * attack)
            else:
                target = coverage_target(state, player, slots[primary] if primary is not None else None)
            x, y = target
            if state.engine.offsides_enabled is not False and state.puck.y * attack < 88:
                y = min(76, y * attack) * attack
            # Separate the guard's target from other currently selected bodies.
            for j, slot in enumerate(slots):
                other = team.get_player_by_scnum(slot)
                if j != i and other is not None and (x-other.x)**2 + (y-other.y)**2 < 24**2:
                    x += 28 if player.x >= other.x else -28
            target = project_target((x, y))
            cost = arrival_time(player, target)
            choices.append((cost, slots[i], i, target))
        if not choices:
            self.guard = None
            self.diagnostics = {'phase': phase, 'guard': None, 'reason': 'all-inputs-reserved'}
            return {}
        best = min(choices)
        current = next((item for item in choices if item[2] == self.guard), None)
        if current is not None and (frame < self.until or best[0] + 12 >= current[0]):
            chosen = current
        else:
            chosen = best
            self.guard, self.until = chosen[2], frame + 24
        role = 'high-slot-outlet' if phase == 'attack' else 'safety-cover'
        self.diagnostics = {'phase': phase, 'guard': self.guard + 1, 'slot': slots[self.guard],
                            'target': chosen[3], 'until': self.until, 'reserved': list(reserved),
                            'target_owner': 'shared-roles', 'turnover_threat': bool(exposed)}
        return {self.guard: (chosen[3], role)}
