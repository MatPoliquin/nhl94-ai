"""Bounded reachable support options using Classic's movement/pass geometry."""
from copy import copy
import math

from nhl94_ai.agents.defense import eligible
from nhl94_ai.agents.motion import arrival_time, bounded_projection
from nhl94_ai.agents.passing import evaluate_pass, pressure_margin, shot_value
from nhl94_ai.env.target_control import project_target


class SupportPlanner:
    def __init__(self):
        self.until = 0
        self.key = None
        self.target = (0, 0)
        self.role = 'coverage'
        self.diagnostics = {}
        self.shots = None
        self.recover_until = 0

    def choose(self, state, slot, frame, primary_slot=None):
        team, opponents, puck = state.team1, state.team2, state.puck
        player = team.get_player_by_scnum(slot)
        owner = state.engine.puck_owner
        carrier = team.get_player_by_scnum(owner)
        attack = 1 if opponents.net.y > team.net.y else -1
        if self.shots is not None and team.stats.shots > self.shots:
            self.recover_until = frame + 90
        self.shots = team.stats.shots
        phase = ('outlet' if team.goalie_haspuck else
                 'attack' if carrier is not None and carrier is not team.goalie and carrier.y * attack >= 88
                 else 'buildup' if carrier is not None and carrier is not team.goalie
                 else 'recovery' if frame < self.recover_until else 'defense')
        key = (slot, owner, attack, phase, primary_slot)
        if key == self.key and frame < self.until:
            return self.target, self.role
        self.key, self.until = key, frame + 8
        if player is None:
            self.target, self.role = (0, 0), 'coverage'
            return self.target, self.role
        if phase in ('defense', 'recovery'):
            self.target = self._coverage(state, player, primary_slot)
            self.role = 'coverage'
            self.diagnostics = {'phase': phase, 'target': self.target}
            return self.target, self.role
        self.role = 'outlet-option' if phase == 'outlet' else 'passing-option'
        if phase == 'outlet':
            carrier = team.goalie
        forward = carrier.y * attack
        flank = -1 if carrier.x >= 0 else 1
        points = [(flank * 45, forward + 40), (flank * 70, forward + 10),
                  (flank * 35, forward - 35), (-flank * 60, forward - 30),
                  (flank * 35, 210 if phase == 'attack' else forward + 65),
                  (player.x, player.y * attack)]
        candidates = []
        for x, y in points:
            y = min(235, y)
            if state.engine.offsides_enabled is not False and puck.y * attack < 88:
                y = min(76, y)
            point = project_target((x, y * attack))
            arrival = arrival_time(player, point)
            spacing = math.dist((carrier.x, carrier.y), point)
            teammate_gap = min((math.dist(point, (other.x, other.y)) for other in team.players
                                if other is not player and other is not carrier and eligible(other)), default=100)
            if arrival > 48 or spacing < 30 or teammate_gap < 18:
                continue
            pressure = pressure_margin(opponents, point, arrival)
            score = (0.3 * (point[1] * attack - forward) - arrival * 0.6
                     + min(18, pressure) * 1.5 - max(0, 30 - teammate_gap))
            candidates.append({'point': point, 'arrival': arrival, 'pressure': pressure, 'score': score})
        # Only three candidate pass forecasts per update; these are estimated
        # future options, never an instruction to the carrier to force a pass.
        finalists = sorted(candidates, key=lambda item: item['score'], reverse=True)[:3]
        index = team.players.index(player) if player in team.players else None
        for candidate in finalists:
            future = copy(state)
            future.team1, future.team2 = copy(team), copy(opponents)
            receiver = copy(player)
            receiver.x, receiver.y = candidate['point']
            receiver.precise_x, receiver.precise_y = candidate['point']
            receiver.motion_x = receiver.motion_y = 0
            future.team1.players = [receiver if other is player else other for other in team.players]
            future.team2.players = []
            for defender in opponents.players:
                projected = copy(defender)
                (projected.x, projected.y), _ = bounded_projection(defender, min(24, candidate['arrival']))
                future.team2.players.append(projected)
            option, details = evaluate_pass(future, carrier, index, receiver, purpose='support') if index is not None else (None, {})
            candidate['pass_status'] = details.get('status', 'unavailable')
            candidate['feasible'] = option is not None
            candidate['next_shot'] = shot_value(future, receiver)
            candidate['score'] += (35 + min(12, option.margin) if option is not None else -15)
            candidate['score'] += candidate['next_shot'] * 0.4
        if finalists:
            self.target = max(finalists, key=lambda item: item['score'])['point']
        else:
            # A nearby legal outlet remains bounded when every forecast fails.
            y = min(76, player.y * attack) if state.engine.offsides_enabled is not False and puck.y * attack < 88 else player.y * attack
            self.target = project_target((player.x, y * attack))
        self.diagnostics = {'phase': phase, 'target': self.target, 'candidates': finalists,
                            'forecast': 'estimated-future-options'}
        return self.target, self.role

    @staticmethod
    def _coverage(state, player, primary_slot):
        puck, team = state.puck, state.team1
        toward = 1 if team.net.y > puck.y else -1
        depth = min(60, max(30, abs(team.net.y - puck.y) * 0.45))
        y = puck.y + toward * depth
        y = min(team.net.y - 24, y) if toward > 0 else max(team.net.y + 24, y)
        middle = (max(-42, min(42, puck.x * 0.4)), y)
        threats = [other for other in state.team2.players if eligible(other)
                   and (other.y - puck.y) * toward > 15 and abs(other.x) < 85]
        if threats:
            threat = min(threats, key=lambda other: math.dist((other.x, other.y), (0, team.net.y)))
            lane = ((puck.x + threat.x) / 2, (puck.y + threat.y) / 2)
            if abs(lane[0]) < 55 and math.dist(lane, (puck.x, puck.y)) >= 30:
                middle = ((middle[0] + lane[0]) / 2, (middle[1] + lane[1]) / 2)
        primary = team.get_player_by_scnum(primary_slot) if primary_slot is not None else None
        if primary is not None and math.dist(middle, (primary.x, primary.y)) < 24:
            middle = (middle[0] + (28 if player.x >= primary.x else -28), middle[1])
        return project_target(middle)
