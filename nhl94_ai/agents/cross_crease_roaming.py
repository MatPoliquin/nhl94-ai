"""Keep skating in possession and use a crossing when traffic permits one."""
import math

import numpy as np

from nhl94_ai.agents.cross_crease_only import CrossCreaseOnly
from nhl94_ai.game.actors import controlled_slot, observable_body
from nhl94_ai.game.motion import coast_projection, velocity
from nhl94_ai.game.shooting import normal_shot_release_frames
from nhl94_ai.env.target_control import project_target, route_waypoint
from nhl94_ai.game.constants import GameConsts as Buttons


class RoamingCrossCrease(CrossCreaseOnly):
    """The only finish remains cross-crease; possession need not force a finish.

    Reconsider traffic every four native frames. A short open crossing is an
    opportunity, not a permanent route assignment. Roaming targets may move
    away from goal, and recently visited ice loses its attraction. Forecasts
    use current positions/velocity, never future emulator state.
    """

    def __init__(self, settings=None, *, select_release=True, prepare_release=True, traffic_aware=False):
        super().__init__(settings)
        self.select_release = select_release
        self.prepare_release = prepare_release
        self.traffic_aware = traffic_aware
        self.release_estimate = {}
        self.phase = 'roam'
        self.target = None
        self.target_frame = -100
        self.visited = []
        self.opportunity = {}

    def _obstacles(self, state, player):
        if not self.traffic_aware:
            return super()._obstacles(state, player)
        # Steering costs stay soft: crowded ice can be crossed when alternatives
        # are worse. Teammates obstruct a route without posing a checking threat.
        return ([(p, 28, 14, 16) for p in state.team2.players if observable_body(p)]
                + [(p, 24, 12, 12) for p in (state.team2.goalie,) if observable_body(p)]
                + [(p, 20, 12, 12) for p in state.team1.players if p is not player and observable_body(p)])

    def _allow_shot(self, state, player, projected):
        if not self.select_release:
            return True
        sign = 1 if state.team2.net.y > 0 else -1
        frames = normal_shot_release_frames(player, self.settings.hold_frames)
        projected = coast_projection(player, frames)
        puck = (state.puck.x + projected[0] - player.x,
                state.puck.y + projected[1] - player.y)
        gap = sign * (state.team2.net.y - puck[1])
        # Stick animation changes its offset during windup; leave lateral room
        # rather than treating this point estimate as exact goal geometry.
        reason = ('too-wide' if abs(puck[0]) > 40 else
                  'too-far' if gap > 65 else 'behind-goal' if gap < 10 else 'ready')
        self.release_estimate = {'projected_puck': puck, 'release_gap': gap,
                                 'release_frames': frames, 'finish': reason}
        return reason == 'ready'

    @staticmethod
    def _bodies(state, player):
        return [(p, 24) for p in (*state.team2.players, state.team2.goalie) if observable_body(p)] + [
            (p, 16) for p in state.team1.players if p is not player and observable_body(p)]

    def _route_pressure(self, state, player, target):
        """Estimate crowding along a possible route; this is not a collision certificate."""
        start = player.x, player.y
        length = math.dist(start, target)
        speed = max(1.2, math.hypot(*velocity(player)))
        pressure, clearance = 0., 1000.
        bodies = self._bodies(state, player)
        for index in range(1, 7):
            fraction = index / 6
            point = tuple(a + (b - a) * fraction for a, b in zip(start, target))
            elapsed = min(20, length * fraction / speed)
            for other, radius in bodies:
                vx, vy = velocity(other)
                separation = math.dist(point, (other.x + vx * elapsed, other.y + vy * elapsed))
                clearance = min(clearance, separation - radius)
                pressure += max(0, radius + 10 - separation) ** 2 / 6
        return pressure, clearance

    def _opening(self, state, player):
        self.opportunity = {}
        sign = 1 if state.team2.net.y > 0 else -1
        gap = sign * (state.team2.net.y - player.y)
        if not 25 <= gap <= 88:
            return None
        if self.prepare_release:
            frames = normal_shot_release_frames(player, self.settings.hold_frames)
            projected = coast_projection(player, frames)
            projected_gap = sign * (state.team2.net.y - projected[1])
            if projected_gap > 65:
                self.opportunity = {'finish': 'too-far', 'projected_gap': projected_gap}
                return None
        vx, _ = velocity(player)
        direction = (1 if vx > 0 else -1) if abs(vx) > .65 else (1 if player.x <= 0 else -1)
        if player.x * direction > 30:
            return None
        depth = state.team2.net.y - sign * self.settings.goal_gap
        target = direction * max(25, player.x * direction + 16), depth
        pressure, clearance = self._route_pressure(state, player, target)
        self.opportunity = {'pressure': pressure, 'clearance': clearance}
        # Permit ordinary close traffic; reject a body occupying the short cut.
        return direction if clearance >= -9 and pressure < 220 else None

    def _roaming_target(self, state, player, frame):
        sign = 1 if state.team2.net.y > 0 else -1
        net = state.team2.net.y
        candidates = [(x, net - sign * gap) for x in (-100, -65, 0, 65, 100)
                      for gap in (50, 100, 150)]
        best, chosen = math.inf, None
        for point in candidates:
            distance = math.dist((player.x, player.y), point)
            if distance < 30:
                continue
            target = route_waypoint((player.x, player.y), project_target(point))
            pressure, _ = self._route_pressure(state, player, target)
            # Nearby open ice matters more than progress toward the goal.
            cost = distance * .7 + pressure * 1.5 + abs(abs(point[0]) - 65) * .15
            cost += max(0, sign * (net - point[1]) - 65) * .6
            for old_frame, old_point in self.visited:
                cost += max(0, 45 - math.dist(point, old_point)) * (1 - (frame - old_frame) / 240)
            if self.target is not None:
                cost += min(25, math.dist(point, self.target)) * .25
            if cost < best:
                best, chosen = cost, target
        return chosen

    def step(self, state, frame):
        owner = state.engine.puck_owner
        player = state.team1.get_player_by_scnum(owner)
        if self.phase == 'shot':
            return super().step(state, frame)
        if (state.engine.clock_stopped or player is None or player is state.team1.goalie
                or player.is_falling or controlled_slot(state.team1) != owner):
            return super().step(state, frame)
        self.release_estimate = {}
        if frame % 4 == 0 or self.target is None:
            self.visited = [(f, p) for f, p in self.visited if frame - f < 240]
            if not self.visited or frame - self.visited[-1][0] >= 24:
                self.visited.append((frame, (player.x, player.y)))
            direction = self._opening(state, player)
            if direction is not None:
                self.phase, self.direction = 'cross', direction
            else:
                self.phase = 'roam'
                if (self.target is None or frame - self.target_frame >= 24
                        or math.dist((player.x, player.y), self.target) < 24):
                    self.target = self._roaming_target(state, player, frame)
                    self.target_frame = frame
        if self.phase == 'cross':
            action = super().step(state, frame)
            self.diagnostics.update(self.release_estimate)
            return action
        action = np.zeros(Buttons.INPUT_MAX, dtype=np.int8)
        target = route_waypoint((player.x, player.y), project_target(self.target))
        self._pad(action, self._skate(state, player, target, frame))
        self.diagnostics = {'phase': 'roam', 'target': target, **self.opportunity}
        return action
