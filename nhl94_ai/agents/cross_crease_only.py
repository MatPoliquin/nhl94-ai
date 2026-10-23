"""Small experimental puck-carrying specialist; no competing offensive tactics."""
from dataclasses import dataclass
import math

import numpy as np

from nhl94_ai.game.motion import coast_projection, velocity
from nhl94_ai.game.actors import controlled_slot, observable_body
from nhl94_ai.game.skating import grounded_step
from nhl94_ai.env.target_control import project_target, route_waypoint
from nhl94_ai.game.constants import GameConsts as Buttons


@dataclass(frozen=True)
class CrossingSettings:
    goal_gap: float = 46
    width: float = 70
    release_x: float = 8
    hold_frames: int = 12
    depth_weight: float = 3

    def __post_init__(self):
        for name, value, low, high in (
            ('goal gap', self.goal_gap, 24, 100), ('width', self.width, 35, 100),
            ('release X', self.release_x, -30, 30), ('depth weight', self.depth_weight, 1, 40),
        ):
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f'Use finite {name} in {low}..{high}.')
        if (isinstance(self.hold_frames, bool) or not isinstance(self.hold_frames, int)
                or not 1 <= self.hold_frames <= 60):
            raise ValueError('Use integer hold duration 1..60.')


class CrossCreaseOnly:
    """Skate into a lateral approach, shoot during it, and retry if still in possession.

    Waypoints are movement targets, not admission certificates. Shot timing is an
    experimental parameter; neither a goalie animation nor body-center crossing
    is required. The controller only reads live state and outputs normal buttons.
    """

    def __init__(self, settings=None):
        self.settings = settings or CrossingSettings()
        self.phase = 'approach'
        self.direction = None
        self.pressed = None
        self.windup = False
        self.attempts = []
        self.diagnostics = {}
        self.pad = (0, 0)

    def _obstacles(self, state, player):
        """Body, avoidance radius, cost weight and velocity forecast horizon."""
        return [(other, 24, 12, 8) for other in (*state.team2.players, state.team2.goalie)
                if observable_body(other)]

    def _skate(self, state, player, target, frame):
        """Choose an ordinary D-pad input using a short native-motion forecast.

        Obstacles incur a cost; they do not cancel the whole offensive attempt.
        No emulator rollout or future goalie state is available to the controller.
        """
        if frame % 4:
            return self.pad
        best = math.inf
        obstacles = self._obstacles(state, player)
        for pad in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
                    (1, 1), (1, -1), (-1, 1), (-1, -1)):
            future, penalty = player, 0.0
            for elapsed in range(1, 25):
                future = grounded_step(future, pad)
                if elapsed % 4:
                    continue
                point = future.precise_x, future.precise_y
                penalty += math.dist(point, project_target(point)) ** 2 * 30
                for other, radius, weight, horizon in obstacles:
                    vx, vy = velocity(other)
                    center = other.x + vx * min(elapsed, horizon), other.y + vy * min(elapsed, horizon)
                    penalty += max(0, radius - math.dist(point, center)) ** 2 * weight
            vx, vy = velocity(future)
            dx, dy = future.precise_x + vx * 14 - target[0], future.precise_y + vy * 14 - target[1]
            cost = dx * dx + dy * dy * (self.settings.depth_weight if self.phase == 'cross' else 1) + penalty
            if cost < best:
                best, self.pad = cost, pad
        return self.pad

    @staticmethod
    def _pad(action, pad):
        for component, positive, negative in (
            (pad[0], Buttons.INPUT_RIGHT, Buttons.INPUT_LEFT),
            (pad[1], Buttons.INPUT_UP, Buttons.INPUT_DOWN),
        ):
            if component:
                action[positive if component > 0 else negative] = 1

    def _allow_shot(self, state, player, projected):
        """Extension point for experiments; the original specialist adds no filter."""
        return True

    def release_ready(self, state, player, direction):
        """The specialist's immediate shot trigger, without approach skating."""
        sign = 1 if state.team2.net.y > state.team1.net.y else -1
        frames = (self.settings.hold_frames + (8 if player.shot_power < 20 else 14)
                  if self.settings.hold_frames >= 8 else self.settings.hold_frames + 2)
        projected = coast_projection(player, frames)
        puck_x = state.puck.x + projected[0] - player.x
        return (velocity(player)[0] * direction > 0.5
                and 12 < sign * (state.team2.net.y - projected[1]) < 95
                and puck_x * direction >= self.settings.release_x
                and self._allow_shot(state, player, projected))

    def step(self, state, frame):
        action = np.zeros(Buttons.INPUT_MAX, dtype=np.int8)
        owner = state.engine.puck_owner
        player = state.team1.get_player_by_scnum(owner)
        if state.engine.clock_stopped:
            return action
        if player is None or player is state.team1.goalie:
            self.diagnostics = {'phase': 'without-puck'}
            if owner < 0:
                selected = state.team1.get_player_by_scnum(controlled_slot(state.team1))
                if selected is not None and selected is not state.team1.goalie and not selected.is_falling:
                    target = project_target((state.puck.x, state.puck.y))
                    target = route_waypoint((selected.x, selected.y), target)
                    self._pad(action, self._skate(state, selected, target, frame))
            return action
        if player.is_falling:
            self.diagnostics = {'phase': 'falling'}
            return action
        if controlled_slot(state.team1) != owner:
            self.diagnostics = {'phase': 'await-control'}
            return action
        sign = 1 if state.team2.net.y > state.team1.net.y else -1
        depth = state.team2.net.y - sign * self.settings.goal_gap
        if self.direction is None:
            self.direction = 1 if player.x <= 0 else -1
        if self.phase == 'shot':
            self.windup |= bool(state.is_shooting)
            elapsed = frame - self.pressed
            self._pad(action, (self.direction, 0))
            if elapsed < self.settings.hold_frames:
                action[Buttons.INPUT_C] = 1
            if elapsed >= max(4, self.settings.hold_frames) and not state.is_shooting:
                self.phase = 'approach'
                self.direction = 1 if player.x <= 0 else -1
            self.diagnostics = {'phase': 'shot', 'elapsed': elapsed}
            return action

        vx, vy = velocity(player)
        target = (-self.direction * self.settings.width, depth)
        if self.phase == 'approach' and math.dist((player.x, player.y), target) < 12:
            self.phase = 'cross'
        if self.phase == 'cross':
            target = (self.direction * 95, depth)
            if self.release_ready(state, player, self.direction):
                self.phase, self.pressed, self.windup = 'shot', frame, False
                self.attempts.append({'frame': frame, 'player': (player.x, player.y),
                                      'puck': (state.puck.x, state.puck.y),
                                      'velocity': (vx, vy), 'goalie': (state.team2.goalie.x, state.team2.goalie.y),
                                      'direction': self.direction})
                action[Buttons.INPUT_C] = 1
                self._pad(action, (self.direction, 0))
                self.diagnostics = {'phase': 'shot', 'elapsed': 0}
                return action
            if player.x * self.direction > 85:
                self.direction *= -1
                self.phase = 'approach'
                target = (-self.direction * self.settings.width, depth)
        target = route_waypoint((player.x, player.y), project_target(target))
        self._pad(action, self._skate(state, player, target, frame))
        self.diagnostics = {'phase': self.phase, 'target': target, 'direction': self.direction}
        return action
