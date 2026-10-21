"""Opt-in deeper pre-shot alignment selected by native snapshot experiments."""
from dataclasses import replace

from nhl94_ai.agents.goalie import GoalieController, goalie_arrival
from nhl94_ai.game.motion import velocity


class DeepGoalie(GoalieController):
    """Preserve Classic handoffs and saves while changing pre-release placement."""

    def step(self, state, *, blocked=False):
        action = super().step(state, blocked=blocked)
        # The base lifecycle publishes its original plan after _position returns.
        if 'positioning_target' in self.diagnostics:
            self.diagnostics['target'] = self.diagnostics['positioning_target']
        return action

    def _position(self, state, plan):
        if plan.crossing_frames is None:
            sign = 1 if state.team1.net.y > 0 else -1
            vx, vy = velocity(state.puck)
            reference = plan.receiver or (state.puck.x+8*vx, state.puck.y+8*vy)
            distance = 260-reference[1]*sign
            if distance > 10:
                target = max(-28, min(28, reference[0]*10/distance)), sign*250
                plan = replace(plan, target=target)
                self.diagnostics.update(positioning_target=target, destination=target, waypoint=target,
                                        reach_frames=goalie_arrival(state.team1.goalie, target),
                                        reason='deeper pre-shot shooting line')
        return super()._position(state, plan)
