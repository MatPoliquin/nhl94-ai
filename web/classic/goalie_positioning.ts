import type { Action, GameState, Vec2 } from "./state";
import { GoalieController, GoaliePlan, goalie_arrival } from "./goalie";
import { velocity } from "./motion";

export class DeepGoalie extends GoalieController {
  override step(state: GameState, { blocked = false } = {}): Action | null {
    const action = super.step(state, { blocked });
    if (this.diagnostics.positioning_target !== undefined) this.diagnostics.target = this.diagnostics.positioning_target;
    return action;
  }

  override _position(state: GameState, plan: GoaliePlan): Action {
    if (plan.crossing_frames === null) {
      const sign = state.team1.net.y > 0 ? 1 : -1, [vx, vy] = velocity(state.puck);
      const reference = plan.receiver ?? [state.puck.x + 8 * vx, state.puck.y + 8 * vy];
      const distance = 260 - reference[1] * sign;
      if (distance > 10) {
        const target: Vec2 = [Math.max(-28, Math.min(28, reference[0] * 10 / distance)), sign * 250];
        plan = Object.assign(new GoaliePlan(plan.target, plan.reason), plan, { target });
        Object.assign(this.diagnostics, { positioning_target: target, destination: target, waypoint: target,
          reach_frames: goalie_arrival(state.team1.goalie, target), reason: "deeper pre-shot shooting line" });
      }
    }
    return super._position(state, plan);
  }
}
