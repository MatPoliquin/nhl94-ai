import { Buttons, type Action, type GameState, type Vec2 } from "./state";
import { required, shiftRight } from "./numeric";
import { GoaliePlan, goalie_steer } from "./goalie";
import { KeeperMotion, PuckFlight, contact_margin } from "./goalie_physics";
import { DeepGoalie } from "./goalie_positioning";

export function contact_frame(state: GameState, action: Action, horizon: number, { save = false } = {}): number | null {
  if (!(0 < horizon && horizon <= 12)) return null;
  let keeper = KeeperMotion.observe(state.team1.goalie), puck = PuckFlight.observe(state.puck);
  if (keeper === null || puck === null) return null;
  const sign = state.team1.net.y > 0 ? 1 : -1;
  const pad: Vec2 = [action[Buttons.INPUT_RIGHT] - action[Buttons.INPUT_LEFT], action[Buttons.INPUT_UP] - action[Buttons.INPUT_DOWN]];
  if (!pad.some(Boolean)) return null;
  for (let frame = 1; frame <= Math.ceil(horizon); frame++) {
    keeper = keeper.step(pad, { save }); puck = puck.step();
    if (Math.abs(shiftRight(puck.x, 16)) >= 128 || shiftRight(puck.y, 16) * sign >= 261) return null;
    if (contact_margin(keeper, puck, { one_timer: state.engine.native_one_timer_shot }) >= 0) return frame;
  }
  return null;
}

export class CSaveGoalie extends DeepGoalie {
  override _position(state: GameState, plan: GoaliePlan): Action {
    if (plan.crossing_frames !== null && this.frames >= this.save_at) {
      const gap = Math.abs(required(plan.crossing_x, "crossing_x") - state.team1.goalie.x);
      if (8 < gap && gap <= 18 && required(plan.height, "crossing height") <= 15) {
        const action = goalie_steer(state.team1.goalie, plan.target);
        const moving = contact_frame(state, action, plan.crossing_frames), saving = contact_frame(state, action, plan.crossing_frames, { save: true });
        Object.assign(this.diagnostics, { native_moving_contact: moving, native_saving_contact: saving });
        if (moving !== null && saving === null) { this.diagnostics.save_decision = "native-body-contact-before-C"; return action; }
      }
    }
    return super._position(state, plan);
  }
}
