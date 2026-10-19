import { on_ice } from "./defense";
import { bounded_projection } from "./motion";
import { distance } from "./numeric";
import { grounded_step } from "./skating";
import { project_target } from "./target";
import { GameState, Player, type Vec2 } from "./state";

export const CARRY_FRAMES = 18;
export const CONTACT_RADIUS = 16;
export const INTERCEPTION_MARGIN = 3;
export const CARRY_SAMPLES = [4, 8, 12, CARRY_FRAMES] as const;

export function carry_pad(player: Player, target: Vec2): Vec2 {
  return [target[0] - player.x, target[1] - player.y].map((delta) => delta > 3 ? 1 : delta < -3 ? -1 : 0) as [number, number];
}

export function carry_path(player: Player, target: Vec2 | null, frames = CARRY_FRAMES, { decision_interval = 4 } = {}): Player[] | null {
  if (frames < 0 || decision_interval < 1) throw new Error("Carry horizons must be nonnegative and decision intervals positive");
  let carrier = Object.assign(new Player(), player), pad: Vec2 = [0, 0];
  const path = [carrier];
  for (let elapsed = 0; elapsed < frames; elapsed++) {
    if (elapsed % decision_interval === 0) pad = target === null ? [0, 0] : carry_pad(carrier, target);
    carrier = grounded_step(carrier, pad);
    if (distance([carrier.x, carrier.y], project_target([carrier.x, carrier.y])) > 1e-6) return null;
    path.push(carrier);
  }
  return path;
}

export function _segment_clearance(start: Vec2, end: Vec2): number {
  const dx = end[0] - start[0], dy = end[1] - start[1];
  const fraction = Math.max(0, Math.min(1, -(start[0] * dx + start[1] * dy) / Math.max(0.01, dx * dx + dy * dy)));
  return Math.hypot(start[0] + dx * fraction, start[1] + dy * fraction);
}

export function body_clearance(state: GameState, player: Player, path: readonly Player[]): number {
  let clearance = Infinity;
  for (const team of [state.team1, state.team2]) {
    for (const other of [...team.players, team.goalie]) {
      if (other === player || !on_ice(other)) continue;
      let previous: Vec2 = [player.x - other.x, player.y - other.y];
      clearance = Math.min(clearance, Math.hypot(...previous) - CONTACT_RADIUS - other.projection_uncertainty);
      for (let elapsed = 1; elapsed < path.length; elapsed++) {
        const carrier = path[elapsed]!, [position, uncertainty] = bounded_projection(other, elapsed, { boards: other.role !== 0 });
        const relative: Vec2 = [carrier.x - position[0], carrier.y - position[1]];
        clearance = Math.min(clearance, _segment_clearance(previous, relative) - CONTACT_RADIUS - uncertainty);
        previous = relative;
      }
    }
  }
  return clearance;
}
