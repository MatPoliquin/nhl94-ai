import { VELOCITY_SCALE, arrival_time, puck_path, velocity } from "./motion";
import { distance, pyRound } from "./numeric";
import { GameState, Player, type Vec2 } from "./state";

export interface ReceptionForecast { gap: number; frames: number; point: Vec2 }

export function passive_reception(state: GameState, receiver: Player, { stick_only = false } = {}): ReceptionForecast | null {
  if ([receiver.selection_flags, receiver.stick_x, receiver.stick_y, receiver.motion_x, receiver.motion_y,
    state.puck.motion_x, state.puck.motion_y, state.puck.height].some((value) => value === null)) return null;
  let x = receiver.precise_x ?? receiver.x, y = receiver.precise_y ?? receiver.y;
  let [vx, vy] = velocity(receiver).map((component) => pyRound(component / VELOCITY_SCALE)) as [number, number];
  const shift = receiver.selection_flags! & 1 ? 9 : 6;
  let closest: ReceptionForecast | null = null, best_gap = Infinity;
  for (const [frame, point, height] of puck_path(state.puck, 40, { sample_interval: 1 })) {
    vx -= (vx >> shift) || Number(!!vx); vy -= (vy >> shift) || Number(!!vy);
    x += vx * VELOCITY_SCALE; y += vy * VELOCITY_SCALE;
    if (height > (stick_only ? 5 : 8)) continue;
    let gap = distance(point, [x + receiver.stick_x!, y + receiver.stick_y!]) - 14;
    if (!stick_only) gap = Math.min(distance(point, [x, y]) - 8, gap);
    if (gap < best_gap) { best_gap = gap; closest = { gap, frames: frame, point }; }
  }
  return closest;
}

export function reception_target(state: GameState, receiver: Player, { stick_only = false } = {}): Vec2 | null {
  const offset = stick_only ? [receiver.stick_x, receiver.stick_y] as const : [0, 0] as const;
  if (offset[0] === null || offset[1] === null) return null;
  let last: Vec2 | null = null;
  for (const [time, point, height] of puck_path(state.puck, 48, { sample_interval: 1 })) {
    if (height > (stick_only ? 5 : 8)) continue;
    last = [point[0] - offset[0], point[1] - offset[1]];
    if (arrival_time(receiver, last) <= time) return last;
  }
  return last;
}
