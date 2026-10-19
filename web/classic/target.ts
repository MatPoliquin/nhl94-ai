import { Buttons, type Action, type Player, type Vec2 } from "./state";
import { copySign, distance, lexCompare, pointEqual, required } from "./numeric";

export const SETTINGS = {
  scale_x: 120, scale_y: 270, arrival_radius: 4, steering_deadzone: 2, braking_frames: 10,
} as const;
const NETS = [[-28, 28, 254, 284], [-28, 28, -284, -254]] as const;
const ROUTE_POINTS: readonly Vec2[] = [[-32, 250], [32, 250], [-32, -250], [32, -250]];

export function project_target(position: Vec2): Vec2 {
  let x = Math.min(Math.max(position[0], -SETTINGS.scale_x), SETTINGS.scale_x);
  let y = Math.min(Math.max(position[1], -SETTINGS.scale_y), SETTINGS.scale_y);
  const corner_x = Math.max(0, Math.abs(x) - 72), corner_y = Math.max(0, Math.abs(y) - 222);
  const radius = Math.hypot(corner_x, corner_y);
  if (radius > 48) {
    x = copySign(72 + corner_x * 48 / radius, x);
    y = copySign(222 + corner_y * 48 / radius, y);
  }
  if (Math.abs(x) < 28 && Math.abs(y) > 254) {
    if (28 - Math.abs(x) < Math.abs(y) - 254) x = copySign(28, x);
    else y = copySign(254, y);
  }
  return [x, y];
}

export function _clear_segment(start: Vec2, end: Vec2): boolean {
  for (const [left, right, bottom, top] of NETS) {
    let enter = 0, leave = 1, broke = false;
    for (const [value, delta, low, high] of [
      [start[0], end[0] - start[0], left, right],
      [start[1], end[1] - start[1], bottom, top],
    ] as const) {
      if (Math.abs(delta) < 1e-9) {
        if (!(low < value && value < high)) { broke = true; break; }
      } else {
        const first = Math.min((low - value) / delta, (high - value) / delta);
        const last = Math.max((low - value) / delta, (high - value) / delta);
        enter = Math.max(enter, first);
        leave = Math.min(leave, last);
      }
      if (enter >= leave) { broke = true; break; }
    }
    if (!broke && enter < leave) return false;
  }
  return true;
}

export function route_waypoint(position: Vec2, target: Vec2): Vec2 {
  const escaped = project_target(position);
  if (distance(position, escaped) > 0.5) return escaped;
  if (_clear_segment(position, target)) return target;
  const points: Vec2[] = [];
  for (const point of [position, target, ...ROUTE_POINTS]) {
    if (!points.some((existing) => pointEqual(existing, point))) points.push(point);
  }
  const queue: { distance: number; index: number; path: number[] }[] = [{ distance: 0, index: 0, path: [] }];
  const visited = new Set<number>();
  while (queue.length) {
    queue.sort((a, b) => a.distance - b.distance || a.index - b.index || lexCompare(a.path, b.path));
    const current = required(queue.shift(), "route queue");
    if (visited.has(current.index)) continue;
    visited.add(current.index);
    if (current.index === 1) return required(points[current.path[0]!], "route waypoint");
    points.forEach((point, neighbor) => {
      if (!visited.has(neighbor) && _clear_segment(points[current.index]!, point)) {
        queue.push({ distance: current.distance + distance(points[current.index]!, point),
          index: neighbor, path: [...current.path, neighbor] });
      }
    });
  }
  throw new Error(`No target route around nets: ${position} -> ${target}`);
}

export function steering_direction(player: Player, waypoint: Vec2): [Vec2, string] {
  const dx = waypoint[0] - player.x, dy = waypoint[1] - player.y;
  const vx = required(player.motion_x, "motion_x"), vy = required(player.motion_y, "motion_y");
  if (Math.hypot(dx, dy) <= SETTINGS.arrival_radius && Math.hypot(vx, vy) < 0.2) return [[0, 0], "holding"];
  const corrections: Vec2 = [dx - SETTINGS.braking_frames * vx, dy - SETTINGS.braking_frames * vy];
  const pad: Vec2 = [Math.abs(corrections[0]) > SETTINGS.steering_deadzone ? copySign(1, corrections[0]) : 0,
    Math.abs(corrections[1]) > SETTINGS.steering_deadzone ? copySign(1, corrections[1]) : 0];
  return [pad, corrections[0] * vx + corrections[1] * vy < 0 ? "braking" : "skating"];
}

export function aim_pass(action: Action, passer: Pick<Player, "x" | "y">, receiver: Pick<Player, "x" | "y">): void {
  const dx = receiver.x - passer.x, dy = receiver.y - passer.y;
  action.fill(0, 4, 8);
  if (Math.abs(dx) > Math.max(3, Math.abs(dy) * 0.41421356237)) action[dx > 0 ? Buttons.INPUT_RIGHT : Buttons.INPUT_LEFT] = 1;
  if (Math.abs(dy) > Math.max(3, Math.abs(dx) * 0.41421356237)) action[dy > 0 ? Buttons.INPUT_UP : Buttons.INPUT_DOWN] = 1;
}
