import type { Player, Vec2 } from "./state";
import { copySign, distance, mod, pointEqual, pyRound } from "./numeric";
import { project_target, route_waypoint } from "./target";

export const VELOCITY_SCALE = 17 / 65536;
export const LANE_HORIZON = 32;
export const SHOT_SPEED_RANGE = [3, 6] as const;
export const BLOCK_RADIUS = 8;
export type Motion = readonly [Vec2, Vec2];

export function rom_direction(dx: number, dy: number): number {
  if (dx === 0 && dy === 0) return 8;
  if (Math.abs(dx) > 2 * Math.abs(dy)) return dx > 0 ? 2 : 6;
  if (Math.abs(dy) > 2 * Math.abs(dx)) return dy > 0 ? 0 : 4;
  return dy > 0 ? (dx > 0 ? 1 : 7) : (dx > 0 ? 3 : 5);
}

export function velocity(player: Player): Vec2 {
  return player.motion_x === null || player.motion_y === null
    ? [player.vx * 256 * VELOCITY_SCALE, player.vy * 256 * VELOCITY_SCALE]
    : [player.motion_x, player.motion_y];
}

export function coast_projection(player: Player, frames: number): Vec2 {
  let x = player.x, y = player.y;
  let [vx, vy] = velocity(player);
  for (let frame = 0; frame < frames; frame++) {
    vx = vx * 63 / 64; vy = vy * 63 / 64;
    x += vx; y += vy;
  }
  return [x, y];
}

export function bounded_projection(player: Player, frames: number, { boards = true } = {}): [Vec2, number] {
  const [vx, vy] = velocity(player);
  const point: Vec2 = [player.x + vx * frames, player.y + vy * frames];
  const bounded = boards ? project_target(point) : point;
  return [bounded, player.projection_uncertainty + distance(point, bounded)];
}

export function skate_step(motion: Motion, heading: number, turning: number, pad: Vec2,
  acceleration: number, limit: number): [Motion, number, number] {
  const [[x, y], previous] = motion;
  let [vx, vy] = previous, [ux, uy] = pad;
  const wanted = ux || uy ? mod(pyRound(Math.atan2(ux, uy) * 4 / Math.PI), 8) : heading;
  if (wanted !== heading) {
    turning = (turning || 6) - 1;
    if (turning === 0) {
      const difference = mod(wanted - heading + 4, 8) - 4;
      heading = mod(heading + (difference > 0 ? 1 : -1), 8);
    }
    ux = uy = 0;
  }
  const length = Math.max(1, Math.hypot(ux, uy));
  const nx = vx * 63 / 64 + acceleration * ux / length, ny = vy * 63 / 64 + acceleration * uy / length;
  if (Math.hypot(nx, ny) <= Math.max(limit, Math.hypot(vx, vy))) { vx = nx; vy = ny; }
  else { vx *= 63 / 64; vy *= 63 / 64; }
  return [[ [x + vx, y + vy], [vx, vy] ], heading, turning];
}

export function blocks_shot_lane(player: Player, start: Vec2, goal: Vec2, delay = 0): boolean {
  if (player.motion_x === null || player.motion_y === null) return false;
  const dx = goal[0] - start[0], dy = goal[1] - start[1], length = Math.hypot(dx, dy);
  if (length < 1) return false;
  const ux = dx / length, uy = dy / length, px = player.x - start[0], py = player.y - start[1];
  const along = px * ux + py * uy, across = px * uy - py * ux;
  if (!(0 < along && along < length) || Math.abs(across) > BLOCK_RADIUS) return false;
  const [vx, vy] = velocity(player), forward = vx * ux + vy * uy, sideways = vx * uy - vy * ux;
  for (const launch of [delay, delay + 4]) for (const speed of SHOT_SPEED_RANGE) {
    if (speed <= forward) return false;
    const travel = (along + forward * launch) / (speed - forward), time = launch + travel;
    if (!(0 < speed * travel && speed * travel < length) || time > LANE_HORIZON) return false;
    if (Math.abs(across + sideways * time) > BLOCK_RADIUS) return false;
  }
  return true;
}

export function skating(player: Player, optimistic = false): [number, number, number] {
  let speed = player.speed ?? (optimistic ? 30 : 10);
  const agility = player.agility ?? (optimistic ? 30 : 10), weight = player.weight ?? (optimistic ? 32 : 88);
  const energy = Math.max(0, Math.min(4096, player.energy ?? (optimistic ? 4096 : 2048)));
  speed = Math.max(0, Math.min(30, speed + player.movement_bonus));
  const index = Math.min(15, Math.floor(Math.floor(speed / 2) * energy / 4096));
  let limit = ((20 + index) * 275) ** 2;
  if (mod(speed, 2) && index < 15) limit = (limit + ((21 + index) * 275) ** 2) / 2;
  const factor = Math.max(1, Math.floor((64 - Math.floor(weight / 4) + agility + player.movement_bonus) / 2));
  return [Math.floor(200 * factor / 32) * VELOCITY_SCALE, Math.sqrt(limit) * VELOCITY_SCALE, energy];
}

export function facing(player: Player): Vec2 {
  const angle = mod(player.facing ?? player.orientation, 8) * Math.PI / 4;
  return [Math.sin(angle), Math.cos(angle)];
}

export function boost_impulse(player: Player): number {
  return Math.floor(Math.max(0, skating(player)[2] - 204) / 128) * 200 * VELOCITY_SCALE;
}

export function burst_velocity(player: Player): Vec2 {
  const [fx, fy] = facing(player), [vx, vy] = velocity(player), impulse = boost_impulse(player);
  return [vx + fx * impulse, vy + fy * impulse];
}

export function stop_projection(player: Player, frames: number, { bounded = false } = {}): Motion | null {
  let x = player.x, y = player.y, [vx, vy] = velocity(player);
  for (let frame = 0; frame < frames; frame++) {
    vx = copySign(Math.max(0, Math.abs(vx) - 150 * VELOCITY_SCALE), vx) * 63 / 64;
    vy = copySign(Math.max(0, Math.abs(vy) - 150 * VELOCITY_SCALE), vy) * 63 / 64;
    x += vx; y += vy;
    if (bounded && distance([x, y], project_target([x, y])) > 1e-6) return null;
  }
  return [[x, y], [vx, vy]];
}

export function check_approach(player: Player, carrier: Player): [number, Vec2, number] | null {
  const [vx, vy] = burst_velocity(player), [tx, ty] = velocity(carrier);
  const dx = carrier.x - player.x, dy = carrier.y - player.y, rx = tx - vx, ry = ty - vy;
  if (Math.max(Math.abs(rx), Math.abs(ry)) >= 32768 * VELOCITY_SCALE) return null;
  const speed2 = rx * rx + ry * ry, dot = dx * rx + dy * ry;
  if (speed2 < 0.01 || dot >= 0) return null;
  const discriminant = dot * dot - speed2 * (dx * dx + dy * dy - 16 ** 2);
  if (discriminant < 0) return null;
  let time = Math.max(0, (-dot - Math.sqrt(discriminant)) / speed2);
  if (time > 8) return null;
  time = Math.ceil(time);
  const separation: Vec2 = [dx + rx * time, dy + ry * time];
  if (Math.hypot(...separation) > 16) return null;
  const impact = Math.max(0, Math.trunc(-(rx * separation[0] + ry * separation[1]) / (4096 * VELOCITY_SCALE)));
  return [time, [player.x + vx * time, player.y + vy * time], impact];
}

export function boost_safe(player: Player, target: Vec2): boolean {
  const dx = target[0] - player.x, dy = target[1] - player.y, dist = Math.hypot(dx, dy);
  const direction = facing(player), [vx, vy] = velocity(player), [acceleration, , energy] = skating(player);
  const speed = Math.hypot(vx, vy) + boost_impulse(player), stopping = speed * speed / (2 * acceleration + speed / 64);
  return energy >= 1024 && dist > Math.max(55, stopping + 12)
    && dx * direction[0] + dy * direction[1] > dist * 0.95
    && (Math.max(Math.abs(player.y), Math.abs(target[1])) < 248 || pointEqual(route_waypoint([player.x, player.y], target), target));
}

export function arrival_time(player: Player, target: Vec2, { optimistic = false, boost = false } = {}): number {
  target = project_target(target);
  const waypoint = Math.max(Math.abs(player.y), Math.abs(target[1])) < 248 ? target : route_waypoint([player.x, player.y], target);
  const dx = waypoint[0] - player.x, dy = waypoint[1] - player.y, dist = Math.hypot(dx, dy);
  const route_extra = distance(waypoint, target);
  if (dist + route_extra < 7) return 0;
  const [vx, vy] = velocity(player), [acceleration, limit] = skating(player, optimistic), [fx, fy] = facing(player);
  let along = (vx * dx + vy * dy) / Math.max(dist, 1), lateral = Math.abs(vx * dy - vy * dx) / Math.max(dist, 1);
  let turn = 4 * (1 - (fx * dx + fy * dy) / Math.max(dist, 1));
  if (optimistic && player.projection_uncertainty) { along = Math.hypot(vx, vy); lateral = turn = 0; }
  if (boost && boost_safe(player, waypoint)) along += boost_impulse(player);
  const effective = Math.max(acceleration * 0.5, acceleration - Math.max(0, along) / 64);
  const remaining = Math.max(0, dist + route_extra - 7 - player.projection_uncertainty);
  let travel: number;
  if (along > limit) travel = remaining / Math.max(limit, along * 0.85);
  else {
    const accelerate = Math.max(0, (limit - along) / effective);
    const covered = along * accelerate + effective * accelerate * accelerate / 2;
    travel = covered >= remaining ? (-along + Math.sqrt(along * along + 2 * effective * remaining)) / effective
      : accelerate + (remaining - covered) / limit;
  }
  return Math.max(0, travel) + turn + Math.min(20, lateral / (2 * acceleration));
}

export function puck_path(puck: Player, horizon = 64, { sample_interval = 4 } = {}): [number, Vec2, number][] {
  let x = puck.x, y = puck.y, [vx, vy] = velocity(puck), z = Math.max(0, puck.height || 0), vz = puck.motion_z || 0;
  const points: [number, Vec2, number][] = [];
  for (let frame = 1; frame <= horizon; frame++) {
    if (z <= 0) { vx *= puck.friction; vy *= puck.friction; }
    x += vx; y += vy;
    if (z > 0 || vz) {
      vz -= 102 * VELOCITY_SCALE; z += vz;
      if (z < 0) { z = 0; vz = -vz / 2; if (Math.abs(vz) < 0.05) vz = 0; }
    }
    if (distance([x, y], project_target([x, y])) > 1e-6) break;
    if (mod(frame, sample_interval) === 0) points.push([frame, [x, y], z]);
  }
  return points;
}
