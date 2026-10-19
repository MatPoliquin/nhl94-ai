import { eligible, on_ice } from "./defense";
import { VELOCITY_SCALE, arrival_time, bounded_projection, facing, rom_direction, skating, velocity } from "./motion";
import { mod, pyRound, distance } from "./numeric";
import { grounded_step } from "./skating";
import { _clear_segment, aim_pass, project_target } from "./target";
import { GameState, Player, Team, neutralAction, type Vec2 } from "./state";

export { rom_direction } from "./motion";
export const PASS_HORIZON = 48;
export const RELEASE_FRAMES = 4;
export const LIVE_RELEASE_FRAMES = 2;
export const SAMPLES = 8;
export const HOCKEY_PASS_PRESS_FRAMES = 4;
// NumPy default_rng(94).uniform(-1, 1, (8, 2)), not a different seeded PRNG.
export const _SCENARIOS: readonly Vec2[] = [
  [-0.5206581488975943, 0.577386089588275],
  [0.44408910178056105, -0.9084440085604344],
  [-0.34108614078236954, -0.4757733928037846],
  [0.12854701686641157, 0.7736261930402584],
  [-0.3888674357851327, 0.9442264172759047],
  [-0.002655188257643326, 0.6164595111044717],
  [0.7465069468200407, -0.7183455211468097],
  [-0.0529619730655031, -0.3083051478291041],
];

export type PassContact = readonly [Vec2, Vec2, number, number];
export type ReceiverOrigin = readonly [number, number, number, number];
export interface PassDiagnostics {
  slot: number;
  status: string;
  recipient?: number | null;
  recipient_model?: string;
  recipient_unstable?: boolean;
  point?: Vec2;
  flight_frames?: number;
  forward_gain?: number;
  direction?: number;
  release_frames?: number;
  geometry?: string;
  cue_frame?: number;
  first_contact_frame?: number | null;
  margin?: number;
  robustness?: number;
  bypassed?: number;
  shot_value?: number;
  value?: number;
  purpose?: string;
  worthwhile?: boolean;
}

export class PassOption {
  continuation_value = 0;
  continuation_target: Vec2 | null = null;
  continuation_frames = 0;
  continuation_position_value = 0;
  continuation_finish_value = 0;
  reception_value: number | null = null;
  continuation_safe: boolean | null = null;
  continuation_risk = 0;
  contact: PassContact | null = null;
  receiver_origin: ReceiverOrigin | null = null;
  release_frames: number | null = null;

  constructor(
    public index: number, public slot: number, public point: Vec2,
    public flight_frames: number, public direction: number, public margin: number,
    public robustness: number, public forward_gain: number, public bypassed: number,
    public shot_value: number, public value: number,
    options: { contact?: PassContact | null; receiver_origin?: ReceiverOrigin | null; release_frames?: number | null } = {},
  ) { Object.assign(this, options); }

  get opportunity(): number {
    return Math.max(this.reception_value ?? this.shot_value, this.continuation_value);
  }
}
export type PassOptionSnapshot = Omit<PassOption, "opportunity">;

export function pad_direction(passer: Player, receiver: Player): number {
  const buttons = neutralAction();
  aim_pass(buttons, passer, receiver);
  const dx = buttons[7] - buttons[6], dy = buttons[4] - buttons[5];
  return dx || dy ? mod(pyRound(Math.atan2(dx, dy) * 4 / Math.PI), 8) : 8;
}

export function selected_receiver(team: Team, puck: Player, passer_slot: number, direction: number): number | null {
  const candidates: [number, number, number][] = [];
  for (const [index, player] of team.players.entries()) {
    const slot = team.skater_scnum_base() + index;
    if (slot === passer_slot || player.role !== null && player.role <= 0 || player.unavailable & 4) continue;
    const dx = player.x - puck.x, dy = player.y - puck.y, sector = rom_direction(dx, dy);
    if (sector === 8) continue;
    const difference = mod(sector - direction + 4, 8) - 4;
    if (Math.abs(difference) <= 1) candidates.push([dx * dx + dy * dy + (difference * 256) ** 2, -slot, slot]);
  }
  candidates.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  return candidates[0]?.[2] ?? null;
}

export function pass_launch_view(state: GameState, passer: Player): [Team, Player] {
  const owner = state.engine.puck_owner, team = Object.assign(new Team(state.team1.controller), state.team1);
  const puck = Object.assign(new Player(), state.puck), carrier = grounded_step(passer, [0, 0]);
  const pads: readonly Vec2[] = [[0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 0], [0, 0]];
  team.players = state.team1.players.map((player, index) => {
    const slot = state.team1.skater_scnum_base() + index;
    let future = Object.assign(new Player(), player);
    if (slot !== owner && on_ice(player)) {
      const known = player.steering !== null && player.steering >= 0 && player.steering < 10 &&
        [player.weight, player.agility, player.speed, player.energy].every((value) => value !== null);
      const pad = known ? pads[player.steering!]! : [0, 0] as const;
      for (let step = 0; step < 1 + Number(slot < owner); step++) {
        future = grounded_step(future, pad, { stop: known && player.steering === 9 });
      }
    }
    return future;
  });
  const shift = puck.friction === 511 / 512 ? 9 : 6;
  for (const [index, axis] of (["x", "y"] as const).entries()) {
    let raw = pyRound(velocity(puck)[index]! / VELOCITY_SCALE);
    raw -= (raw >> shift) || Number(!!raw);
    const precise = axis === "x" ? puck.precise_x : puck.precise_y;
    const integrated = Math.floor((precise ?? puck[axis]) + raw * VELOCITY_SCALE);
    const hotspot = carrier[axis] + (axis === "x" ? passer.stick_x! : passer.stick_y!);
    puck[axis] = integrated + ((hotspot - integrated) >> 2);
  }
  return [team, puck];
}

export function option_receiver(receiver: Player, option: PassOption): Player {
  if (option.receiver_origin === null) return receiver;
  const future = Object.assign(new Player(), receiver);
  [future.x, future.y, future.motion_x, future.motion_y] = option.receiver_origin;
  return future;
}

export function _nominal_speed(passing: number, goalie = false): number {
  const speed = 160 + 2 * (goalie ? 8 : passing);
  return passing & 1 ? speed + Math.floor(speed / 16) : speed;
}

export function pass_speed(passer: Player): number | null {
  return passer.passing === null ? null :
    _nominal_speed(passer.passing, passer.role === 0) * 65536 / (16 * 60) * VELOCITY_SCALE;
}

function _word(value: number): number { return mod(value + 32768, 65536) - 32768; }
function _divide(value: number, divisor: number): number { return Math.trunc(value / divisor); }

export function _rom_sqrt(value: number): number {
  value = mod(value, 0x100000000);
  if (value <= 1600) return Math.floor(Math.sqrt(value));
  if (value > 0xf00000) {
    let low = 0, high = 65535;
    for (;;) {
      const middle = Math.floor((low + high) / 2);
      if (middle === low) return low;
      if (middle * middle > value) high = middle;
      else low = middle;
    }
  }
  let guess = ((value >>> 8) + 2) & 65535;
  for (let index = 0; index < 10; index++) {
    const next = (Math.floor(value / guess) + guess & 65535) >>> 1;
    if (next === guess) break;
    guess = next;
  }
  return guess;
}

export function rom_pass_vector(
  dx: number, dy: number, vx: number, vy: number, passing: number, { goalie = false } = {},
): Vec2 | null {
  const speed = _nominal_speed(passing, goalie);
  // Multiplication is deliberately not a JS bitwise operation: Python shifts an unbounded integer.
  const qx = _word(Math.floor(vx * 240 / 65536)), qy = _word(Math.floor(vy * 240 / 65536));
  const ax = dx >> 2, ay = dy >> 2;
  const j = _word(2 * _word(qx * ax + qy * ay)), k = qx * qx + qy * qy - (speed >> 2) ** 2;
  const root = _rom_sqrt(j * j - 4 * _word(k) * _word(ax * ax + ay * ay));
  const divisor = (_word(k) >> 2) || 1;
  let time = _divide(_word(root - j), divisor);
  if (time < 0) time = _divide(_word(-root - j), divisor);
  time ||= 1;
  if ((time & 65535) > 24) time = 24;
  const result: number[] = [];
  for (const [separation, component] of [[dx, qx], [dy, qy]]) {
    const lead = component! * time >> 1;
    const register = ((lead & 0xffff0000) | ((lead + separation!) & 65535)) >>> 0;
    const signed = (register << 16) | (register >>> 16);
    const output = _divide(signed, time * 120);
    if (output < -32768 || output > 32767) return null;
    result.push(output);
  }
  return [result[0]!, result[1]!];
}

function _live_pass_contact(puck: Player, passer: Player, receiver: Player, delay: number): PassContact | null {
  const [vx, vy] = velocity(receiver), [px, py] = velocity(passer);
  const start: Vec2 = [Math.floor(puck.x + px * delay), Math.floor(puck.y + py * delay)];
  const target: Vec2 = [Math.floor(receiver.x + vx * delay) + receiver.stick_x!, Math.floor(receiver.y + vy * delay) + receiver.stick_y!];
  const vector = rom_pass_vector(target[0] - start[0], target[1] - start[1],
    pyRound(vx / VELOCITY_SCALE), pyRound(vy / VELOCITY_SCALE), passer.passing!, { goalie: passer.role === 0 });
  if (vector === null) return null;
  const sx = vector[0] * VELOCITY_SCALE, sy = vector[1] * VELOCITY_SCALE;
  let closest = Infinity, point: Vec2 = start, frame = 0;
  for (let elapsed = 1; elapsed <= PASS_HORIZON; elapsed++) {
    const travel = puck.friction !== 1 ? (1 - puck.friction ** elapsed) / (1 - puck.friction) : elapsed;
    const current: Vec2 = [start[0] + sx * travel, start[1] + sy * travel];
    const gap = distance(current, [target[0] + vx * elapsed, target[1] + vy * elapsed]);
    if (gap < closest) { closest = gap; point = current; frame = elapsed; }
  }
  if (closest > 14 || distance(point, project_target(point)) > 1e-6) return null;
  return [start, point, frame, Math.hypot(sx, sy) * puck.friction ** frame];
}

export function pass_release_frames(receiver: Player, delay = RELEASE_FRAMES): number {
  return receiver.stick_x !== null && receiver.stick_y !== null ? Math.min(delay, LIVE_RELEASE_FRAMES) : delay;
}

export function pass_contact(puck: Player, passer: Player, receiver: Player, delay = RELEASE_FRAMES): PassContact | null {
  const speed = pass_speed(passer);
  if (speed === null || receiver.motion_x === null || receiver.motion_y === null) return null;
  if (receiver.stick_x !== null && receiver.stick_y !== null) return _live_pass_contact(puck, passer, receiver, pass_release_frames(receiver, delay));
  const [vx, vy] = velocity(receiver), [px, py] = velocity(passer), [fx, fy] = facing(receiver);
  const start: Vec2 = [puck.x + px * delay, puck.y + py * delay];
  const dx = receiver.x + vx * delay + fx * 10 - start[0], dy = receiver.y + vy * delay + fy * 10 - start[1];
  const a = vx * vx + vy * vy - speed * speed, b = 2 * (dx * vx + dy * vy), c = dx * dx + dy * dy;
  const discriminant = b * b - 4 * a * c;
  if (discriminant < 0 || Math.abs(a) < 0.01) return null;
  const roots = [(-b - Math.sqrt(discriminant)) / (2 * a), (-b + Math.sqrt(discriminant)) / (2 * a)].filter((time) => time > 0);
  const time = Math.min(...roots);
  if (!roots.length || time > PASS_HORIZON) return null;
  const point: Vec2 = [start[0] + dx + vx * time, start[1] + dy + vy * time];
  return distance(point, project_target(point)) > 1e-6 ? null : [start, point, time, speed];
}

export function pass_point_at(puck: Player, receiver: Player, contact: PassContact, time: number): Vec2 {
  const [start, point, flight] = contact, live = receiver.stick_x !== null && receiver.stick_y !== null;
  const fraction = live && puck.friction !== 1 ? (1 - puck.friction ** time) / (1 - puck.friction ** flight) : time / flight;
  return [start[0] + (point[0] - start[0]) * fraction, start[1] + (point[1] - start[1]) * fraction];
}

export function one_timer_contact_frame(puck: Player, receiver: Player, contact: PassContact, release: number): number | null {
  const flight = contact[2], [vx, vy] = velocity(receiver), [fx, fy] = facing(receiver);
  const offset: Vec2 = receiver.stick_x !== null && receiver.stick_y !== null ? [receiver.stick_x, receiver.stick_y] : [fx * 10, fy * 10];
  const times = Array.from({ length: Math.max(0, Math.ceil(flight) - 1) }, (_, index) => index + 1);
  times.push(flight);
  for (const time of times) {
    const position = pass_point_at(puck, receiver, contact, time);
    const body: Vec2 = [receiver.x + vx * (release + time), receiver.y + vy * (release + time)];
    if (distance(position, body) <= 8 || distance(position, [body[0] + offset[0], body[1] + offset[1]]) <= 14) return release + time;
  }
  return null;
}

export function _contact_offsets(player: Player): readonly Vec2[] {
  return player.stick_x !== null && player.stick_y !== null && (player.stick_x || player.stick_y) ?
    [[0, 0], [player.stick_x, player.stick_y]] : [[0, 0]];
}

export function pressure_margin(opponents: Team, point: Vec2, time: number): number {
  let result = 60, found = false;
  for (const player of opponents.players) {
    if (player.role !== null && player.role <= 0) continue;
    for (const offset of _contact_offsets(player)) {
      const margin = arrival_time(player, [point[0] - offset[0], point[1] - offset[1]], { optimistic: true, boost: true }) - time;
      result = found ? Math.min(result, margin) : margin;
      found = true;
    }
  }
  return result;
}

export function _line_intersects_circle(start: Vec2, end: Vec2, center: Vec2, radius: number): boolean {
  const dx = end[0] - start[0], dy = end[1] - start[1], length = dx * dx + dy * dy;
  const projection = (center[0] - start[0]) * dx + (center[1] - start[1]) * dy;
  const fraction = length !== 0 ? Math.max(0, Math.min(1, projection / length)) : 0;
  return (center[0] - (start[0] + fraction * dx)) ** 2 + (center[1] - (start[1] + fraction * dy)) ** 2 <= radius ** 2;
}

function _swept_contact(state: GameState, player: Player, segment: readonly [Vec2, Vec2], times: Vec2, radius: number, offset: Vec2 = [0, 0]): boolean {
  const boards = player !== state.team1.goalie && player !== state.team2.goalie;
  const first = bounded_projection(player, times[0], { boards }), last = bounded_projection(player, times[1], { boards });
  return _line_intersects_circle(
    [segment[0][0] - first[0][0] - offset[0], segment[0][1] - first[0][1] - offset[1]],
    [segment[1][0] - last[0][0] - offset[0], segment[1][1] - last[0][1] - offset[1]],
    [0, 0], radius + Math.max(first[1], last[1]));
}

export function shot_value(state: GameState, shooter: Player, point: Vec2 | null = null, delay = 0): number {
  point ??= [shooter.x, shooter.y];
  const sign = state.team2.net.y > state.team1.net.y ? 1 : -1;
  if (Math.abs(point[0]) > 71 || !(150 < point[1] * sign && point[1] * sign < 245)) return 0;
  const goalie = state.team2.goalie, [gx] = velocity(goalie);
  let clear = 0;
  for (const goal of state._shot_target_points_for_net(state.team2.net)) {
    const travel = distance(point, goal) / 4;
    let blocked = false;
    for (const other of [...state.team2.players, goalie]) {
      if (!on_ice(other)) continue;
      const [position, uncertainty] = bounded_projection(other, delay + travel / 2, { boards: other !== goalie });
      if (_line_intersects_circle(point, goal, position, (other === goalie ? 12 : 8) + uncertainty)) { blocked = true; break; }
    }
    clear += Number(!blocked);
  }
  return clear * 12 + Math.max(0, point[1] * sign - 170) * 0.3 + (shooter.shot_accuracy ?? 15) * 0.3 +
    Math.min(20, Math.abs(point[0] - goalie.x - gx * delay) / 3);
}

export function one_timer_position_ok(option: PassOption, passer: Player, current_shot: number): boolean {
  return !(!(option.point[0] * passer.x <= 0 && Math.abs(option.point[0] - passer.x) > 30) && option.shot_value <= current_shot + 12);
}

export interface EvaluatePassOptions {
  decision_interval?: number;
  release_prediction?: boolean;
  rank_risk?: boolean;
  one_timer_window?: boolean;
}

export function evaluate_pass(
  state: GameState, passer: Player, index: number, originalReceiver: Player, purpose = "advance",
  { decision_interval = 4, release_prediction = false, rank_risk = false, one_timer_window = true }: EvaluatePassOptions = {},
): [PassOption | null, PassDiagnostics] {
  if (decision_interval < 1) throw new Error("Pass decision interval must be positive");
  if (rank_risk || release_prediction || !one_timer_window) throw new Error("Experimental pass profiles are not supported by baseline Classic");
  const receiver = originalReceiver, slot = state.team1.skater_scnum_base() + index;
  const details: PassDiagnostics = { slot, status: "missing-feedback" };
  const reject = (status: string): [null, PassDiagnostics] => [null, { ...details, status }];
  const required = ["motion_x", "motion_y", "speed", "agility", "weight", "energy", "facing", "stick"] as const;
  if (state.puck.height === null || !eligible(receiver) || [passer, receiver].some((p) => required.some((field) => p[field] === null)) || passer.passing === null) return [null, details];
  if (receiver.projection_uncertainty) return reject("uncertain-reception");
  const owner = state.engine.puck_owner, direction = pad_direction(passer, receiver), release = pass_release_frames(receiver);
  const recipient = selected_receiver(state.team1, state.puck, owner, direction);
  Object.assign(details, { recipient, recipient_model: "current-position", recipient_unstable: false });
  if (recipient !== slot) return reject("different-rom-recipient");
  const contact = pass_contact(state.puck, passer, receiver);
  if (contact === null) return reject("unreachable-reception");
  const [start, point, flight, speed] = contact;
  if (state.puck.height > 8 || !_clear_segment(start, point)) return reject("airborne-or-unsafe-route");
  const total = flight + release, sign = state.team2.net.y > state.team1.net.y ? 1 : -1, gain = (point[1] - passer.y) * sign;
  const live_geometry = receiver.stick_x !== null && receiver.stick_y !== null;
  Object.assign(details, { point, flight_frames: flight, forward_gain: gain, direction, release_frames: release,
    geometry: live_geometry ? "rom-sprite-and-integer-launch" : "facing-hotspot-estimate" });
  if (purpose === "one-timer") {
    const accuracy = receiver.shot_accuracy ?? 15;
    if (Math.abs(point[0]) > 70 || !(175 + Math.max(0, 15 - accuracy) <= point[1] * sign && point[1] * sign < 245)) return reject("outside-one-timer-window");
    const cue = Math.ceil(Math.max(HOCKEY_PASS_PRESS_FRAMES, release) / decision_interval) * decision_interval;
    details.cue_frame = cue;
    const first = one_timer_contact_frame(state.puck, receiver, contact, release);
    details.first_contact_frame = first;
    if (first === null) return reject("unreachable-reception");
    if (flight < 4 || first <= cue) return reject("too-short-for-one-timer-cue");
  }
  const crosses = state.puck.y * sign < 88 && 88 <= point[1] * sign;
  if (crosses && state.engine.offsides_enabled === null) return reject("unknown-offside-rule");
  if (crosses && state.engine.offsides_enabled && state.team1.players.some((p) => p !== passer &&
    (p.y + velocity(p)[1] * (release + flight * (sign * 88 - start[1]) / (point[1] - start[1]))) * sign >= 88)) return reject("offside-receiver");
  if (purpose === "advance" && gain < 20) return reject("insufficient-progress");
  if (purpose !== "one-timer" && speed > (13000 + 350 * receiver.stick!) * VELOCITY_SCALE) return reject("too-fast-to-control");
  if (state.team2.players.some((p) => on_ice(p) && (p.motion_x === null || p.motion_y === null))) return reject("unknown-opponent-motion");
  let margin = 60;
  const safe = Array<boolean>(SAMPLES).fill(true), steps: number[] = [];
  for (let time = 4; time < Math.ceil(flight); time += 4) steps.push(time);
  if (!steps.includes(flight)) steps.push(flight);
  steps.sort((a, b) => a - b);
  let previous: Vec2 = start, previous_elapsed = release;
  for (const time of steps) {
    const fraction = live_geometry && state.puck.friction !== 1 ? (1 - state.puck.friction ** time) / (1 - state.puck.friction ** flight) : time / flight;
    const puck: Vec2 = [start[0] + (point[0] - start[0]) * fraction, start[1] + (point[1] - start[1]) * fraction], elapsed = time + release;
    for (const friend of [...state.team1.players, state.team1.goalie]) {
      if (friend === passer || friend === state.team1.players[index] || !on_ice(friend)) continue;
      if (_swept_contact(state, friend, [previous, puck], [previous_elapsed, elapsed], 8)) return reject("friendly-obstruction");
      for (const offset of _contact_offsets(friend).slice(1)) {
        if (_swept_contact(state, friend, [previous, puck], [previous_elapsed, elapsed], 14, offset)) return reject("friendly-stick-obstruction");
      }
    }
    for (const other of [...state.team2.players, state.team2.goalie]) {
      if (!on_ice(other)) continue;
      const [center, uncertainty] = bounded_projection(other, elapsed, { boards: other !== state.team2.goalie });
      const body_radius = other === state.team2.goalie ? 16 : 14, radius = body_radius + uncertainty;
      if (_swept_contact(state, other, [previous, puck], [previous_elapsed, elapsed], body_radius)) return reject("moving-interception");
      const offsets = _contact_offsets(other);
      for (const offset of offsets.slice(1)) {
        if (_swept_contact(state, other, [previous, puck], [previous_elapsed, elapsed], 14, offset)) return reject("moving-stick-interception");
      }
      if (other !== state.team2.goalie) {
        const arrival = Math.min(...offsets.map((offset) => arrival_time(other, [puck[0] - offset[0], puck[1] - offset[1]], { optimistic: true, boost: true })));
        margin = Math.min(margin, arrival - elapsed);
        const [acceleration] = skating(other, true), spread = acceleration * Math.min(elapsed, 12) ** 2 / 4;
        for (const [sample, offset] of _SCENARIOS.entries()) {
          safe[sample] = safe[sample]! && offsets.every((contact) => distance(
            [center[0] + contact[0] + offset[0] * spread, center[1] + contact[1] + offset[1] * spread], puck) > radius);
        }
      }
    }
    previous = puck; previous_elapsed = elapsed;
  }
  margin = Math.min(margin, pressure_margin(state.team2, point, total + (purpose === "one-timer" ? 0 : 4)));
  const robustness = safe.filter(Boolean).length / SAMPLES;
  Object.assign(details, { margin, robustness });
  if (margin < 3 || robustness < 0.875) return reject("contested-reception-or-lane");
  const bypassed = state.team2.players.filter((p) => on_ice(p) && passer.y * sign < p.y * sign && p.y * sign < point[1] * sign).length;
  const shooting = shot_value(state, receiver, point, total), value = gain * 0.4 + bypassed * 16 + Math.min(margin, 20) + shooting * 0.6;
  const option = new PassOption(index, slot, point, flight, direction, margin, robustness, gain, bypassed, shooting, value,
    { contact, receiver_origin: [receiver.x, receiver.y, ...velocity(receiver)], release_frames: release });
  return [option, { ...details, status: "safe", bypassed, shot_value: shooting, value }];
}
