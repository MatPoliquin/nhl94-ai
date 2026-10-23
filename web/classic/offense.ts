import { body_clearance, carry_path, _segment_clearance } from "./carry";
import { controlled_slot, on_ice } from "./defense";
import { clone_forecast, project_body, set_position } from "./forecast";
import { PassLifecycle, type TacticalDiagnostics } from "./lifecycle";
import { bounded_projection, coast_projection, facing, skate_step, skating, stop_projection, velocity } from "./motion";
import { distance, lexCompare, mod, pointEqual, pyRound } from "./numeric";
import { evaluate_pass, one_timer_position_ok, pressure_margin, rom_direction, shot_value, type PassDiagnostics, type PassOption } from "./passing";
import { GameState, Player, type Vec2 } from "./state";
import { project_target, route_waypoint } from "./target";

export const FEINT_FRAMES = 18;
export const GOALIE_CLEARANCE = 32;
export const GOALIE_SHOT_CLEARANCE = 24;
export const GOALIE_HORIZON = 32;
export const SHOT_RELEASE_DELAY = 2;
export const CARRY_SAMPLES = [4, 8, 12, FEINT_FRAMES] as const;
export type OffensePlan = readonly [string, Vec2, PassOption | null];
export interface GoalieAvoidanceDetails {
  goalie_clearance: number;
  original_goalie_clearance: number;
  goalie_avoidance_safe: boolean;
  goalie_carry_model: string;
  reason: string;
}
export type GoalieEscape = readonly [Vec2, GoalieAvoidanceDetails];
export interface CarryDetails extends TacticalDiagnostics {
  mode: string;
  reason: string;
  carry_safe: boolean | null;
}
export interface EstimatedCarryDetails extends TacticalDiagnostics {
  carry_safe: null;
  carry_estimated_clear: boolean;
  carry_bounded: boolean;
  carry_clearance: number;
  carry_pressure: number;
  carry_progress: number;
  carry_position_value: number;
  carry_shot_value: number;
  carry_pressure_model: string;
  carry_status: string;
  carry_movement_model: string;
}

export function normal_shot_release_frames(player: Player, hold_frames: number): number {
  const durations = player.shot_durations?.length ? player.shot_durations : Array<number>(8).fill(4);
  let index = 0, timer = -1;
  const bound = hold_frames + 2 * durations.reduce((sum, duration) => sum + duration + 1, 0) + 3;
  for (let elapsed = 1; elapsed < bound; elapsed++) {
    if (index >= 7) return elapsed + 1;
    if (timer < 0) timer = durations[index]!;
    else {
      timer--;
      if (timer < 0) { index++; if (index < 7) timer = durations[index]!; }
    }
    if (index >= 7) return elapsed + 1;
    if (index < 4 && (elapsed === hold_frames || player.shot_power !== null && player.shot_power < 20 && index > 2)) index = 7 - index;
  }
  throw new Error("Shot animation did not reach native contact.");
}

export function shot_release_in_front(state: GameState, player: Player, hold_frames: number): boolean {
  const sign = state.team2.net.y > state.team1.net.y ? 1 : -1;
  const point = state.engine.puck_owner_known ? state.puck : player;
  const [, y] = coast_projection(player, normal_shot_release_frames(player, hold_frames));
  const reach = Math.max(...(player.shot_offsets_y ?? [-20, 20]).map((offset) => offset * sign));
  return Math.max(point.y * sign, player.y * sign + reach, y * sign + reach) < state.team2.net.y * sign;
}

export function ordinary_shot_conditions(state: GameState, player: Player, hold_frames: number, escape: GoalieEscape | null): [boolean, boolean] {
  const sign = state.team2.net.y > state.team1.net.y ? 1 : -1, accuracy = Math.max(0, Math.min(30, player.shot_accuracy ?? 15));
  const safe = shot_release_in_front(state, player, hold_frames) &&
    goalie_contact_time(player, state.team2.goalie, GOALIE_SHOT_CLEARANCE) > GOALIE_HORIZON;
  const eligible = Math.abs(player.x) < 48 && safe;
  return [player.y * sign > 218 + Math.max(0, 15 - accuracy) * 0.4 && eligible, player.y * sign > 175 && escape !== null && eligible];
}

export function carry_projection(player: Player, target: Vec2, frames = FEINT_FRAMES): readonly [Vec2, Vec2] | null {
  let x = player.x, y = player.y, [vx, vy] = velocity(player);
  const [acceleration, limit] = skating(player);
  let direction = player.facing ?? player.orientation, turn_left = 0;
  for (let elapsed = 0; elapsed < frames; elapsed++) {
    const dx = target[0] - x, dy = target[1] - y;
    const ux = Math.abs(dx) > 3 ? Math.sign(dx) : 0, uy = Math.abs(dy) > 3 ? Math.sign(dy) : 0;
    const [motion, heading, turning] = skate_step([[x, y], [vx, vy]], direction, turn_left, [ux, uy], acceleration, limit);
    [[x, y], [vx, vy]] = motion;
    direction = heading; turn_left = turning;
    if (distance([x, y], project_target([x, y])) > 1e-6) return null;
  }
  return [[x, y], [vx, vy]];
}

export function _braking(player: Player, target: Vec2): boolean {
  const [vx, vy] = velocity(player), dx = target[0] - player.x, dy = target[1] - player.y;
  const ux = Math.abs(dx) > 3 ? Math.sign(dx) : 0, uy = Math.abs(dy) > 3 ? Math.sign(dy) : 0;
  const wanted = ux || uy ? mod(pyRound(Math.atan2(ux, uy) * 4 / Math.PI), 8) : null;
  const direction = player.facing ?? player.orientation;
  return wanted === mod(direction + 4, 8) && mod(rom_direction(vx, vy) - direction + 2, 8) < 4 && !((player.selection_flags ?? 0) & 0x10);
}

export function _native_carrier(player: Player): boolean {
  return [player.motion_x, player.motion_y, player.weight, player.agility, player.speed, player.energy].every((value) => value !== null);
}

export function _carry_motion(player: Player, target: Vec2, frames = FEINT_FRAMES, { decision_interval = null }: { decision_interval?: number | null } = {}): readonly [Vec2, Vec2] | null {
  if (decision_interval !== null && _native_carrier(player)) {
    const path = carry_path(player, target, frames, { decision_interval }), end = path?.at(-1);
    return end ? [[end.x, end.y], velocity(end)] : null;
  }
  return _braking(player, target) ? stop_projection(player, frames, { bounded: true }) : carry_projection(player, target, frames);
}

export function projected_state(state: GameState, target: Vec2, frames = FEINT_FRAMES, { decision_interval = null }: { decision_interval?: number | null } = {}): GameState | null {
  const original = state.team1.get_player_by_scnum(state.engine.puck_owner)!;
  if (decision_interval !== null && _native_carrier(original)) {
    const path = carry_path(original, target, frames, { decision_interval });
    return path === null ? null : carry_future(state, original, path.at(-1)!, { frames });
  }
  const motion = _carry_motion(original, target, frames);
  if (motion === null) return null;
  const future = clone_forecast(state);
  for (const name of ["team1", "team2"] as const) {
    const team = future[name];
    const carrier = name === "team1" ? team.get_player_by_scnum(state.engine.puck_owner) : null;
    for (const player of [...team.players, team.goalie]) {
      if (player === carrier || !on_ice(player)) continue;
      project_body(player, frames, player !== team.goalie);
    }
    future[name] = team;
  }
  const player = future.team1.get_player_by_scnum(state.engine.puck_owner)!;
  [[player.x, player.y], [player.motion_x, player.motion_y]] = motion;
  set_position(player, [player.x, player.y]);
  set_position(future.puck, [state.puck.x + player.x - original.x, state.puck.y + player.y - original.y]);
  return future;
}

export function carry_future(state: GameState, player: Player, carrier: Player, { frames = 18 } = {}): GameState {
  const future = clone_forecast(state);
  for (const name of ["team1", "team2"] as const) {
    const team = future[name];
    for (const [index, other] of team.players.entries()) {
      if (state[name].players[index] === player) team.players[index] = carrier;
      else if (on_ice(other)) project_body(other, frames);
    }
    if (on_ice(team.goalie)) project_body(team.goalie, frames, false);
    future[name] = team;
  }
  set_position(future.puck, [state.puck.x + carrier.x - player.x, state.puck.y + carrier.y - player.y]);
  return future;
}

export function carry_clearance(state: GameState, player: Player, target: Vec2, { decision_interval = null }: { decision_interval?: number | null } = {}): number {
  if (decision_interval !== null && _native_carrier(player)) {
    const path = carry_path(player, target, 18, { decision_interval });
    return path === null ? -Infinity : body_clearance(state, player, path);
  }
  let clearance = Infinity;
  for (const frames of CARRY_SAMPLES) {
    const motion = _carry_motion(player, target, frames);
    if (motion === null) return -Infinity;
    for (const team of [state.team1, state.team2]) {
      for (const other of [...team.players, team.goalie]) {
        if (other === player || !on_ice(other)) continue;
        const [position, uncertainty] = bounded_projection(other, frames, { boards: other !== team.goalie });
        clearance = Math.min(clearance, distance(motion[0], position) - 16 - uncertainty);
      }
    }
  }
  return clearance;
}

export function carry_clear(state: GameState, player: Player, target: Vec2, options: { decision_interval?: number | null } = {}): boolean {
  return carry_clearance(state, player, target, options) > 0;
}

export function goalie_contact_time(player: Player, goalie: Player, clearance = GOALIE_CLEARANCE): number {
  if (!on_ice(goalie)) return Infinity;
  const [vx, vy] = velocity(player), [gx, gy] = velocity(goalie);
  const dx = player.x - goalie.x, dy = player.y - goalie.y, rx = vx - gx, ry = vy - gy;
  const separation = dx * dx + dy * dy - clearance ** 2;
  if (separation <= 0) return 0;
  const speed = rx * rx + ry * ry, dot = dx * rx + dy * ry, discriminant = dot * dot - speed * separation;
  return speed < 0.01 || dot >= 0 || discriminant < 0 ? Infinity : (-dot - Math.sqrt(discriminant)) / speed;
}

export function _goalie_clearance(player: Player, goalie: Player, target: Vec2, { decision_interval = null }: { decision_interval?: number | null } = {}): number {
  if (!on_ice(goalie)) return Infinity;
  const [gx, gy] = velocity(goalie);
  let previous: Vec2 = [player.x - goalie.x, player.y - goalie.y], clearance = Math.hypot(...previous);
  if (decision_interval !== null) {
    const path = carry_path(player, target, GOALIE_HORIZON, { decision_interval });
    if (path === null) return -Infinity;
    for (let frames = 1; frames < path.length; frames++) {
      const carrier = path[frames]!, relative: Vec2 = [carrier.x - goalie.x - gx * frames, carrier.y - goalie.y - gy * frames];
      clearance = Math.min(clearance, _segment_clearance(previous, relative)); previous = relative;
    }
    return clearance;
  }
  for (let frames = 4; frames <= GOALIE_HORIZON; frames += 4) {
    const motion = _carry_motion(player, target, frames);
    if (motion === null) return -Infinity;
    const relative: Vec2 = [motion[0][0] - goalie.x - gx * frames, motion[0][1] - goalie.y - gy * frames];
    clearance = Math.min(clearance, _segment_clearance(previous, relative)); previous = relative;
  }
  return clearance;
}

export function goalie_avoidance(state: GameState, player: Player | null, target: Vec2, { decision_interval = null }: { decision_interval?: number | null } = {}): GoalieEscape | null {
  const goalie = state.team2.goalie;
  if (player === null || player === state.team1.goalie || !on_ice(goalie) || distance([player.x, player.y], [goalie.x, goalie.y]) > 120) return null;
  const native = decision_interval !== null && _native_carrier(player), interval = native ? decision_interval : null;
  let clearance = _goalie_clearance(player, goalie, target, { decision_interval: interval });
  const contact = goalie_contact_time(player, goalie);
  if (clearance >= GOALIE_CLEARANCE && (native || contact > GOALIE_HORIZON)) return null;
  if (!native && contact <= GOALIE_HORIZON) {
    const [vx, vy] = velocity(player), [gx, gy] = velocity(goalie);
    const dx = player.x - goalie.x, dy = player.y - goalie.y, rx = vx - gx, ry = vy - gy;
    const time = Math.max(0, Math.min(GOALIE_HORIZON, -(dx * rx + dy * ry) / Math.max(0.01, rx * rx + ry * ry)));
    clearance = Math.min(clearance, Math.hypot(dx + rx * time, dy + ry * time));
  }
  const sign = state.team2.net.y > state.team1.net.y ? 1 : -1, [fx, fy] = facing(player);
  const points: Vec2[] = [[player.x - fx * 48, player.y - fy * 48]];
  for (const [dx, dy] of [[-48, 0], [48, 0], [-48, -24], [48, -24], [0, -48]] as const) points.push([player.x + dx, player.y + sign * dy]);
  const candidates: [number, number, Vec2][] = [];
  for (const proposal of points) {
    const point = project_target(proposal);
    if (!pointEqual(route_waypoint([player.x, player.y], point), point)) continue;
    const margin = _goalie_clearance(player, goalie, point, { decision_interval: interval });
    if (margin > clearance) candidates.push([margin, -distance(point, target), point]);
  }
  candidates.sort((a, b) => lexCompare([b[0], b[1], ...b[2]], [a[0], a[1], ...a[2]]));
  const best = candidates[0], point = best?.[2] ?? project_target([player.x, player.y - sign * 48]), margin = best?.[0] ?? clearance;
  return [point, { goalie_clearance: margin, original_goalie_clearance: clearance, goalie_avoidance_safe: margin >= GOALIE_CLEARANCE,
    goalie_carry_model: native ? "native-held-input" : "legacy-cut-or-missing-feedback",
    reason: margin >= GOALIE_CLEARANCE ? "move clear of projected goalie contact" : "maximize goalie clearance; contact may already be unavoidable" }];
}

export class OffenseController {
  readonly one_timers: boolean;
  decision_interval: number;
  one_timer_at = 0;
  pass_action = new PassLifecycle();
  pass_at = 0;
  feint_until = 0;
  feint_at = 0;
  feint_target: Vec2 | null = null;
  feint_mode = "feint";
  diagnostics: TacticalDiagnostics = {};
  readonly pass_timing = false;
  readonly carry_motion = false;
  readonly rank_pass_risk = false;
  constructor({ one_timers = true, decision_interval = 4, allow_uncertified = false, chance_creation = false } = {}) {
    if (allow_uncertified || chance_creation) throw new Error("Classic tactical experiments are unsupported");
    this.one_timers = one_timers; this.decision_interval = decision_interval;
  }
  get uses_lookahead(): boolean { return true; }
  get movement_interval(): number | null { return this.carry_motion ? this.decision_interval : null; }
  get shot_hold_frames(): number { return Math.max(4, this.decision_interval); }
  clear_carry(state: GameState, player: Player, target: Vec2): boolean { return carry_clear(state, player, target, { decision_interval: this.movement_interval }); }
  cancel(): void {
    this.pass_action.cancel(); this.feint_target = null; this.feint_until = 0; this.diagnostics = {};
  }
  observe(state: GameState, frame: number) { return this.pass_action.observe(state, frame); }
  start_pass(state: GameState, option: PassOption, frame: number, purpose: string): void {
    this.pass_action.start(state, option, frame, purpose); this.pass_at = frame + 24; this.feint_target = null; this.feint_until = 0;
  }
  carry_target(state: GameState, player: Player): Vec2 {
    return OffenseController.carry_target(state, player);
  }
  static carry_target(state: GameState, player: Player): Vec2 {
    const sign = state.team2.net.y > state.team1.net.y ? 1 : -1;
    return [player.y * sign <= 140 ? player.x >= 0 ? 35 : -35 : state.team2.goalie.x >= 0 ? -35 : 35, sign * 235];
  }
  breakaway(state: GameState, player: Player): boolean {
    const sign = state.team2.net.y > state.team1.net.y ? 1 : -1;
    if (state.team2.players.some((other) => (other.role === null || other.role > 0) && other.y * sign >= player.y * sign - 8 && Math.abs(other.x - player.x) < 90)) return false;
    const target = this.carry_target(state, player);
    if (!this.clear_carry(state, player, target)) return false;
    for (const frames of [8, 16, 24]) {
      const motion = this.carry_motion ? _carry_motion(player, target, frames, { decision_interval: this.movement_interval }) : carry_projection(player, target, frames);
      if (motion === null || pressure_margin(state.team2, motion[0], frames) < 6) return false;
    }
    return true;
  }
  passes(state: GameState, purpose = "advance", { continuations = true } = {}): [PassOption[], PassDiagnostics[]] {
    if (purpose === "one-timer" && !this.one_timers) return [[], []];
    const player = state.team1.get_player_by_scnum(state.engine.puck_owner);
    if (player === null) return [[], []];
    const choices: PassOption[] = [], diagnostics: PassDiagnostics[] = [], current = purpose === "one-timer" ? shot_value(state, player) : 0;
    for (const [index, receiver] of state.team1.players.entries()) {
      if (receiver === player) continue;
      let [option, details] = evaluate_pass(state, player, index, receiver, purpose, { decision_interval: this.decision_interval });
      details.purpose = purpose;
      if (option !== null && purpose !== "one-timer" && continuations && this.uses_lookahead) throw new Error("Offense lookahead is not part of baseline Classic");
      if (purpose === "one-timer" && option !== null && !one_timer_position_ok(option, player, current)) {
        option = null; details = { ...details, status: "weaker-one-timer-position" };
      }
      diagnostics.push(details); if (option !== null) choices.push(option);
    }
    choices.sort(purpose === "one-timer" ? (a, b) => b.shot_value - a.shot_value || b.margin - a.margin || a.slot - b.slot :
      (a, b) => b.value - a.value || a.slot - b.slot);
    return [choices, diagnostics];
  }
  _continue_cut(state: GameState, player: Player, frame: number): OffensePlan | null {
    if (this.feint_target === null) return null;
    const target = this._cut_target(state, player, this.feint_target);
    if (frame < this.feint_until && pressure_margin(state.team2, [player.x, player.y], 4) >= 3 &&
        target !== null && pointEqual(target, this.feint_target) && this.clear_carry(state, player, this.feint_target)) {
      return this._plan(state, this.feint_mode, this.feint_target,
        this.feint_mode === "one-timer-setup" ? "continue bounded one-timer setup" : "continue bounded opportunity-creating cut");
    }
    this.diagnostics.cut_cancelled = "route or pressure changed; discard predicted opportunity";
    this.feint_target = null; return null;
  }
  _cut_target(state: GameState, player: Player, target: Vec2): Vec2 | null {
    const escape = goalie_avoidance(state, player, target);
    return escape !== null ? escape[1].goalie_avoidance_safe ? escape[0] : null : target;
  }
  _cut_targets(state: GameState, player: Player, sign: number): Vec2[] {
    const proposals = [-1, 1].map((direction) => project_target([player.x + direction * 26, player.y + sign * 8]));
    const targets = proposals.map((point) => this._cut_target(state, player, point));
    if (targets.some((target, index) => target === null || !pointEqual(target, proposals[index]!))) {
      for (const direction of [-1, 1]) for (const width of [26, 48]) targets.push(this._cut_target(state, player, project_target([player.x + direction * width, player.y])));
    }
    const unique: Vec2[] = [];
    for (const point of targets) if (point !== null && !unique.some((other) => pointEqual(point, other))) unique.push(point);
    return unique;
  }
  _feint(state: GameState, player: Player, frame: number, initial_value: number, { one_timer_only = false } = {}): OffensePlan | null {
    if (frame < this.feint_at) return null;
    const sign = state.team2.net.y > state.team1.net.y ? 1 : -1;
    if (!(88 < player.y * sign && player.y * sign < 218)) return null;
    const forward: Vec2 = [player.x, sign * 235], straight_target = this._cut_target(state, player, forward);
    const straight = straight_target !== null && pointEqual(straight_target, forward) && this.clear_carry(state, player, straight_target) ?
      projected_state(state, straight_target, FEINT_FRAMES, { decision_interval: this.movement_interval }) : null;
    let one_timer_open = !!this.passes(state, "one-timer")[0].length, current = initial_value;
    if (straight !== null) {
      const mover = straight.team1.get_player_by_scnum(state.engine.puck_owner)!, [passes] = this.passes(straight, "position", { continuations: false });
      one_timer_open ||= !!this.passes(straight, "one-timer")[0].length;
      current = Math.max(current, shot_value(straight, mover), passes.length ? passes[0]!.shot_value + Math.min(passes[0]!.margin, 12) : 0);
    }
    const cuts: [number, Vec2, string, string][] = [];
    for (const target of this._cut_targets(state, player, sign)) {
      if (!pointEqual(route_waypoint([player.x, player.y], target), target) || !this.clear_carry(state, player, target)) continue;
      const future = projected_state(state, target, FEINT_FRAMES, { decision_interval: this.movement_interval });
      if (future === null) continue;
      const mover = future.team1.get_player_by_scnum(state.engine.puck_owner)!;
      if (pressure_margin(state.team2, [mover.x, mover.y], FEINT_FRAMES) < 5) continue;
      const [passes] = this.passes(future, "position", { continuations: false }), one_timers = !one_timer_open ? this.passes(future, "one-timer")[0] : [];
      const best = passes[0], value = Math.max(shot_value(future, mover), best ? best.shot_value + Math.min(best.margin, 12) : 0);
      if (one_timers.length) {
        const option = one_timers[0]!, one_timer_value = option.shot_value + Math.min(option.margin, 12);
        if (one_timer_value >= current) cuts.push([one_timer_value, target, "open a safe one-timer lane", "one-timer-setup"]);
      }
      if (!one_timer_only && value > current + 10) cuts.push([value, target, best ? "open stronger receiving position" : "open shooting lane", "feint"]);
    }
    let best: typeof cuts[number] | null = null;
    for (const cut of cuts) if (best === null || cut[0] > best[0]) best = cut;
    if (best === null) return null;
    this.feint_target = best[1]; this.feint_mode = best[3]; this.feint_until = frame + FEINT_FRAMES; this.feint_at = frame + 72;
    return this._plan(state, this.feint_mode, this.feint_target, best[2]);
  }
  _plan(state: GameState, mode: string, target: Vec2, reason: string, option: PassOption | null = null): OffensePlan {
    const actual = controlled_slot(state.team1);
    Object.assign(this.diagnostics, { mode, decision: mode, reason, target, destination: target, waypoint: target,
      actual_slot: actual, desired_slot: option ? option.slot : actual, receiver: option ? { ...option } : null, last_pass: this.pass_action.last_pass });
    return [mode, target, option];
  }
  choose(_state: GameState, _frame: number): OffensePlan | null { throw new Error("Use the baseline PossessionOffenseController; offense lookahead is unsupported"); }
  fallback_carry(_state: GameState, _player: Player): [Vec2, CarryDetails] { throw new Error("Use the baseline PossessionOffenseController; offense lookahead is unsupported"); }
}
