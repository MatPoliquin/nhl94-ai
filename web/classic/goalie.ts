import { Buttons, neutralAction, type Action, type GameState, type Player, type Vec2 } from "./state";
import { distance, required } from "./numeric";
import { DefenseController, controlled_slot, eligible, owns_puck, type ControllerDiagnostics } from "./defense";
import { VELOCITY_SCALE, arrival_time, blocks_shot_lane, skating, velocity } from "./motion";
import { evaluate_pass, one_timer_contact_frame, pass_contact, pass_release_frames, type PassDiagnostics, type PassOption } from "./passing";
import { grounded_step } from "./skating";
import { aim_pass } from "./target";

export const TAKEOVER_FRAMES = 24;
export const HANDOFF_TIMEOUT = 60;
export const SAVE_ANIMATIONS = new Set([0x146, 0x178, 0x1aa, 0x1ec, 0x21e, 0x250, 0x2a2, 0x148e, 0x14c0, 0x14f2, 0x1544]);
export const DIVE_ANIMATION = 0x2f4;
export type PassWindup = readonly [number, number, number];

export function outlet_options(state: GameState,
  { decision_interval = 4, limit = null, release_prediction = false }: {
    decision_interval?: number; limit?: number | null; release_prediction?: boolean;
  } = {}): [PassOption[], ReturnType<typeof evaluate_pass>[1][]] {
  const candidates = state.team1.players.slice(0, limit ?? undefined).map((player, index) =>
    evaluate_pass(state, state.team1.goalie, index, player, "outlet", { decision_interval, release_prediction }));
  const options = candidates.flatMap(([option]) => option === null ? [] : [option]).sort((a, b) => b.value - a.value || a.slot - b.slot);
  return [options, candidates.map(([, details]) => details)];
}

export class GoaliePlan {
  crossing_frames: number | null = null;
  crossing_x: number | null = null;
  height: number | null = null;
  deadline: number | null = null;
  receiver: Vec2 | null = null;
  reception_frames: number | null = null;
  constructor(public target: Vec2, public reason: string, options: Partial<Omit<GoaliePlan, "target" | "reason">> = {}) {
    Object.assign(this, options);
  }
}

export function goalie_motion(player: Player): [number, number] {
  let [, limit] = skating(player);
  const factor = Math.max(1, Math.floor((64 - Math.floor(required(player.weight, "weight") / 4)
    + 2 * required(player.agility, "agility") + 16 + player.movement_bonus) / 2));
  let acceleration = Math.floor(200 * factor / 32) * VELOCITY_SCALE;
  if (player.has_puck) { acceleration /= 2; limit /= Math.sqrt(2); }
  return [acceleration, limit];
}

export function goalie_arrival(player: Player, target: Vec2): number {
  const [acceleration, limit] = goalie_motion(player), [vx, vy] = velocity(player);
  const dx = target[0] - player.x, dy = target[1] - player.y, dist = Math.hypot(dx, dy);
  const along = (vx * dx + vy * dy) / Math.max(1, dist), remaining = Math.max(0, dist - 5);
  const effective = Math.max(acceleration / 2, acceleration - Math.max(0, along) / 64);
  const ramp = Math.max(0, (limit - along) / effective), covered = along * ramp + effective * ramp * ramp / 2;
  if (covered >= remaining) return Math.max(0, (-along + Math.sqrt(along * along + 2 * effective * remaining)) / effective);
  return ramp + (remaining - covered) / limit;
}

export function _receiver_contact(state: GameState, windup: PassWindup | null, receiver_slot: number | null = null): [number, Vec2] | null {
  const target = receiver_slot ?? state.engine.pass_target;
  const receiver = target === null ? null : state.team2.get_player_by_scnum(target);
  if (receiver === null) return null;
  const winding_up = receiver_slot !== null && receiver.is_one_timer &&
    (receiver.live_anim === 0x7fc || receiver.live_anim === 0x92e) &&
    (receiver.role === null || receiver.role > 0) && !(receiver.unavailable & 4);
  if (!eligible(receiver) && !winding_up) return null;
  let contact: ReturnType<typeof pass_contact>, release: number;
  if (windup !== null) {
    const passer = state.team2.get_player_by_scnum(windup[0]);
    if (passer === null || state.engine.puck_owner !== windup[0]) return null;
    contact = pass_contact(state.puck, passer, receiver); release = pass_release_frames(receiver);
  } else if (state.engine.puck_owner < 0 && state.engine.last_puck_player !== null && state.team2.owns_scnum(state.engine.last_puck_player)) {
    if (target === state.engine.last_puck_player || required(state.puck.height, "puck.height") > 8) return null;
    const [vx, vy] = velocity(state.puck), [rx, ry] = velocity(receiver);
    const offset: Vec2 = [receiver.stick_x || 0, receiver.stick_y || 0];
    const dx = receiver.x + offset[0] - state.puck.x, dy = receiver.y + offset[1] - state.puck.y;
    if (dx * (vx - rx) + dy * (vy - ry) <= 0) return null;
    let closest: [number, Vec2 | null, number] = [Infinity, null, 0];
    for (let frame = 1; frame <= 40; frame++) {
      const friction = state.puck.friction;
      const travel = friction !== 1 ? friction * (1 - friction ** frame) / (1 - friction) : frame;
      const point: Vec2 = [state.puck.x + vx * travel, state.puck.y + vy * travel];
      const stick: Vec2 = [receiver.x + rx * frame + offset[0], receiver.y + ry * frame + offset[1]];
      const dist = distance(point, stick);
      if (dist < closest[0]) closest = [dist, point, frame];
    }
    if (closest[0] > 14) return null;
    contact = [[state.puck.x, state.puck.y], required(closest[1], "receiver contact"), closest[2], Math.hypot(vx, vy)]; release = 0;
  } else return null;
  if (contact === null) return null;
  const time = one_timer_contact_frame(state.puck, receiver, contact, release);
  if (time === null) return null;
  const [rx, ry] = velocity(receiver);
  return [time, [receiver.x + rx * time, receiver.y + ry * time]];
}

export function goalie_target(state: GameState, windup: PassWindup | null = null, receiver_slot: number | null = null): GoaliePlan {
  const sign = state.team1.net.y > 0 ? 1 : -1, puck = state.puck;
  let [vx, vy] = velocity(puck), x = puck.x, y = puck.y, z = Math.max(0, required(puck.height, "puck.height")), vz = puck.motion_z || 0;
  const incoming = _receiver_contact(state, windup, receiver_slot);
  let depth = Math.max(236, Math.min(254, Math.max(state.team1.goalie.y * sign, y * sign)));
  let crossing: [number, number, number] | null = y * sign >= depth ? [0, x, z] : null;
  if (state.engine.puck_owner < 0 && vy * sign > 0.05 && y * sign < 264) {
    for (let frame = 1; frame <= 96; frame++) {
      if (z <= 0) { vx *= puck.friction; vy *= puck.friction; }
      const previous = [x, y, z] as const;
      x += vx; y += vy;
      if (z > 0 || vz) { vz -= 102 * VELOCITY_SCALE; z += vz; if (z < 0) { z = 0; vz = -vz / 2; } }
      if (Math.abs(x) >= 128) break;
      if (crossing === null && y * sign >= depth) {
        const fraction = (sign * depth - previous[1]) / (y - previous[1]);
        crossing = [frame - 1 + fraction, previous[0] + (x - previous[0]) * fraction, Math.max(0, previous[2] + (z - previous[2]) * fraction)];
      }
      if (y * sign >= 264) {
        const fraction = (sign * 264 - previous[1]) / (y - previous[1]);
        const goal_x = previous[0] + (x - previous[0]) * fraction, goal_z = Math.max(0, previous[2] + (z - previous[2]) * fraction);
        if (Math.abs(goal_x) <= 25 && goal_z <= 24) {
          const [time, cross_x, height] = required(crossing, "crease crossing");
          if (incoming !== null && incoming[0] < time) break;
          return new GoaliePlan([Math.max(-32, Math.min(32, cross_x)), sign * depth], "released puck threatens goal", {
            crossing_frames: time, crossing_x: cross_x, height, deadline: frame - 1 + fraction,
          });
        }
        break;
      }
    }
  }
  [vx, vy] = velocity(puck);
  let reference: Vec2 = [puck.x + vx * 8, Math.max(-259, Math.min(259, puck.y + vy * 8))], reception: number | null = null;
  if (incoming !== null) { reception = incoming[0]; reference = [incoming[1][0], Math.max(-259, Math.min(259, incoming[1][1]))]; }
  const dx = reference[0], dy = reference[1] - sign * 260, length = Math.max(1, Math.hypot(dx, dy));
  const target: Vec2 = [Math.max(-28, Math.min(28, 26 * dx / length)), sign * Math.max(236, Math.min(254, 260 + sign * 18 * dy / length))];
  depth = puck.y * sign;
  let deadline: number | null = null;
  if (state.team2.owns_scnum(state.engine.puck_owner) && 88 < depth && depth < 264 && Math.abs(puck.x) < 100) {
    deadline = Math.max(4, (185 - depth) / Math.max(0.7, vy * sign)) + distance([puck.x, puck.y], [0, sign * 264]) / 6;
  }
  if (reception !== null && reference[1] * sign > 88) deadline = reception + distance(reference, [0, sign * 264]) / 6;
  return new GoaliePlan(target, reception !== null ? "anticipate incoming receiver shooting angle" : "protect predicted shooting angle", {
    deadline, receiver: reception !== null ? reference : null, reception_frames: reception,
  });
}

export function goalie_steer(player: Player, target: Vec2): Action {
  const action = neutralAction(), [vx, vy] = velocity(player);
  for (const [position, speed, destination, negative, positive] of [
    [player.x, vx, target[0], Buttons.INPUT_LEFT, Buttons.INPUT_RIGHT],
    [player.y, vy, target[1], Buttons.INPUT_DOWN, Buttons.INPUT_UP],
  ] as const) {
    const braking = speed * Math.abs(speed) / (2 * 2000 * VELOCITY_SCALE), error = destination - position - braking;
    if (Math.abs(error) > 2) action[error > 0 ? positive : negative] = 1;
  }
  return action;
}

export function moving_body_contact(state: GameState, action: Action, horizon: number, { save = false } = {}): number | null {
  let goalie = state.team1.goalie;
  const puck = state.puck;
  if (!(0 < horizon && horizon <= 12) || (["speed", "agility", "weight", "energy"] as const).some((name) => goalie[name] === null)) return null;
  const pad: Vec2 = [action[Buttons.INPUT_RIGHT] - action[Buttons.INPUT_LEFT], action[Buttons.INPUT_UP] - action[Buttons.INPUT_DOWN]];
  if (!pad.some(Boolean)) return null;
  let x = puck.x, y = puck.y, z = Math.max(0, required(puck.height, "puck.height")), [vx, vy] = velocity(puck), vz = puck.motion_z || 0;
  let previous: Vec2 = [x - goalie.x, y - goalie.y];
  for (let frame = 1; frame <= Math.ceil(horizon); frame++) {
    const previous_height = z;
    goalie = grounded_step(goalie, save ? [0, 0] : pad, { goalie: !save });
    if (save && frame === 1) { goalie.motion_x = required(goalie.motion_x, "motion_x") / 4; goalie.motion_y = required(goalie.motion_y, "motion_y") / 4; }
    if (z <= 0) { vx *= puck.friction; vy *= puck.friction; }
    x += vx; y += vy;
    if (z > 0 || vz) { vz -= 102 * VELOCITY_SCALE; z += vz; if (z < 0) { z = 0; vz = -vz / 2; } }
    const relative: Vec2 = [x - goalie.x, y - goalie.y], dx = relative[0] - previous[0], dy = relative[1] - previous[1];
    const fraction = Math.max(0, Math.min(1, -(previous[0] * dx + previous[1] * dy) / Math.max(1e-9, dx * dx + dy * dy)));
    const dist = Math.hypot(previous[0] + fraction * dx, previous[1] + fraction * dy), contact_height = previous_height + fraction * (z - previous_height);
    const radius = save ? (state.engine.native_one_timer_shot ? 12 : 15) : 11;
    if (contact_height <= 15 && dist <= radius && frame - 1 + fraction <= horizon) return frame - 1 + fraction;
    previous = relative;
  }
  return null;
}

interface OutletRequest {
  receiver: number | null; started: number; passes: number | null; launched: boolean; deadline: number; buttons: Action;
}
interface IncomingPass {
  passer: number; receiver: number; passes: number; expires: number; period: number;
}
export interface GoalieOptions {
  policy?: "selective" | "always";
  pass_timing?: boolean;
  save_timing?: "classic" | "contact";
}

export interface GoalieDiagnostics extends ControllerDiagnostics {
  outlet_candidates?: PassDiagnostics[];
  positioning_target?: Vec2;
}

export class GoalieController {
  readonly policy: "selective" | "always";
  readonly save_timing: "classic" | "contact";
  readonly pass_timing: boolean;
  frames = 0;
  phase = "skater";
  started = 0;
  release_until = 0;
  retry_at = 0;
  save_at = 0;
  possession_since: number | null = null;
  outlet: OutletRequest | null = null;
  pending_save: [string, number] | null = null;
  last_anim: number | null = null;
  last_owner: number | null = null;
  last_effective = false;
  last_passes: number | null = null;
  pass_windup: PassWindup | null = null;
  incoming_pass: IncomingPass | null = null;
  hold_sample: number | null = null;
  hold_tick: number | null = null;
  hold_stride = 1;
  metrics: Record<string, number> = {};
  last_event: Record<string, unknown> | null = null;
  diagnostics: GoalieDiagnostics = {};

  constructor({ policy = "selective", pass_timing = false, save_timing = "classic" }: GoalieOptions = {}) {
    if (policy !== "selective" && policy !== "always") throw new Error("Enabled goalie policy must be selective or always");
    if (save_timing !== "classic" && save_timing !== "contact") throw new Error("Goalie save timing must be classic or contact");
    this.policy = policy; this.pass_timing = pass_timing; this.save_timing = save_timing;
  }

  static validate(state: GameState): number {
    const engine = state.engine, team = state.team1;
    if (team.players.length !== 5) throw new Error("Manual goalie AI supports full-team NHL94 only");
    if ((["goalie_modes", "goalie_hold_counts", "controller_teams", "input_block", "clock_stopped", "camera"] as const).some((name) => engine[name] === null)) {
      throw new Error("Manual goalie AI requires live goalie/control telemetry");
    }
    const controllers = required(engine.controller_teams, "controller_teams").flatMap((side, index) => side === team.controller ? [index] : []);
    if (controllers.length !== 1) throw new Error("Manual goalie AI requires exactly one joystick on the controlled team");
    const controller = controllers[0]!, modes = required(engine.goalie_modes, "goalie_modes");
    if (modes[controller % 2] !== 0 || modes[team.controller - 1] !== 0) {
      throw new Error("Manual goalie AI requires Manual goalie settings for joystick and team; use SabresVsMightyDucks.ManualGoalie.Start");
    }
    const fields = ["role", "selection_flags", "live_state_flags", "live_anim", "speed", "agility", "weight", "energy",
      "motion_x", "motion_y", "passing", "stick"] as const;
    if (fields.some((name) => team.goalie[name] === null) || (["height", "motion_x", "motion_y"] as const).some((name) => state.puck[name] === null)) {
      throw new Error("Missing live goalie motion, ratings or assignment feedback");
    }
    return controller;
  }
  validate(state: GameState): number { return GoalieController.validate(state); }

  _event(name: string, details: Record<string, unknown> = {}): void {
    this.metrics[name] = (this.metrics[name] ?? 0) + 1;
    this.last_event = { event: name, frame: this.frames, ...details };
  }

  _timeout(request: string): void {
    this._event("handoff_timeouts", { request });
    console.warn(`Manual goalie ${request} not confirmed within ${HANDOFF_TIMEOUT} frames; releasing input and backing off.`);
    this.phase = "backoff"; this.retry_at = this.frames + 180; this.release_until = this.frames + 2;
  }

  _takeover(state: GameState, plan: GoaliePlan, blocked: boolean, takeover_frames = TAKEOVER_FRAMES): [boolean, string] {
    if (blocked || this.frames < this.retry_at) return [false, "preserve pending skater action"];
    if (owns_puck(state.team1, state.engine.puck_owner)) return [false, "retain attacking skater"];
    const goalie = state.team1.goalie;
    if (required(goalie.selection_flags, "selection_flags") & 0x20 || goalie.unavailable & 2) return [false, "retain CPU during goalie animation recovery"];
    const camera = required(state.engine.camera, "camera");
    if (Math.abs(goalie.x - camera[0]) >= 108 || Math.abs(goalie.y - camera[1]) >= 92) return [false, "goalie outside effective manual viewport"];
    const sign = state.team1.net.y > 0 ? 1 : -1, depth = state.puck.y * sign;
    if (!(88 < depth && depth < 264)) return [false, "retain skater outside defensive zone"];
    if (this.policy === "always") return [true, "always-manual defensive ablation"];
    if (plan.deadline === null) return [false, "no projected goal threat"];
    const reach = goalie_arrival(goalie, plan.target), deadline = plan.crossing_frames ?? plan.deadline;
    if (deadline < takeover_frames + reach + 6) return [false, "too late for goalie takeover plus reach"];
    const player = state.team1.get_player_by_scnum(controlled_slot(state.team1));
    if (player !== null && eligible(player)) {
      if (plan.receiver === null && state.engine.puck_owner >= 0 && arrival_time(player, [state.puck.x, state.puck.y], { boost: true }) <= TAKEOVER_FRAMES / 2) {
        return [false, "skater can contest before takeover"];
      }
      if (plan.receiver !== null && state.engine.puck_owner < 0) {
        const race = new DefenseController().choose_target(state);
        if (race.mode === "intercept-pass" && arrival_time(player, race.target) + 5 <= required(race.puck_arrival, "puck_arrival")) return [false, "skater has a reachable pass interception"];
      }
      const origin = plan.receiver ?? [state.puck.x, state.puck.y], delay = plan.reception_frames || 0;
      if ([-12, 0, 12].every((x) => blocks_shot_lane(player, origin, [x, state.team1.net.y], delay))) return [false, "retain useful controlled shooting-lane block"];
    }
    return [true, "reachable goal threat without useful controlled skater cover"];
  }

  _remaining_hold(count: number): number {
    if (this.hold_sample !== null && count < this.hold_sample) {
      if (this.hold_tick !== null) this.hold_stride = Math.max(this.hold_stride, (this.frames - this.hold_tick) / (this.hold_sample - count));
      this.hold_tick = this.frames;
    }
    this.hold_sample = count;
    return Math.max(0, count) * this.hold_stride + 2;
  }

  _pass_feedback(state: GameState): void {
    const passes = state.team2.pass_attempts;
    if (passes !== null && this.last_passes !== null && passes > this.last_passes && state.team2.owns_scnum(state.engine.puck_owner) && state.engine.pass_target !== null) {
      this.pass_windup = [state.engine.puck_owner, state.engine.pass_target, this.frames];
    }
    this.last_passes = passes;
    if (this.pass_windup !== null && (state.engine.puck_owner !== this.pass_windup[0] || state.engine.pass_target !== this.pass_windup[1]
      || this.frames - this.pass_windup[2] > 6)) this.pass_windup = null;
  }

  faceoff(): void {
    this.phase = "stoppage"; this.outlet = this.pending_save = null;
    this.possession_since = null; this.last_effective = false;
    this.pass_windup = null; this.incoming_pass = null;
  }

  _retained_receiver(state: GameState): number | null {
    const engine = state.engine, opponent = state.team2;
    if (engine.clock_stopped || engine.puck_owner >= 0 || engine.last_puck_player === null ||
        !opponent.owns_scnum(engine.last_puck_player) || opponent.pass_attempts === null) {
      this.incoming_pass = null; return null;
    }
    const pending = this.incoming_pass;
    if (pending !== null && (engine.last_puck_player !== pending.passer || opponent.pass_attempts !== pending.passes ||
        state.period !== pending.period || this.frames > pending.expires)) this.incoming_pass = null;
    const target = engine.pass_target;
    if (target !== null && target >= 0) {
      const contact = _receiver_contact(state, null);
      this.incoming_pass = contact === null ? null : { passer: engine.last_puck_player, receiver: target,
        passes: opponent.pass_attempts, expires: this.frames + Math.min(40, Math.ceil(contact[0]) + 4), period: state.period };
      return null;
    }
    if (this.incoming_pass !== null) {
      if (_receiver_contact(state, null, this.incoming_pass.receiver) !== null) return this.incoming_pass.receiver;
      this.incoming_pass = null;
    }
    return null;
  }

  step(state: GameState, { blocked = false } = {}): Action | null {
    const controller = this.validate(state);
    this.frames++;
    const team = state.team1, goalie = team.goalie, slot = team.defense_goalie, actual = controlled_slot(team);
    const selected = actual === slot && !!(required(goalie.selection_flags, "selection_flags") & 8);
    const effective = selected && !(required(goalie.live_state_flags, "live_state_flags") & 4);
    if (state.engine.puck_owner === slot && this.last_owner !== slot && effective && this.last_effective) this._event("controlled_catches");
    this._pass_feedback(state);
    const receiver_slot = this._retained_receiver(state);
    const plan = goalie_target(state, this.pass_windup, receiver_slot);
    this.diagnostics = { phase: "goalie", mode: this.phase, decision: this.phase, target: null,
      destination: plan.target, waypoint: plan.target, actual_slot: actual, desired_slot: slot, reason: plan.reason, policy: this.policy,
      crossing_frames: plan.crossing_frames, crossing_x: plan.crossing_x, deadline: plan.deadline, reach_frames: goalie_arrival(goalie, plan.target),
      receiver: plan.receiver, reception_frames: plan.reception_frames, retained_receiver_slot: receiver_slot,
      hold_count: state.engine.goalie_hold_counts?.[controller] ?? null,
      animation: goalie.live_anim, assignment: goalie.assignment, cpu_fallback: !!(required(goalie.live_state_flags, "live_state_flags") & 4),
      metrics: { ...this.metrics }, last_event: this.last_event };
    const action = this._step(state, plan, slot, selected, blocked);
    Object.assign(this.diagnostics, { mode: this.phase, decision: this.phase, metrics: { ...this.metrics }, last_event: this.last_event });
    if (action !== null) {
      this.diagnostics.target = plan.target; this.diagnostics.buttons = [...action];
      if (!action.slice(4, 8).some(Boolean)) {
        const idle = this.diagnostics.idle_reason ?? (action[Buttons.INPUT_C] ? "save-commitment" : action[Buttons.INPUT_A] ? "dive-commitment"
          : action[Buttons.INPUT_B] ? "waiting-for-control" : this.phase);
        this.diagnostics.idle_reason = idle;
        const key = `idle:${String(idle)}`; this.metrics[key] = (this.metrics[key] ?? 0) + 1;
      }
    }
    this.last_anim = goalie.live_anim; this.last_owner = state.engine.puck_owner; this.last_effective = effective;
    return action;
  }

  _step(state: GameState, plan: GoaliePlan, slot: number | null, selected: boolean, blocked: boolean): Action | null {
    const goalie = state.team1.goalie, owner = state.engine.puck_owner, neutral = neutralAction();
    const flags = required(goalie.selection_flags, "selection_flags"), live_flags = required(goalie.live_state_flags, "live_state_flags");
    if (this.pending_save !== null) {
      const [request, started] = this.pending_save;
      const expected = request === "save" ? SAVE_ANIMATIONS : new Set([DIVE_ANIMATION]);
      if (selected && !(live_flags & 4) && expected.has(required(goalie.live_anim, "live_anim")) && goalie.live_anim !== this.last_anim) {
        this._event(`${request}_animations`); this.pending_save = null;
      } else if (!selected || owns_puck(state.team1, owner) || live_flags & 4 || flags & 0x20 || goalie.unavailable & 6 || goalie.role !== 0) {
        this._event(`${request}_cancelled`); this.pending_save = null; this.save_at = this.frames + 2;
      } else if (this.frames - started > 6) { this._event(`${request}_unconfirmed`); this.pending_save = null; this.save_at = this.frames + 2; }
    }
    if (state.engine.clock_stopped || state.engine.input_block) {
      if (!["skater", "stoppage"].includes(this.phase)) this.release_until = this.frames + 2;
      this.phase = "stoppage"; this.outlet = null; this.pending_save = null; this.possession_since = null;
      return selected || this.frames < this.release_until ? neutral : null;
    }
    if (this.phase === "stoppage") { this.phase = selected ? "positioning" : "skater"; this.retry_at = this.frames + 4; }
    if (this.phase === "backoff") { if (this.frames < this.release_until) return neutral; this.phase = selected ? "positioning" : "skater"; }
    if (["prepare-goalie", "request-goalie"].includes(this.phase) && !selected) {
      const remaining = this.phase === "request-goalie" && state.engine.goalie_hold_counts !== null
        ? this._remaining_hold(required(state.engine.goalie_hold_counts[this.validate(state)], "goalie hold count")) : TAKEOVER_FRAMES;
      this.diagnostics.remaining_hold_frames = remaining;
      const [take, reason] = this._takeover(state, plan, blocked, remaining);
      if (!take) {
        this._event("takeover_cancelled", { reason }); this.diagnostics.reason = reason;
        this.phase = "neutral-handoff"; this.release_until = this.frames + 2; this.retry_at = this.frames + 36; return neutral;
      }
    }
    if (this.phase === "prepare-goalie") {
      if (this.frames < this.release_until) return neutral;
      this.phase = "request-goalie"; this.started = this.frames; this.hold_sample = this.hold_tick = null; this.hold_stride = 1;
    }
    if (this.phase === "request-goalie") {
      if (selected) { this._event("takeovers", { elapsed_frames: this.frames - this.started }); this.phase = "release-goalie"; this.release_until = this.frames + 2; return neutral; }
      if (this.frames - this.started >= HANDOFF_TIMEOUT) { this._timeout("takeover"); return neutral; }
      neutral[Buttons.INPUT_B] = 1; return neutral;
    }
    if (this.phase === "release-goalie" && this.frames < this.release_until) return neutral;
    if (this.phase === "request-skater") {
      const actual = controlled_slot(state.team1), base = state.team1.skater_scnum_base();
      if (base <= actual && actual < base + state.team1.players.length) {
        this._event("returns", { actual_slot: actual }); this.phase = "neutral-handoff"; this.release_until = this.frames + 2;
        this.retry_at = this.frames + 36; return neutral;
      }
      if (this.frames - this.started >= HANDOFF_TIMEOUT) this._timeout("return");
      else if (this.frames - this.started < 2) neutral[Buttons.INPUT_B] = 1;
      return neutral;
    }
    if (this.phase === "neutral-handoff") { if (this.frames < this.release_until) return neutral; this.phase = "skater"; }
    if (this.outlet !== null) return this._outlet_feedback(state);
    if (this.pending_save !== null && this.frames - this.pending_save[1] < 2) {
      if (this.pending_save[0] === "save") neutral[Buttons.INPUT_C] = 1;
      else { neutral[Buttons.INPUT_A] = 1; neutral[plan.target[0] > goalie.x ? Buttons.INPUT_RIGHT : Buttons.INPUT_LEFT] = 1; }
      return neutral;
    }
    if (goalie.role !== 0 || goalie.unavailable & 4) {
      this.diagnostics.reason = "goalie pulled or unavailable";
      if (selected) return this._return();
      this.phase = "skater"; return null;
    }
    if (!selected) {
      this.possession_since = null;
      if (!["skater", "release-goalie"].includes(this.phase)) {
        this._event("automatic_handoffs", { actual_slot: controlled_slot(state.team1) }); this.phase = "neutral-handoff"; this.release_until = this.frames + 2; return neutral;
      }
      this.phase = "skater";
      const [take, reason] = this._takeover(state, plan, blocked); this.diagnostics.reason = reason;
      if (take) {
        this._event("takeover_requests"); this.phase = "prepare-goalie"; this.started = this.frames; this.release_until = this.frames + 2; return neutral;
      }
      return null;
    }
    const fallback = !!(live_flags & 4);
    if (fallback) this.metrics.cpu_fallback_frames = (this.metrics.cpu_fallback_frames ?? 0) + 1;
    if (flags & 0x20 || goalie.unavailable & 2) { this.phase = fallback ? "cpu-fallback" : "save-recovery"; return neutral; }
    const sign = state.team1.net.y > 0 ? 1 : -1;
    if (owner !== slot && (owns_puck(state.team1, owner) || state.puck.y * sign < 70 && plan.deadline === null)) return this._return();
    if (fallback) { this.phase = "cpu-fallback"; return neutral; }
    if (owner === slot) return this._possess(state);
    this.possession_since = null; this.phase = "positioning";
    return this._position(state, plan);
  }

  _position(state: GameState, plan: GoaliePlan): Action {
    const goalie = state.team1.goalie, action = goalie_steer(goalie, plan.target);
    if (plan.crossing_frames !== null && this.frames >= this.save_at) {
      const gap = Math.abs(required(plan.crossing_x, "crossing_x") - goalie.x), reach = goalie_arrival(goalie, plan.target);
      const height = required(plan.height, "crossing height");
      const can_align = gap > 8 && height <= 15 && plan.crossing_frames >= 2 && reach + 1 <= plan.crossing_frames
        && !!(action[Buttons.INPUT_LEFT] || action[Buttons.INPUT_RIGHT]);
      if (can_align) {
        Object.assign(this.diagnostics, { save_decision: "align-before-save", alignment_gap: gap, alignment_frames: reach }); return action;
      }
      if (this.save_timing === "contact" && gap > 8 && height <= 15) {
        const contact = moving_body_contact(state, action, plan.crossing_frames);
        if (contact !== null && moving_body_contact(state, action, plan.crossing_frames, { save: true }) === null) {
          Object.assign(this.diagnostics, { save_decision: "body-contact-before-save", body_contact_frames: contact }); return action;
        }
      }
      if (plan.crossing_frames <= 9 && gap > 18 && height <= 5) {
        action.fill(0); action[Buttons.INPUT_A] = 1; action[required(plan.crossing_x, "crossing_x") > goalie.x ? Buttons.INPUT_RIGHT : Buttons.INPUT_LEFT] = 1;
        this._request_save("dive");
      } else if (plan.crossing_frames <= 12 && gap <= 18) {
        action.fill(0); action[Buttons.INPUT_C] = 1; this._request_save("save");
      }
    }
    if (!action.some(Boolean)) this.diagnostics.idle_reason = "holding-position";
    return action;
  }

  _return(): Action {
    this.phase = "request-skater"; this.started = this.frames; this._event("return_requests");
    const action = neutralAction(); action[Buttons.INPUT_B] = 1; return action;
  }

  _request_save(request: string): void {
    this.phase = `${request}-request`; this.pending_save = [request, this.frames]; this.save_at = this.frames + 40;
    this._event(`${request}_requests`);
  }

  _possess(state: GameState): Action {
    this.phase = "possession-outlet";
    this.possession_since ??= this.frames;
    const action = neutralAction();
    if (this.frames - this.possession_since < 8) return action;
    const [options, candidates] = outlet_options(state, { decision_interval: 1, release_prediction: this.pass_timing });
    this.diagnostics.outlet_candidates = candidates;
    if (options.length) {
      const option = options[0]!;
      aim_pass(action, state.team1.goalie, required(state.team1.get_player_by_scnum(option.slot), "outlet receiver"));
      action[Buttons.INPUT_B] = 1;
      this.outlet = { receiver: option.slot, started: this.frames, passes: state.team1.pass_attempts, launched: false,
        deadline: this.frames + Math.ceil(option.flight_frames) + 20, buttons: [...action] };
      this._event("outlet_requests", { receiver: option.slot });
    } else if (this.frames - this.possession_since >= 90) {
      action[Buttons.INPUT_A] = 1;
      const sign = state.team2.net.y > state.team1.net.y ? 1 : -1;
      action[sign > 0 ? Buttons.INPUT_UP : Buttons.INPUT_DOWN] = 1;
      this.outlet = { receiver: null, started: this.frames, passes: state.team1.pass_attempts, launched: false,
        deadline: this.frames + 60, buttons: [...action] };
      this._event("clear_requests");
    }
    return action;
  }

  _outlet_feedback(state: GameState): Action {
    const request = required(this.outlet, "outlet request"), owner = state.engine.puck_owner, neutral = neutralAction();
    this.phase = "outlet-flight";
    const launched = state.team1.pass_attempts !== null && request.passes !== null && state.team1.pass_attempts > request.passes;
    if (!request.launched && launched) {
      request.launched = true; this._event(request.receiver !== null ? "outlets_launched" : "clears_launched");
    } else if (!request.launched && owner !== state.team1.defense_goalie) {
      this._event(request.receiver !== null ? "outlet_interrupted" : "clear_possession_ended");
      this.outlet = null; this.phase = "neutral-handoff"; this.release_until = this.frames + 2; return neutral;
    }
    if (request.launched && owner >= 0 && owner !== state.team1.defense_goalie) {
      this._event(owner === request.receiver ? "outlet_receptions" : "outlet_other_owner", { owner });
      this.outlet = null; this.phase = "neutral-handoff"; this.release_until = this.frames + 2;
    } else if (this.frames >= request.deadline || !request.launched && this.frames - request.started > 8) {
      this._event("outlet_timeouts"); this.outlet = null; this.phase = "positioning"; this.possession_since = this.frames - 90;
    } else if (!request.launched && (request.receiver !== null || this.frames - request.started < 2) &&
        controlled_slot(state.team1) === state.team1.defense_goalie) return [...request.buttons];
    return neutral;
  }
}
