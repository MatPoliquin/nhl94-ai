import { Buttons, neutralAction, type Action, type GameState, type Player, type Team, type Vec2 } from "./state";
import { distance, lexCompare, pointEqual, required } from "./numeric";
import { project_target, route_waypoint } from "./target";
import {
  LANE_HORIZON, SHOT_SPEED_RANGE, arrival_time, blocks_shot_lane,
  boost_safe, burst_velocity, check_approach, facing, puck_path, skating, velocity,
} from "./motion";

export const SWITCH_DELAY = 6;
export const SWITCH_INTERVAL = 12;
export const SWITCH_TIMEOUT = 8;
export const SWITCH_MARGIN = 4;

export interface ControllerDiagnostics {
  [key: string]: unknown;
  target?: Vec2 | null;
  destination?: Vec2;
  waypoint?: Vec2;
  decision?: string;
  mode?: string;
  phase?: string;
  reason?: string;
  buttons?: Action;
}

export function defensive_steering(player: Player, target: Vec2): [Vec2, Vec2] {
  const waypoint = route_waypoint([player.x, player.y], target);
  const [vx, vy] = velocity(player), [acceleration] = skating(player);
  const brake_x = vx * Math.abs(vx) / (2 * acceleration + Math.abs(vx) / 64);
  const brake_y = vy * Math.abs(vy) / (2 * acceleration + Math.abs(vy) / 64);
  const corrections: Vec2 = [waypoint[0] - player.x - brake_x, waypoint[1] - player.y - brake_y];
  return [[corrections[0] > 3 ? 1 : corrections[0] < -3 ? -1 : 0,
    corrections[1] > 3 ? 1 : corrections[1] < -3 ? -1 : 0], waypoint];
}

export function controlled_slot(team: Team): number {
  return team.defense_control ?? team.controlled_scnum();
}

export function owns_puck(team: Team, owner: number): boolean {
  const goalie = team.defense_goalie ?? team.goalie_scnum(), base = team.skater_scnum_base();
  return base <= owner && owner < base + team.players.length || owner === goalie;
}

export function eligible(player: Player): boolean {
  const locked = player.selection_flags === null ? player.is_falling : player.selection_flags & 0x20;
  return !locked && (player.role === null || player.role > 0) && !(player.unavailable & 4);
}

export function on_ice(player: Player): boolean { return player.role === null || player.role >= 0; }
export function skaters(team: Team): [number, Player][] {
  return team.players.flatMap((player, index) => eligible(player) ? [[team.skater_scnum_base() + index, player] as [number, Player]] : []);
}

export class ShotLane {
  constructor(public readonly origin: Vec2, public readonly goal: Vec2, public readonly threat_slot: number | null,
    public readonly blockers: readonly number[], public readonly release_delay: number) {}
}

export class DefensePlan {
  threat_slot: number | null = null;
  puck_arrival: number | null = null;
  opponent_arrival: number | null = null;
  receiver: Vec2 | null = null;
  shot_goal: Vec2 | null = null;
  lanes: readonly ShotLane[] = [];
  reserved_slots: readonly number[] = [];
  recovery_slot: number | null = null;
  skating_arrival: number | null = null;
  constructor(public target: Vec2, public mode: string, public reason: string,
    options: Partial<Omit<DefensePlan, "target" | "mode" | "reason">> = {}) { Object.assign(this, options); }
}

interface SwitchRequest { from_slot: number; to_slot: number; frame: number }
interface SwitchResult extends SwitchRequest { actual_slot: number; elapsed_frames: number; outcome: string }
interface CheckRequest { from_slot: number; carrier_slot: number; until: number }
interface CheckDetails {
  status: string; carrier_slot: number; contact_frames?: number; contact_point?: Vec2;
  impact_estimate?: number; weight_threshold?: number;
}
type Reception = [number, Vec2, number, Player];

export class DefenseController {
  readonly verify_interceptions = false;
  frames = 0;
  plan: DefensePlan | null = null;
  desired_slot: number | null = null;
  switch_at = 0;
  poke_at = 0;
  boost_at = 0;
  switch_attempts = 0;
  pending_switch: SwitchRequest | null = null;
  pending_check: CheckRequest | null = null;
  last_switch_result: SwitchResult | null = null;
  last_selection: number | null = null;
  b_down = false;
  c_down = false;
  last_owner = -1;
  diagnostics: ControllerDiagnostics = {};

  constructor({ verify_interceptions = false } = {}) {
    if (verify_interceptions) throw new Error("Experimental interception certification is not included in the Genesis baseline port.");
  }

  idle(elapsed = 0, { frame = null }: { frame?: number | null } = {}): void {
    this.frames = frame ?? this.frames + elapsed;
    this.plan = null; this.desired_slot = null; this.switch_attempts = 0;
    this.pending_switch = null; this.last_switch_result = null; this.last_selection = null;
    this.pending_check = null; this.b_down = this.c_down = false; this.diagnostics = {};
  }

  static _goal_side(team: Team, point: Vec2, threat: Vec2, slack = 0): boolean {
    const sign = team.net.y > 0 ? 1 : -1;
    return point[1] * sign >= threat[1] * sign - slack;
  }

  _shot_lanes(state: GameState, threat: Vec2, threat_slot: number | null = null,
    excluding: number | null = null, delay = 0): ShotLane[] {
    const sign = state.team1.net.y > 0 ? 1 : -1, in_front = threat[1] * sign < Math.abs(state.team1.net.y);
    const friends = skaters(state.team1);
    return state._shot_target_points_for_net(state.team1.net).map((goal) => new ShotLane(threat, goal, threat_slot,
      friends.filter(([slot, player]) => slot !== excluding && in_front
        && (this.pending_switch === null || slot !== this.pending_switch.to_slot)
        && blocks_shot_lane(player, threat, goal, delay)).map(([slot]) => slot), delay));
  }

  _covered(state: GameState, target: Vec2, excluding: number | null = null, delay = 0): boolean {
    return this._shot_lanes(state, target, null, excluding, delay).every((lane) => lane.blockers.length > 0);
  }

  _open_receiver(state: GameState, threat: Vec2, carrier_slot: number, primary: readonly ShotLane[]):
    [number, number, Vec2, ShotLane[], ShotLane[]] | null {
    const actual = controlled_slot(state.team1), sign = state.team1.net.y > 0 ? 1 : -1;
    const carrier = required(state.team2.get_player_by_scnum(carrier_slot), "carrier");
    const [carrier_vx, carrier_vy] = velocity(carrier);
    const immediate = this._shot_lanes(state, [carrier.x, carrier.y], carrier_slot, actual);
    const choices: [number, number, Vec2, ShotLane[], ShotLane[]][] = [];
    for (const [slot, player] of skaters(state.team2)) {
      if (slot === carrier_slot) continue;
      const [vx, vy] = velocity(player);
      let delay = distance(threat, [player.x, player.y]) / SHOT_SPEED_RANGE[0];
      const point: Vec2 = [player.x + vx * delay, player.y + vy * delay];
      delay = distance(threat, point) / SHOT_SPEED_RANGE[0];
      if (delay > LANE_HORIZON || Math.abs(point[0]) > Buttons.SLOT_BAND_MAX_X
        || !(Buttons.SLOT_BAND_MIN_Y < point[1] * sign && point[1] * sign < 244)) continue;
      const lanes = this._shot_lanes(state, point, slot, actual, delay), open = lanes.filter((lane) => !lane.blockers.length);
      if (!open.length) continue;
      const future_origin: Vec2 = [carrier.x + carrier_vx * delay, carrier.y + carrier_vy * delay];
      if (distance(future_origin, project_target(future_origin)) > 1e-6) continue;
      const future = this._shot_lanes(state, future_origin, carrier_slot, actual, delay);
      const retained = primary.map((lane, index) => new ShotLane(lane.origin, lane.goal, lane.threat_slot,
        lane.blockers.filter((blocker) => immediate[index]!.blockers.includes(blocker) && future[index]!.blockers.includes(blocker)),
        lane.release_delay));
      if (!retained.every((lane) => lane.blockers.length > 0)) continue;
      const danger_time = delay + Math.min(...open.map((lane) => distance(point, lane.goal))) / SHOT_SPEED_RANGE[1];
      choices.push([danger_time, slot, point, lanes, retained]);
    }
    choices.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    return choices[0] ?? null;
  }

  _lane(state: GameState, threat: Vec2, slot: number | null = null, reason = "protect shooting lane",
    receiver: Vec2 | null = null, delay = 0, check_receivers = false): DefensePlan {
    let lanes = this._shot_lanes(state, threat, slot, controlled_slot(state.team1), delay);
    let open = lanes.filter((lane) => !lane.blockers.length), primary: ShotLane[] = [];
    if (!open.length && check_receivers) {
      const alternative = this._open_receiver(state, threat, required(slot, "carrier slot"), lanes);
      if (alternative) {
        [, slot, threat, lanes, primary] = alternative;
        open = lanes.filter((lane) => !lane.blockers.length);
        receiver = threat; reason = "teammate covers carrier; deny uncovered receiver lane";
      } else reason = "teammate covers shot lanes; retain goal-side support";
    }
    const sign = state.team1.net.y > 0 ? 1 : -1, depth = threat[1] * sign;
    let goal_x = Math.max(-12, Math.min(12, -state.team1.goalie.x * 0.5)), reserved: number[] = [];
    if (open.length && (primary.length || open.length < lanes.length)) {
      goal_x = [...open].sort((a, b) => Math.abs(b.goal[0] - state.team1.goalie.x) - Math.abs(a.goal[0] - state.team1.goalie.x))[0]!.goal[0];
      reserved = [...new Set([...primary, ...lanes].filter((lane) => lane.blockers.length).map((lane) => lane.blockers[0]!))].sort((a, b) => a - b);
      if (!primary.length) reason = "cover goal-mouth gap beside teammate";
    }
    const delta: Vec2 = [goal_x - threat[0], state.team1.net.y - threat[1]], length = Math.max(1, Math.hypot(...delta));
    const gap = Math.min(34, Math.max(14, (264 - depth) * 0.18));
    let target: Vec2 = [threat[0] + delta[0] * gap / length, sign * Math.min(244, depth + Math.abs(delta[1]) * gap / length)];
    let mode = receiver === null ? "protect-lane" : "deny-reception";
    if (receiver === null && !reserved.length && Math.abs(threat[0]) > 82 && -50 < depth && depth < 232) {
      const inside = skaters(state.team1).filter(([, player]) => Math.abs(player.x) < Math.abs(threat[0]) - 8
        && distance([player.x, player.y], threat) < 75);
      const receiver_lanes = skaters(state.team2).flatMap(([other_slot, player]) =>
        other_slot !== slot && Math.abs(player.x) < 55 && depth + 15 < player.y * sign && player.y * sign < 244
          ? this._shot_lanes(state, [player.x, player.y], other_slot, controlled_slot(state.team1),
            distance(threat, [player.x, player.y]) / SHOT_SPEED_RANGE[0]) : []);
      if (inside.length && receiver_lanes.every((lane) => lane.blockers.length > 0)) {
        target = [threat[0] - (threat[0] < 0 ? -20 : 20), threat[1] + sign * 12];
        mode = "contain-boards"; reason = "close inside escape; retain goal-side position";
        primary = [...primary, ...receiver_lanes];
        reserved = [...new Set(receiver_lanes.map((lane) => required(lane.blockers[0], "receiver blocker")))].sort((a, b) => a - b);
      }
    }
    return new DefensePlan(project_target(target), mode, reason, { threat_slot: slot, receiver,
      shot_goal: [goal_x, state.team1.net.y], lanes: [...primary, ...lanes], reserved_slots: reserved });
  }

  _switch_delay(state: GameState, slot: number): number {
    let delay = 0;
    if (slot !== controlled_slot(state.team1)) {
      delay = SWITCH_DELAY + Math.max(0, this.switch_at - this.frames);
      if (this.pending_switch && slot === this.pending_switch.to_slot) delay = Math.max(0, this.pending_switch.frame + SWITCH_DELAY - this.frames);
    }
    return delay;
  }

  _cost(state: GameState, slot: number, player: Player, target: Vec2): number {
    let cost = arrival_time(player, target, { boost: this.frames >= this.boost_at }) + this._switch_delay(state, slot);
    if (!DefenseController._goal_side(state.team1, target, [state.puck.x, state.puck.y], 12)
      && !this._covered(state, [state.puck.x, state.puck.y], slot)) cost += 24;
    return cost;
  }

  static _likely_switch(state: GameState): number | null {
    const [vx, vy] = velocity(state.puck), point: Vec2 = [state.puck.x + Math.floor(vx * 256 / 17), state.puck.y + Math.floor(vy * 256 / 17)];
    const actual = controlled_slot(state.team1);
    const candidates = skaters(state.team1).filter(([slot, player]) => player.selection_flags === null
      || !(player.selection_flags & 0x20) && (slot === actual || !(player.selection_flags & 8)));
    candidates.sort((a, b) => distance([a[1].x, a[1].y], point) - distance([b[1].x, b[1].y], point) || b[0] - a[0]);
    return candidates[0]?.[0] ?? null;
  }

  _observe_switch(actual: number): void {
    if (actual >= 0 && actual !== this.last_selection) this.switch_attempts = 0;
    this.last_selection = actual;
    const request = this.pending_switch;
    if (request === null) return;
    const elapsed = this.frames - request.frame;
    let outcome: string;
    if (actual >= 0 && actual !== request.from_slot) outcome = actual === request.to_slot ? "confirmed" : "unexpected-selection";
    else if (elapsed >= SWITCH_TIMEOUT) { outcome = "unconfirmed"; this.switch_at = this.frames + Math.min(24, 4 * this.switch_attempts); }
    else return;
    this.last_switch_result = { ...request, actual_slot: actual, elapsed_frames: elapsed, outcome };
    this.pending_switch = null;
  }

  static _race_margin(player: Player): number {
    return player.motion_x === null || player.speed === null || player.energy === null ? 10 : 5;
  }

  choose_target(state: GameState): DefensePlan {
    const owner = state.engine.puck_owner, opponents = skaters(state.team2), goalie = state.team2.goalie;
    const goalie_slot = state.team2.defense_goalie ?? state.team2.goalie_scnum();
    if (owner === goalie_slot) {
      if (opponents.length) {
        const [slot, receiver] = [...opponents].sort((a, b) =>
          distance([a[1].x, a[1].y], [goalie.x, goalie.y]) - distance([b[1].x, b[1].y], [goalie.x, goalie.y]))[0]!;
        return Object.assign(this._lane(state, [receiver.x, receiver.y], slot), { mode: "deny-goalie-outlet",
          reason: "goalie owns puck; cover outlet instead of chasing crease" });
      }
      const sign = state.team1.net.y > state.team2.net.y ? 1 : -1;
      return new DefensePlan(project_target([goalie.x, goalie.y + sign * 48]), "deny-goalie-outlet",
        "goalie owns puck; retreat clear of crease", { threat_slot: goalie_slot });
    }
    const carrier = opponents.find(([slot]) => slot === owner);
    if (carrier) {
      const [slot, player] = carrier, [vx, vy] = velocity(player);
      return this._lane(state, project_target([player.x + vx * 8, player.y + vy * 8]), slot, undefined, null, 0, true);
    }
    const path = puck_path(state.puck), friends = skaters(state.team1);
    const actual = controlled_slot(state.team1), likely = DefenseController._likely_switch(state);
    const reachable = friends.filter(([slot]) => slot === actual || actual >= 0 && slot === likely);
    let reception: Reception | null = null;
    for (const [time, point, height] of path) {
      if (height > 8) continue;
      const arrivals: [number, number, Player][] = opponents.map(([slot, player]) => [arrival_time(player, point, { optimistic: true }), slot, player]);
      arrivals.sort((a, b) => a[0] - b[0]);
      if (arrivals.length && arrivals[0]![0] <= time) {
        const [, slot, player] = arrivals[0]!; reception = [time, point, slot, player]; break;
      }
    }
    const previous = state.engine.last_puck_player ?? this.last_owner;
    const passing = owns_puck(state.team2, previous) && Math.hypot(...velocity(state.puck)) > 0.8;
    if (passing && state.engine.pass_target !== null) {
      const intended = opponents.find(([slot]) => slot === state.engine.pass_target);
      if (intended) {
        const [slot, player] = intended, [vx, vy] = velocity(player);
        for (const [time, point, height] of path) {
          const receiver: Vec2 = [player.x + vx * time, player.y + vy * time];
          if (height <= 8 && distance(point, receiver) <= 20) {
            if (reception === null || time <= reception[0] + 4) reception = [time, point, slot, player];
            break;
          }
        }
      }
    }
    let reason = reception ? "no safe race before opponent reception" : "uncertain or unreachable puck path";
    const puck_known = state.puck.motion_x !== null && state.puck.motion_y !== null && state.puck.height !== null;
    for (const [time, point, height] of path) {
      if (!puck_known || height > 8 || reception && time >= reception[0] - 3) continue;
      const candidates: [number, number, Player][] = reachable.map(([slot, player]) => [this._cost(state, slot, player, point), slot, player]);
      if (!candidates.length) break;
      const enemy = Math.min(...opponents.map(([, rival]) => arrival_time(rival, point, { optimistic: true })));
      candidates.sort((a, b) => a[0] - b[0]);
      const [ours, slot, player] = candidates[0]!, margin = DefenseController._race_margin(player);
      const settle = 4 + Math.max(0, 15 - (player.stick ?? 5)) / 3;
      if (ours + margin > time || enemy <= time + settle + margin) continue;
      if (!DefenseController._goal_side(state.team1, point, [state.puck.x, state.puck.y], 12)
        && !this._covered(state, [state.puck.x, state.puck.y], slot, time)) {
        reason = "recovery would abandon the last goal-side defender"; continue;
      }
      if (distance(velocity(player), velocity(state.puck)) > 3.5 && time < ours + 12) {
        reason = "insufficient time to control the incoming puck"; continue;
      }
      return new DefensePlan(point, passing ? "intercept-pass" : "recover-safe",
        "win the puck race with control and coverage margin", {
          threat_slot: reception?.[2] ?? null, puck_arrival: time, opponent_arrival: Number.isFinite(enemy) ? enemy : null,
        });
    }
    if (reception) {
      const [time, point, slot] = reception;
      return Object.assign(this._lane(state, point, slot, reason, point, time), { puck_arrival: time, opponent_arrival: time });
    }
    if (opponents.length) {
      const [slot, player] = [...opponents].sort((a, b) => arrival_time(a[1], [state.puck.x, state.puck.y], { optimistic: true })
        - arrival_time(b[1], [state.puck.x, state.puck.y], { optimistic: true }))[0]!;
      const [vx, vy] = velocity(player);
      return this._lane(state, project_target([player.x + vx * 8, player.y + vy * 8]), slot, reason);
    }
    return this._lane(state, [state.puck.x, state.puck.y], null, reason);
  }

  select_player(state: GameState, target: Vec2): [number | null, Record<number, number>] {
    const reserved = this.plan?.reserved_slots ?? [];
    const candidates = skaters(state.team1).filter(([slot]) => !reserved.includes(slot));
    const costs: Record<number, number> = {};
    for (const [slot, player] of candidates) costs[slot] = this._cost(state, slot, player, target);
    const actual = controlled_slot(state.team1), available = [actual, DefenseController._likely_switch(state)];
    const reachable = candidates.filter(([slot]) => available.includes(slot)).map(([slot]) => slot);
    if (this.plan?.recovery_slot !== null && this.plan?.recovery_slot !== undefined && reachable.includes(this.plan.recovery_slot)) {
      this.desired_slot = this.plan.recovery_slot; return [this.desired_slot, costs];
    }
    let best = [...reachable].sort((a, b) => costs[a]! - costs[b]!)[0] ?? null, current_can_wait = true;
    if (this.plan && ["recover-safe", "intercept-pass"].includes(this.plan.mode) && actual in costs) {
      const player = required(candidates.find(([slot]) => slot === actual), "controlled defender")[1];
      current_can_wait = costs[actual]! + DefenseController._race_margin(player) <= required(this.plan.puck_arrival, "puck_arrival");
    }
    if (actual in costs && costs[actual]! <= required(costs[required(best, "best defender")], "best defender cost") + 4 && current_can_wait) best = actual;
    this.desired_slot = best;
    return [best, costs];
  }

  _body_check(state: GameState, actual: number, player: Player | null): CheckDetails {
    const owner = state.engine.puck_owner, carrier = skaters(state.team2).find(([slot]) => slot === owner)?.[1];
    let details: CheckDetails = { status: "no-active-carrier", carrier_slot: owner };
    const status = (value: string): CheckDetails => ({ ...details, status: value });
    if (!state.engine.puck_owner_known || !carrier) return details;
    if (this.pending_check) return status("follow-through");
    if (player === null) return status("unavailable-checker");
    const fields = ["motion_x", "motion_y", "weight", "selection_flags", "facing"] as const;
    if ([player, carrier].some((p) => fields.some((name) => p[name] === null)) || player.energy === null) return status("missing-feedback");
    const flags = required(player.selection_flags, "selection_flags"), carrier_flags = required(carrier.selection_flags, "selection_flags");
    if (!(flags & 8) || [player, carrier].some((p) => !!((p === player ? flags : carrier_flags) & 4) || !!(p.unavailable & 0x21))) return status("unavailable-contact");
    if (this.pending_switch || this.frames < this.switch_at) return status("selection-busy");
    if (this.frames < this.boost_at) return status("cooldown");
    if (this.b_down || this.c_down) return status("release-buttons");
    if (player.energy < 1024) return status("low-energy");
    const dx = carrier.x - player.x, dy = carrier.y - player.y, dist = Math.hypot(dx, dy);
    if (dist > 30) return status("outside-range");
    if (!DefenseController._goal_side(state.team1, [player.x, player.y], [carrier.x, carrier.y])) return status("not-goal-side");
    const [fx, fy] = facing(player);
    if (dx * fx + dy * fy < dist * 0.9) return status("not-facing-carrier");
    if (this.plan && this.plan.threat_slot !== null && this.plan.threat_slot !== owner) return status("covering-other-threat");
    const approach = check_approach(player, carrier);
    if (!approach) return status("no-contact-course");
    const [time, point, impact] = approach;
    const threshold = ((240 - required(player.weight, "weight") + required(carrier.weight, "weight")) & 0xff) >> 1;
    details = { ...details, contact_frames: time, contact_point: point, impact_estimate: impact, weight_threshold: threshold };
    if (impact < 20 || impact < threshold + 4) return status("unfavorable-impact");
    const duration = time + 2, [vx, vy] = burst_velocity(player), end: Vec2 = [player.x + vx * duration, player.y + vy * duration];
    const [tx, ty] = velocity(carrier), carrier_end: Vec2 = [carrier.x + tx * duration, carrier.y + ty * duration];
    if (!DefenseController._goal_side(state.team1, end, carrier_end, 4) || distance(end, project_target(end)) > 1e-6
      || !pointEqual(route_waypoint([player.x, player.y], end), end)) return status("unsafe-route");
    for (const team of [state.team1, state.team2]) for (const other of [...team.players, team.goalie]) {
      if (other === player || other === carrier || other.role !== null && other.role < 0) continue;
      const [ox, oy] = velocity(other), rx = vx - ox, ry = vy - oy, dx = player.x - other.x, dy = player.y - other.y;
      const closest = Math.max(0, Math.min(duration, -(dx * rx + dy * ry) / Math.max(0.01, rx * rx + ry * ry)));
      if (Math.hypot(dx + rx * closest, dy + ry * closest) <= 16) return status("obstructed-contact");
    }
    const sign = state.team1.net.y > 0 ? 1 : -1;
    for (const [slot, receiver] of skaters(state.team2)) {
      if (slot === owner || Math.abs(receiver.x) > Buttons.SLOT_BAND_MAX_X || !(Buttons.SLOT_BAND_MIN_Y < receiver.y * sign && receiver.y * sign < 244)) continue;
      if (this._shot_lanes(state, [receiver.x, receiver.y], slot).some((lane) => lane.blockers.length === 1 && lane.blockers[0] === actual)) return status("leaves-receiver-lane");
    }
    return status("ready");
  }

  step(state: GameState, { elapsed = 1, frame = null }: { elapsed?: number; frame?: number | null } = {}): Action {
    this.frames = frame ?? this.frames + elapsed;
    const actual = controlled_slot(state.team1);
    this._observe_switch(actual);
    if (owns_puck(state.team1, state.engine.puck_owner)) { this.idle(0); return neutralAction(); }
    if (this.pending_check && (this.frames >= this.pending_check.until || actual !== this.pending_check.from_slot
      || state.engine.puck_owner !== this.pending_check.carrier_slot)) this.pending_check = null;
    let plan = this.choose_target(state);
    if (this.plan && ["protect-lane", "contain-boards"].includes(plan.mode) && plan.mode === this.plan.mode
      && plan.threat_slot === this.plan.threat_slot && (plan.shot_goal === null ? this.plan.shot_goal === null
        : this.plan.shot_goal !== null && pointEqual(plan.shot_goal, this.plan.shot_goal))
      && lexCompare(plan.reserved_slots, this.plan.reserved_slots) === 0 && distance(plan.target, this.plan.target) < 3) {
      plan = Object.assign(new DefensePlan(plan.target, plan.mode, plan.reason), plan, { target: this.plan.target });
    }
    this.plan = plan;
    const [best, costs] = this.select_player(state, plan.target), action = neutralAction();
    let mode = "waiting-for-selection", waypoint = plan.target;
    const player = skaters(state.team1).find(([slot]) => slot === actual)?.[1] ?? null, check = this._body_check(state, actual, player);
    const [puck_vx, puck_vy] = velocity(state.puck), likely = DefenseController._likely_switch(state);
    const switch_status = actual < 0 ? "no-selection" : this.pending_check ? "check-follow-through"
      : this.pending_switch ? "awaiting-selection" : best === null ? "no-eligible-skater"
      : likely === actual ? "rom-keeps-current" : best === actual ? "no-useful-switch"
      : this.frames < this.switch_at ? "cooldown" : this.b_down ? "release-b" : "request";
    if (switch_status === "request") {
      action[Buttons.INPUT_B] = 1; this.switch_at = this.frames + SWITCH_INTERVAL; this.poke_at = this.frames + 16;
      this.switch_attempts++; this.pending_switch = { from_slot: actual, to_slot: required(best, "best defender"), frame: this.frames };
      mode = "switch-request";
    } else if (this.pending_check) mode = "check-follow-through";
    else if (player) {
      const [pad, next_waypoint] = defensive_steering(player, plan.target); waypoint = next_waypoint;
      const [vx, vy] = velocity(player);
      if (pad[0]) action[pad[0] > 0 ? Buttons.INPUT_RIGHT : Buttons.INPUT_LEFT] = 1;
      if (pad[1]) action[pad[1] > 0 ? Buttons.INPUT_UP : Buttons.INPUT_DOWN] = 1;
      mode = action.some(Boolean) ? "skating" : "holding";
      const direction = facing(player), dx = state.puck.x - player.x, dy = state.puck.y - player.y, dist = Math.hypot(dx, dy);
      const rel_v: Vec2 = [puck_vx - vx, puck_vy - vy];
      const closest_time = Math.max(0, Math.min(4, -(dx * rel_v[0] + dy * rel_v[1]) / Math.max(0.01, rel_v[0] ** 2 + rel_v[1] ** 2)));
      const contact = Math.hypot(dx + rel_v[0] * closest_time, dy + rel_v[1] * closest_time);
      const aligned = dx * direction[0] + dy * direction[1] > dist * 0.65;
      const can_poke = this.frames >= Math.max(this.poke_at, this.switch_at) && !this.b_down && this.pending_switch === null
        && likely === actual && state.engine.puck_owner !== (state.team2.defense_goalie ?? state.team2.goalie_scnum());
      const goal_side = DefenseController._goal_side(state.team1, [player.x, player.y], [state.puck.x, state.puck.y], 5);
      const reachable = dist < 22 && contact < 16 && (state.puck.height || 0) < 8;
      if (check.status === "ready") {
        action.fill(0, 4, 8); action[Buttons.INPUT_C] = 1;
        this.boost_at = this.frames + 36; this.poke_at = this.frames + 16;
        this.pending_check = { from_slot: actual, carrier_slot: state.engine.puck_owner,
          until: this.frames + required(check.contact_frames, "check contact_frames") + 2 };
        mode = "check-request";
      } else if (reachable && aligned && goal_side && can_poke) {
        action[Buttons.INPUT_B] = 1; this.poke_at = this.frames + 16; mode = "poke-request";
      } else if (this.pending_switch === null && this.frames >= this.boost_at && !this.c_down && boost_safe(player, waypoint)) {
        action[Buttons.INPUT_C] = 1; this.boost_at = this.frames + 36; mode = "boost-request";
      }
    }
    this.b_down = !!action[Buttons.INPUT_B]; this.c_down = !!action[Buttons.INPUT_C];
    if (state.engine.puck_owner >= 0) this.last_owner = state.engine.puck_owner;
    const fields = ["motion_x", "motion_y", "speed", "agility", "weight", "energy", "selection_flags"] as const;
    const missing: string[] = [...new Set([...skaters(state.team1), ...skaters(state.team2)].flatMap(([, p]) => fields.filter((name) => p[name] === null)))].sort();
    missing.push(...(["motion_x", "motion_y", "height"] as const).filter((name) => state.puck[name] === null).map((name) => `puck.${name}`));
    const ideal = Object.keys(costs).map(Number).sort((a, b) => costs[a]! - costs[b]!)[0] ?? null;
    this.diagnostics = { ...plan, decision: plan.mode, mode, frame: this.frames, destination: plan.target, waypoint,
      desired_slot: best, actual_slot: actual, acting_slot: actual, switch_attempts: this.switch_attempts, ideal_slot: ideal,
      switch_status, pending_switch: this.pending_switch, check, pending_check: this.pending_check, last_switch_result: this.last_switch_result,
      likely_switch_slot: likely, arrival_frames: costs, missing_feedback: missing, buttons: [...action],
      puck_path: puck_path(state.puck, 32).map(([time, point]) => [time, point]), selection_available: actual >= 0 };
    return action;
  }
}
