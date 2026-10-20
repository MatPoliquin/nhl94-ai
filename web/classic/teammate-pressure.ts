import { DefensePlan, defensive_steering, eligible, skaters } from "./defense";
import { arrival_time, blocks_shot_lane, bounded_projection } from "./motion";
import { distance } from "./numeric";
import { grounded_step } from "./skating";
import { Player, type GameState, type Vec2 } from "./state";
import { project_target } from "./target";

const HORIZON = 24;
export const TRAP_ROLES = ["contain", "close", "cover"];
export interface PressureAssignment { plan: DefensePlan; contact: boolean; covered_closer: boolean }
type Identity = [number, number[], number, number, number | null, number];
const same = (a: Identity, b: Identity | null) => JSON.stringify(a) === JSON.stringify(b);

/** One shared trap; observed coverage must survive both attackers leaving. */
export class SharedPressure {
  private key: Identity | null = null;
  private started = 0;
  private until = 0;
  private retryAt = 0;
  diagnostics: Record<string, unknown> = { phase: "idle", reason: "no-opponent-carrier" };

  reset(reason: string, frame = 0): void {
    if (this.key !== null) this.retryAt = frame + 12;
    this.key = null; this.diagnostics = { phase: "idle", reason };
  }

  private known(player: Player): boolean {
    return eligible(player) && !player.is_falling &&
      (["motion_x", "motion_y", "speed", "agility", "weight", "energy", "facing"] as const)
        .every((field) => player[field] !== null);
  }

  private screen(point: Vec2, threat: Vec2, net: number): boolean {
    const dx = -threat[0], dy = net - threat[1], length = Math.max(1, Math.hypot(dx, dy));
    const px = point[0] - threat[0], py = point[1] - threat[1];
    const along = (px * dx + py * dy) / length, across = Math.abs(px * dy - py * dx) / length;
    return 24 <= along && along <= length - 16 && across <= 26;
  }

  private forecast(player: Player, target?: Vec2): Player[] {
    if (target) {
      const result = [player];
      for (let t = 0; t < HORIZON; t++) {
        const previous = result.at(-1)!;
        result.push(grounded_step(previous, defensive_steering(previous, target)[0]));
      }
      return result;
    }
    return Array.from({ length: HORIZON + 1 }, (_, time) => {
      const [x, y] = bounded_projection(player, time)[0];
      return Object.assign(new Player(), player, { x, y });
    });
  }

  private coverage(state: GameState, attackers: number[], guard: number | null, target: Vec2, carrier: Player): boolean {
    // CPU teammates get only observed-motion credit, never assumed steering.
    const guards = skaters(state.team1).filter(([slot, player]) => !attackers.includes(slot) && this.known(player))
      .map(([slot, player]) => this.forecast(player, slot === guard ? target : undefined));
    const sign = state.team1.net.y > 0 ? 1 : -1;
    const threats = skaters(state.team2).map(([, player]) => player).filter((player) =>
      player === carrier || Math.abs(player.x) < 85 && (player.y - carrier.y) * sign > 15);
    for (const time of [0, 8, 16, 24]) for (const threat of threats) {
      if (!this.known(threat)) return false;
      const origin = bounded_projection(threat, time)[0], available = guards.map((path) => path[time]!);
      if (!available.some((p) => this.screen([p.x, p.y], origin, state.team1.net.y))) return false;
      if (origin[1] * sign > 130 && Math.abs(origin[0]) < 85 &&
          !state._shot_target_points_for_net(state.team1.net).every((goal) =>
            available.some((p) => blocks_shot_lane(p, origin, goal)))) return false;
    }
    return guards.length > 0;
  }

  private clearRoutes(state: GameState, targets: Map<number, Vec2>, carrierSlot: number): boolean {
    const paths: [number, Player[]][] = [
      ...skaters(state.team1).map(([slot, player]): [number, Player[]] => [slot, this.forecast(player, targets.get(slot))]),
      ...skaters(state.team2).filter(([slot]) => slot !== carrierSlot)
        .map(([slot, player]): [number, Player[]] => [slot, this.forecast(player)]),
    ];
    for (let i = 0; i < paths.length; i++) {
      const [slot, path] = paths[i]!;
      for (const [otherSlot, other] of paths.slice(i + 1)) {
        if (!targets.has(slot) && !targets.has(otherSlot)) continue;
        if (path.some((a, time) => distance([a.x, a.y], [other[time]!.x, other[time]!.y]) < 18)) return false;
      }
    }
    return true;
  }

  choose(state: GameState, slots: number[], frame: number,
    { busy = false, pendingContact = null }: { busy?: boolean; pendingContact?: number | null } = {}): PressureAssignment[] | null {
    const owner = state.engine.puck_owner, carrier = skaters(state.team2).find(([slot]) => slot === owner)?.[1];
    const selected = slots.map((slot) => state.team1.get_player_by_scnum(slot));
    if (busy || state.engine.clock_stopped || state.time === 0 || state.engine.sflags & 0x0100 ||
        !state.engine.puck_owner_known || !carrier) {
      this.reset("no-opponent-carrier-or-busy", frame); return null;
    }
    if (new Set(slots).size !== slots.length || selected.some((p) => !p || !this.known(p)) || !this.known(carrier)) {
      this.reset("unavailable-or-missing-feedback", frame); return null;
    }
    const players = selected as Player[];
    if (this.key && (owner !== this.key[0] || JSON.stringify(slots) !== JSON.stringify(this.key[1]))) {
      this.reset("possession-or-selection-changed", frame); return null;
    }
    const sign = state.team1.net.y > 0 ? 1 : -1, point = bounded_projection(carrier, 8)[0];
    const depth = Math.max(carrier.y * sign, bounded_projection(carrier, HORIZON)[0][1] * sign);
    const boards = Math.abs(point[0]) > 82;
    if (depth > (boards ? 150 : 88)) { this.reset("too-close-to-own-goal", frame); return null; }
    if (this.key && frame - this.started >= 120) { this.reset("trap-timeout", frame); this.retryAt = frame + 36; }
    if (frame < this.retryAt) { this.diagnostics = { phase: "idle", reason: "regroup" }; return null; }
    const coverTarget = project_target([point[0] * 0.45, point[1] + sign * 72]);
    let guardIndex = slots.length === 3 ? [0, 1, 2].sort((a, b) =>
      arrival_time(players[a]!, coverTarget) - arrival_time(players[b]!, coverTarget) || a - b)[0]! : null;
    if (this.key && frame < this.until) guardIndex = this.key[4];
    const attackers = slots.map((_, i) => i).filter((i) => i !== guardIndex);
    const guard = guardIndex === null ? null : slots[guardIndex]!;
    if (!this.coverage(state, attackers.map((i) => slots[i]!), guard, coverTarget, carrier)) {
      this.reset("exposed-lane", frame); return null;
    }
    const choices: { score: number; identity: Identity; ready: boolean; targets: Map<number, Vec2> }[] = [];
    for (const pair of [attackers, [...attackers].reverse()]) for (const flank of [-1, 1]) {
      const [contain, close] = pair as [number, number];
      const identity: Identity = [owner, [...slots], contain, close, guardIndex, flank];
      if (this.key && frame < this.until && !same(identity, this.key)) continue;
      const ready = (players[contain]!.y - carrier.y) * sign >= 8 &&
        distance([players[contain]!.x, players[contain]!.y], point) <= 42 &&
        distance([players[close]!.x, players[close]!.y], point) <= 48;
      const gap = ready ? 12 : 22, side = point[0] > 0 ? 1 : -1;
      const first = project_target([point[0] + (boards ? -side : flank) * gap, point[1] + sign * gap]);
      const second = project_target([point[0] - (boards ? side * 6 : flank * gap), point[1] - sign * gap]);
      const times = [arrival_time(players[contain]!, first), arrival_time(players[close]!, second)];
      if (Math.max(...times) > HORIZON || Math.abs(times[0]! - times[1]!) > 14) continue;
      const targets = new Map<number, Vec2>([[slots[contain]!, first], [slots[close]!, second]]);
      if (guard !== null) targets.set(guard, coverTarget);
      if (!this.clearRoutes(state, targets, owner)) continue;
      choices.push({ score: times[0]! + times[1]! - (same(identity, this.key) ? 6 : 0), identity, ready, targets });
    }
    const best = choices.sort((a, b) => a.score - b.score)[0];
    if (!best) { this.reset("approach-unreachable-or-obstructed", frame); return null; }
    const { identity, ready, targets } = best;
    if (!same(identity, this.key)) { if (!this.key) this.started = frame; this.until = frame + 24; }
    this.key = identity;
    const [, , contain, close] = identity;
    const contact = ready ? pendingContact !== null && [contain, close].includes(pendingContact) ? pendingContact :
      [contain, close].sort((a, b) => arrival_time(players[a]!, [carrier.x, carrier.y]) -
        arrival_time(players[b]!, [carrier.x, carrier.y]))[0]! : null;
    const result = slots.map((slot, index) => {
      const role = index === contain ? "contain" : index === close ? "close" : "cover";
      return { plan: new DefensePlan(targets.get(slot)!, role, "coordinated carrier trap", { threat_slot: owner }),
        contact: contact === index, covered_closer: role === "close" };
    });
    this.diagnostics = { phase: ready ? "close" : "contain", reason: "covered-approach", carrier: owner,
      cover_slot: guard, contact_controller: contact, roles: result.map((item) => item.plan.mode), started: this.started };
    return result;
  }
}
