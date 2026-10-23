import { eligible } from "./defense";
import { facing } from "./motion";
import { distance, lexCompare, pointEqual } from "./numeric";
import { CARRY_SAMPLES, FEINT_FRAMES, OffenseController, _carry_motion, carry_clearance, goalie_avoidance, shot_release_in_front,
  type CarryDetails, type EstimatedCarryDetails, type OffensePlan } from "./offense";
import { pressure_margin, shot_value, type PassOption } from "./passing";
import { GameState, Player, type Vec2 } from "./state";
import { project_target, route_waypoint } from "./target";

export class PossessionOffenseController extends OffenseController {
  override get uses_lookahead(): boolean { return false; }

  _estimated_carry(state: GameState, player: Player, proposal: Vec2): [Vec2, EstimatedCarryDetails] {
    let target = route_waypoint([player.x, player.y], project_target(proposal));
    const escape = goalie_avoidance(state, player, target), details = escape?.[1] ?? {};
    if (escape !== null) target = route_waypoint([player.x, player.y], escape[0]);
    const clearance = carry_clearance(state, player, target, { decision_interval: this.movement_interval });
    const motions = CARRY_SAMPLES.map((frames) => _carry_motion(player, target, frames, { decision_interval: this.movement_interval }));
    const bounded = motions.every((motion) => motion !== null);
    const margin = bounded ? Math.min(...motions.map((motion, index) => pressure_margin(state.team2, motion![0], CARRY_SAMPLES[index]!))) : -Infinity;
    const clear = clearance > 0 && margin >= 3 && (escape?.[1].goalie_avoidance_safe ?? true);
    const sign = state.team2.net.y > state.team1.net.y ? 1 : -1;
    return [target, { ...details, carry_safe: null, carry_estimated_clear: clear, carry_bounded: bounded,
      carry_clearance: clearance, carry_pressure: margin, carry_progress: bounded ? (motions.at(-1)![0][1] - player.y) * sign : -Infinity,
      carry_position_value: clear ? shot_value(state, player, motions.at(-1)![0], FEINT_FRAMES) : 0,
      carry_shot_value: 0, carry_pressure_model: "arrival-estimate", carry_status: clear ? "estimated-clear" : "fallback",
      carry_movement_model: this.carry_motion ? "native-held-input" : "sampled-legacy" }];
  }

  override fallback_carry(state: GameState, player: Player): [Vec2, CarryDetails] {
    const preferred = this.carry_target(state, player);
    if (player.motion_x === null || player.motion_y === null || !state.engine.puck_owner_known) {
      return [route_waypoint([player.x, player.y], preferred), { mode: "carry", carry_safe: null, reason: "missing feedback; retain legacy carry/setup" }];
    }
    const first = this._estimated_carry(state, player, preferred);
    if (first[1].carry_estimated_clear) return [first[0], { ...first[1], mode: first[1].goalie_clearance !== undefined ? "goalie-avoid" : "carry",
      reason: "model-based forward route; defender safety uncertified" }];
    const sign = state.team2.net.y > state.team1.net.y ? 1 : -1, [fx, fy] = facing(player), brake: Vec2 = [player.x - fx * 48, player.y - fy * 48];
    const targets: Vec2[] = [brake, [player.x, player.y]];
    for (const [dx, dy] of [[-48, 0], [48, 0], [0, -48], [-48, -24], [48, -24], [-48, 24], [48, 24]] as const) targets.push([player.x + dx, player.y + sign * dy]);
    const bounded = [first, ...targets.map((target) => this._estimated_carry(state, player, target))].filter((option) => option[1].carry_bounded);
    if (!bounded.length) return [route_waypoint([player.x, player.y], project_target(brake)), { mode: "carry-escape", carry_safe: null,
      carry_bounded: false, carry_estimated_clear: false, carry_pressure_model: "arrival-estimate",
      reason: "no bounded carry; brake before unavoidable wall or net contact" }];
    const key = (option: [Vec2, EstimatedCarryDetails]) => {
      const details = option[1], clear = details.carry_estimated_clear;
      return [Number(clear), details.carry_position_value, clear ? details.carry_progress : Math.min(details.carry_pressure, 12),
        clear ? Math.min(details.carry_pressure, 12) : Math.min(details.carry_clearance, 16), Math.min(details.carry_clearance, 16), -distance(option[0], preferred)];
    };
    let best = bounded[0]!;
    for (const option of bounded.slice(1)) if (lexCompare(key(option), key(best)) > 0) best = option;
    return [best[0], { ...best[1], mode: first[1].goalie_clearance !== undefined ? "goalie-avoid" : pointEqual(best[0], first[0]) ? "carry" : "carry-escape",
      reason: best[1].carry_estimated_clear ? "model-based escape; defender safety uncertified" : "no estimated-clear carry; maximize contact and interception clearance" }];
  }

  _immediate_gain(option: PassOption, player: Player, sign: number, purpose: string, current: number): boolean {
    return purpose === "advance" ? option.forward_gain >= 20 &&
      (option.bypassed > 0 || option.forward_gain >= 50 || player.y * sign < 88 && 88 <= option.point[1] * sign) : option.shot_value > current + 12;
  }

  override choose(state: GameState, frame: number): OffensePlan | null {
    const player = state.team1.get_player_by_scnum(state.engine.puck_owner);
    this.diagnostics = { candidates: [], scenario_samples: 8, evaluation_frame: frame, evaluation_carrier: state.engine.puck_owner };
    if (player === null || !eligible(player) || player.motion_x === null || player.motion_y === null || !state.engine.puck_owner_known) {
      this.diagnostics.status = "missing-feedback"; return null;
    }
    const sign = state.team2.net.y > state.team1.net.y ? 1 : -1, purpose = player.y * sign < 140 ? "advance" : "position";
    let [choices, candidates] = this.passes(state, purpose); this.diagnostics.candidates = candidates;
    const finish_depth = 198 + Math.max(0, 15 - (player.shot_accuracy ?? 15)) * 0.4;
    if (player.y * sign > finish_depth && Math.abs(player.x) < 48 && shot_release_in_front(state, player, this.shot_hold_frames) &&
        pressure_margin(state.team2, [player.x, player.y], 8) >= 3) {
      this.feint_target = null; this.diagnostics.status = "close-to-finish"; return null;
    }
    if (this.breakaway(state, player)) {
      this.feint_target = null;
      return this._plan(state, "carry-breakaway", this.carry_target(state, player), "no reachable skater interception in the arrival estimate");
    }
    const current = shot_value(state, player);
    choices = choices.filter((option) => this._immediate_gain(option, player, sign, purpose, current));
    const worthwhile = new Set(choices.map((option) => option.slot));
    for (const candidate of candidates) if (candidate.status === "safe") candidate.worthwhile = worthwhile.has(candidate.slot);
    const opportunity = Math.max(current, choices.length ? choices[0]!.shot_value + Math.min(choices[0]!.margin, 12) : 0);
    if (this.feint_mode === "one-timer-setup") {
      const continuation = this._continue_cut(state, player, frame); if (continuation !== null) return continuation;
    }
    if (choices.length && frame >= this.pass_at) {
      const best = choices[0]!;
      if (purpose === "position") {
        const setup = this._feint(state, player, frame, opportunity, { one_timer_only: true }); if (setup !== null) return setup;
      }
      return this._plan(state, purpose === "advance" ? "advance-pass" : "position-pass", best.point, "advance beyond defenders with reception space", best);
    }
    const continuation = this._continue_cut(state, player, frame); if (continuation !== null) return continuation;
    const cut = this._feint(state, player, frame, opportunity); if (cut !== null) return cut;
    this.diagnostics.status = "no-better-safe-option"; return null;
  }
}
