import { DefenseController, controlled_slot, defensive_steering, eligible, owns_puck } from "./defense";
import { HOCKEY_PASS_PRESS_FRAMES, type PassDiagnostics, type PassOption, type PassOptionSnapshot } from "./passing";
import { passive_reception, reception_target, type ReceptionForecast } from "./receiving";
import { aim_pass } from "./target";
import { Buttons, GameState, neutralAction, type Action, type Vec2 } from "./state";

export const ONE_TIMER_TIMEOUT_FRAMES = 72;
export const ONE_TIMER_CONTACT_GRACE = 20;
export const ONE_TIMER_RETRY_FRAMES = 96;

export interface TacticalDiagnostics {
  phase?: string;
  mode?: string;
  decision?: string;
  reason?: string;
  status?: string;
  target?: Vec2;
  destination?: Vec2;
  waypoint?: Vec2;
  actual_slot?: number;
  desired_slot?: number | null;
  actual_receiver?: number | null;
  receiver?: PassOptionSnapshot | null;
  buttons?: Action;
  candidates?: PassDiagnostics[];
  one_timer_candidates?: PassDiagnostics[];
  outlet_candidates?: PassDiagnostics[];
  scenario_samples?: number;
  evaluation_frame?: number;
  evaluation_carrier?: number;
  cut_cancelled?: string;
  shot_release_model?: string;
  pending_pass?: PassSnapshot;
  last_pass?: PassOutcome | null;
  reception_correction?: ReceptionCorrectionSnapshot;
  teammate_scores?: TeammateScore[];
  carry_safe?: boolean | null;
  carry_estimated_clear?: boolean;
  carry_bounded?: boolean;
  carry_clearance?: number;
  carry_pressure?: number;
  carry_progress?: number;
  carry_position_value?: number;
  carry_shot_value?: number;
  carry_pressure_model?: string;
  carry_status?: string;
  carry_movement_model?: string;
  carry_viable?: boolean;
  goalie_clearance?: number;
  original_goalie_clearance?: number;
  goalie_avoidance_safe?: boolean;
  goalie_carry_model?: string;
}

export interface TeammateScore {
  slot: number;
  selected: boolean;
  pass: PassDiagnostics | { status: string };
  one_timer: PassDiagnostics | null;
}

export class ShotLifecycle {
  readonly rebound_recovery = false;
  rebound_wait_for_animation = true;
  until_decision = 0;
  hold_until_frame = 0;
  shots_before = 0;
  slot = -1;
  side = 1;
  extended_hold = false;
  crossing = false;
  release_observed = false;
  recoveries = 0;

  active(decision: number): boolean { return decision < this.until_decision; }
  phase(decision: number, frame: number): string {
    return !this.active(decision) ? "idle" : frame < this.hold_until_frame ? "holding" : "follow-through";
  }
  start(state: GameState, decision: number, frame: number, hold_frames: number, follow_through_decisions = 7): void {
    this.extended_hold = this.crossing = this.release_observed = false;
    this.until_decision = decision + follow_through_decisions;
    this.hold_until_frame = frame + hold_frames;
    this.shots_before = state.team1.stats.shots;
    this.slot = state.engine.puck_owner;
  }
  cancel(): void {
    this.until_decision = 0;
    this.extended_hold = this.crossing = this.release_observed = false;
  }
  observe(state: GameState, decision: number): boolean {
    if (!this.active(decision)) return false;
    const owner = state.engine.puck_owner;
    if (state.engine.clock_stopped) this.release_observed = false;
    else if (owner < 0 && this.released(state)) this.release_observed = true;
    if (owns_puck(state.team1, owner) && owner !== this.slot) { this.cancel(); return true; }
    return false;
  }
  released(state: GameState): boolean {
    return state.team1.stats.shots > this.shots_before && state.engine.puck_owner !== this.slot &&
      [-1, this.slot].includes(state.engine.shot_player);
  }
  aim(action: Action): void {
    if (this.side) action[this.side > 0 ? Buttons.INPUT_RIGHT : Buttons.INPUT_LEFT] = 1;
  }
}

export class SetupState {
  until_decision = 0;
  retry_at_decision = 0;
  slot: number | null = null;
}

export class ActionEnd {
  constructor(public readonly reason: string, public readonly interrupt = true, public readonly neutral_frame = false) {}
}

export class ActionStep {
  constructor(
    public buttons: Action, public intent: number, public decision: string,
    public target: Vec2 | null = null, public diagnostics: TacticalDiagnostics = {},
  ) {}
}

export interface ReceptionCorrectionSnapshot {
  started: number;
  forecast: ReceptionForecast;
  from_slot: number;
  switch_frame: number | null;
}

export class ReceptionCorrection {
  switch_frame: number | null = null;
  constructor(public started: number, public forecast: ReceptionForecast, public from_slot: number) {}
  snapshot(): ReceptionCorrectionSnapshot { return { ...this }; }
}

export interface PassSnapshot {
  passer: number;
  receiver: number;
  frame: number;
  deadline: number;
  purpose: string;
  point: Vec2;
  passes_before: number | null;
  direction: number;
  actual_receiver: number | null;
  launched: boolean;
  flight_observed: boolean;
  reception_correction?: ReceptionCorrectionSnapshot;
}
export interface PassOutcome extends PassSnapshot { outcome: string; owner: number; end_frame: number }
export interface PassIntention {
  passer: number; receiver: number; frame: number; deadline: number; purpose: string; point: Vec2;
  flight_frames?: number | null;
}

export class PassState {
  actual_receiver: number | null = null;
  launched = false;
  flight_observed = false;
  reception_correction: ReceptionCorrection | null = null;
  constructor(
    public passer: number, public receiver: number, public frame: number, public deadline: number,
    public purpose = "advance-pass", public point: Vec2 = [0, 0], public passes_before: number | null = null,
    public direction = 8,
  ) {}
  get phase(): string { return this.reception_correction !== null ? "reception-correction" : this.launched ? "flight" : "release"; }
  snapshot(): PassSnapshot {
    const result: PassSnapshot = {
      passer: this.passer, receiver: this.receiver, frame: this.frame, deadline: this.deadline,
      purpose: this.purpose, point: this.point, passes_before: this.passes_before, direction: this.direction,
      actual_receiver: this.actual_receiver, launched: this.launched, flight_observed: this.flight_observed,
    };
    if (this.reception_correction !== null) result.reception_correction = this.reception_correction.snapshot();
    return result;
  }
}

export class PassLifecycle {
  readonly stick_control = false;
  pending: PassState | null = null;
  last_pass: PassOutcome | null = null;
  last_request: PassSnapshot | null = null;
  reception_metrics: Record<string, number> = {};

  start(state: GameState, option: PassOption, frame: number, purpose: string): void {
    this.pending = new PassState(state.engine.puck_owner, option.slot, frame, frame + Math.ceil(option.flight_frames) + 20,
      purpose, option.point, state.team1.pass_attempts, option.direction);
    this.last_request = this.pending.snapshot();
  }
  cancel(): void { this.pending = null; }
  finish(state: GameState, frame: number, reason: string): ActionEnd | null {
    if (this.pending === null) return null;
    if (this.pending.reception_correction !== null) {
      const key = "corrected:" + reason;
      this.reception_metrics[key] = (this.reception_metrics[key] ?? 0) + 1;
    }
    this.last_pass = { ...this.pending.snapshot(), outcome: reason, owner: state.engine.puck_owner, end_frame: frame };
    this.pending = null;
    return new ActionEnd(reason);
  }
  observe(state: GameState, frame: number): ActionEnd | null {
    const request = this.pending;
    if (request === null) return null;
    const owner = state.engine.puck_owner;
    let result: string;
    if (state.team2.owns_scnum(owner)) result = request.launched ? "intercepted" : "lost-before-release";
    else if (owner === request.passer && request.flight_observed) result = "recovered-by-passer";
    else if (frame >= request.deadline) result = request.launched ? "flight-timeout" : "not-released";
    else if (owner >= 0 && owner !== request.passer) result = owner === request.receiver ? "received" : "other-receiver";
    else {
      if (owner < 0) request.flight_observed = true;
      if (request.actual_receiver === null && state.team1.pass_attempts !== null && request.passes_before !== null &&
          state.team1.pass_attempts > request.passes_before) {
        request.actual_receiver = state.engine.pass_target;
        request.launched = true;
      }
      if (owner < 0 && state.engine.last_puck_player === request.passer) request.launched = true;
      return null;
    }
    return this.finish(state, frame, result);
  }
  receive(state: GameState, frame: number, { b_down = false } = {}): ActionStep | ActionEnd | null {
    const request = this.pending;
    if (request === null || !request.launched) return null;
    if (state.engine.puck_owner >= 0 || state.engine.clock_stopped || controlled_slot(state.team1) < 0 ||
        frame - request.frame < HOCKEY_PASS_PRESS_FRAMES) return null;
    const slot = request.actual_receiver, receiver = slot === null ? null : state.team1.get_player_by_scnum(slot);
    let correction = request.reception_correction;
    if (receiver === null || !eligible(receiver)) return correction !== null ? this.finish(state, frame, "receiver-unavailable") : null;
    const actual = controlled_slot(state.team1);
    if (correction === null) {
      const forecast = passive_reception(state, receiver);
      const wait = receiver.decision_timer === null || forecast !== null && forecast.frames <= receiver.decision_timer;
      if (forecast === null || forecast.gap <= -2 || receiver.assignment !== 19 || wait) return null;
      correction = new ReceptionCorrection(frame, forecast, actual);
      request.reception_correction = correction;
      this.reception_metrics.corrections = (this.reception_metrics.corrections ?? 0) + 1;
    }
    const action = neutralAction();
    let intent = 0;
    const target = reception_target(state, receiver);
    if (target === null || actual !== slot && (actual !== correction.from_slot || frame - correction.started >= 16)) {
      return this.finish(state, frame, "reception-unreachable");
    }
    let mode = "receive-pass-wait", waypoint = target;
    if (actual === slot) {
      this.reception_metrics["steered-frames"] = (this.reception_metrics["steered-frames"] ?? 0) + 1;
      const [pad, point] = defensive_steering(receiver, target);
      waypoint = point;
      if (pad[0]) action[pad[0] > 0 ? Buttons.INPUT_RIGHT : Buttons.INPUT_LEFT] = 1;
      if (pad[1]) action[pad[1] > 0 ? Buttons.INPUT_UP : Buttons.INPUT_DOWN] = 1;
      mode = "receive-pass";
    } else if (!b_down && DefenseController._likely_switch(state) === slot &&
        (correction.switch_frame === null || frame - correction.switch_frame >= 8)) {
      action[Buttons.INPUT_B] = 1;
      correction.switch_frame = frame;
      this.reception_metrics.switches = (this.reception_metrics.switches ?? 0) + 1;
      mode = "receive-pass-switch"; intent = 6;
    }
    return new ActionStep(action, intent, mode, waypoint, {
      phase: "offense", mode, decision: mode, reason: "correct marginal reception after native receive timer",
      target: waypoint, destination: waypoint, waypoint, actual_slot: actual, desired_slot: slot,
      reception_correction: correction.snapshot(), buttons: [...action],
    });
  }
}

export class OneTimerState {
  actual_receiver: number | null = null;
  launched = false;
  constructor(
    public passer: number, public receiver: number, public deadline: number, public started = 0,
    public passes_before: number | null = null, public attempts_before: number | null = null, public shots_before = 0,
  ) {}
  get phase(): string { return this.launched ? "awaiting-shot" : "release"; }
}

export class OneTimerLifecycle {
  readonly execution = null;
  pending: OneTimerState | null = null;
  retry_at_frame = 0;
  starts = 0;
  metrics: Record<string, number> = {};
  option: PassOption | null = null;
  flight_frames: number | null = null;
  needs_frame(_state: GameState, _frame: number): boolean { return false; }
  start(state: GameState, receiver: number, frame: number, duration = ONE_TIMER_TIMEOUT_FRAMES): void {
    this.pending = new OneTimerState(state.engine.puck_owner, receiver, frame + duration, frame,
      state.team1.pass_attempts, state.team1.one_timer_attempts, state.team1.stats.shots);
    this.starts++;
    this.retry_at_frame = frame + ONE_TIMER_RETRY_FRAMES;
  }
  finish(reason: string, { interrupt = true, neutral_frame = false } = {}): ActionEnd | null {
    if (this.pending === null) return null;
    this.metrics[reason] = (this.metrics[reason] ?? 0) + 1;
    this.pending = null;
    return new ActionEnd(reason, interrupt, neutral_frame);
  }
  observe(state: GameState, frame: number): ActionEnd | null {
    const request = this.pending;
    if (request === null) return null;
    if (state.engine.clock_stopped) return this.finish("play-stopped", { neutral_frame: true });
    const attempts = state.team1.one_timer_attempts;
    const released = attempts !== null && request.attempts_before !== null && attempts > request.attempts_before;
    const recorded = state.team1.stats.shots > request.shots_before, owner = state.engine.puck_owner;
    if (state.engine.shot_player === request.receiver && (released || recorded)) return this.finish("shot-released");
    if (owner === request.passer && request.launched) return this.finish("recovered-by-passer");
    if (owns_puck(state.team1, owner) && owner !== request.passer) return this.finish("possession-changed");
    if (frame >= request.deadline) return this.finish("timeout");
    if (owner < 0) request.launched = true;
    return null;
  }
  step(state: GameState, frame: number, { c_down = false } = {}): ActionStep | ActionEnd {
    const request = this.pending;
    if (request === null) throw new Error("No pending one-timer");
    const receiver = state.team1.get_player_by_scnum(request.receiver), owner = state.engine.puck_owner;
    if (request.actual_receiver === null && request.passes_before !== null && state.team1.pass_attempts !== null &&
        state.team1.pass_attempts > request.passes_before && state.engine.pass_target !== null) request.actual_receiver = state.engine.pass_target;
    if (request.actual_receiver !== null && request.actual_receiver !== request.receiver) return this.finish("receiver-mismatch")!;
    if (receiver === null || receiver.is_falling || frame >= request.deadline || owner >= 0 && owner !== request.passer) {
      return this.finish(frame >= request.deadline ? "timeout" : receiver === null || receiver.is_falling ? "receiver-unavailable" : "possession-changed")!;
    }
    const action = neutralAction();
    let mode = "one-timer-wait", target: Vec2 | null = null;
    if (frame - request.started < HOCKEY_PASS_PRESS_FRAMES) {
      aim_pass(action, state.team1.get_player_by_scnum(request.passer)!, receiver);
      action[Buttons.INPUT_B] = 1;
    } else if (owner === request.passer) {
      target = [receiver.x, receiver.y];
      aim_pass(action, state.team1.get_player_by_scnum(request.passer)!, receiver);
    } else if (!c_down && !receiver.is_one_timer) {
      mode = "one-timer-shoot";
      action[Buttons.INPUT_C] = 1;
    }
    return new ActionStep(action, 0, mode, target, {
      desired_slot: request.receiver, reason: "wait for one-timer contact", actual_receiver: request.actual_receiver,
    });
  }
}
