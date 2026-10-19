import { defensive_steering, eligible } from "./defense";
import { ActionStep, OneTimerLifecycle, OneTimerState, type PassIntention } from "./lifecycle";
import { HOCKEY_PASS_PRESS_FRAMES, shot_value } from "./passing";
import { one_timer_reception, passive_reception, reachable_reception } from "./receiving";
import { Buttons, neutralAction, type Action, type GameState, type Vec2 } from "./state";
import { project_target } from "./target";

export interface PassCommitment {
  passer: number;
  receiver: number;
  passer_controller: number | null;
  receiver_controller: number | null;
  point: Vec2;
  started: number;
  expiry: number;
  purpose: string;
  passes_before: number | null;
  shots_before: number;
  attempts_before: number | null;
  goals_before: number;
  phase: "release" | "flight" | "received" | "shot";
  released: number | null;
  received: number | null;
  cue: number | null;
  shot: number | null;
}

export class SharedReceiving {
  pending: PassCommitment | null = null;
  last_owner = -1;
  last_count: number | null = null;
  ended: { passer_controller: number | null; reason: string }[] = [];

  finish(reason: string): void {
    if (this.pending) this.ended.push({ passer_controller: this.pending.passer_controller, reason });
    this.pending = null;
  }

  propose(request: PassIntention | null, controller: number | null, slots: readonly number[],
    state: GameState, frame: number): void {
    if (!request) return;
    if (this.pending) {
      if (request.frame <= this.pending.started) return;
      this.finish("next-pass-requested");
    }
    if (!state.team1.get_player_by_scnum(request.receiver)) return;
    this.pending = {
      passer: request.passer, receiver: request.receiver, passer_controller: controller,
      receiver_controller: slots.includes(request.receiver) ? slots.indexOf(request.receiver) : null,
      point: request.point, started: frame,
      expiry: frame + Math.max(8, Math.min(72, request.deadline - request.frame)),
      purpose: request.purpose, passes_before: state.team1.pass_attempts, shots_before: state.team1.stats.shots,
      attempts_before: state.team1.one_timer_attempts, goals_before: state.team1.stats.score,
      phase: "release", released: null, received: null, cue: null, shot: null,
    };
  }

  observe(state: GameState, slots: readonly number[], frame: number): void {
    this.ended = [];
    const count = state.team1.pass_attempts, owner = state.engine.puck_owner;
    const released = count !== null && this.last_count !== null && count > this.last_count;
    if (state.engine.clock_stopped) this.finish("stoppage");
    else if (released && state.team1.owns_scnum(this.last_owner)) {
      const actual = state.engine.pass_target;
      if (this.pending && this.pending.phase !== "release") this.finish("next-native-pass");
      const receiver = actual === null ? null : state.team1.get_player_by_scnum(actual);
      if (!this.pending && receiver && actual !== null) {
        this.propose({ passer: this.last_owner, receiver: actual, point: [receiver.x, receiver.y],
          frame, deadline: frame + 72, purpose: "native-pass" },
        slots.includes(this.last_owner) ? slots.indexOf(this.last_owner) : null, slots, state, frame);
      }
      if (this.pending?.phase === "release") {
        if (actual !== this.pending.receiver) {
          if (actual === null) this.finish("receiver-unavailable");
          else {
            this.pending.receiver = actual;
            this.pending.purpose = "native-pass";
            this.pending.receiver_controller = slots.includes(actual) ? slots.indexOf(actual) : null;
          }
        }
        if (this.pending) { this.pending.phase = "flight"; this.pending.released = frame; }
      }
    }
    const plan = this.pending;
    if (plan) {
      const receiver = state.team1.get_player_by_scnum(plan.receiver), attempts = state.team1.one_timer_attempts;
      const shot = state.team1.stats.shots > plan.shots_before ||
        attempts !== null && plan.attempts_before !== null && attempts > plan.attempts_before;
      if (plan.shot === null && plan.released !== null && shot && state.engine.shot_player === plan.receiver) {
        plan.shot = frame; plan.phase = "shot"; plan.expiry = frame + 150;
      }
      if (plan.phase === "shot") {
        if (state.team1.stats.score > plan.goals_before) this.finish("goal");
        else if (owner >= 0) this.finish(state.team1.owns_scnum(owner) ? "shot-recovered-team" : "shot-recovered-opponent");
      } else if (state.team2.owns_scnum(owner)) this.finish(plan.released !== null ? "intercepted" : "lost-before-release");
      else if (plan.released !== null && owner === plan.receiver) {
        if (plan.received === null) { plan.received = frame; plan.phase = "received"; plan.expiry = frame + 90; }
      } else if (plan.released !== null && owner >= 0 && owner !== plan.passer) this.finish("other-receiver");
      else if (plan.phase === "received" && owner !== plan.receiver) this.finish("reception-possession-ended");
      else if (!receiver || receiver.is_falling || !eligible(receiver) && !receiver.is_one_timer) this.finish("receiver-unavailable");
      else if (plan.receiver_controller !== null && !slots.includes(plan.receiver)) this.finish("receiver-selection-lost");
      else if (plan.released !== null && owner === plan.passer) this.finish("recovered-by-passer");
      if (this.pending && frame >= plan.expiry) this.finish(`unresolved-${plan.phase}`);
    }
    this.last_count = count; this.last_owner = owner;
  }

  receiverAction(state: GameState, controller: number, frame: number, previous: Action): { action: Action; target: Vec2 } | null {
    const plan = this.pending;
    if (!plan || plan.receiver_controller !== controller || !["release", "flight"].includes(plan.phase)) return null;
    const player = state.team1.get_player_by_scnum(plan.receiver);
    if (!player || !eligible(player) && !player.is_one_timer) return null;
    const action = neutralAction();
    if (plan.phase === "release" || player.is_one_timer) return { action, target: [player.x, player.y] };
    const forecast = passive_reception(state, player), interception = reachable_reception(state, player);
    let target: Vec2 = [player.x, player.y];
    if (!forecast || forecast.gap > -2) {
      const [pad, waypoint] = defensive_steering(player, project_target(interception?.point ?? plan.point));
      target = waypoint;
      action[Buttons.INPUT_LEFT] = Number(pad[0] < 0); action[Buttons.INPUT_RIGHT] = Number(pad[0] > 0);
      action[Buttons.INPUT_DOWN] = Number(pad[1] < 0); action[Buttons.INPUT_UP] = Number(pad[1] > 0);
    }
    plan.point = target;
    const attack = state.team2.net.y > state.team1.net.y ? 1 : -1;
    const contact = plan.purpose === "one-timer" ? one_timer_reception(state, player) : null;
    if (contact && contact.gap <= -2 && contact.frames > 1 && contact.frames <= 16 &&
        player.y * attack > 175 && player.y * attack < 245 && Math.abs(player.x) < 70 &&
        shot_value(state, player) > 0 && plan.cue === null && !previous[Buttons.INPUT_C]) {
      const lifecycle = new OneTimerLifecycle();
      lifecycle.pending = new OneTimerState(plan.passer, plan.receiver, plan.expiry,
        frame - HOCKEY_PASS_PRESS_FRAMES, plan.passes_before);
      lifecycle.pending.actual_receiver = plan.receiver; lifecycle.pending.launched = true;
      const step = lifecycle.step(state, frame);
      if (step instanceof ActionStep) { plan.cue = frame; return { action: step.buttons, target }; }
    }
    return { action, target };
  }
}
