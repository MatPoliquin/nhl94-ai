import type { ControllerPort } from "../controllers";
import type { Observation } from "../observations";
import { actionMask } from "./bot";
import { ClassicController } from "./classic";
import { controlInterval, decisionState, type ClassicConfig } from "./config";
import { controllerView } from "./decoder";
import { defensive_steering, eligible } from "./defense";
import { arrival_time } from "./motion";
import { SharedReceiving } from "./teammate-receiving";
import { SharedPressure, TRAP_ROLES } from "./teammate-pressure";
import { Buttons, neutralAction, type Action, type GameState, type Vec2 } from "./state";
import { project_target } from "./target";

export class TeammateCoordinator {
  readonly controllers: ClassicController[];
  readonly receiving = new SharedReceiving();
  readonly pressure = new SharedPressure();
  primary: number | null = null;
  roles: string[];
  targets: Vec2[];
  private frames = 0;
  private primaryUntil = 0;
  private switchAt = 0;
  private lastMenu = -120;
  private remaining = 0;
  private previous: Action[];
  private lastSlots: number[];

  constructor(readonly ports: readonly ControllerPort[], private readonly config: ClassicConfig) {
    if (ports.length < 2 || ports.length > 3) throw new Error("Classic coordination requires two or three teammates.");
    this.controllers = ports.map(() => new ClassicController({ ...config, ai_goalie: false }));
    this.previous = ports.map(neutralAction);
    this.lastSlots = ports.map(() => -1);
    this.roles = ports.map(() => "idle");
    this.targets = ports.map(() => [0, 0]);
  }

  private choosePrimary(state: GameState, slots: number[]): number | null {
    const owner = state.engine.puck_owner;
    const receiver = this.receiving.pending?.phase === "flight" ? this.receiving.pending.receiver : -1;
    if (slots.includes(owner) && owner >= 0 || slots.includes(receiver) && receiver >= 0) {
      this.primary = slots.indexOf(owner >= 0 && slots.includes(owner) ? owner : receiver);
      this.primaryUntil = this.frames + 24;
      return this.primary;
    }
    const choices = slots.flatMap((slot, index) => {
      const player = state.team1.get_player_by_scnum(slot);
      return player && slot !== state.team1.goalie_scnum() && eligible(player) ?
        [{ time: arrival_time(player, [state.puck.x, state.puck.y]), index }] : [];
    }).sort((a, b) => a.time - b.time || a.index - b.index);
    const best = choices[0];
    if (!best) return this.primary = null;
    const current = choices.find((choice) => choice.index === this.primary)?.time ?? Infinity;
    if (this.primary === null || current === Infinity || this.frames >= this.primaryUntil && best.time + 6 < current) {
      this.primary = best.index; this.primaryUntil = this.frames + 24;
    }
    return this.primary;
  }

  private supportTarget(state: GameState, index: number): [Vec2, string] {
    const team = state.team1, puck = state.puck;
    const supporters = this.ports.map((_, i) => i).filter((i) => i !== this.primary);
    if (this.ports.length === 3 && supporters.length > 1 && index === supporters.at(-1)) {
      return [project_target([Math.max(-30, Math.min(30, puck.x * 0.25)),
        puck.y + (team.net.y - puck.y) * 0.65]), "deep-coverage"];
    }
    const attack = state.team2.net.y > team.net.y ? 1 : -1;
    const carrier = team.get_player_by_scnum(state.engine.puck_owner);
    if (carrier && carrier !== team.goalie) {
      const forward = carrier.y * attack;
      let y = Math.min(225, forward + 45);
      if (state.engine.offsides_enabled !== false && forward < 88) y = Math.min(y, 76);
      return [project_target([carrier.x >= 0 ? -45 : 45, y * attack]), "passing-option"];
    }
    if (team.goalie_haspuck) return [project_target([puck.x < 0 ? 45 : -45, team.net.y * 0.55]), "outlet-option"];
    return [project_target([Math.max(-48, Math.min(48, puck.x * 0.45)),
      puck.y + (team.net.y - puck.y) * 0.45]), "coverage"];
  }

  act(observation: Observation, input: GameState): ReadonlyMap<ControllerPort, number> {
    const views = this.ports.map((port) =>
      decisionState(controllerView(input, port, observation.controllers[port].slot), this.config.player_ratings_consideration));
    const state = views[0]!, slots = views.map((view) => view.team1.controlled_scnum());
    this.frames++;
    const unavailable = observation.transient || state.engine.highlight;
    if (unavailable) {
      this.remaining = 0; this.primary = null; this.receiving.finish("native-transition");
      this.receiving.last_owner = -1; this.receiving.last_count = null;
    }
    const wasTrapping = this.roles.some((role) => TRAP_ROLES.includes(role));
    const pendingContact = this.controllers.findIndex((controller) => controller.defense.pending_check !== null);
    const assignments = this.remaining === 0 || wasTrapping || unavailable ?
      this.pressure.choose(state, slots, this.frames, { busy: !!unavailable,
        pendingContact: pendingContact < 0 ? null : pendingContact }) : null;
    if (wasTrapping && assignments === null) {
      this.controllers.forEach((controller) => controller.cancel_actions("trap-ended"));
      this.remaining = 0; this.primary = null;
    }
    if (this.remaining > 0 && !unavailable) {
      this.remaining--;
      this.controllers.forEach((controller) => controller.advance_held_frame());
      return new Map(this.ports.map((port, i) => [port, actionMask(this.previous[i]!) ]));
    }
    this.receiving.observe(state, slots, this.frames);
    for (const event of this.receiving.ended) {
      if (event.passer_controller !== null) this.controllers[event.passer_controller]!.delegate_reception(
        views[event.passer_controller]!, event.reason);
    }
    const primary = this.choosePrimary(state, slots), actions = this.ports.map(neutralAction);
    const intermission = state.time === 0 || !!(state.engine.sflags & 0x0100);
    if (intermission) this.receiving.finish("intermission");
    const order = primary === null ? this.ports.map((_, i) => i) :
      [primary, ...this.ports.map((_, i) => i).filter((i) => i !== primary)];
    for (const i of order) {
      const view = views[i]!, slot = slots[i]!, player = view.team1.get_player_by_scnum(slot);
      const controller = this.controllers[i]!, plan = this.receiving.pending;
      let [target, role] = this.supportTarget(view, i);
      let external = true;
      if (plan?.phase === "flight" && plan.passer_controller === i && plan.receiver_controller !== null &&
          plan.receiver_controller !== i) controller.delegate_reception();
      const reception = this.receiving.receiverAction(view, i, this.frames, this.previous[i]!);
      if (unavailable || intermission) {
        role = state.engine.highlight ? "highlight" : "intermission";
        if (!observation.transient && i === 0 && this.frames - this.lastMenu >= 120 &&
            (state.engine.highlight || state.period <= 3 && state.time > 0)) {
          actions[i]![Buttons.INPUT_START] = 1; this.lastMenu = this.frames;
        }
      } else if (slot < 0) role = "unavailable";
      else if (view.engine.clock_stopped) {
        role = i === 0 ? "faceoff" : "faceoff-support";
        if (i === 0) actions[i]![Buttons.INPUT_B] = Number(this.frames % 8 < 4);
      } else if (!player || slot === view.team1.goalie_scnum() || player.is_falling ||
          player.unavailable & 4 || player.role !== null && player.role <= 0 ||
          !eligible(player) && !player.is_one_timer && !controller.has_committed_action()) role = "unavailable";
      else if (assignments !== null) {
        const assignment = assignments[i]!;
        if (!TRAP_ROLES.includes(this.roles[i]!)) controller.cancel_actions("trap-started");
        actions[i] = controller.predict_team_defense(view, assignment);
        role = assignment.plan.mode; target = assignment.plan.target; external = false;
      }
      else if (reception) { actions[i] = reception.action; target = reception.target; role = "receive"; }
      else if (i === primary || controller.has_committed_action()) {
        role = state.engine.puck_owner === slot ? "carrier" : view.team2.owns_scnum(state.engine.puck_owner) ? "pressure" : "retrieve";
        if (this.lastSlots[i] !== slot) controller.scheduler.interrupt();
        actions[i] = controller.predict_frame(view, 4); external = false;
        this.receiving.propose(controller.pass_intention(), i, slots, state, this.frames);
        target = controller._last_target;
        if (state.engine.puck_owner !== slot && actions[i]![Buttons.INPUT_B]) {
          if (this.frames < this.switchAt || plan && ["release", "flight"].includes(plan.phase)) actions[i]![Buttons.INPUT_B] = 0;
          else if (!this.previous[i]![Buttons.INPUT_B]) this.switchAt = this.frames + 8;
        }
      } else {
        const [pad, waypoint] = defensive_steering(player, target);
        target = waypoint;
        actions[i]![Buttons.INPUT_UP] = Number(pad[1] > 0); actions[i]![Buttons.INPUT_DOWN] = Number(pad[1] < 0);
        actions[i]![Buttons.INPUT_LEFT] = Number(pad[0] < 0); actions[i]![Buttons.INPUT_RIGHT] = Number(pad[0] > 0);
      }
      if (external) {
        if (["intermission", "highlight", "unavailable", "faceoff", "faceoff-support"].includes(role)) controller.cancel_actions(role);
        controller.observe_external_frame(view, actions[i]!, { frame_skip: 4 });
      }
      controller.accept_applied_action(actions[i]!);
      this.roles[i] = role; this.targets[i] = target; this.lastSlots[i] = slot;
    }
    if (!intermission && !state.engine.clock_stopped && !unavailable) {
      this.controllers.forEach((controller, i) => this.receiving.propose(controller.pass_intention(), i, slots, state, this.frames));
    }
    this.previous = actions; this.remaining = controlInterval(this.config) - 1;
    return new Map(this.ports.map((port, i) => [port, actionMask(actions[i]!) ]));
  }
}
