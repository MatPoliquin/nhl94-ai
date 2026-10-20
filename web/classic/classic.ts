import { DEFAULT_CLASSIC_CONFIG, decisionState, parseClassicConfig, type ClassicConfig } from "./config";
import { carry_pad } from "./carry";
import { DefenseController, controlled_slot, eligible, on_ice, owns_puck } from "./defense";
import { CSaveGoalie } from "./goalie_csave";
import { outlet_options } from "./goalie";
import { ActionEnd, OneTimerLifecycle, ShotLifecycle, SetupState, ONE_TIMER_CONTACT_GRACE, ONE_TIMER_TIMEOUT_FRAMES,
  type PassIntention, type PassSnapshot, type TacticalDiagnostics, type TeammateScore } from "./lifecycle";
import { goalie_avoidance, ordinary_shot_conditions, type GoalieEscape } from "./offense";
import { HOCKEY_PASS_PRESS_FRAMES, pass_release_frames } from "./passing";
import { PossessionOffenseController } from "./possession";
import { aim_pass } from "./target";
import { Buttons, GameState, Player, Team, neutralAction, type Action, type Vec2 } from "./state";
import type { PressureAssignment } from "./teammate-pressure";

export class ActionScheduler {
  frames = 0;
  decisions = 0;
  interval = 4;
  elapsed = 4;
  remaining = 0;
  was_defending = false;
  action = neutralAction();
  configure(interval: number): void {
    if (!Number.isInteger(interval) || interval < 1) throw new Error("Scripted decision interval must be positive");
    this.interval = interval;
  }
  advance(frames: number): number { this.frames += frames; return this.frames; }
  interrupt(): void { this.remaining = 0; }
  cache(action: Action, remaining: number | null = null): void {
    this.action = action; this.remaining = remaining ?? this.interval;
  }
}

interface OneTimerPassRequest { frame: number; passer: number; receiver: number; purpose: "one-timer" }

export class ClassicController {
  readonly config: ClassicConfig;
  readonly scheduler = new ActionScheduler();
  readonly buttons = { b_down: false, c_down: false };
  readonly defense = new DefenseController();
  readonly offense = new PossessionOffenseController();
  readonly shot = new ShotLifecycle();
  readonly one_timer = new OneTimerLifecycle();
  readonly setup = new SetupState();
  readonly goalie: CSaveGoalie | null;
  recovery_switch_at_decision = 0;
  input_owner = "idle";
  _last_decision = "init";
  _last_target: Vec2 = [0, 0];
  _last_action_preferences = neutralAction();
  _last_pass_request: OneTimerPassRequest | PassSnapshot | null = null;
  defense_diagnostics: DefenseController["diagnostics"] = {};
  offense_diagnostics: TacticalDiagnostics = {};
  goalie_diagnostics: CSaveGoalie["diagnostics"] = {};
  carry_metrics: Record<string, number> = {};

  constructor(config: ClassicConfig = DEFAULT_CLASSIC_CONFIG) {
    this.config = parseClassicConfig(config);
    this.goalie = this.config.ai_goalie ? new CSaveGoalie({ policy: "selective", pass_timing: false }) : null;
  }

  get diagnostics() {
    return { input_owner: this.input_owner, decision: this._last_decision, target: this._last_target,
      frames: this.scheduler.frames, decisions: this.scheduler.decisions,
      defense: this.defense_diagnostics, offense: this.offense_diagnostics, goalie: this.goalie_diagnostics,
      one_timer: this.one_timer.metrics, carry: this.carry_metrics };
  }

  _encode(action: Action, _intent = 0): Action {
    this.buttons.c_down = !!action[Buttons.INPUT_C]; this.buttons.b_down = !!action[Buttons.INPUT_B];
    this._last_action_preferences = [...action];
    return action;
  }
  _neutral(owner: string): Action { this.input_owner = owner; return this._encode(neutralAction()); }
  _idle(elapsed: number): void { this.defense.idle(0, { frame: this.scheduler.advance(elapsed) }); }

  advance_held_frame(): void {
    this.defense.frames = this.scheduler.advance(1);
    this.scheduler.remaining = Math.max(0, this.scheduler.remaining - 1);
    if (this.goalie !== null) this.goalie.frames++;
  }
  has_committed_action(): boolean {
    return this.offense.pass_action.pending !== null || this.one_timer.pending !== null || this.shot.active(this.scheduler.decisions);
  }
  pass_intention(): PassIntention | PassSnapshot | null {
    if (this.offense.pass_action.pending !== null) return this.offense.pass_action.pending.snapshot();
    const request = this.one_timer.pending;
    return request === null ? null : { passer: request.passer, receiver: request.receiver, frame: request.started,
      deadline: request.deadline, purpose: "one-timer", point: this._last_target };
  }
  delegate_reception(state: GameState | null = null, reason = "teammate-receiver"): void {
    if (state === null) this.offense.cancel();
    else this.offense.pass_action.finish(state, this.scheduler.frames, reason);
    this.end_one_timer(reason); this.scheduler.interrupt();
  }
  cancel_actions(reason: string): void {
    this.offense.cancel(); this.end_one_timer(reason); this.shot.cancel(); this.setup.until_decision = 0;
    this.defense.idle(0, { frame: this.scheduler.frames }); this.scheduler.interrupt(); this.scheduler.was_defending = false;
  }
  accept_applied_action(action: Action): void {
    this.scheduler.action = [...action];
    this.buttons.b_down = !!action[Buttons.INPUT_B]; this.buttons.c_down = !!action[Buttons.INPUT_C];
    this.defense.b_down = this.buttons.b_down; this.defense.c_down = this.buttons.c_down;
    this._last_action_preferences = [...action];
  }
  observe_external_frame(state: GameState, action: Action, { frame_skip = 4 } = {}): void {
    this.scheduler.configure(frame_skip); this.offense.decision_interval = frame_skip;
    if (state.engine.clock_stopped) this.cancel_actions("stoppage");
    else { this._observe_actions(state, 1); this._observe_ordinary_pass(state); }
    this._idle(1); this.scheduler.interrupt(); this.scheduler.elapsed = 1; this.scheduler.was_defending = false;
    this.input_owner = "external"; this.accept_applied_action(action);
  }

  predict_team_defense(state: GameState, assignment: PressureAssignment, frame_skip = 4): Action {
    this.scheduler.configure(frame_skip); this.scheduler.interrupt();
    this.scheduler.elapsed = 1; this.scheduler.was_defending = true;
    this.offense.cancel(); this.end_one_timer("team-defense", { interrupt: false });
    this.shot.cancel(); this.setup.until_decision = 0;
    this.defense.b_down = this.buttons.b_down; this.defense.c_down = this.buttons.c_down;
    const action = this.defense.step(state, { frame: this.scheduler.advance(1), assignment });
    this.defense_diagnostics = this.defense.diagnostics; this.offense_diagnostics = {};
    this.input_owner = "team-defense"; this._last_decision = assignment.plan.mode; this._last_target = assignment.plan.target;
    this.accept_applied_action(action);
    return action;
  }

  predict_frame(input: GameState, frame_skip = 4, _deterministic = true): Action {
    const state = decisionState(input, this.config.player_ratings_consideration);
    this.scheduler.configure(frame_skip); this.offense.decision_interval = frame_skip;
    if (state.engine.faceoff) {
      if (this.goalie !== null) {
        this.goalie.phase = "stoppage"; this.goalie.outlet = this.goalie.pending_save = null;
        this.goalie.possession_since = null; this.goalie.last_effective = false; this.goalie.pass_windup = null;
        this.goalie_diagnostics = {};
      }
      this.offense.cancel(); this.shot.cancel(); this.end_one_timer("stoppage", { interrupt: false }); this.setup.until_decision = 0;
      this._idle(1); this.scheduler.interrupt(); this._last_decision = this.input_owner = "faceoff";
      this._last_target = [state.puck.x, state.puck.y]; this.defense_diagnostics = {};
      this.offense_diagnostics = { phase: "faceoff", mode: "faceoff", reason: "capture the puck when the referee drops it" };
      const action = neutralAction(); action[Buttons.INPUT_B] = Number(this.scheduler.frames % 8 < 4);
      return this._encode(action);
    }
    if (this._observe_actions(state, 1)) return this._neutral("transition");
    this.scheduler.elapsed = 1;
    const goalie_action = this._goalie_frame(state);
    if (goalie_action !== null) { this.scheduler.elapsed = frame_skip; return goalie_action; }
    this._observe_ordinary_pass(state);
    const reception = this._receive_pass(state);
    if (reception !== null) {
      this._prepare_exclusive_frame();
      return this._encode(reception);
    }
    return this._scheduled_frame(state);
  }

  _observe_actions(state: GameState, elapsed: number): boolean {
    this._observe_follow_through(state);
    if (this._observe_one_timer(state, this.scheduler.frames + elapsed)) { this._idle(elapsed); return true; }
    return false;
  }
  _observe_follow_through(state: GameState): void {
    if (this.shot.observe(state, this.scheduler.decisions)) this.scheduler.interrupt();
  }
  _observe_ordinary_pass(state: GameState): void {
    const event = this.offense.observe(state, this.scheduler.frames);
    if (event !== null && event.reason === "recovered-by-passer") this.scheduler.interrupt();
  }
  _apply_one_timer_end(event: ActionEnd | null): boolean {
    if (event === null) return false;
    if (event.interrupt) this.scheduler.interrupt();
    this._last_decision = "one-timer-ended"; this.offense_diagnostics = { reason: event.reason };
    return event.neutral_frame;
  }
  end_one_timer(reason: string, { interrupt = true } = {}): void {
    this._apply_one_timer_end(this.one_timer.finish(reason, { interrupt }));
  }
  _observe_one_timer(state: GameState, frame: number): boolean { return this._apply_one_timer_end(this.one_timer.observe(state, frame)); }
  _receive_pass(state: GameState): Action | null {
    const step = this.offense.pass_action.receive(state, this.scheduler.frames, { b_down: this.buttons.b_down });
    if (step instanceof ActionEnd) { this.scheduler.interrupt(); return null; }
    if (step === null) return null;
    this.input_owner = "pass"; this._last_decision = step.decision; this._last_target = step.target!;
    this.defense_diagnostics = {}; this.offense_diagnostics = step.diagnostics; return step.buttons;
  }
  _continue_one_timer(state: GameState): Action | null {
    this.input_owner = "one-timer";
    const step = this.one_timer.step(state, this.scheduler.frames, { c_down: this.buttons.c_down });
    if (step instanceof ActionEnd) { this._apply_one_timer_end(step); return null; }
    this._last_decision = step.decision;
    if (step.target !== null) this._last_target = step.target;
    this.offense_diagnostics = step.diagnostics;
    return step.buttons;
  }
  _goalie_frame(state: GameState): Action | null {
    if (this.goalie === null) return null;
    const blocked = this.one_timer.pending !== null || this.offense.pass_action.pending !== null ||
      this.shot.active(this.scheduler.decisions) || this.defense.pending_check !== null || this.defense.pending_switch !== null;
    const action = this.goalie.step(state, { blocked });
    this.goalie_diagnostics = this.goalie.diagnostics;
    if (action === null) return null;
    this.input_owner = "goalie"; this._idle(1); this.defense_diagnostics = {}; this.offense_diagnostics = {};
    this.end_one_timer("goalie-control"); this.shot.cancel(); this.setup.until_decision = 0; this.setup.slot = null;
    this.offense.cancel(); this.scheduler.interrupt(); this.scheduler.was_defending = false;
    this._last_decision = "goalie-" + this.goalie.phase; this._last_target = this.goalie_diagnostics.target!;
    return this._encode(action);
  }
  _defending(state: GameState): boolean {
    const owner = state.engine.puck_owner;
    if (owns_puck(state.team1, owner)) return false;
    return owner >= 0 || this.one_timer.pending === null && this.offense.pass_action.pending === null && !this.shot.active(this.scheduler.decisions);
  }
  _defend(state: GameState): Action {
    this.input_owner = "defense"; this.offense.cancel(); this.offense_diagnostics = {};
    if (state.engine.puck_owner >= 0) { this.end_one_timer("possession-changed", { interrupt: false }); this.shot.cancel(); this.setup.until_decision = 0; }
    this.defense.b_down = this.buttons.b_down; this.defense.c_down = this.buttons.c_down;
    const frame = this.scheduler.advance(this.scheduler.elapsed), action = this.defense.step(state, { frame });
    const player = state.team1.get_player_by_scnum(controlled_slot(state.team1));
    const escape = goalie_avoidance(state, player, this.defense.diagnostics.waypoint!);
    if (escape !== null && !action[Buttons.INPUT_B] && this.defense.pending_check === null) {
      for (let index = 4; index < 9; index++) action[index] = 0;
      this._steer(action, player!, ...escape[0]); this.defense.c_down = false;
      Object.assign(this.defense.diagnostics, escape[1], { mode: "goalie-avoid", decision: "goalie-avoid",
        target: escape[0], destination: escape[0], waypoint: escape[0], buttons: [...action] });
    }
    this.defense_diagnostics = this.defense.diagnostics;
    this._last_target = this.defense_diagnostics.target!; this._last_decision = this.defense_diagnostics.decision!;
    return action;
  }
  _prepare_exclusive_frame(): void {
    this._idle(1); this.scheduler.interrupt(); this.scheduler.was_defending = false; this.scheduler.elapsed = this.scheduler.interval;
  }
  _scheduled_frame(state: GameState): Action {
    let defending = this._defending(state);
    if (!defending && this.one_timer.needs_frame(state, this.scheduler.frames + 1)) {
      this._prepare_exclusive_frame();
      const step = this._continue_one_timer(state);
      return step !== null ? this._encode(step) : this._neutral("one-timer-ended");
    }
    if (this.scheduler.remaining === 0) {
      this.scheduler.cache(this._predict_decision(state)); defending = Object.keys(this.defense_diagnostics).length > 0;
    } else if (defending) this.scheduler.action = this._encode(this._defend(state));
    else if (this.scheduler.was_defending) { this.scheduler.action = this._neutral("handoff"); this._idle(1); }
    else this._idle(1);
    if (!defending) this.defense_diagnostics = {};
    this.scheduler.was_defending = defending; this.scheduler.remaining--; this.scheduler.elapsed = this.scheduler.interval;
    if ((this._last_decision === "shoot" || this.shot.extended_hold) && this.scheduler.frames >= this.shot.hold_until_frame) {
      this.scheduler.action = [...this.scheduler.action]; this.scheduler.action[Buttons.INPUT_C] = 0;
      this.buttons.c_down = false; this.shot.extended_hold = false;
    }
    return [...this.scheduler.action];
  }
  _predict_decision(state: GameState): Action {
    this.scheduler.decisions++;
    const action = this._decide(state);
    if (!Object.keys(this.defense_diagnostics).length) {
      const actual = controlled_slot(state.team1), details = this.offense_diagnostics;
      this.offense_diagnostics = { ...details, phase: "offense", mode: this._last_decision, decision: this._last_decision,
        target: this._last_target, destination: this._last_target, waypoint: this._last_target, actual_slot: actual,
        desired_slot: details.desired_slot ?? actual, reason: details.reason ?? details.status ?? this._last_decision,
        buttons: [...action], last_pass: this.offense.pass_action.last_pass };
      this.offense_diagnostics.teammate_scores = this._teammate_scores(state);
    }
    return this._encode(action);
  }
  _teammate_scores(state: GameState): TeammateScore[] {
    return state.team1.players.flatMap((_, index) => {
      const slot = state.team1.skater_scnum_base() + index;
      return slot === state.engine.puck_owner ? [] : [{ slot, selected: slot === this.offense_diagnostics.desired_slot,
        pass: this.offense_diagnostics.candidates?.find((item) => item.slot === slot) ?? { status: "not-evaluated" },
        one_timer: this.offense_diagnostics.one_timer_candidates?.find((item) => item.slot === slot) ?? null }];
    });
  }
  _goalie_escape(state: GameState, player: Player | null, target: Vec2): GoalieEscape | null { return goalie_avoidance(state, player, target); }
  _steer(action: Action, player: Player, x: number, y: number, state: GameState | null = null): void {
    let point: Vec2 = [x, y];
    const escape = state !== null ? this._goalie_escape(state, player, point) : null;
    if (escape !== null) { point = escape[0]; this._last_decision = "goalie-avoid"; Object.assign(this.offense_diagnostics, escape[1]); }
    this._last_target = point;
    const [dx, dy] = carry_pad(player, point);
    if (dx) action[dx > 0 ? Buttons.INPUT_RIGHT : Buttons.INPUT_LEFT] = 1;
    if (dy) action[dy > 0 ? Buttons.INPUT_UP : Buttons.INPUT_DOWN] = 1;
  }
  _carry(state: GameState, player: Player, action: Action, { breakaway = false } = {}): Action {
    const [point, details] = this.offense.fallback_carry(state, player);
    Object.assign(this.offense_diagnostics, details);
    this._last_decision = breakaway && details.mode === "carry" ? "carry-breakaway" : details.mode;
    const estimated = details.carry_pressure_model === "arrival-estimate";
    const outcome = estimated ? details.carry_estimated_clear ? "estimated-clear" : "estimated-fallback" :
      details.carry_safe === null ? "missing-feedback" : !details.carry_safe && details.carry_viable ? "uncertified-viable" :
      !details.carry_safe ? "least-risk" : details.mode === "carry" ? "safe-default" : "safe-escape";
    this.carry_metrics[outcome] = (this.carry_metrics[outcome] ?? 0) + 1;
    this._steer(action, player, ...point, details.carry_safe === null && !estimated ? state : null);
    return action;
  }
  _accuracy(player: Player): number { return Math.max(0, Math.min(30, player.shot_accuracy ?? 15)); }
  _one_timer_target(team: Team, opponents: Team, attack: number, prepare = false, state: GameState | null = null): [number, number] | null {
    this.one_timer.flight_frames = null; this.one_timer.option = null;
    const passer = state !== null ? team.get_player_by_scnum(state.engine.puck_owner)! : team.get_controlled_player()!;
    if (this.scheduler.frames < this.one_timer.retry_at_frame || passer.y * attack < 100) return null;
    if (state !== null && passer.passing !== null && !prepare) {
      const [choices, details] = this.offense.passes(state, "one-timer"); this.offense.diagnostics.one_timer_candidates = details;
      if (!choices.length) return null;
      const best = choices[0]!; this.one_timer.option = best;
      this.one_timer.flight_frames = best.flight_frames + pass_release_frames(team.players[best.index]!);
      const indices = team.players.flatMap((player, index) => player !== passer ? [index] : []);
      return [indices.indexOf(best.index), best.slot];
    }
    const teammates = team.players.flatMap((player, index) => index !== team.control - 1 ? [{ slot: team.skater_scnum_base() + index, player }] : []);
    for (const [index, { slot, player: receiver }] of teammates.entries()) {
      if (!eligible(receiver) || Math.abs(receiver.x) > 70 || !(175 + Math.max(0, 15 - this._accuracy(receiver)) < receiver.y * attack && receiver.y * attack < 245)) continue;
      const dx = receiver.x - passer.x, dy = receiver.y - passer.y, length2 = dx * dx + dy * dy;
      if (prepare) { if (Math.abs(dy) < 75 && receiver.y * attack >= passer.y * attack - 25) return [index, slot]; continue; }
      if (!(900 < length2 && length2 < 22500 && Math.abs(dy) < 80 && receiver.x * passer.x <= 0 && Math.abs(dx) > 30)) continue;
      let clear = true;
      for (const defender of opponents.players) {
        if (!on_ice(defender)) continue;
        const along = Math.max(0, Math.min(1, ((defender.x - passer.x) * dx + (defender.y - passer.y) * dy) / length2));
        if (Math.hypot(defender.x - passer.x - along * dx, defender.y - passer.y - along * dy) < 14) { clear = false; break; }
      }
      if (clear) return [index, slot];
    }
    return null;
  }
  _decide(state: GameState): Action {
    this.defense_diagnostics = {}; this.offense_diagnostics = {}; this._observe_ordinary_pass(state);
    if (this._defending(state)) return this._defend(state);
    this._idle(this.scheduler.elapsed);
    const action = neutralAction(), team = state.team1, actual = controlled_slot(team), player = team.get_player_by_scnum(actual);
    const attack = state.team2.net.y > team.net.y ? 1 : -1, owner = state.engine.puck_owner;
    if (this.offense.pass_action.pending !== null) return this._continue_pass(state, action, player, actual);
    if (this.one_timer.pending !== null) { const pending = this._continue_one_timer(state); if (pending !== null) return pending; }
    if (owner === (team.defense_goalie ?? team.goalie_scnum())) return this._goalie_outlet(state, action);
    if (owner >= 0 && !team.owns_scnum(owner)) { this.shot.cancel(); this.setup.until_decision = 0; }
    if (this.shot.active(this.scheduler.decisions)) return this._continue_shot(state, action, player);
    if (owner === actual && player !== null && player !== team.goalie) return this._attack(state, action, player, actual, attack);
    return this._recover(state, action, player);
  }
  _continue_pass(state: GameState, action: Action, player: Player | null, actual: number): Action {
    this.input_owner = "pass";
    const team = state.team1, owner = state.engine.puck_owner, request = this.offense.pass_action.pending!;
    this._last_decision = request.launched ? "pass-flight" : "pass-release";
    this.offense_diagnostics = { ...this.offense.diagnostics, mode: this._last_decision, decision: this._last_decision,
      reason: "wait for observed reception; do not boost or switch through a pass", target: request.point, destination: request.point,
      waypoint: request.point, actual_slot: actual, desired_slot: request.receiver, pending_pass: request.snapshot() };
    if (owner === request.passer && !request.launched && this.scheduler.frames - request.frame < HOCKEY_PASS_PRESS_FRAMES) {
      const receiver = team.get_player_by_scnum(request.receiver);
      if (receiver !== null) { aim_pass(action, team.get_player_by_scnum(owner)!, receiver); action[Buttons.INPUT_B] = 1; return action; }
    }
    if (request.launched && owner < 0 && player !== null) {
      const escape = this._goalie_escape(state, player, [player.x, player.y]);
      if (escape !== null) {
        this._steer(action, player, ...escape[0]); this._last_decision = "goalie-avoid"; Object.assign(this.offense_diagnostics, escape[1]);
      }
    }
    return action;
  }
  _goalie_outlet(state: GameState, action: Action): Action {
    this.input_owner = "goalie-outlet"; const team = state.team1; this.shot.cancel();
    const [options, candidates] = outlet_options(state, { decision_interval: this.scheduler.interval, release_prediction: false });
    this.offense_diagnostics = { outlet_candidates: candidates };
    if (this.buttons.b_down || !options.length) {
      this._last_decision = "goalie-hold"; this._last_target = [team.goalie.x, team.goalie.y];
      this.offense_diagnostics.reason = this.buttons.b_down ? "release B before requesting an outlet" : "no evaluated outlet; allow the automatic goalie to cover";
      return action;
    }
    const option = options[0]!, receiver = team.players[option.index]!;
    this._last_decision = "goalie-outlet"; this._last_target = [receiver.x, receiver.y];
    this.offense.start_pass(state, option, this.scheduler.frames, "goalie-outlet"); this._last_pass_request = { ...this.offense.pass_action.last_request! };
    Object.assign(this.offense_diagnostics, { desired_slot: option.slot, reason: "evaluated safe goalie outlet" });
    aim_pass(action, team.goalie, receiver); action[Buttons.INPUT_B] = 1; return action;
  }
  _continue_shot(state: GameState, action: Action, player: Player | null): Action {
    this.input_owner = "shot"; this._last_decision = "shot-follow-through";
    if (!this.shot.released(state) || player === null) this.shot.aim(action);
    else this._steer(action, player, player.x + this.shot.side * 40, player.y, state);
    if (this.shot.extended_hold && this.scheduler.frames < this.shot.hold_until_frame) action[Buttons.INPUT_C] = 1;
    return action;
  }
  _attack(state: GameState, action: Action, player: Player, actual: number, attack: number): Action {
    this.input_owner = "offense"; const team = state.team1, owner = state.engine.puck_owner, progress = player.y * attack;
    this.offense.one_timer_at = this.one_timer.retry_at_frame;
    const plan = this.offense.choose(state, this.scheduler.frames);
    this.offense_diagnostics = this.offense.diagnostics;
    const escape = this._goalie_escape(state, player, this.offense.carry_target(state, player));
    this.offense_diagnostics.shot_release_model = player.shot_offsets_y !== null ? "native-animation-envelope" : "conservative-animation-envelope";
    const [normal_finish, early_finish] = ordinary_shot_conditions(state, player, this.scheduler.interval, escape);
    const target = this._one_timer_target(team, state.team2, attack, false, state);
    if (target !== null && !normal_finish) {
      this.input_owner = "one-timer";
      if (this.buttons.b_down) { this._last_decision = "pass-button-release"; return action; }
      const slot = target[1], receiver = team.get_player_by_scnum(slot)!;
      const duration = this.one_timer.flight_frames !== null ? Math.ceil(this.one_timer.flight_frames) + ONE_TIMER_CONTACT_GRACE : ONE_TIMER_TIMEOUT_FRAMES;
      this.one_timer.start(state, slot, this.scheduler.frames, duration); this.setup.until_decision = 0;
      this._last_decision = "one-timer-pass"; this._last_target = [receiver.x, receiver.y];
      Object.assign(this.offense_diagnostics, { desired_slot: slot, receiver: null,
        reason: player.passing !== null ? "moving one-timer pass with reception margin" : "legacy one-timer geometry; motion safety unverified" });
      this._last_pass_request = { frame: this.scheduler.frames, passer: owner, receiver: slot, purpose: "one-timer" };
      aim_pass(action, player, receiver); action[Buttons.INPUT_B] = 1; return action;
    }
    const goalie_danger = progress > 175 && escape !== null, preparing = plan !== null && plan[0] === "one-timer-setup";
    if (normal_finish || early_finish && !preparing) {
      this.input_owner = "shot"; this._last_decision = "shoot"; this.shot.side = state.team2.goalie.x > 0 ? -1 : 1;
      this._last_target = [this.shot.side * 13, state.team2.net.y];
      Object.assign(this.offense_diagnostics, { desired_slot: actual, receiver: null,
        reason: goalie_danger ? "finish before projected goalie contact" : "close-range finishing opportunity" });
      this.shot.aim(action);
      if (this.buttons.c_down) return action;
      this.shot.start(state, this.scheduler.decisions, this.scheduler.frames, this.scheduler.interval);
      action[Buttons.INPUT_C] = 1; return action;
    }
    if (plan !== null) {
      const [mode, point, option] = plan; this._last_decision = mode; this._last_target = point;
      if (option !== null) {
        this.input_owner = "pass";
        if (this.buttons.b_down) { this._last_decision = "pass-button-release"; return action; }
        this._last_target = option.point;
        this.offense.start_pass(state, option, this.scheduler.frames, mode); this._last_pass_request = { ...this.offense.pass_action.last_request! };
        aim_pass(action, player, team.get_player_by_scnum(option.slot)!); action[Buttons.INPUT_B] = 1; return action;
      }
      if (mode === "carry-breakaway") return this._carry(state, player, action, { breakaway: true });
      this._steer(action, player, ...point, ["feint", "one-timer-setup", "carry-opportunity", "create-chance"].includes(mode) ? null : state);
      return action;
    }
    if (player.passing === null && this.scheduler.decisions >= this.setup.retry_at_decision && 140 < progress && progress < 215) {
      const setup = this._one_timer_target(team, state.team2, attack, true);
      if (setup !== null) { this.setup.slot = setup[1]; this.setup.until_decision = this.scheduler.decisions + 8; this.setup.retry_at_decision = this.scheduler.decisions + 48; }
    }
    if (this.scheduler.decisions < this.setup.until_decision) {
      const receiver = this.setup.slot === null ? null : team.get_player_by_scnum(this.setup.slot);
      if (receiver !== null && !receiver.is_falling) {
        this._last_decision = "one-timer-setup";
        this._steer(action, player, receiver.x >= 0 ? -65 : 65, attack * Math.min(215, Math.max(progress, receiver.y * attack - 15)), state);
        return action;
      }
      this.setup.until_decision = 0;
    }
    return this._carry(state, player, action);
  }
  _recover(state: GameState, action: Action, player: Player | null): Action {
    this.input_owner = "recovery"; const team = state.team1, puck = state.puck; this._last_decision = "recover";
    if (player === null) return action;
    const distance = Math.hypot(puck.x - player.x, puck.y - player.y);
    const closest = Math.min(...team.players.map((p) => Math.hypot(puck.x - p.x, puck.y - p.y)));
    if (this.scheduler.decisions >= this.recovery_switch_at_decision && (!team.control || closest + 20 < distance)) {
      this._last_decision = "switch"; this.recovery_switch_at_decision = this.scheduler.decisions + 6; action[Buttons.INPUT_B] = 1; return action;
    }
    const lead = Math.min(distance / 12, 6) * 0.45;
    this._steer(action, player, Math.max(-112, Math.min(112, puck.x + puck.vx * lead)), Math.max(-246, Math.min(246, puck.y + puck.vy * lead)), state);
    action[Buttons.INPUT_C] = Number(this.scheduler.decisions % 2 === 0 && this._last_decision !== "goalie-avoid"); return action;
  }
}
