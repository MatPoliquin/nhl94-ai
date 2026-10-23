import { describe, expect, it, vi } from "vitest";
import { DEFAULT_CLASSIC_CONFIG } from "../classic/config";
import { controllerView } from "../classic/decoder";
import { SharedReceiving } from "../classic/teammate-receiving";
import * as reception from "../classic/receiving";
import { TeammateCoordinator } from "../classic/teammates";
import { ClassicController } from "../classic/classic";
import { TeamControlContext } from "../classic/teammate-context";
import { faceoff_center, FaceoffInput } from "../classic/teammate-faceoff";
import { Buttons, GameState, neutralAction } from "../classic/state";
import { decodeObservation, type Observation } from "../observations";
import { Button } from "../inputs";
import { parseReplay } from "../replay";

function state(): GameState {
  const state = new GameState();
  state.time = 300;
  state.team1.net.y = -264; state.team2.net.y = 264;
  state.team1.pass_attempts = state.team1.one_timer_attempts = 0;
  state.engine.controller_teams = [1, 1, 1, 2];
  state.engine.puck_owner = 0; state.engine.clock_stopped = false;
  state.engine.offsides_enabled = true;
  for (const team of [state.team1, state.team2]) {
    team.players.forEach((player, index) => Object.assign(player, {
      x: index * 20, y: index * -20, motion_x: 0, motion_y: 0,
      speed: 15, agility: 15, weight: 64, energy: 4096, role: 1,
    }));
  }
  return state;
}

function observation(): Observation {
  return { frame: 0, seconds: 300, period: 0, score: [0, 0], puck: { x: 0, y: 0 },
    puckOwner: 0, ramHash: "", controllers: [
      { team: 1, slot: 0, player: { x: 0, y: 0 } }, { team: 1, slot: 1, player: { x: 20, y: -20 } },
      { team: 1, slot: 2, player: { x: 40, y: -40 } }, { team: 2, slot: 6, player: { x: 0, y: 0 } },
    ] };
}

describe("Classic teammate coordination", () => {
  it("uses one carrier, a passing option and distinct deep coverage without mutating the shared state", () => {
    const team = new TeammateCoordinator([0, 1, 2], DEFAULT_CLASSIC_CONFIG), input = state();
    const before = JSON.stringify(input);
    team.controllers.forEach((controller) => vi.spyOn(controller, "predict_frame").mockReturnValue(neutralAction()));
    expect([...team.act(observation(), input).keys()]).toEqual([0, 1, 2]);
    expect(team.primary).toBe(0);
    expect(team.controllers.every((controller) => controller.goalie === null)).toBe(true);
    expect(team.roles).toEqual(["carrier", "passing-option", "deep-coverage"]);
    expect(team.targets[1]).not.toEqual(team.targets[2]);
    expect(team.controllers.map((controller) => controller.scheduler.frames)).toEqual([0, 1, 1]);
    expect(JSON.stringify(input)).toBe(before);
    const view = controllerView(input, 2, 2);
    expect(view.team1.players[0]?.rel_controlled_x).toBe(-40);
    expect(view.team1.players[2]?.rel_controlled_x).toBe(0);
    expect(view.team1.players[0]).not.toBe(input.team1.players[0]);
  });

  it("hands primary ownership to the actual carrier, not the first controller", () => {
    const team = new TeammateCoordinator([0, 1, 2], DEFAULT_CLASSIC_CONFIG), input = state();
    input.engine.puck_owner = 2;
    team.controllers.forEach((controller) => vi.spyOn(controller, "predict_frame").mockReturnValue(neutralAction()));
    team.act(observation(), input);
    expect(team.primary).toBe(2);
    expect(team.roles).toEqual(["passing-option", "deep-coverage", "carrier"]);
  });

  it("holds configured low-frequency actions but immediately neutralizes transient ownership", () => {
    const team = new TeammateCoordinator([0, 1, 2], { ...DEFAULT_CLASSIC_CONFIG, decision_hz: 6 }), input = state();
    const action = neutralAction(); action[8] = 1;
    const predict = vi.spyOn(team.controllers[0]!, "predict_frame").mockReturnValue(action);
    const first = team.act(observation(), input);
    for (let frame = 1; frame < 10; frame++) expect(team.act(observation(), input)).toEqual(first);
    expect(predict).toHaveBeenCalledTimes(1);
    expect([...team.act({ ...observation(), transient: true }, input).values()]).toEqual([0, 0, 0]);
  });

  it("tracks native pass release and transfers pressure to the selected receiver", () => {
    const receiving = new SharedReceiving(), input = state(), slots = [0, 1, 2];
    receiving.observe(input, slots, 0);
    receiving.propose({ passer: 0, receiver: 1, point: [20, -20], frame: 1, deadline: 60, purpose: "advance-pass" },
      0, slots, input, 1);
    expect(receiving.pending?.phase).toBe("release");
    expect(receiving.receiverAction(input, 1, 1, neutralAction())?.action).toEqual(neutralAction());
    input.team1.pass_attempts = 1; input.engine.pass_target = 2; input.engine.puck_owner = -1;
    receiving.observe(input, slots, 2);
    expect(receiving.pending).toMatchObject({ phase: "flight", receiver: 2, receiver_controller: 2, released: 2 });
    input.engine.puck_owner = 2;
    receiving.observe(input, slots, 3);
    expect(receiving.pending?.phase).toBe("received");
    input.engine.puck_owner = 6;
    receiving.observe(input, slots, 4);
    expect(receiving.pending).toBeNull();
    expect(receiving.ended).toEqual([{ passer_controller: 0, reason: "intercepted" }]);
  });

  function releasedPass(purpose = "advance-pass") {
    const input = state(), receiving = new SharedReceiving();
    receiving.observe(input, [0, 1, 2], 0);
    receiving.propose({ passer: 0, receiver: 1, point: [35, 210], frame: 1, deadline: 60, purpose },
      0, [0, 1, 2], input, 1);
    input.team1.pass_attempts = 1; input.engine.pass_target = 1; input.engine.puck_owner = -1;
    Object.assign(input.team1.players[1]!, { x: 35, y: 210 });
    receiving.observe(input, [0, 1, 2], 2);
    return { input, receiving };
  }

  it("preserves a released pass through an entirely unknown table and resumes reception", () => {
    const { input, receiving } = releasedPass(), plan = receiving.pending;
    receiving.observe(input, [-1, -1, -1], 3);
    expect(receiving.pending).toBe(plan);
    receiving.observe(input, [0, 1, 2], 4);
    input.engine.puck_owner = 1;
    receiving.observe(input, [0, 1, 2], 5);
    expect(receiving.pending).toMatchObject({ phase: "received", received: 5 });
  });

  it.each(["known-loss", "partial-unknown", "interception", "stoppage", "expiry", "unreleased"])(
    "still cancels a pass for %s during an unknown controller table", (reason) => {
      const { input, receiving } = releasedPass();
      let slots = [-1, -1, -1], frame = 3;
      if (reason === "known-loss") slots = [0, 2, 3];
      if (reason === "partial-unknown") slots = [0, 2, -1];
      if (reason === "interception") input.engine.puck_owner = 6;
      if (reason === "stoppage") input.engine.clock_stopped = true;
      if (reason === "expiry") frame = receiving.pending!.expiry;
      if (reason === "unreleased") {
        receiving.pending!.phase = "release"; receiving.pending!.released = null;
      }
      receiving.observe(input, slots, frame);
      expect(receiving.pending).toBeNull();
    },
  );

  it("keeps unknown controllers neutral without erasing the coordinator's released pass", () => {
    const team = new TeammateCoordinator([0, 1, 2], DEFAULT_CLASSIC_CONFIG);
    const { input, receiving } = releasedPass();
    team.act(observation(), input);
    team.receiving.pending = receiving.pending;
    const missing = { team: 1, slot: -1, player: null };
    const unknown: Observation = { ...observation(), transient: true,
      controllers: [missing, missing, missing, { ...missing, team: 2 }] };
    expect([...team.act(unknown, input).values()]).toEqual([0, 0, 0]);
    expect(team.receiving.pending).toBe(receiving.pending);
    expect(team.roles).toEqual(["unavailable", "unavailable", "unavailable"]);
    expect(team.controllers.map((controller) => controller.scheduler.frames)).toEqual([2, 2, 2]);
    team.act(observation(), input);
    expect(team.receiving.pending).toBe(receiving.pending);
  });

  it("spends a one-timer cue only when its current receiver applies a fresh C edge", () => {
    const contact = vi.spyOn(reception, "one_timer_reception").mockReturnValue({ gap: -4, frames: 8, point: [35, 210] });
    try {
      const { input, receiving } = releasedPass("one-timer");
      const request = receiving.receiverAction(input, 1, 3, neutralAction())!;
      expect(request.action[Buttons.INPUT_C]).toBe(1);
      expect(receiving.pending!.cue).toBeNull();
      receiving.acceptAppliedAction(1, -1, 3, request.action, neutralAction());
      receiving.acceptAppliedAction(0, 1, 3, request.action, neutralAction());
      receiving.acceptAppliedAction(1, 1, 3, neutralAction(), neutralAction());
      receiving.acceptAppliedAction(1, 1, 3, request.action, request.action);
      expect(receiving.pending!.cue).toBeNull();
      const retry = receiving.receiverAction(input, 1, 4, neutralAction())!;
      expect(retry.role).toBe("one-timer-receiver");
      receiving.acceptAppliedAction(1, 1, 4, retry.action, neutralAction());
      expect(receiving.pending!.cue).toBe(4);
      expect(receiving.receiverAction(input, 1, 5, neutralAction())!.action[Buttons.INPUT_C]).toBe(0);
    } finally { contact.mockRestore(); }
  });

  it("reroutes a receiver to its actual controller and observes native arming without another cue", () => {
    const contact = vi.spyOn(reception, "one_timer_reception").mockReturnValue({ gap: -4, frames: 8, point: [35, 210] });
    try {
      const { input, receiving } = releasedPass("one-timer");
      receiving.pending!.receiver_controller = null;
      input.team1.players[1]!.is_one_timer = 1;
      receiving.observe(input, [1, 0, 2], 3);
      expect(receiving.pending).toMatchObject({ receiver_controller: 0, cue: null, armed: 3 });
      expect(receiving.receiverAction(input, 1, 3, neutralAction())).toBeNull();
      expect(receiving.receiverAction(input, 0, 3, neutralAction())!.action).toEqual(neutralAction());
      input.team1.players[1]!.is_one_timer = 0;
      receiving.observe(input, [1, 0, 2], 4);
      expect(receiving.pending!.armed).toBe(3);
      expect(receiving.receiverAction(input, 0, 4, neutralAction())!.action[Buttons.INPUT_C]).toBe(0);
    } finally { contact.mockRestore(); }
  });

  it("accounts for the coordinator's applied one-timer edge without cueing it again", () => {
    const contact = vi.spyOn(reception, "one_timer_reception").mockReturnValue({ gap: -4, frames: 8, point: [35, 210] });
    try {
      const team = new TeammateCoordinator([0, 1, 2], DEFAULT_CLASSIC_CONFIG);
      const { input, receiving } = releasedPass("one-timer");
      team.act(observation(), input);
      team.receiving.pending = receiving.pending;
      expect(team.act(observation(), input).get(1)! & Button.C).toBe(Button.C);
      expect(team.roles[1]).toBe("one-timer-receiver");
      expect(team.receiving.pending!.cue).toBe(2);
      expect(team.act(observation(), input).get(1)! & Button.C).toBe(0);
      expect(team.receiving.pending!.cue).toBe(2);
    } finally { contact.mockRestore(); }
  });

  it("assigns faceoff B only to the native draw participant, not the first port", () => {
    const team = new TeammateCoordinator([0, 1, 2], DEFAULT_CLASSIC_CONFIG), input = state();
    input.engine.puck_owner = -1; input.engine.clock_stopped = true;
    Object.assign(input.team1.players[1]!, { x: 0, y: -12, role: 4, assignment: 0x17 });
    Object.assign(input.team2.players[0]!, { x: 0, y: 12, role: 4, assignment: 0x17 });
    expect(faceoff_center(input)).toBe(1);
    const masks = team.act(observation(), input);
    expect([...masks.values()]).toEqual([0, Button.B, 0]);
    expect(team.roles).toEqual(["faceoff-support", "faceoff", "faceoff-support"]);
    input.team1.players[1]!.assignment = 0x16;
    expect([...team.act(observation(), input).values()]).toEqual([0, 0, 0]);
    expect(team.roles).toEqual(["stoppage", "stoppage", "stoppage"]);
  });

  it("releases B on draw ownership handoff and rejects unknown or ambiguous participants", () => {
    const input = state(), faceoff = new FaceoffInput(), previous = [neutralAction(), neutralAction()];
    input.engine.puck_owner = -1; input.engine.clock_stopped = true;
    Object.assign(input.team1.players[1]!, { x: 0, y: -12, role: 4, assignment: 0x17 });
    Object.assign(input.team2.players[0]!, { x: 0, y: 12, role: 4, assignment: 0x17 });
    previous[1]![Buttons.INPUT_B] = 1;
    expect(faceoff.choose(input, [0, 1], 1, false, previous).action).toEqual(neutralAction());
    expect(faceoff.choose(input, [0, 1], 2, false, previous).action[Buttons.INPUT_B]).toBe(1);
    expect(faceoff.choose(input, [0, 1], 3, true, previous).controller).toBeNull();
    input.team1.players[0]!.assignment = 0x17; input.team1.players[0]!.role = 4;
    expect(faceoff_center(input)).toBeNull();
  });

  it("lets only the selected goalie execute an outlet without inventing shared reception ownership", () => {
    const team = new TeammateCoordinator([0, 1, 2], DEFAULT_CLASSIC_CONFIG), input = state();
    input.engine.puck_owner = 5; input.team1.goalie_haspuck = true;
    const observed: Observation = { ...observation(), controllers: [
      observation().controllers[0], { team: 1, slot: 5, player: { x: 0, y: -250 } },
      observation().controllers[2], observation().controllers[3],
    ] };
    const outlet = neutralAction(); outlet[Buttons.INPUT_B] = 1;
    const predict = vi.spyOn(team.controllers[1]!, "predict_frame").mockReturnValue(outlet);
    expect(team.act(observed, input).get(1)).toBe(Button.B);
    expect(predict).toHaveBeenCalledOnce();
    expect(team.roles[1]).toBe("goalie-outlet");
    const receiving = new SharedReceiving();
    receiving.propose({ passer: 5, receiver: 0, point: [0, 0], frame: 1, deadline: 50, purpose: "goalie-outlet" },
      1, [0, 5, 2], input, 1);
    expect(receiving.pending).toBeNull();
  });

  it("supports a confirmed CPU carrier without switch/pass input and falls back when ownership is unknown", () => {
    const input = state(), controller = new ClassicController({ ...DEFAULT_CLASSIC_CONFIG, ai_goalie: false });
    const view = controllerView(input, 0, 0);
    view.engine.puck_owner = 3; view.engine.puck_owner_known = true;
    controller.team_context = new TeamControlContext([0, 1, 2], true, [-45, 130], true);
    const action = controller._decide(view);
    expect(controller._last_decision).toBe("cpu-carrier-support");
    expect(controller._last_target).toEqual([-45, 76]);
    expect(action[Buttons.INPUT_B]).toBe(0);
    expect(action[Buttons.INPUT_C]).toBe(0);
    expect(controller.team_context.cpu_carrier(view)).toBe(true);
    view.engine.puck_owner_known = false;
    expect(controller.team_context.cpu_carrier(view)).toBe(false);
    controller.team_context = new TeamControlContext([0, 1, 2], false, [-45, 130], true);
    view.engine.puck_owner_known = true;
    expect(controller.team_context.cpu_carrier(view)).toBe(false);
  });

  it("does not let an unselected skater issue or continue a goalie's B release", () => {
    const input = state(), controller = new ClassicController({ ...DEFAULT_CLASSIC_CONFIG, ai_goalie: false });
    const view = controllerView(input, 0, 0);
    view.engine.puck_owner = 5; view.team1.goalie_haspuck = true;
    controller.team_context = new TeamControlContext([0, 1], true, [45, -145], true);
    expect(controller._decide(view)[Buttons.INPUT_B]).toBe(0);
    expect(controller._last_decision).toBe("goalie-outlet-support");
  });
});

describe("four-port native ownership and replays", () => {
  it("suppresses temporary ROM controller aliases without repairing live RAM", () => {
    const ram = new Uint8Array(65536), words = new DataView(ram.buffer);
    for (let port = 0; port < 4; port++) {
      words.setUint16(0xc328 + port * 2, port === 3 ? 2 : 1, true);
      words.setInt16(0xc320 + port * 2, port === 3 ? 6 : port, true);
      ram[(0xb04a + (port === 3 ? 6 : port) * 0x80 + 0x62) ^ 1] = 8;
    }
    expect(decodeObservation(ram, 0, [1, 1, 1, 2]).transient).toBe(false);
    words.setInt16(0xc320, 2, true);
    const before = ram.slice(), observed = decodeObservation(ram, 1, [1, 1, 1, 2]);
    expect(observed.transient).toBe(true);
    expect(observed.controllers.map((controller) => controller.slot)).toEqual([-1, -1, -1, -1]);
    expect(ram).toEqual(before);
    ram[0xc2ea ^ 1] = 0x10;
    expect(decodeObservation(ram, 2, [1, 1, 1, 2]).transient).toBe(false);
  });

  it("round-trips all four input masks and paused/final assignments with strict v4 validation", () => {
    const replay = { format: "nhl94-browser-inputs-v4", coreBuild: "a".repeat(64), romHash: "b".repeat(64),
      startStateHash: "c".repeat(64), inputs: [[Button.A, Button.B, Button.C, Button.RIGHT]],
      configurationEvents: [
        { frame: 0, teams: [1, 1, 1, 0], goalieModes: [1, 0] },
        { frame: 0, teams: [1, 1, 1, 2], goalieModes: [1, 0] },
        { frame: 1, teams: [1, 1, 1, 0], goalieModes: [1, 0] },
      ] };
    expect(parseReplay(JSON.parse(JSON.stringify(replay)))).toEqual(replay);
    for (const inputs of [[[0, 0]], [[0, 0, 0]], [[0, 0, 0, 2]], [[0, 0, 0, -1]], [new Array(4)]]) {
      expect(() => parseReplay({ ...replay, inputs })).toThrow("inputs");
    }
    expect(() => parseReplay({ ...replay, configurationEvents: [{ frame: 0, teams: [1, 1], goalieModes: [1, 0] }] })).toThrow("assignment");
  });
});
