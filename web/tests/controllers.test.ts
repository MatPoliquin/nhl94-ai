import { describe, expect, it } from "vitest";
import { AVAILABLE_BOTS, DiagnosticBot, type Bot } from "../bot";
import { createBot } from "../bot-factory";
import { ClassicBot } from "../classic/bot";
import { DEFAULT_CLASSIC_CONFIG } from "../classic/config";
import {
  assignedTeams, controllerInputs, controllerRouting, DEFAULT_CONTROLLERS, parseControllerSetup,
  type ControllerPort, type ControllerSetup,
} from "../controllers";
import { Button } from "../inputs";
import type { Observation } from "../observations";

function observation(): Observation {
  return {
    frame: 24, period: 1, seconds: 300, score: [0, 0],
    puck: { x: 10, y: 20 }, puckOwner: -1, ramHash: "00000000",
    controllers: [
      { team: 1, slot: 0, player: { x: -30, y: 40 } },
      { team: 2, slot: 6, player: { x: 20, y: -60 } },
      { team: 0, slot: -1, player: null }, { team: 0, slot: -1, player: null },
    ],
  };
}

class ProbeBot implements Bot {
  readonly calls: { state: Observation; port: ControllerPort }[] = [];
  constructor(private readonly action: number) {}
  reset(): void { this.calls.length = 0; }
  act(state: Observation, port: ControllerPort): number {
    this.calls.push({ state, port });
    return this.action;
  }
}

describe("four-slot controller configuration", () => {
  it("offers ROM AI before Classic AI in the bot catalogue", () => {
    expect(AVAILABLE_BOTS.map((bot) => bot.id)).toEqual(["rom", "classic"]);
    expect(AVAILABLE_BOTS.map((bot) => bot.label)).toEqual(["ROM AI", "no ml bot"]);
  });

  it("only constructs a scripted bot for Classic AI", () => {
    expect(createBot("classic", DEFAULT_CLASSIC_CONFIG)).toBeInstanceOf(ClassicBot);
    for (const agent of ["human", "rom", "none"] as const) {
      expect(createBot(agent, DEFAULT_CLASSIC_CONFIG)).toBeNull();
    }
  });

  it("defaults to Human/Home, Classic/Away and two available None slots", () => {
    expect(parseControllerSetup(DEFAULT_CONTROLLERS)).toEqual([
      { agent: "human", team: 1 }, { agent: "classic", team: 2 },
      { agent: "none", team: 1 }, { agent: "none", team: 2 },
    ]);
  });

  it("rejects invalid agents, teams and shapes", () => {
    for (const value of [
      null, [], DEFAULT_CONTROLLERS.slice(0, 2),
      [{ agent: "unsupported", team: 1 }, ...DEFAULT_CONTROLLERS.slice(1)],
      [{ agent: "diagnostic", team: 1 }, ...DEFAULT_CONTROLLERS.slice(1)],
      [{ agent: "human", team: "1" }, ...DEFAULT_CONTROLLERS.slice(1)],
    ]) expect(() => parseControllerSetup(value)).toThrow();
  });

  it("combines keyboard and gamepad for one Human and evaluates the opposing bot", () => {
    expect(controllerInputs(DEFAULT_CONTROLLERS, observation(), [Button.A, Button.B],
      [new DiagnosticBot(), new DiagnosticBot()]))
      .toEqual([Button.A | Button.B, Button.LEFT | Button.UP | Button.C, 0, 0]);
  });

  it("routes occupants by Home/Away instead of assuming port 1 always means Home", () => {
    const state = observation();
    const reversed = { ...state, controllers: [state.controllers[1], state.controllers[0], state.controllers[2], state.controllers[3]] as const };
    const bots = [new ProbeBot(Button.B), new ProbeBot(Button.C)] as const;
    expect(controllerRouting(DEFAULT_CONTROLLERS, reversed).ports).toEqual([1, 0, 2, 3]);
    expect(controllerInputs(DEFAULT_CONTROLLERS, reversed, [Button.A, 0], bots)).toEqual([Button.C, Button.A, 0, 0]);
    expect(bots[1].calls[0]).toEqual({ state: reversed, port: 0 });
  });

  it("applies a swapped team selection to human and bot inputs", () => {
    const setup: ControllerSetup = [
      { agent: "human", team: 2 }, { agent: "classic", team: 1 },
      DEFAULT_CONTROLLERS[2], DEFAULT_CONTROLLERS[3],
    ];
    expect(controllerInputs(setup, observation(), [Button.START, 0],
      [new ProbeBot(Button.A), new ProbeBot(Button.B)])).toEqual([Button.B, Button.START, 0, 0]);
  });

  it("keeps two bot instances separate and gives both the same pre-frame observation", () => {
    const setup: ControllerSetup = [
      { agent: "classic", team: 1 }, { agent: "classic", team: 2 },
      DEFAULT_CONTROLLERS[2], DEFAULT_CONTROLLERS[3],
    ];
    const state = observation(), bots = [new ProbeBot(Button.B), new ProbeBot(Button.C)] as const;
    expect(controllerInputs(setup, state, [Button.A, Button.START], bots)).toEqual([Button.B, Button.C, 0, 0]);
    expect(bots[0].calls[0]?.state).toBe(state);
    expect(bots[1].calls[0]?.state).toBe(state);
    expect(bots[0].calls[0]?.port).toBe(0);
    expect(bots[1].calls[0]?.port).toBe(1);
  });

  it("separates keyboard/on-screen and gamepad when both slots are Human", () => {
    const setup: ControllerSetup = [
      { agent: "human", team: 1 }, { agent: "human", team: 2 },
      DEFAULT_CONTROLLERS[2], DEFAULT_CONTROLLERS[3],
    ];
    expect(controllerInputs(setup, observation(), [Button.A, Button.B],
      [new DiagnosticBot(), new DiagnosticBot()])).toEqual([Button.A, Button.B, 0, 0]);
  });

  it("releases None slots and routes combined human sources to a sole second-slot Human", () => {
    const setup: ControllerSetup = [
      { agent: "none", team: 1 }, { agent: "human", team: 2 },
      DEFAULT_CONTROLLERS[2], DEFAULT_CONTROLLERS[3],
    ];
    expect(controllerInputs(setup, observation(), [Button.A, Button.B],
      [new DiagnosticBot(), new DiagnosticBot()])).toEqual([0, Button.A | Button.B, 0, 0]);
  });

  it("releases ROM AI ownership without reserving a port or running a scripted bot", () => {
    const setup: ControllerSetup = [
      { agent: "rom", team: 1 }, { agent: "human", team: 2 },
      DEFAULT_CONTROLLERS[2], DEFAULT_CONTROLLERS[3],
    ];
    const state = { ...observation(), controllers: [
      { team: 0, slot: -1, player: null }, observation().controllers[1],
      observation().controllers[2], observation().controllers[3],
    ] as const };
    expect(parseControllerSetup(setup)).toEqual(setup);
    expect(assignedTeams(setup)).toEqual([0, 2, 0, 0]);
    expect(controllerRouting(setup, state)).toEqual({ ports: [0, 1, 2, 3], message: "" });
    const bots = [new ProbeBot(Button.C), new ProbeBot(Button.C)] as const;
    expect(controllerInputs(setup, state, [Button.A, Button.B], bots)).toEqual([0, Button.A | Button.B, 0, 0]);
    expect(bots.map((bot) => bot.calls)).toEqual([[], []]);
  });

  it("supports Classic against ROM AI and neutral inputs for ROM AI against ROM AI", () => {
    const setup: ControllerSetup = [
      { agent: "classic", team: 1 }, { agent: "rom", team: 2 },
      DEFAULT_CONTROLLERS[2], DEFAULT_CONTROLLERS[3],
    ];
    const bot = new ProbeBot(Button.C);
    expect(assignedTeams(setup)).toEqual([1, 0, 0, 0]);
    expect(controllerInputs(setup, observation(), [Button.A, Button.B], [bot, null])).toEqual([Button.C, 0, 0, 0]);
    const allRom: ControllerSetup = [{ agent: "rom", team: 1 }, setup[1], setup[2], setup[3]];
    expect(assignedTeams(allRom)).toEqual([0, 0, 0, 0]);
    expect(controllerRouting(allRom, observation()).message).toBe("");
    expect(controllerInputs(allRom, observation(), [Button.A, Button.B], [null, null])).toEqual([0, 0, 0, 0]);
  });

  it("keeps assignment initialization explicit before a match is loaded", () => {
    const empty = { ...observation(), seconds: 0, controllers: [
      { team: 0, slot: -1, player: null }, { team: 0, slot: -1, player: null },
      observation().controllers[2], observation().controllers[3],
    ] as const };
    expect(controllerRouting(DEFAULT_CONTROLLERS, empty).message).toContain("applied to the loaded match");
    expect(controllerInputs(DEFAULT_CONTROLLERS, empty, [Button.START, 0],
      [new DiagnosticBot(), new DiagnosticBot()])).toEqual([Button.START, 0, 0, 0]);
  });

  it("supports two same-team assignments only when the game has assigned both ports", () => {
    const setup: ControllerSetup = [
      { agent: "human", team: 1 }, { agent: "classic", team: 1 },
      DEFAULT_CONTROLLERS[2], DEFAULT_CONTROLLERS[3],
    ];
    expect(controllerRouting(setup, observation()).message).toContain("applied to the loaded match");
    const state = observation();
    const sameTeam = { ...state, controllers: [state.controllers[0], { ...state.controllers[1], team: 1 }, state.controllers[2], state.controllers[3]] as const };
    expect(controllerRouting(setup, sameTeam)).toEqual({ ports: [0, 1, 2, 3], message: "" });
  });

  it("rejects invalid human and bot masks before advancing the core", () => {
    expect(() => controllerInputs(DEFAULT_CONTROLLERS, observation(), [-1, 0],
      [new DiagnosticBot(), new DiagnosticBot()])).toThrow("human");
    expect(() => controllerInputs(DEFAULT_CONTROLLERS, observation(), [0, 0],
      [new ProbeBot(0), new ProbeBot(2)])).toThrow("Bot on controller 2");
    expect(() => controllerInputs(DEFAULT_CONTROLLERS, observation(), [0, 0],
      [null, null])).toThrow("Bot on controller 2 is not initialized");
  });

  it("runs three bots against a Human in controller 4 without sharing human inputs", () => {
    const setup: ControllerSetup = [
      { agent: "classic", team: 1 }, { agent: "classic", team: 1 },
      { agent: "classic", team: 1 }, { agent: "human", team: 2 },
    ];
    const state: Observation = { ...observation(), controllers: [
      { team: 1, slot: 0, player: { x: 0, y: 0 } }, { team: 1, slot: 1, player: { x: 1, y: 0 } },
      { team: 1, slot: 2, player: { x: 2, y: 0 } }, { team: 2, slot: 6, player: { x: 0, y: 0 } },
    ] };
    expect(parseControllerSetup(setup)).toEqual(setup);
    expect(assignedTeams(setup)).toEqual([1, 1, 1, 2]);
    const bots = [new ProbeBot(Button.C), new ProbeBot(Button.B), new ProbeBot(Button.UP), null];
    expect(controllerInputs(setup, state, [Button.A, Button.START], bots)).toEqual([
      Button.C, Button.B, Button.UP, Button.A | Button.START,
    ]);
    expect(bots[2]?.calls[0]?.port).toBe(2);
  });

  it("assigns the two human sources in Human-slot order, including slots 3 and 4", () => {
    const setup: ControllerSetup = [
      { agent: "none", team: 1 }, { agent: "none", team: 2 },
      { agent: "human", team: 1 }, { agent: "human", team: 2 },
    ];
    const state: Observation = { ...observation(), controllers: [
      observation().controllers[2], observation().controllers[3],
      observation().controllers[0], observation().controllers[1],
    ] };
    expect(controllerInputs(setup, state, [Button.A, Button.B], [null, null, null, null])).toEqual([0, 0, Button.A, Button.B]);
    expect(() => parseControllerSetup(setup.map(() => ({ agent: "human", team: 1 })))).toThrow("two Human");
    expect(() => parseControllerSetup(setup.map(() => ({ agent: "classic", team: 1 })))).toThrow("three bots");
  });
});
