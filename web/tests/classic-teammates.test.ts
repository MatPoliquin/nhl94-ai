import { describe, expect, it, vi } from "vitest";
import { DEFAULT_CLASSIC_CONFIG } from "../classic/config";
import { controllerView } from "../classic/decoder";
import { SharedReceiving } from "../classic/teammate-receiving";
import { TeammateCoordinator } from "../classic/teammates";
import { GameState, neutralAction } from "../classic/state";
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
