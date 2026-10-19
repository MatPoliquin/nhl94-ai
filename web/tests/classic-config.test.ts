import { describe, expect, it } from "vitest";
import { DECISION_RATES, DEFAULT_CLASSIC_CONFIG, controlInterval, decisionState, parseClassicConfig, parseClassicConfigText } from "../classic/config";
import { GameState } from "../classic/state";
import { GenesisRam } from "../classic/ram";
import { parseReplay } from "../replay";

describe("Classic configuration", () => {
  it("reads the commented Python configuration and uses the same defaults and input cadence", () => {
    expect(DEFAULT_CLASSIC_CONFIG).toEqual({ decision_hz: 60, ai_goalie: true, player_ratings_consideration: true });
    expect(parseClassicConfigText('{/* test */ "ai_goalie": false // test\n}')).toEqual({ ...DEFAULT_CLASSIC_CONFIG, ai_goalie: false });
    expect(parseClassicConfig({})).toEqual(DEFAULT_CLASSIC_CONFIG);
    expect(DECISION_RATES.map((decision_hz) => controlInterval({ ...DEFAULT_CLASSIC_CONFIG, decision_hz }))).toEqual([1, 4, 6, 10]);
    for (const value of [null, [], { other: true }, { decision_hz: true }, { decision_hz: 12 }, { ai_goalie: 0 }, { player_ratings_consideration: "false" }]) {
      expect(() => parseClassicConfig(value)).toThrow("Classic");
    }
    expect(() => parseClassicConfigText('{"ai_goalie": true,}')).toThrow();
  });

  it("neutralizes both teams without changing game state, energy, handedness or velocities", () => {
    const state = new GameState();
    for (const team of [state.team1, state.team2]) {
      for (const player of [...team.players, team.goalie]) {
        Object.assign(player, { speed: 30, agility: 2, weight: 120, energy: 1234, handedness: 1, motion_x: -1.25, shot_accuracy: 24 });
        player.input_state = { speed: 30, weight: 120, handedness: 1, vx: 777 };
      }
    }
    const before = JSON.stringify(state);
    expect(decisionState(state, true)).toBe(state);
    const neutral = decisionState(state, false);
    for (const team of [neutral.team1, neutral.team2]) {
      for (const player of [...team.players, team.goalie]) {
        expect(player).toMatchObject({ speed: 15, agility: 15, weight: 64, shot_accuracy: 15, energy: 1234, handedness: 1, motion_x: -1.25 });
        expect(player.input_state).toEqual({ speed: 15, weight: 64, handedness: 1, vx: 777 });
      }
    }
    expect(JSON.stringify(state)).toBe(before);
  });
});

describe("Classic RAM and replay configuration", () => {
  it("reads CPU bytes, words and signed fixed-point longs from word-swapped memory", () => {
    const bytes = new Uint8Array(65536), view = new DataView(bytes.buffer);
    view.setUint16(0x100, 0xfffe, true);
    view.setUint16(0x102, 0x8001, true);
    const ram = new GenesisRam(bytes);
    expect(ram.u8(0xff0100)).toBe(255);
    expect(ram.u8(0xff0101)).toBe(254);
    expect(ram.i8(0xff0101)).toBe(-2);
    expect(ram.i16(0xff0100)).toBe(-2);
    expect(ram.u32(0xff0100)).toBe(0xfffe8001);
    expect(ram.i32(0xff0100)).toBe(-98303);
    expect(() => new GenesisRam(new Uint8Array(1))).toThrow("64 KiB");
  });

  it("validates v3 goalie changes including events after the last input, while retaining v2", () => {
    const base = { coreBuild: "a".repeat(64), romHash: "b".repeat(64), startStateHash: "c".repeat(64), inputs: [[0, 0]] };
    const value = { ...base, format: "nhl94-browser-inputs-v3", configurationEvents: [
      { frame: 0, teams: [1, 2], goalieModes: [0, 1] },
      { frame: 1, teams: [1, 2], goalieModes: [1, 0] },
    ] };
    expect(parseReplay(value)).toEqual(value);
    for (const modes of [[0, 2], [true, false], [0], null]) {
      expect(() => parseReplay({ ...value, configurationEvents: [{ frame: 0, teams: [1, 2], goalieModes: modes }] })).toThrow("goalie");
    }
    expect(parseReplay({ ...base, format: "nhl94-browser-inputs-v2", controllerEvents: [{ frame: 0, teams: [1, 2] }] }).format).toBe("nhl94-browser-inputs-v2");
  });
});
