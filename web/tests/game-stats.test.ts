import { readFile } from "node:fs/promises";
import { describe, expect, it } from "vitest";
import { decodeGameStats, GAME_STAT_ROWS, type TeamStats } from "../game-stats";
import { identifyRom } from "../rom";

const offsets = {
  score: 0x0c, shots: 0x00, powerPlayGoals: 0x02, powerPlayOpportunities: 0x04,
  powerPlaySeconds: 0x352, powerPlayShots: 0x354, shorthandedGoals: 0x356,
  breakawayGoals: 0x35a, breakawayAttempts: 0x358,
  oneTimerGoals: 0x35e, oneTimerAttempts: 0x35c,
  penaltyShotGoals: 0x362, penaltyShotAttempts: 0x360,
  faceoffsWon: 0x0e, checks: 0x10, penalties: 0x06, penaltyMinutes: 0x08,
  attackZoneSeconds: 0x0a, passesCompleted: 0x14, passAttempts: 0x12,
} satisfies Record<keyof TeamStats, number>;

const example: TeamStats = {
  score: 2, shots: 9, powerPlayGoals: 1, powerPlayOpportunities: 3,
  powerPlaySeconds: 61, powerPlayShots: 4, shorthandedGoals: 1,
  breakawayGoals: 1, breakawayAttempts: 5, oneTimerGoals: 2, oneTimerAttempts: 7,
  penaltyShotGoals: 1, penaltyShotAttempts: 2, faceoffsWon: 12, checks: 18,
  penalties: 2, penaltyMinutes: 4, attackZoneSeconds: 125, passesCompleted: 11, passAttempts: 17,
};

describe("native pause-menu team statistics", () => {
  it("reads every home/away counter as a word with the correct byte offset", () => {
    const storage = new Uint8Array(65544), ram = storage.subarray(4, 65540);
    const words = new DataView(ram.buffer, ram.byteOffset, ram.byteLength);
    for (const [name, offset] of Object.entries(offsets)) {
      const key = name as keyof TeamStats;
      words.setUint16(0xc6ce + offset, example[key], true);
      words.setUint16(0xca32 + offset, example[key] + 100, true);
    }
    words.setUint16(0xc6cc, 0xbeef, true);
    const [home, away] = decodeGameStats(ram);
    expect(home).toEqual(example);
    for (const name of Object.keys(offsets)) {
      const key = name as keyof TeamStats;
      expect(away[key]).toBe(example[key] + 100);
    }
    expect(home.shots).toBe(9);
    expect(home.attackZoneSeconds).toBe(125);
    expect(away.faceoffsWon).not.toBe(away.score);
  });

  it("does not sign-extend counters or accept a truncated RAM snapshot", () => {
    const ram = new Uint8Array(65536);
    new DataView(ram.buffer).setUint16(0xc6ce, 65535, true);
    expect(decodeGameStats(ram)[0].shots).toBe(65535);
    expect(() => decodeGameStats(new Uint8Array(65535))).toThrow("65536");
  });

  it("formats all 15 menu rows with goals/attempts, percentages and actual clock seconds", () => {
    expect(GAME_STAT_ROWS.map((row) => [row.label, row.format(example)])).toEqual([
      ["Score", "2"],
      ["Shots on goal", "9"],
      ["Shooting %", "22%"],
      ["Power play (goals/opportunities)", "1 / 3"],
      ["Power play time", "1:01"],
      ["Power play shots", "4"],
      ["Shorthanded goals", "1"],
      ["Breakaways (goals/attempts)", "1 / 5"],
      ["One-timers (goals/attempts)", "2 / 7"],
      ["Penalty shots (goals/attempts)", "1 / 2"],
      ["Faceoffs won", "12"],
      ["Body checks", "18"],
      ["Penalties / minutes", "2 / 4"],
      ["Attack zone time", "2:05"],
      ["Passes (completed/attempted)", "11 / 17 (64%)"],
    ]);
  });

  it("matches the native zero-denominator branch without NaN or invented precision", () => {
    const empty = decodeGameStats(new Uint8Array(65536))[0];
    expect(GAME_STAT_ROWS.find((row) => row.id === "shooting")?.format(empty)).toBe("0%");
    expect(GAME_STAT_ROWS.find((row) => row.id === "passing")?.format(empty)).toBe("0 / 0 (0%)");
    expect(GAME_STAT_ROWS.find((row) => row.id === "shooting")?.format({ ...empty, score: 1 })).toBe("100%");
  });
});

describe.skipIf(!process.env.NHL94_ROM)("original ROM game-stat table", () => {
  it("verifies the complete native row list and offset pairs against the local ROM", async () => {
    const rom = new Uint8Array(await readFile(process.env.NHL94_ROM!));
    await identifyRom(rom);
    const view = new DataView(rom.buffer, rom.byteOffset, rom.byteLength);
    const rows: [string, number, number][] = [];
    let address = 0x0fdfaa;
    for (let index = 0; index < 15; index++) {
      const size = view.getUint16(address);
      rows.push([
        new TextDecoder("ascii").decode(rom.subarray(address + 2, address + size)).replace(/\0+$/, ""),
        view.getUint16(address + size), view.getUint16(address + size + 2),
      ]);
      address += size + 4;
    }
    expect(rows).toEqual([
      ["Score", offsets.score, 0xffff], ["Shots", offsets.shots, 0xffff],
      ["Shooting Pct", 0xffff, 0xffff],
      ["Power Play", offsets.powerPlayGoals, offsets.powerPlayOpportunities],
      ["PP Minutes", offsets.powerPlaySeconds, 0xffff],
      ["PP Shots", offsets.powerPlayShots, 0xffff],
      ["SH Goals", offsets.shorthandedGoals, 0xffff],
      ["Breakaways", offsets.breakawayGoals, offsets.breakawayAttempts],
      ["One-Timers", offsets.oneTimerGoals, offsets.oneTimerAttempts],
      ["Penalty Shots", offsets.penaltyShotGoals, offsets.penaltyShotAttempts],
      ["Faceoffs Won", offsets.faceoffsWon, 0xffff], ["Body Checks", offsets.checks, 0xffff],
      ["Penalties", offsets.penalties, offsets.penaltyMinutes],
      ["Attack Zone", offsets.attackZoneSeconds, 0xffff],
      ["Passing", offsets.passesCompleted, offsets.passAttempts],
    ]);
    expect(GAME_STAT_ROWS).toHaveLength(rows.length);
  });
});
