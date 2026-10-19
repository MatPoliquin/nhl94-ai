import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { describe, expect, it } from "vitest";
import { Engine, type CoreFactory } from "../engine";
import { assignedTeams, DEFAULT_CONTROLLERS } from "../controllers";
import { Button } from "../inputs";
import {
  decodeDefaultState, DEFAULT_STATE_BYTES, DEFAULT_STATE_FILE, DEFAULT_STATE_HASH,
} from "../start-state";
import { sha256 } from "../rom";

describe("bundled native starting state", () => {
  it("keeps the supplied compressed file unchanged and validates its exact identity", async () => {
    const compressed = new Uint8Array(await readFile(`public/states/${DEFAULT_STATE_FILE}`));
    expect(await sha256(compressed)).toBe(DEFAULT_STATE_HASH);
    const decoded = await decodeDefaultState(compressed);
    expect(decoded.length).toBe(DEFAULT_STATE_BYTES);
    const words = new DataView(decoded.buffer);
    expect(words.getUint16(16 + 0xc468, true)).toBe(300);
    expect(words.getUint16(16 + 0xc328, true)).toBe(1);
    expect(words.getUint16(16 + 0xc32a, true)).toBe(0);
  });

  it("rejects corrupt or substituted states instead of silently booting another game", async () => {
    const compressed = new Uint8Array(await readFile(`public/states/${DEFAULT_STATE_FILE}`));
    compressed[100] = compressed[100]! ^ 1;
    await expect(decodeDefaultState(compressed)).rejects.toThrow("damaged or incompatible");
    await expect(decodeDefaultState(new Uint8Array(0))).rejects.toThrow("starting state");
  });
});

describe.skipIf(!process.env.NHL94_ROM)("default match and native controller assignment", () => {
  async function match() {
    const rom = new Uint8Array(await readFile(process.env.NHL94_ROM!));
    const wasm = new Uint8Array(await readFile("public/core/genesis.wasm"));
    const imported: { default: CoreFactory } =
      await import(/* @vite-ignore */ pathToFileURL(resolve("public/core/genesis.js")).href);
    const core = await imported.default({ wasmBinary: wasm, printErr: (message) => console.error(message) });
    const engine = await Engine.create(async () => core, wasm, rom);
    const snapshot = await decodeDefaultState(new Uint8Array(await readFile(`public/states/${DEFAULT_STATE_FILE}`)));
    engine.start(snapshot);
    const word = (address: number) => new DataView(core.HEAPU8.buffer)
      .getUint16(core._arena_ram() + address - 0xff0000, true);
    const flag = (slot: number) => core.HEAPU8[core._arena_ram() + ((0xb04a + slot * 0x80 + 0x62) ^ 1)]!;
    return { engine, core, word, flag };
  }

  it("resumes the exact native save, assigns both teams and plays without native pointer corruption", async () => {
    const { engine, word, flag } = await match();
    try {
      expect(engine.frame).toBe(0);
      expect(engine.observe().seconds).toBe(300);
      expect(engine.video().pixels.some((byte, index) => index % 4 !== 3 && byte > 0)).toBe(true);
      engine.assignControllers([1, 2]);
      const initial = engine.observe();
      expect(initial.controllers.map((controller) => controller.team)).toEqual([1, 2, 0, 0]);
      expect(initial.controllers[0].slot).toBe(2);
      expect(initial.controllers[1].slot).toBe(8);
      expect(flag(2) & 8).toBe(8);
      expect(flag(8) & 8).toBe(8);
      expect(word(0xffd04a)).toBe(4);
      expect([word(0xffd05a), word(0xffd05c)]).toEqual([0, 0]);
      for (let frame = 0; frame < 1800; frame++) engine.step(frame % 60 < 15 ? Button.RIGHT : 0, 0);
      expect(engine.observe().seconds).toBeLessThan(300);
      expect(engine.observe().controllers.map((controller) => controller.team)).toEqual([1, 2, 0, 0]);
      expect(engine.audio().length).toBeGreaterThan(0);
    } finally { engine.dispose(); }
  }, 120000);

  it("swaps sides, supports teammates and hands None slots back to native CPU ownership", async () => {
    const { engine, word, flag } = await match();
    try {
      for (const teams of [[1, 2], [2, 1], [1, 1], [2, 2], [0, 1], [0, 0], [1, 2]] as const) {
        engine.assignControllers(teams);
        const state = engine.observe();
        expect(state.controllers.map((controller) => controller.team)).toEqual([...teams, 0, 0]);
        const selections = state.controllers.map((controller) => controller.slot);
        for (const port of [0, 1] as const) {
          const slot = selections[port]!;
          if (teams[port] === 0) expect(slot).toBe(-1);
          else {
            expect(slot).toBeGreaterThanOrEqual((teams[port] - 1) * 6);
            expect(slot).toBeLessThan((teams[port] - 1) * 6 + 6);
            expect(flag(slot) & 8).toBe(8);
          }
        }
        if (teams[0] && teams[1]) expect(selections[0]).not.toBe(selections[1]);
        for (let slot = 0; slot < 12; slot++) {
          if (!selections.includes(slot)) expect(flag(slot) & 8).toBe(0);
        }
        expect(word(0xffc32c)).toBe(0);
        expect(word(0xffc32e)).toBe(0);
        for (let frame = 0; frame < 30; frame++) engine.step(0, 0);
      }
    } finally { engine.dispose(); }
  }, 120000);

  it("applies directional input to the newly assigned away controller", async () => {
    const { engine } = await match();
    try {
      engine.assignControllers([2, 0]);
      const before = engine.save();
      for (let frame = 0; frame < 240; frame++) engine.step(0, 0);
      const neutral = engine.observe();
      engine.restore(before, 0);
      for (let frame = 0; frame < 240; frame++) engine.step(Button.RIGHT, 0);
      const skating = engine.observe();
      expect(skating.controllers[0].team).toBe(2);
      expect(skating.controllers[0].player).not.toEqual(neutral.controllers[0].player);
    } finally { engine.dispose(); }
  }, 120000);

  it("lets both ROM AI teams skate autonomously with no controller inputs", async () => {
    const { engine, flag } = await match();
    try {
      engine.assignControllers([1, 2]);
      engine.assignControllers(assignedTeams([
        { agent: "rom", team: 1 }, { agent: "rom", team: 2 },
        DEFAULT_CONTROLLERS[2], DEFAULT_CONTROLLERS[3],
      ]));
      expect(engine.observe().controllers.map((controller) => [controller.team, controller.slot]))
        .toEqual([[0, -1], [0, -1], [0, -1], [0, -1]]);
      for (let slot = 0; slot < 12; slot++) expect(flag(slot) & 8).toBe(0);
      const before = engine.classicState();
      const positions = [before.team1, before.team2].map((team) => team.players.map((player) => [player.x, player.y]));
      for (let frame = 0; frame < 600; frame++) engine.step(0, 0);
      const after = engine.classicState();
      for (const [index, team] of [after.team1, after.team2].entries()) {
        expect(team.players.map((player) => [player.x, player.y])).not.toEqual(positions[index]);
      }
      expect(engine.observe().seconds).toBeLessThan(300);
      expect(engine.observe().controllers.map((controller) => controller.team)).toEqual([0, 0, 0, 0]);
    } finally { engine.dispose(); }
  }, 120000);

  it("changes native goalie modes without advancing time or changing player ratings", async () => {
    const { engine, word, core } = await match();
    try {
      engine.assignControllers([1, 2]);
      const ram = () => core.HEAPU8.slice(core._arena_ram(), core._arena_ram() + 65536);
      const before = ram();
      expect(engine.goalieModes()).toEqual([0, 0]);
      engine.assignGoalies([0, 1]);
      expect(engine.frame).toBe(0);
      expect([word(0xffd05a), word(0xffd05c)]).toEqual([0, 1]);
      const after = ram();
      const changes = [...after.keys()].filter((index) => before[index] !== after[index]);
      expect(changes).toEqual([0xd05c]);
      engine.assignGoalies([1, 0]);
      expect(engine.goalieModes()).toEqual([1, 0]);
      for (let frame = 0; frame < 120; frame++) engine.step(0, 0);
      expect(engine.goalieModes()).toEqual([1, 0]);
      expect(engine.observe().controllers.map((controller) => controller.team)).toEqual([1, 2, 0, 0]);
    } finally { engine.dispose(); }
  }, 120000);

  it("delivers distinct native inputs to all four EA 4-Way Play controllers", async () => {
    const { engine, word, flag, core } = await match();
    try {
      engine.assignControllers([1, 1, 1, 2]);
      engine.assignGoalies([1, 0]);
      expect(word(0xffd046)).toBe(1);
      const initial = engine.observe();
      expect(initial.controllers.map((controller) => controller.team)).toEqual([1, 1, 1, 2]);
      const slots = initial.controllers.map((controller) => controller.slot);
      expect(new Set(slots).size).toBe(4);
      for (const slot of slots) expect(flag(slot) & 8).toBe(8);
      const initialSnapshot = engine.save();
      for (const port of [0, 1, 2, 3] as const) {
        for (const [button, native] of [[Button.RIGHT, 8], [Button.B, 16], [Button.C, 32]]) {
          engine.restore(initialSnapshot, 0);
          const inputs: [number, number, number, number] = [0, 0, 0, 0];
          inputs[port] = button!;
          for (let frame = 0; frame < 3; frame++) engine.step(...inputs);
          const channels = [0, 1, 2, 3].map((p) => core.HEAPU8[core._arena_ram() + ((0xbef6 + p) ^ 1)]);
          expect(channels, `controller ${port + 1}, button ${button}`).toEqual([0, 1, 2, 3].map((p) => p === port ? 255 ^ native! : 255));
        }
      }
      engine.restore(initialSnapshot, 0);
      let allMoved = false;
      for (let frame = 0; frame < 2000 && !allMoved; frame++) {
        engine.step(frame % 8 < 4 ? Button.B : 0, 0, 0, 0);
        if (frame < 200 || frame % 40 || engine.observe().transient) continue;
        const snapshot = engine.save(), savedFrame = engine.frame;
        const liveSlots = engine.observe().controllers.map((controller) => controller.slot);
        if (new Set(liveSlots).size !== 4 || liveSlots.some((slot) => slot < 0 || slot % 6 === 5)) continue;
        const distances = [];
        for (const port of [0, 1, 2, 3] as const) {
          const positions = [];
          for (const direction of [Button.LEFT, Button.RIGHT]) {
            engine.restore(snapshot, savedFrame);
            const inputs: [number, number, number, number] = [0, 0, 0, 0];
            inputs[port] = direction;
            for (let tick = 0; tick < 40; tick++) engine.step(...inputs);
            const moved = engine.classicState();
            const player = (port === 3 ? moved.team2 : moved.team1).get_player_by_scnum(liveSlots[port]!)!;
            positions.push([player.x, player.y]);
          }
          distances.push(Math.hypot(positions[0]![0]! - positions[1]![0]!, positions[0]![1]! - positions[1]![1]!));
        }
        engine.restore(snapshot, savedFrame);
        allMoved = distances.every((distance) => distance > 2);
      }
      expect(allMoved, "all four controllers must independently move a live skater").toBe(true);
      engine.assignControllers([1, 1, 1, 0]);
      expect(engine.observe().controllers.map((controller) => controller.team)).toEqual([1, 1, 1, 0]);
      expect(word(0xffc326)).toBe(0xffff);
      engine.assignControllers([1, 2, 0, 0]);
      expect(word(0xffd046)).toBe(0);
      for (let frame = 0; frame < 120; frame++) engine.step(0, Button.LEFT);
      expect(engine.observe().controllers.map((controller) => controller.team)).toEqual([1, 2, 0, 0]);
      engine.restore(initialSnapshot, 0);
      expect(engine.observe().controllers.map((controller) => controller.team)).toEqual([1, 1, 1, 2]);
      for (let frame = 0; frame < 3; frame++) engine.step(0, 0, 0, Button.C);
      expect([0, 1, 2, 3].map((p) => core.HEAPU8[core._arena_ram() + ((0xbef6 + p) ^ 1)]))
        .toEqual([255, 255, 255, 223]);
    } finally { engine.dispose(); }
  }, 120000);
});
