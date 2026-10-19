import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { describe, expect, it } from "vitest";
import { Engine, type CoreFactory } from "../engine";
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
      expect(initial.controllers.map((controller) => controller.team)).toEqual([1, 2]);
      expect(initial.controllers[0].slot).toBe(2);
      expect(initial.controllers[1].slot).toBe(8);
      expect(flag(2) & 8).toBe(8);
      expect(flag(8) & 8).toBe(8);
      expect(word(0xffd04a)).toBe(4);
      expect([word(0xffd05a), word(0xffd05c)]).toEqual([0, 0]);
      for (let frame = 0; frame < 1800; frame++) engine.step(frame % 60 < 15 ? Button.RIGHT : 0, 0);
      expect(engine.observe().seconds).toBeLessThan(300);
      expect(engine.observe().controllers.map((controller) => controller.team)).toEqual([1, 2]);
      expect(engine.audio().length).toBeGreaterThan(0);
    } finally { engine.dispose(); }
  }, 120000);

  it("swaps sides, supports teammates and hands None slots back to native CPU ownership", async () => {
    const { engine, word, flag } = await match();
    try {
      for (const teams of [[1, 2], [2, 1], [1, 1], [2, 2], [0, 1], [0, 0], [1, 2]] as const) {
        engine.assignControllers(teams);
        const state = engine.observe();
        expect(state.controllers.map((controller) => controller.team)).toEqual(teams);
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
});
