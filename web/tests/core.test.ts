import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { describe, expect, it } from "vitest";
import { Engine, type CoreFactory } from "../engine";
import { Button } from "../inputs";
import { identifyRom } from "../rom";

const romPath = process.env.NHL94_ROM;

describe.skipIf(!romPath)("real Genesis Plus GX WASM (local ROM only)", () => {
  it("replays 10,000 frames identically and restores a local snapshot", async () => {
    const rom = new Uint8Array(await readFile(romPath!));
    await identifyRom(rom);
    const wasm = new Uint8Array(await readFile("public/core/genesis.wasm"));
    const imported: { default: CoreFactory } =
      await import(/* @vite-ignore */ pathToFileURL(resolve("public/core/genesis.js")).href);
    const first = await Engine.create(imported.default, wasm, rom);
    const second = await Engine.create(imported.default, wasm, rom);
    try {
      for (let frame = 0; frame < 10000; frame++) {
        const a = frame % 240 < 2 ? Button.START : 0;
        const b = frame % 80 < 20 ? Button.LEFT | Button.C : 0;
        first.step(a, b);
        second.step(a, b);
        if (frame % 1000 === 999) {
          expect(first.observe()).toEqual(second.observe());
          expect(first.stats()).toEqual(second.stats());
        }
      }
      expect(first.frame).toBe(10000);
      expect([256, 320]).toContain(first.video().width);
      expect(first.video().height).toBe(224);
      expect(first.video().pixels).toEqual(second.video().pixels);
      expect(first.video().pixels.some((byte, index) => index % 4 !== 3 && byte > 0)).toBe(true);
      expect(first.audio().length).toBeGreaterThan(0);
      expect(first.fps).toBeGreaterThan(59);
      expect(first.fps).toBeLessThan(61);
      const snapshot = first.save(), savedFrame = first.frame;
      for (let index = 0; index < 30; index++) first.step(Button.RIGHT, Button.B);
      const observation = first.observe(), image = first.video().pixels, stats = first.stats();
      first.restore(snapshot, savedFrame);
      for (let index = 0; index < 30; index++) first.step(Button.RIGHT, Button.B);
      expect(first.observe()).toEqual(observation);
      expect(first.video().pixels).toEqual(image);
      expect(first.stats()).toEqual(stats);
    } finally {
      first.dispose(); second.dispose();
    }
    expect(() => first.step(0, 0)).toThrow("disposed");
  }, 120000);
});
