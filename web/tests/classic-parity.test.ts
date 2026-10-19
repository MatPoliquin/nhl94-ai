import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { spawnSync } from "node:child_process";
import { gzipSync } from "node:zlib";
import { describe, expect, it } from "vitest";
import { DiagnosticBot } from "../bot";
import { ClassicBot } from "../classic/bot";
import { DEFAULT_CLASSIC_CONFIG, type ClassicConfig } from "../classic/config";
import { Engine, type CoreFactory } from "../engine";
import { decodeDefaultState, DEFAULT_STATE_FILE } from "../start-state";

type Json = null | boolean | number | string | Json[] | { [key: string]: Json };
interface OracleResult {
  actions: number[];
  decisions: string[];
  snapshots: { frame: number; state: Json }[];
}
type Capture = { ram: string } | { changes: number[] };

async function match() {
  const rom = new Uint8Array(await readFile(process.env.NHL94_ROM!));
  const wasm = new Uint8Array(await readFile("public/core/genesis.wasm"));
  const imported: { default: CoreFactory } =
    await import(/* @vite-ignore */ pathToFileURL(resolve("public/core/genesis.js")).href);
  const core = await imported.default({ wasmBinary: wasm, printErr: (message) => console.error(message) });
  const engine = await Engine.create(async () => core, wasm, rom);
  engine.start(await decodeDefaultState(new Uint8Array(await readFile(`public/states/${DEFAULT_STATE_FILE}`))));
  engine.assignControllers([1, 2]);
  return { core, engine };
}

function oracle(config: ClassicConfig, port: 0 | 1, frames: Capture[], snapshots: number[]): OracleResult {
  const input = gzipSync(JSON.stringify({ rom: process.env.NHL94_ROM, config, port, frames, snapshots }));
  const result = spawnSync(process.env.NHL94_PYTHON!, [resolve("scripts/classic-reference.py")], {
    input, encoding: "utf8", timeout: 120000, maxBuffer: 16 * 1024 * 1024,
  });
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(`Python Classic reference failed:\n${result.stderr}`);
  return JSON.parse(result.stdout) as OracleResult;
}

function compareState(actual: Json, expected: Json, path = "state"): void {
  if (typeof actual === "number" && typeof expected === "number" && !Number.isInteger(expected)) {
    expect(actual, path).toBeCloseTo(expected, 11);
  } else if (Array.isArray(expected)) {
    expect(Array.isArray(actual), path).toBe(true);
    if (!Array.isArray(actual)) throw new Error(`${path} must be an array.`);
    expect(actual.length, path).toBe(expected.length);
    expected.forEach((item, index) => compareState(actual[index]!, item, `${path}[${index}]`));
  } else if (expected !== null && typeof expected === "object") {
    if (actual === null || typeof actual !== "object" || Array.isArray(actual)) throw new Error(`${path} must be an object.`);
    for (const [key, value] of Object.entries(expected)) {
      expect(key in actual, `${path}.${key}`).toBe(true);
      compareState(actual[key]!, value, `${path}.${key}`);
    }
  } else {
    expect(actual, path).toEqual(expected);
  }
}

async function compareTrace(config: ClassicConfig, port: 0 | 1, count: number): Promise<void> {
  const { core, engine } = await match();
  try {
    engine.assignGoalies(config.ai_goalie ? [0, 0] : [1, 1]);
    const bot = new ClassicBot(config), opponent = new DiagnosticBot();
    const frames: Capture[] = [], actions: number[] = [], snapshots = new Map<number, Json>();
    let previous: Uint8Array | undefined;
    for (let frame = 0; frame < count; frame++) {
      const bytes = core.HEAPU8.slice(core._arena_ram(), core._arena_ram() + 65536);
      if (!previous) frames.push({ ram: Buffer.from(bytes).toString("base64") });
      else {
        const changes: number[] = [];
        for (let index = 0; index < bytes.length; index++) {
          if (bytes[index] !== previous[index]) changes.push(index, bytes[index]!);
        }
        frames.push({ changes });
      }
      previous = bytes;
      const observation = engine.observe(), state = engine.classicState();
      if ([0, Math.floor(count / 2), count - 1].includes(frame)) snapshots.set(frame, JSON.parse(JSON.stringify(state)) as Json);
      const action = bot.act(observation, port, state);
      const other = opponent.act(observation, port === 0 ? 1 : 0);
      actions.push(action);
      engine.step(port === 0 ? action : other, port === 0 ? other : action);
    }
    const reference = oracle(config, port, frames, [...snapshots.keys()]);
    for (const snapshot of reference.snapshots) compareState(snapshots.get(snapshot.frame)!, snapshot.state);
    expect(reference.actions).toHaveLength(count);
    for (let frame = 0; frame < count; frame++) {
      expect(actions[frame], `port ${port}, frame ${frame}, Python decision ${reference.decisions[frame]}`).toBe(reference.actions[frame]);
    }
  } finally { engine.dispose(); }
}

describe.skipIf(!process.env.NHL94_ROM || !process.env.NHL94_PYTHON)("Classic Python/native-WASM parity", () => {
  it("matches both independent Classic ports throughout a complete first period", async () => {
    const { core, engine } = await match();
    try {
      const bots = [new ClassicBot(DEFAULT_CLASSIC_CONFIG), new ClassicBot(DEFAULT_CLASSIC_CONFIG)] as const;
      const frames: Capture[] = [], actions: [number[], number[]] = [[], []];
      let previous: Uint8Array | undefined;
      while (engine.observe().seconds > 0 && engine.frame < 30000) {
        const bytes = core.HEAPU8.slice(core._arena_ram(), core._arena_ram() + 65536);
        if (!previous) frames.push({ ram: Buffer.from(bytes).toString("base64") });
        else {
          const changes: number[] = [];
          for (let index = 0; index < bytes.length; index++) {
            if (bytes[index] !== previous[index]) changes.push(index, bytes[index]!);
          }
          frames.push({ changes });
        }
        previous = bytes;
        const observation = engine.observe(), state = engine.classicState();
        const before = engine.frame % 100 === 0 ? JSON.stringify(state) : undefined;
        const first = bots[0].act(observation, 0, state), second = bots[1].act(observation, 1, state);
        if (before) expect(JSON.stringify(state)).toBe(before);
        actions[0].push(first); actions[1].push(second);
        engine.step(first, second);
      }
      expect(engine.observe().seconds, "Both bots must play to the end of the period.").toBe(0);
      expect(engine.frame).toBeGreaterThan(1800);
      for (const port of [0, 1] as const) {
        const reference = oracle(DEFAULT_CLASSIC_CONFIG, port, frames, []);
        for (let frame = 0; frame < frames.length; frame++) {
          expect(actions[port][frame], `complete period: port ${port}, frame ${frame}, ${reference.decisions[frame]}`)
            .toBe(reference.actions[frame]);
        }
      }
    } finally { engine.dispose(); }
  }, 300000);

  for (const port of [0, 1] as const) {
    it(`matches decoded state and every native-frame input on ${port === 0 ? "Home" : "Away"}`, async () => {
      await compareTrace(DEFAULT_CLASSIC_CONFIG, port, 1800);
    }, 180000);
  }
  for (const decision_hz of [60, 15, 10, 6] as const) {
    for (const ai_goalie of [false, true]) {
      for (const player_ratings_consideration of [false, true]) {
        it(`matches ${decision_hz} Hz, goalie ${ai_goalie}, ratings ${player_ratings_consideration}`, async () => {
          await compareTrace({ decision_hz, ai_goalie, player_ratings_consideration }, decision_hz === 10 ? 1 : 0, 300);
        }, 180000);
      }
    }
  }
});
