import { describe, expect, it } from "vitest";
import { DiagnosticBot } from "../bot";
import {
  Button, gamepadMask, keyboardMask, newlyPressedInput, validMask, type GamepadReading,
} from "../inputs";
import { decodeObservation, ramDigest } from "../observations";
import { MAX_CONTROLLER_EVENTS, MAX_REPLAY_FRAMES, parseReplay } from "../replay";
import { inspectRom } from "../rom";

function gameRam(): Uint8Array<ArrayBuffer> {
  const ram = new Uint8Array(65536);
  const words = new DataView(ram.buffer);
  const set = (address: number, value: number) => words.setInt16(address - 0xff0000, value, true);
  set(0xffc328, 1); set(0xffc320, 0);
  set(0xffc32a, 2); set(0xffc322, 6);
  set(0xffb04a, -30); set(0xffb05e, 40);
  set(0xffb34a, 20); set(0xffb35e, -60);
  set(0xffb74a, 10); set(0xffb75e, 20);
  set(0xffb7aa, -1); set(0xffc466, 1); set(0xffc468, 300);
  set(0xffc6da, 2); set(0xffca3e, 3);
  return ram;
}

describe("Genesis controller contract", () => {
  it("maps Genesis A/B/C onto libretro Y/B/A and preserves release edges", () => {
    expect(keyboardMask(["KeyZ", "KeyX", "KeyC", "Enter"])).toBe(512 | 1 | 256 | 8);
    expect(keyboardMask([])).toBe(0);
    expect(keyboardMask(["ArrowUp", "ArrowLeft"])).toBe(16 | 64);
  });
  it("rejects unknown, fractional and negative mask bits", () => {
    for (const value of [-1, 2, 1024, 65536, 1.5, NaN, "1", null]) expect(validMask(value)).toBe(false);
    expect(validMask(Button.A | Button.C)).toBe(true);
    expect(gamepadMask(null)).toBe(0);
  });
  it("accepts an explicitly mapped non-standard receiver instead of silently ignoring it", () => {
    const gamepad = {
      connected: true, mapping: "" as const, axes: [0, 0],
      buttons: [{ pressed: true, touched: true, value: 1 }],
    };
    const bindings = {
      UP: [], DOWN: [], LEFT: [], RIGHT: [],
      A: [{ kind: "button" as const, index: 0 }], B: [], C: [], START: [],
    };
    expect(gamepadMask(gamepad, bindings)).toBe(Button.A);
    expect(gamepadMask(gamepad)).toBe(0);
    expect(gamepadMask({ ...gamepad, connected: false }, bindings)).toBe(0);
  });
  it("preserves standard gamepad actions, axes, deadzone and released inputs", () => {
    const gamepad: GamepadReading = {
      connected: true, mapping: "standard", axes: [-1, 1],
      buttons: Array.from({ length: 16 }, (_, index) => ({
        pressed: [0, 1, 2, 9].includes(index), touched: false, value: 0,
      })),
    };
    expect(gamepadMask(gamepad)).toBe(Button.A | Button.B | Button.C | Button.START | Button.LEFT | Button.DOWN);
    expect(gamepadMask({ ...gamepad, axes: [-0.3, 0.2], buttons: [] })).toBe(0);
    expect(gamepadMask({ ...gamepad, connected: false })).toBe(0);
  });
  it("learns only fresh presses and axis directions, including opposite directions", () => {
    const neutral: GamepadReading = {
      connected: true, mapping: "", axes: [0, 0],
      buttons: [{ pressed: false, touched: false, value: 0 }],
    };
    const pressed = { ...neutral, buttons: [{ pressed: true, touched: true, value: 1 }] };
    expect(newlyPressedInput(neutral, pressed)).toEqual({ kind: "button", index: 0 });
    expect(newlyPressedInput(pressed, pressed)).toBeNull();
    expect(newlyPressedInput(pressed, neutral)).toBeNull();
    expect(newlyPressedInput(neutral, { ...neutral, axes: [0, -1] }))
      .toEqual({ kind: "axis", index: 1, direction: -1 });
    expect(newlyPressedInput({ ...neutral, axes: [0, -1] }, { ...neutral, axes: [0, 1] }))
      .toEqual({ kind: "axis", index: 1, direction: 1 });
    for (const value of [0.2, 3.28, NaN]) {
      expect(newlyPressedInput(neutral, { ...neutral, axes: [value, 0] })).toBeNull();
    }
  });
});

describe("word-swapped Genesis RAM", () => {
  it("decodes signed coordinates, clock, scores and controller assignments", () => {
    const observation = decodeObservation(gameRam(), 123);
    expect(observation.frame).toBe(123);
    expect(observation.puck).toEqual({ x: 10, y: 20 });
    expect(observation.puckOwner).toBe(-1);
    expect(observation.score).toEqual([2, 3]);
    expect(observation.period).toBe(1);
    expect(observation.seconds).toBe(300);
    expect(observation.controllers[0].player).toEqual({ x: -30, y: 40 });
    expect(observation.controllers[1].player).toEqual({ x: 20, y: -60 });
  });
  it("does not expose an unassigned or invalid selected player", () => {
    const ram = gameRam();
    new DataView(ram.buffer).setInt16(0xc322, -1, true);
    expect(decodeObservation(ram, 0).controllers[1].player).toBeNull();
    expect(() => decodeObservation(new Uint8Array(8), 0)).toThrow("65536");
  });
  it("changes the diagnostic checksum when work RAM changes", () => {
    const ram = gameRam(), before = ramDigest(ram);
    ram[0] = 1;
    expect(ramDigest(ram)).not.toBe(before);
  });
});

describe("diagnostic bot, not Classic", () => {
  it("steers the actual controller-selected player and pulses C", () => {
    const bot = new DiagnosticBot(), observation = decodeObservation(gameRam(), 24);
    expect(bot.act(observation, 0)).toBe(Button.RIGHT | Button.DOWN | Button.C);
    expect(bot.act(observation, 1)).toBe(Button.LEFT | Button.UP | Button.C);
    expect(bot.act({ ...observation, frame: 30 }, 0) & Button.C).toBe(0);
  });
  it("stays neutral before a game/assignment is available", () => {
    expect(new DiagnosticBot().act(decodeObservation(new Uint8Array(65536), 0), 0)).toBe(0);
  });
});

describe("local input replays", () => {
  const replay = {
    format: "nhl94-browser-inputs-v1", coreBuild: "a".repeat(64),
    romHash: "b".repeat(64), inputs: [[0, 0], [Button.START, Button.C]],
  };
  it("round trips button press and release masks without code or ROM data", () => {
    expect(parseReplay(JSON.parse(JSON.stringify(replay)))).toEqual(replay);
  });
  it("rejects malformed headers, oversized frame counts and unsupported inputs", () => {
    for (const value of [
      null, {}, { ...replay, coreBuild: "wrong" }, { ...replay, inputs: [] },
      { ...replay, inputs: [[2, 0]] }, { ...replay, inputs: [[0, 0, 0]] },
      { ...replay, inputs: new Array(MAX_REPLAY_FRAMES + 1).fill([0, 0]) },
    ]) expect(() => parseReplay(value)).toThrow();
  });
  it("preserves a saved-state replay and ordered controller changes, including paused and final changes", () => {
    const match = {
      ...replay, format: "nhl94-browser-inputs-v2", startStateHash: "c".repeat(64),
      controllerEvents: [
        { frame: 0, teams: [1, 2] }, { frame: 1, teams: [2, 2] },
        { frame: 1, teams: [2, 1] }, { frame: 2, teams: [0, 1] },
      ],
    };
    expect(parseReplay(JSON.parse(JSON.stringify(match)))).toEqual(match);
  });
  it("rejects unknown starts, invalid assignments and out-of-order controller events", () => {
    const match = {
      ...replay, format: "nhl94-browser-inputs-v2", startStateHash: "c".repeat(64),
      controllerEvents: [{ frame: 0, teams: [1, 2] }],
    };
    for (const value of [
      { ...match, startStateHash: "not-a-hash" }, { ...match, controllerEvents: [] },
      { ...match, controllerEvents: [{ frame: 1, teams: [1, 2] }] },
      { ...match, controllerEvents: [{ frame: -1, teams: [1, 2] }] },
      { ...match, controllerEvents: [{ frame: 0, teams: [true, 2] }] },
      { ...match, controllerEvents: [{ frame: 0, teams: [1, 3] }] },
      { ...match, controllerEvents: [{ frame: 0, teams: [1] }] },
      { ...match, controllerEvents: [{ frame: 0, teams: [1, 2] }, { frame: 3, teams: [0, 0] }] },
      { ...match, controllerEvents: [{ frame: 0, teams: [1, 2] }, { frame: 2, teams: [0, 0] }, { frame: 1, teams: [1, 2] }] },
      { ...match, controllerEvents: new Array(MAX_CONTROLLER_EVENTS + 1).fill({ frame: 0, teams: [1, 2] }) },
    ]) expect(() => parseReplay(value)).toThrow();
  });
});

describe("ROM validation", () => {
  it("accepts a raw NHL94 header and rejects other formats", () => {
    const bytes = new Uint8Array(1024);
    bytes.set(new TextEncoder().encode("SEGA GENESIS"), 0x100);
    bytes.set(new TextEncoder().encode("NHL Hockey '94"), 0x120);
    expect(inspectRom(bytes)).toContain("NHL Hockey");
    expect(() => inspectRom(new Uint8Array(8))).toThrow();
    bytes[0x100] = 0;
    expect(() => inspectRom(bytes)).toThrow("raw Genesis");
  });
});
