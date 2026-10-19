import { afterEach, describe, expect, it, vi } from "vitest";
import { automaticGamepadProfile, GAMEPAD_PROFILES_KEY, GamepadProfiles, parseGamepadProfiles } from "../gamepad-profiles";
import { Button, gamepadMask } from "../inputs";

function receiver() {
  return {
    id: "6B controller (Vendor: 0ca3 Product: 0024)", index: 0, mapping: "" as const, connected: true,
    buttons: Array.from({ length: 10 }, () => ({ pressed: false, touched: false, value: 0 })),
    axes: [0, 0, 0],
  };
}

function memoryStorage(): Storage {
  const values = new Map<string, string>();
  return {
    get length() { return values.size; },
    clear: () => values.clear(), key: (index) => [...values.keys()][index] ?? null,
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => { values.set(key, value); },
    removeItem: (key) => { values.delete(key); },
  };
}

afterEach(() => { vi.unstubAllGlobals(); vi.restoreAllMocks(); });

describe("automatic M30 receiver profile", () => {
  it("recognizes Firefox and Chromium identities only with the supported raw layout", () => {
    const pad = receiver();
    for (const id of [pad.id, "0ca3-0024-6B controller", "6B controller (Vendor: 0CA3 Product: 0024)"]) {
      expect(automaticGamepadProfile({ ...pad, id })?.label).toBe("M30 / 6B receiver");
    }
    for (const device of [
      { ...pad, id: "Unknown M30" }, { ...pad, id: "6B controller (Vendor: ffff Product: 0024)" },
      { ...pad, axes: [0, 0] }, { ...pad, buttons: pad.buttons.slice(0, 9) },
    ]) expect(automaticGamepadProfile(device)).toBeUndefined();
    expect(automaticGamepadProfile({ ...pad, mapping: "standard" })?.label).toBe("standard");
  });

  it("uses physical A/B/C, Start 9 and the two digital D-pad axes, ignoring axis 2", () => {
    const pad = receiver(), profile = automaticGamepadProfile(pad)!;
    for (const [index, expected] of [[0, Button.A], [1, Button.B], [5, Button.C], [9, Button.START]] as const) {
      pad.buttons[index]!.pressed = true;
      expect(gamepadMask(pad, profile.bindings)).toBe(expected);
      pad.buttons[index]!.pressed = false;
    }
    for (const [axes, expected] of [
      [[-1, -1, 1], Button.UP | Button.LEFT], [[1, 1, -1], Button.DOWN | Button.RIGHT],
      [[0, 0, 1], 0], [[0.3, -0.3, 0], 0],
    ] as const) {
      pad.axes = [...axes];
      expect(gamepadMask(pad, profile.bindings)).toBe(expected);
    }
    pad.buttons[8]!.pressed = true;
    pad.buttons[2]!.pressed = true;
    expect(gamepadMask(pad, profile.bindings)).toBe(0);
  });
});

describe("persistent gamepad mappings", () => {
  it("restores custom mappings across reloads and controller index changes, without matching other layouts", () => {
    const storage = memoryStorage();
    vi.stubGlobal("localStorage", storage);
    const pad = receiver(), bindings = automaticGamepadProfile(pad)!.bindings;
    const saved = new GamepadProfiles();
    saved.set(pad, bindings);
    expect(storage.getItem(GAMEPAD_PROFILES_KEY)).not.toBeNull();
    const restored = new GamepadProfiles();
    const reindexed = { ...pad, index: 3 };
    expect(restored.get(reindexed)).toEqual(bindings);
    expect(restored.get({ ...pad, axes: [0, 0] })).toBeUndefined();
    expect(restored.get({ ...pad, mapping: "standard" })).toBeUndefined();
    expect(restored.get({ ...pad, id: "Other receiver" })).toBeUndefined();
    restored.delete(pad);
    expect(new GamepadProfiles().get(pad)).toBeUndefined();
  });

  it("rejects invalid versions, missing actions and out-of-range bindings", () => {
    const pad = receiver(), bindings = automaticGamepadProfile(pad)!.bindings;
    const profile = { device: { id: pad.id, mapping: pad.mapping, buttons: 10, axes: 3 }, bindings };
    for (const value of [
      null, { version: 2, profiles: [] }, { version: 1, profiles: [profile, profile] },
      { version: 1, profiles: [{ ...profile, bindings: { ...bindings, UP: [] } }] },
      { version: 1, profiles: [{ ...profile, bindings: { ...bindings, A: [{ kind: "button", index: 10 }] } }] },
      { version: 1, profiles: [{ ...profile, bindings: { ...bindings, UP: [{ kind: "axis", index: 3, direction: -1 }] } }] },
      { version: 1, profiles: [{ ...profile, bindings: { ...bindings, UP: [{ kind: "axis", index: 1, direction: 0 }] } }] },
      { version: 1, profiles: [{ ...profile, device: { ...profile.device, buttons: -1 } }] },
    ]) expect(() => parseGamepadProfiles(JSON.stringify(value))).toThrow();
    expect(() => parseGamepadProfiles("{")).toThrow();
  });

  it("reports blocked storage and retains the new mapping in memory", () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    vi.stubGlobal("localStorage", {
      getItem() { throw new DOMException("Storage blocked", "SecurityError"); },
      setItem() { throw new DOMException("Storage blocked", "SecurityError"); },
    });
    const pad = receiver(), profiles = new GamepadProfiles(), bindings = automaticGamepadProfile(pad)!.bindings;
    expect(profiles.error).toContain("could not be loaded");
    profiles.set(pad, bindings);
    expect(profiles.error).toContain("only last for this tab");
    expect(profiles.get(pad)).toEqual(bindings);
    expect(console.error).toHaveBeenCalledTimes(2);
  });
});
