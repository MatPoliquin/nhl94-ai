import { GAMEPAD_ACTIONS, STANDARD_BINDINGS, type GamepadBindings } from "./inputs";

type Device = Pick<Gamepad, "id" | "mapping" | "buttons" | "axes">;
interface DeviceLayout {
  id: string;
  mapping: Gamepad["mapping"];
  buttons: number;
  axes: number;
}
interface SavedProfile { device: DeviceLayout; bindings: GamepadBindings }
export interface AutomaticProfile { label: string; bindings: GamepadBindings }
export const GAMEPAD_PROFILES_KEY = "nhl94-arena-gamepad-profiles-v1";

// M30 2.4G receiver: C 5 observed directly; Start 9 confirmed on this layout.
const M30_6B_BINDINGS: GamepadBindings = {
  UP: [{ kind: "axis", index: 1, direction: -1 }],
  DOWN: [{ kind: "axis", index: 1, direction: 1 }],
  LEFT: [{ kind: "axis", index: 0, direction: -1 }],
  RIGHT: [{ kind: "axis", index: 0, direction: 1 }],
  A: [{ kind: "button", index: 0 }], B: [{ kind: "button", index: 1 }],
  C: [{ kind: "button", index: 5 }], START: [{ kind: "button", index: 9 }],
};

export function automaticGamepadProfile(pad: Device): AutomaticProfile | undefined {
  if (pad.mapping === "standard") return { label: "standard", bindings: STANDARD_BINDINGS };
  const receiver = /vendor:\s*0ca3\s+product:\s*0024\b/i.test(pad.id) ||
    /^0ca3-0024(?:-|$)/i.test(pad.id);
  if (receiver && pad.buttons.length === 10 && pad.axes.length === 3) {
    return { label: "M30 / 6B receiver", bindings: M30_6B_BINDINGS };
  }
  return undefined;
}

function layout(pad: Device): DeviceLayout {
  return { id: pad.id, mapping: pad.mapping, buttons: pad.buttons.length, axes: pad.axes.length };
}

function key(device: DeviceLayout): string {
  return JSON.stringify([device.id, device.mapping, device.buttons, device.axes]);
}

function record(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function bindingsValid(value: unknown, device: DeviceLayout): value is GamepadBindings {
  if (!record(value) || Object.keys(value).length !== GAMEPAD_ACTIONS.length) return false;
  return GAMEPAD_ACTIONS.every((action) => {
    const bindings: unknown = value[action];
    return Array.isArray(bindings) && bindings.length > 0 && bindings.length <= 4 && bindings.every((binding: unknown) => {
      if (!record(binding) || !Number.isSafeInteger(binding.index) || typeof binding.index !== "number" || binding.index < 0) return false;
      if (binding.kind === "button") return Object.keys(binding).length === 2 && binding.index < device.buttons;
      return binding.kind === "axis" && Object.keys(binding).length === 3 && binding.index < device.axes &&
        (binding.direction === -1 || binding.direction === 1);
    });
  });
}

export function parseGamepadProfiles(text: string): Map<string, SavedProfile> {
  const value: unknown = JSON.parse(text);
  if (!record(value) || value.version !== 1 || !Array.isArray(value.profiles) || value.profiles.length > 64) {
    throw new Error("Unsupported saved gamepad profile format.");
  }
  const result = new Map<string, SavedProfile>();
  for (const profile of value.profiles) {
    if (!record(profile) || !record(profile.device)) throw new Error("Invalid saved gamepad device.");
    const device = profile.device;
    if (typeof device.id !== "string" || device.id.length > 1024 ||
        (device.mapping !== "" && device.mapping !== "standard") ||
        typeof device.buttons !== "number" || !Number.isSafeInteger(device.buttons) || device.buttons < 1 || device.buttons > 256 ||
        typeof device.axes !== "number" || !Number.isSafeInteger(device.axes) || device.axes < 0 || device.axes > 64) {
      throw new Error("Invalid saved gamepad layout.");
    }
    const descriptor: DeviceLayout = { id: device.id, mapping: device.mapping, buttons: device.buttons, axes: device.axes };
    if (!bindingsValid(profile.bindings, descriptor)) throw new Error("Invalid saved gamepad bindings.");
    const identity = key(descriptor);
    if (result.has(identity)) throw new Error("Duplicate saved gamepad profile.");
    result.set(identity, { device: descriptor, bindings: profile.bindings });
  }
  return result;
}

export class GamepadProfiles {
  private profiles = new Map<string, SavedProfile>();
  error: string | undefined;

  constructor() {
    try {
      const text = localStorage.getItem(GAMEPAD_PROFILES_KEY);
      if (text !== null) this.profiles = parseGamepadProfiles(text);
    } catch (error) {
      this.report("Saved mappings could not be loaded", error);
    }
  }

  get(pad: Device): GamepadBindings | undefined { return this.profiles.get(key(layout(pad)))?.bindings; }

  set(pad: Device, bindings: GamepadBindings): void {
    const device = layout(pad);
    this.profiles.set(key(device), { device, bindings });
    this.save();
  }

  delete(pad: Device): void {
    this.profiles.delete(key(layout(pad)));
    this.save();
  }

  private save(): void {
    try {
      if (this.profiles.size > 64) throw new Error("Saved gamepad profile limit reached.");
      localStorage.setItem(GAMEPAD_PROFILES_KEY, JSON.stringify({ version: 1, profiles: [...this.profiles.values()] }));
      this.error = undefined;
    } catch (error) {
      this.report("Mappings could not be saved; custom controls will only last for this tab", error);
    }
  }

  private report(message: string, error: unknown): void {
    console.error(message, error);
    this.error = `${message}: ${error instanceof Error ? error.message : String(error)}`;
  }
}
