export const Button = {
  B: 1 << 0,
  START: 1 << 3,
  UP: 1 << 4,
  DOWN: 1 << 5,
  LEFT: 1 << 6,
  RIGHT: 1 << 7,
  C: 1 << 8,
  A: 1 << 9,
} as const;

export const ALLOWED_BUTTONS = Object.values(Button).reduce((mask, bit) => mask | bit, 0);

export const KEY_BUTTONS: Readonly<Record<string, number>> = {
  ArrowUp: Button.UP, ArrowDown: Button.DOWN,
  ArrowLeft: Button.LEFT, ArrowRight: Button.RIGHT,
  KeyZ: Button.A, KeyX: Button.B, KeyC: Button.C, Enter: Button.START,
};

export function validMask(value: unknown): value is number {
  return typeof value === "number" && Number.isInteger(value) &&
    value >= 0 && value <= ALLOWED_BUTTONS && (value & ~ALLOWED_BUTTONS) === 0;
}

export function keyboardMask(keys: Iterable<string>): number {
  let result = 0;
  for (const key of keys) result |= KEY_BUTTONS[key] ?? 0;
  return result;
}

export const GAMEPAD_ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "A", "B", "C", "START"] as const;
export type GamepadAction = typeof GAMEPAD_ACTIONS[number];
export type GamepadReading = Pick<Gamepad, "connected" | "mapping" | "buttons" | "axes">;
export type GamepadBinding =
  | { readonly kind: "button"; readonly index: number }
  | { readonly kind: "axis"; readonly index: number; readonly direction: -1 | 1 };
export type GamepadBindings = Readonly<Record<GamepadAction, readonly GamepadBinding[]>>;

export const STANDARD_BINDINGS: GamepadBindings = {
  UP: [{ kind: "button", index: 12 }, { kind: "axis", index: 1, direction: -1 }],
  DOWN: [{ kind: "button", index: 13 }, { kind: "axis", index: 1, direction: 1 }],
  LEFT: [{ kind: "button", index: 14 }, { kind: "axis", index: 0, direction: -1 }],
  RIGHT: [{ kind: "button", index: 15 }, { kind: "axis", index: 0, direction: 1 }],
  A: [{ kind: "button", index: 2 }], B: [{ kind: "button", index: 0 }],
  C: [{ kind: "button", index: 1 }], START: [{ kind: "button", index: 9 }],
};

export function gamepadMask(gamepad: GamepadReading | null, bindings?: GamepadBindings): number {
  if (!gamepad?.connected) return 0;
  const mapping = bindings ?? (gamepad.mapping === "standard" ? STANDARD_BINDINGS : undefined);
  if (!mapping) return 0;
  let mask = 0;
  for (const action of GAMEPAD_ACTIONS) {
    if (mapping[action].some((binding) => binding.kind === "button"
      ? gamepad.buttons[binding.index]?.pressed
      : (gamepad.axes[binding.index] ?? 0) * binding.direction > 0.4)) mask |= Button[action];
  }
  return mask;
}

export function newlyPressedInput(previous: GamepadReading, current: GamepadReading): GamepadBinding | null {
  for (let index = 0; index < current.buttons.length; index++) {
    if (current.buttons[index]?.pressed && !previous.buttons[index]?.pressed) {
      return { kind: "button", index };
    }
  }
  for (let index = 0; index < current.axes.length; index++) {
    const value = current.axes[index]!, before = previous.axes[index] ?? value;
    if (!Number.isFinite(value) || Math.abs(value) > 1) continue;
    const direction = value < 0 ? -1 : 1;
    if (value * direction > 0.7 && before * direction < 0.4) {
      return { kind: "axis", index, direction };
    }
  }
  return null;
}
