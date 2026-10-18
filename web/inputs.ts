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

export function gamepadMask(gamepad: Gamepad | null): number {
  if (!gamepad || gamepad.mapping !== "standard") return 0;
  const pressed = (index: number) => gamepad.buttons[index]?.pressed ?? false;
  let mask = 0;
  if (pressed(0)) mask |= Button.B;
  if (pressed(1)) mask |= Button.C;
  if (pressed(2)) mask |= Button.A;
  if (pressed(9)) mask |= Button.START;
  if (pressed(12) || (gamepad.axes[1] ?? 0) < -0.4) mask |= Button.UP;
  if (pressed(13) || (gamepad.axes[1] ?? 0) > 0.4) mask |= Button.DOWN;
  if (pressed(14) || (gamepad.axes[0] ?? 0) < -0.4) mask |= Button.LEFT;
  if (pressed(15) || (gamepad.axes[0] ?? 0) > 0.4) mask |= Button.RIGHT;
  return mask;
}
