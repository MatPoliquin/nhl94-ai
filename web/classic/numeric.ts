import type { Vec2 } from "./state";

/** Python's round-to-even, including negative half-integers. */
export function pyRound(value: number): number {
  const floor = Math.floor(value);
  const fraction = value - floor;
  return fraction < 0.5 ? floor : fraction > 0.5 ? floor + 1 : floor % 2 === 0 ? floor : floor + 1;
}

export function mod(a: number, b: number): number {
  return a - Math.floor(a / b) * b;
}

export function distance(a: Vec2, b: Vec2): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

export function pointEqual(a: Vec2, b: Vec2): boolean {
  return a[0] === b[0] && a[1] === b[1];
}

export function lexCompare(a: readonly number[], b: readonly number[]): number {
  for (let index = 0; index < Math.min(a.length, b.length); index += 1) {
    if (a[index]! < b[index]!) return -1;
    if (a[index]! > b[index]!) return 1;
  }
  return a.length - b.length;
}

export function required<T>(value: T | null | undefined, name: string): T {
  if (value === null || value === undefined) throw new TypeError(`Missing required Classic feedback: ${name}`);
  return value;
}

/** Arithmetic shifts without JavaScript's implicit signed-32-bit truncation. */
export function shiftRight(value: number, bits: number): number {
  return Math.floor(value / 2 ** bits);
}

export function copySign(value: number, sign: number): number {
  return sign < 0 || Object.is(sign, -0) ? -Math.abs(value) : Math.abs(value);
}
