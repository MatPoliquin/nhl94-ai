import type { Player, Vec2 } from "./state";
import { mod, pointEqual, pyRound, required, shiftRight } from "./numeric";
import { skating, VELOCITY_SCALE, velocity } from "./motion";

export const FIXED = 65536;
export const PADS: readonly Vec2[] = [[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [-1, 1], [1, -1], [1, 1]];
export function signed_word(value: number): number { return mod(value + 32768, 65536) - 32768; }
export function friction(value: number, shift: number): number {
  return signed_word(value - (shiftRight(value, shift) || Number(!!value)));
}
export function brake(value: number, amount: number): number {
  return value >= 0 ? Math.max(0, value - amount) : Math.min(0, value + amount);
}

export class PuckFlight {
  constructor(public readonly x: number, public readonly y: number, public readonly z: number,
    public readonly vx: number, public readonly vy: number, public readonly vz: number, public readonly shift = 9) {}

  static observe(puck: Player): PuckFlight | null {
    if (puck.motion_x === null || puck.motion_y === null || puck.motion_z === null || puck.height === null) return null;
    return new PuckFlight(pyRound((puck.precise_x ?? puck.x) * FIXED), pyRound((puck.precise_y ?? puck.y) * FIXED),
      pyRound(puck.height * FIXED), pyRound(puck.motion_x / VELOCITY_SCALE), pyRound(puck.motion_y / VELOCITY_SCALE),
      pyRound(puck.motion_z / VELOCITY_SCALE), puck.friction === 511 / 512 ? 9 : 6);
  }

  step(): PuckFlight {
    let vx = this.vx, vy = this.vy;
    if (shiftRight(this.z, 16) === 0) { vx = friction(vx, this.shift); vy = friction(vy, this.shift); }
    let z = this.z, vz = this.vz;
    if (z >= 0 && (shiftRight(z, 16) || vz)) {
      vz = signed_word(vz - 102); z += vz * 17;
      if (z < 0) { z = 0; vz = shiftRight(signed_word(-vz), 1); }
    }
    return new PuckFlight(this.x + vx * 17, this.y + vy * 17, z, vx, vy, vz, this.shift);
  }

  get point(): readonly [number, number, number] { return [this.x / FIXED, this.y / FIXED, this.z / FIXED]; }
}

export class KeeperMotion {
  constructor(public readonly x: number, public readonly y: number, public readonly vx: number, public readonly vy: number,
    public readonly acceleration: number, public readonly speed_squared: number, public readonly shift = 6,
    public readonly saving = false) {}

  static observe(goalie: Player): KeeperMotion | null {
    if ((["speed", "weight", "agility", "energy", "motion_x", "motion_y"] as const).some((name) => goalie[name] === null)) return null;
    const [, limit] = skating(goalie), [vx, vy] = velocity(goalie);
    const acceleration = Math.floor((64 - Math.floor(required(goalie.weight, "weight") / 4)
      + 2 * required(goalie.agility, "agility") + 16 + goalie.movement_bonus) / 2);
    return new KeeperMotion(pyRound((goalie.precise_x ?? goalie.x) * FIXED), pyRound((goalie.precise_y ?? goalie.y) * FIXED),
      pyRound(vx / VELOCITY_SCALE), pyRound(vy / VELOCITY_SCALE), acceleration, pyRound((limit / VELOCITY_SCALE) ** 2),
      (goalie.selection_flags || 0) & 1 ? 9 : 6);
  }

  step(pad: Vec2 = [0, 0], { save = false } = {}): KeeperMotion {
    let vx = friction(this.vx, this.shift), vy = friction(this.vy, this.shift);
    const x = this.x + vx * 17, y = this.y + vy * 17, saving = this.saving || save;
    if (save && !this.saving) { vx = shiftRight(vx, 2); vy = shiftRight(vy, 2); }
    if (!saving) {
      const px = shiftRight(x, 16), py = shiftRight(y, 16);
      let blocked = false;
      if (px > 36 && vx > 0 || px < -36 && vx < 0) { vx = 0; blocked = true; }
      if (-231 < py && py <= 231 && vy && (py < 0) !== (vy < 0)) { vy = 0; blocked = true; }
      if (!blocked) {
        if (pointEqual(pad, [0, 0])) { vx = brake(vx, 2000); vy = brake(vy, 2000); }
        else {
          const scale = pad.every(Boolean) ? 141 : 200;
          const nx = signed_word(vx + shiftRight(pad[0] * scale * this.acceleration, 5));
          const ny = signed_word(vy + shiftRight(pad[1] * scale * this.acceleration, 5));
          if (nx * nx + ny * ny <= this.speed_squared) { vx = nx; vy = ny; }
        }
      }
    }
    return new KeeperMotion(x, y, vx, vy, this.acceleration, this.speed_squared, this.shift, saving);
  }

  get point(): Vec2 { return [this.x / FIXED, this.y / FIXED]; }
}

export function contact_margin(keeper: KeeperMotion, puck: PuckFlight, { one_timer = false } = {}): number {
  if (shiftRight(puck.z, 16) > 15) return -Infinity;
  const dx = shiftRight(keeper.x, 16) - shiftRight(puck.x, 16), dy = shiftRight(keeper.y, 16) - shiftRight(puck.y, 16);
  return (one_timer ? 12 : 15) - Math.hypot(dx, dy);
}

export function free_trajectory(puck: PuckFlight, frames: number, defended_sign: number): PuckFlight[] {
  const result: PuckFlight[] = [];
  for (let frame = 0; frame < frames; frame++) {
    puck = puck.step(); result.push(puck);
    if (Math.abs(shiftRight(puck.x, 16)) >= 128 || shiftRight(puck.y, 16) * defended_sign >= 261) break;
  }
  return result;
}
