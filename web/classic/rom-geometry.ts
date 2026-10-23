import { GenesisRam } from "./ram";
import type { Player, Vec2 } from "./state";

interface AnimationFrame { sprite: number; duration: number }
interface ShotGeometry { durations: readonly number[]; offsets: readonly Vec2[] }

export class RomGeometry {
  private readonly view: DataView;
  private readonly hotspots: number;
  private readonly animations = new Map<number, readonly (readonly AnimationFrame[])[]>();
  private readonly shots: readonly ShotGeometry[];
  private readonly oneTimers: readonly (readonly Vec2[])[];
  private readonly projection: Vec2;
  private readonly projectionDurations: readonly number[];

  constructor(private readonly rom: Uint8Array) {
    this.view = new DataView(rom.buffer, rom.byteOffset, rom.byteLength);
    const signature = [0x48, 0xe7, 0x00, 0xc0, 0x20, 0x6f, 0x00, 0x0c, 0x42, 0x40, 0x42, 0x41, 0x4a, 0x68, 0x00, 0x06, 0x6f, 0x00];
    let entry = -1;
    for (let offset = 0; offset <= rom.length - signature.length; offset++) {
      if (!signature.every((byte, index) => rom[offset + index] === byte)) continue;
      if (entry >= 0) throw new Error("Classic cannot identify a unique ROM stick-geometry routine.");
      entry = offset;
    }
    if (entry < 0 || this.word(entry + 20) !== 0x227c) {
      throw new Error("Classic cannot identify the ROM stick-geometry routine.");
    }
    this.hotspots = this.view.getUint32(entry + 22);
    if (this.hotspots <= 0 || this.hotspots >= rom.length - 2) throw new Error("Invalid Classic stick-geometry table.");
    const shots: ShotGeometry[] = [], oneTimers: Vec2[][] = [];
    for (let direction = 0; direction < 8; direction++) {
      const times: number[][] = [], offsets: Vec2[] = [], contact: Vec2[] = [];
      for (const animation of [0x7fc, 0x92e, 0x50c]) {
        const frames = this.animation(animation)[direction]!;
        const durations: number[] = [];
        for (const frame of frames) {
          durations.push(Math.abs(frame.duration));
          offsets.push(this.hotspot(frame.sprite));
        }
        if (animation !== 0x50c) {
          if (frames.length < 8) throw new Error("Classic found a truncated shot animation.");
          times.push(durations.slice(0, 8));
          contact.push(this.hotspot(frames[6]!.sprite));
        }
      }
      shots.push({ durations: Array.from({ length: 8 }, (_, index) => Math.max(...times.map((time) => time[index]!))), offsets });
      oneTimers.push(contact);
    }
    this.shots = shots;
    this.oneTimers = oneTimers;
    this.projection = [
      Math.max(...shots.flatMap((shot) => shot.offsets.map((point) => Math.abs(point[0])))),
      Math.max(...shots.flatMap((shot) => shot.offsets.map((point) => Math.abs(point[1])))),
    ];
    this.projectionDurations = Array.from({ length: 8 }, (_, index) =>
      Math.max(...shots.map((shot) => shot.durations[index]!)));
  }

  private word(address: number, signed = false): number {
    if (!Number.isInteger(address) || address < 0 || address > this.rom.length - 2) {
      throw new Error("Classic animation address is outside the selected ROM.");
    }
    return signed ? this.view.getInt16(address) : this.view.getUint16(address);
  }

  private hotspot(sprite: number): Vec2 {
    const address = this.hotspots + sprite * 2;
    if (address < this.hotspots || address > this.rom.length - 2) {
      throw new Error("Classic sprite index is outside the ROM stick table.");
    }
    return [this.view.getInt8(address), this.view.getInt8(address + 1)];
  }

  private animation(animation: number): readonly (readonly AnimationFrame[])[] {
    if (animation === 0) return Array.from({ length: 8 }, () => []);
    const cached = this.animations.get(animation);
    if (cached) return cached;
    const base = 0x5b1c + animation;
    if (base + 18 > this.rom.length) throw new Error("Classic animation header is outside the selected ROM.");
    const result: AnimationFrame[][] = [];
    for (let direction = 0; direction < 8; direction++) {
      const start = base + this.word(base + direction * 2, true), frames: AnimationFrame[] = [];
      let terminated = false;
      for (let offset = 0; offset < 256; offset += 4) {
        const sprite = this.word(start + offset), duration = this.word(start + offset + 2, true);
        frames.push({ sprite, duration });
        if (duration < 0) { terminated = true; break; }
      }
      if (!terminated) throw new Error("Classic found an unterminated ROM animation.");
      result.push(frames);
    }
    this.animations.set(animation, result);
    return result;
  }

  update(player: Player, ram: GenesisRam, base: number, rotated: boolean, teammates: boolean): void {
    const sprite = ram.i16(base + 6), flags = ram.u8(base + 4);
    const flipX = !!(flags & 8), flipY = !!(flags & 16);
    const transform = ([x, y]: Vec2): Vec2 => {
      x = flipX ? -x : x;
      y = flipY ? y : -y;
      return rotated ? [y, -x] : [x, y];
    };
    const stick: Vec2 = sprite > 0 ? transform(this.hotspot(sprite)) : [0, 0];
    player.stick_x = stick[0];
    player.stick_y = stick[1];
    player.sprite_flipped_x = flipX;
    let direction = ((ram.u16(base + 0x54) - (rotated ? 2 : 0)) % 8 + 8) % 8;
    if (flipX) direction = (8 - direction) % 8;
    const shot = this.shots[direction]!;
    player.one_timer_offsets = teammates ? this.oneTimers[direction]!.map(transform) : null;
    player.shot_durations = shot.durations;
    const previous = this.animation(ram.u16(base + 0x58))[direction]!;
    const offsets = [...shot.offsets, ...previous.map((frame) => this.hotspot(frame.sprite))].map(transform);
    const xs = offsets.map(([x]) => x), ys = offsets.map(([, y]) => y);
    player.shot_offsets_x = [Math.min(...xs), Math.max(...xs)];
    player.shot_offsets_y = [Math.min(...ys), Math.max(...ys)];
    const px = Math.max(rotated ? this.projection[1] : this.projection[0], ...xs.map(Math.abs));
    const py = Math.max(rotated ? this.projection[0] : this.projection[1], ...ys.map(Math.abs));
    player.shot_projection_offsets_x = [-px, px];
    player.shot_projection_offsets_y = [-py, py];
    player.shot_projection_durations = this.projectionDurations;
  }
}
