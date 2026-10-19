import { validMask } from "./inputs";
import { decodeObservation, type Observation } from "./observations";
import { decodeGameStats, type GameStats } from "./game-stats";
import type { ControllerTeams, GoalieModes } from "./controllers";
import { ClassicDecoder } from "./classic/decoder";
import type { GameState } from "./classic/state";

export interface CoreModule {
  HEAPU8: Uint8Array<ArrayBuffer>;
  HEAP16: Int16Array<ArrayBuffer>;
  _arena_alloc(size: number): number;
  _arena_free(pointer: number): void;
  _arena_load(pointer: number, size: number): number;
  _arena_step(first: number, second: number): number;
  _arena_pixels(): number;
  _arena_width(): number;
  _arena_height(): number;
  _arena_audio(): number;
  _arena_audio_frames(): number;
  _arena_fps(): number;
  _arena_sample_rate(): number;
  _arena_ram(): number;
  _arena_state_size(): number;
  _arena_save(pointer: number): number;
  _arena_restore(pointer: number): number;
  _arena_set_controllers(first: number, second: number): number;
  _arena_dispose(): void;
}

export type CoreFactory = (options: {
  wasmBinary: Uint8Array<ArrayBuffer>;
  printErr: (message: string) => void;
}) => Promise<CoreModule>;

export class Engine {
  frame = 0;
  private disposed = false;
  private classicDecoder: ClassicDecoder | undefined;

  private constructor(private readonly core: CoreModule, private readonly rom: Uint8Array) {}

  static async create(factory: CoreFactory, wasm: Uint8Array<ArrayBuffer>,
    rom: Uint8Array<ArrayBuffer>): Promise<Engine> {
    const core = await factory({
      wasmBinary: wasm,
      printErr: (message) => console.error(`[Genesis Plus GX] ${message}`),
    });
    const pointer = core._arena_alloc(rom.length);
    if (!pointer) throw new Error("Could not allocate ROM memory.");
    try {
      core.HEAPU8.set(rom, pointer);
      if (!core._arena_load(pointer, rom.length)) throw new Error("Genesis Plus GX could not load the ROM.");
    } finally {
      core._arena_free(pointer);
    }
    return new Engine(core, rom);
  }

  get fps(): number { return this.core._arena_fps(); }
  get sampleRate(): number { return this.core._arena_sample_rate(); }

  start(snapshot: Uint8Array<ArrayBuffer>): void {
    this.restore(snapshot, 0);
    // Native saves do not include the frontend framebuffer; initialize it before recording.
    this.step(0, 0);
    this.frame = 0;
  }

  assignControllers(teams: ControllerTeams): void {
    this.assertActive();
    if (teams.length !== 2 || !teams.every((team) => Number.isInteger(team) && team >= 0 && team <= 2) ||
        !this.core._arena_set_controllers(teams[0], teams[1])) {
      throw new Error("Unable to assign controllers in this game state. A full-team exhibition match is required.");
    }
  }

  observe(): Observation {
    return decodeObservation(this.ram(), this.frame);
  }

  classicState(): GameState {
    this.classicDecoder ??= new ClassicDecoder(this.rom);
    return this.classicDecoder.decode(this.ram());
  }

  goalieModes(): GoalieModes {
    const ram = this.ram(), view = new DataView(ram.buffer, ram.byteOffset, ram.byteLength);
    const first = view.getUint16(0xd05a, true), second = view.getUint16(0xd05c, true);
    if ((first !== 0 && first !== 1) || (second !== 0 && second !== 1)) {
      throw new Error("Unsupported native goalie configuration.");
    }
    return [first, second];
  }

  assignGoalies(modes: GoalieModes): void {
    if (modes.length !== 2 || !modes.every((mode) => mode === 0 || mode === 1)) {
      throw new Error("Goalie modes must select manual or automatic control for both teams.");
    }
    const ram = this.ram(), view = new DataView(ram.buffer, ram.byteOffset, ram.byteLength);
    if (view.getUint16(0xd048, true) !== 0) throw new Error("Goalie configuration requires an exhibition match.");
    view.setUint16(0xd05a, modes[0], true);
    view.setUint16(0xd05c, modes[1], true);
  }

  stats(): GameStats {
    return decodeGameStats(this.ram());
  }

  private ram(): Uint8Array<ArrayBuffer> {
    this.assertActive();
    const pointer = this.core._arena_ram();
    if (!pointer) throw new Error("Genesis work RAM is unavailable.");
    return this.core.HEAPU8.subarray(pointer, pointer + 65536);
  }

  step(first: number, second: number): void {
    this.assertActive();
    if (!validMask(first) || !validMask(second)) throw new Error("Invalid Genesis controller mask.");
    if (!this.core._arena_step(first, second)) throw new Error("Emulator frame failed.");
    this.frame++;
  }

  video(): { width: number; height: number; pixels: Uint8Array<ArrayBuffer> } {
    this.assertActive();
    const width = this.core._arena_width(), height = this.core._arena_height();
    const pointer = this.core._arena_pixels();
    return { width, height, pixels: this.core.HEAPU8.slice(pointer, pointer + width * height * 4) };
  }

  audio(): Float32Array<ArrayBuffer> {
    this.assertActive();
    const pointer = this.core._arena_audio() / 2;
    const result = new Float32Array(this.core._arena_audio_frames() * 2);
    for (let index = 0; index < result.length; index++) {
      result[index] = this.core.HEAP16[pointer + index]! / 32768;
    }
    return result;
  }

  save(): Uint8Array<ArrayBuffer> {
    this.assertActive();
    const size = this.core._arena_state_size();
    const pointer = this.core._arena_alloc(size);
    if (!pointer) throw new Error("Could not allocate snapshot memory.");
    try {
      if (!this.core._arena_save(pointer)) throw new Error("Snapshot save failed.");
      return this.core.HEAPU8.slice(pointer, pointer + size);
    } finally {
      this.core._arena_free(pointer);
    }
  }

  restore(snapshot: Uint8Array<ArrayBuffer>, frame: number): void {
    this.assertActive();
    if (snapshot.length !== this.core._arena_state_size() ||
        !Number.isSafeInteger(frame) || frame < 0) throw new Error("Invalid local snapshot.");
    const pointer = this.core._arena_alloc(snapshot.length);
    if (!pointer) throw new Error("Could not allocate snapshot memory.");
    try {
      this.core.HEAPU8.set(snapshot, pointer);
      if (!this.core._arena_restore(pointer)) throw new Error("Snapshot restore failed.");
      this.frame = frame;
    } finally {
      this.core._arena_free(pointer);
    }
  }

  dispose(): void {
    if (!this.disposed) this.core._arena_dispose();
    this.disposed = true;
  }

  private assertActive(): void {
    if (this.disposed) throw new Error("Emulator has been disposed.");
  }
}
