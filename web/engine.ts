import { validMask } from "./inputs";
import { decodeObservation, type Observation } from "./observations";

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
  _arena_dispose(): void;
}

export type CoreFactory = (options: {
  wasmBinary: Uint8Array<ArrayBuffer>;
  printErr: (message: string) => void;
}) => Promise<CoreModule>;

export class Engine {
  frame = 0;
  private disposed = false;

  private constructor(private readonly core: CoreModule) {}

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
    return new Engine(core);
  }

  get fps(): number { return this.core._arena_fps(); }
  get sampleRate(): number { return this.core._arena_sample_rate(); }

  observe(): Observation {
    this.assertActive();
    const pointer = this.core._arena_ram();
    if (!pointer) throw new Error("Genesis work RAM is unavailable.");
    return decodeObservation(this.core.HEAPU8.subarray(pointer, pointer + 65536), this.frame);
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
