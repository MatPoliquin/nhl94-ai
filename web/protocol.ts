import type { Observation } from "./observations";
import type { ControllerSetup } from "./controllers";
import type { GameStats } from "./game-stats";

export type WorkerCommand =
  | { type: "load"; rom: ArrayBuffer; coreBase: string; controllers: ControllerSetup }
  | { type: "restart"; controllers: ControllerSetup }
  | { type: "configure"; controllers: ControllerSetup }
  | { type: "advance"; frames: number; humanInputs: readonly [number, number]; controllers: ControllerSetup }
  | { type: "export" }
  | { type: "replay"; value: unknown };

export type WorkerRequest = WorkerCommand & { id: number };

export interface FramePacket {
  frame: number;
  width: number;
  height: number;
  pixels: ArrayBuffer;
  audio: ArrayBuffer;
  observation: Observation;
  stats: GameStats;
  fps: number;
  sampleRate: number;
  inputs: readonly [number, number];
  replaying: boolean;
  finished: boolean;
}

export type WorkerResponse =
  | { id: number; type: "frame"; packet: FramePacket }
  | { id: number; type: "export"; text: string }
  | { id: number; type: "error"; message: string };
