import type { Observation } from "./observations";

export type WorkerCommand =
  | { type: "load"; rom: ArrayBuffer; coreBase: string }
  | { type: "restart" }
  | { type: "advance"; frames: number; mask: number; port: 0 | 1; bot: boolean }
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
