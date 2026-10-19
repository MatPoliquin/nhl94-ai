import { validMask } from "./inputs";
import type { ControllerTeams } from "./controllers";

export const MAX_REPLAY_FRAMES = 108000;
export const MAX_REPLAY_BYTES = 4 * 1024 * 1024;
export const MAX_CONTROLLER_EVENTS = 4096;

interface ReplayBase {
  coreBuild: string;
  romHash: string;
  inputs: [number, number][];
}

export interface ControllerEvent {
  frame: number;
  teams: ControllerTeams;
}

export type Replay = ReplayBase & (
  | { format: "nhl94-browser-inputs-v1" }
  | { format: "nhl94-browser-inputs-v2"; startStateHash: string; controllerEvents: ControllerEvent[] }
);

export function parseReplay(value: unknown): Replay {
  if (typeof value !== "object" || value === null) throw new Error("Replay must be an object.");
  const replay = value as Record<string, unknown>;
  if ((replay.format !== "nhl94-browser-inputs-v1" && replay.format !== "nhl94-browser-inputs-v2") ||
      typeof replay.coreBuild !== "string" || !/^[a-f0-9]{64}$/.test(replay.coreBuild) ||
      typeof replay.romHash !== "string" || !/^[a-f0-9]{64}$/.test(replay.romHash) ||
      !Array.isArray(replay.inputs) || replay.inputs.length === 0 ||
      replay.inputs.length > MAX_REPLAY_FRAMES) {
    throw new Error("Invalid or unsupported replay header/frame count.");
  }
  const inputs: [number, number][] = [];
  for (const frame of replay.inputs) {
    if (!Array.isArray(frame) || frame.length !== 2 ||
        !validMask(frame[0]) || !validMask(frame[1])) {
      throw new Error("Replay contains invalid controller inputs.");
    }
    inputs.push([frame[0], frame[1]]);
  }
  const base = { coreBuild: replay.coreBuild, romHash: replay.romHash, inputs };
  if (replay.format === "nhl94-browser-inputs-v1") return { ...base, format: replay.format };
  if (typeof replay.startStateHash !== "string" || !/^[a-f0-9]{64}$/.test(replay.startStateHash) ||
      !Array.isArray(replay.controllerEvents) || replay.controllerEvents.length < 1 ||
      replay.controllerEvents.length > MAX_CONTROLLER_EVENTS) {
    throw new Error("Invalid replay starting state or controller event list.");
  }
  const controllerEvents: ControllerEvent[] = [];
  let previousFrame = 0;
  for (const event of replay.controllerEvents) {
    if (typeof event !== "object" || event === null || typeof event.frame !== "number" ||
        !Number.isSafeInteger(event.frame) || event.frame < previousFrame || event.frame > inputs.length ||
        !Array.isArray(event.teams) || event.teams.length !== 2 ||
        !event.teams.every((team: unknown) => team === 0 || team === 1 || team === 2)) {
      throw new Error("Invalid replay controller assignment.");
    }
    controllerEvents.push({ frame: event.frame, teams: [event.teams[0], event.teams[1]] });
    previousFrame = event.frame;
  }
  if (controllerEvents[0]!.frame !== 0) throw new Error("Replay must identify its initial controllers.");
  return { ...base, format: replay.format, startStateHash: replay.startStateHash, controllerEvents };
}
