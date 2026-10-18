import { validMask } from "./inputs";

export const MAX_REPLAY_FRAMES = 108000;
export const MAX_REPLAY_BYTES = 4 * 1024 * 1024;

export interface Replay {
  format: "nhl94-browser-inputs-v1";
  coreBuild: string;
  romHash: string;
  inputs: [number, number][];
}

export function parseReplay(value: unknown): Replay {
  if (typeof value !== "object" || value === null) throw new Error("Replay must be an object.");
  const replay = value as Record<string, unknown>;
  if (replay.format !== "nhl94-browser-inputs-v1" ||
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
  return {
    format: "nhl94-browser-inputs-v1", coreBuild: replay.coreBuild,
    romHash: replay.romHash, inputs,
  };
}
