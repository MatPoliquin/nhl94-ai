import { validMask } from "./inputs";
import type { ControllerInputs, ControllerTeam, ControllerTeams, GoalieModes } from "./controllers";

export const MAX_REPLAY_FRAMES = 108000;
export const MAX_REPLAY_BYTES = 4 * 1024 * 1024;
export const MAX_CONTROLLER_EVENTS = 4096;

interface ReplayBase {
  coreBuild: string;
  romHash: string;
  inputs: (ControllerInputs | [number, number])[];
}

export interface ControllerEvent {
  frame: number;
  teams: ControllerTeams | readonly [ControllerTeam, ControllerTeam];
}

export interface ConfigurationEvent extends ControllerEvent {
  goalieModes: GoalieModes;
}

export type Replay = ReplayBase & (
  | { format: "nhl94-browser-inputs-v1" }
  | { format: "nhl94-browser-inputs-v2"; startStateHash: string; controllerEvents: ControllerEvent[] }
  | { format: "nhl94-browser-inputs-v3"; startStateHash: string; configurationEvents: ConfigurationEvent[] }
  | { format: "nhl94-browser-inputs-v4"; startStateHash: string; configurationEvents: ConfigurationEvent[] }
);

export function parseReplay(value: unknown): Replay {
  if (typeof value !== "object" || value === null) throw new Error("Replay must be an object.");
  const replay = value as Record<string, unknown>;
  if ((replay.format !== "nhl94-browser-inputs-v1" && replay.format !== "nhl94-browser-inputs-v2" &&
       replay.format !== "nhl94-browser-inputs-v3" && replay.format !== "nhl94-browser-inputs-v4") ||
      typeof replay.coreBuild !== "string" || !/^[a-f0-9]{64}$/.test(replay.coreBuild) ||
      typeof replay.romHash !== "string" || !/^[a-f0-9]{64}$/.test(replay.romHash) ||
      !Array.isArray(replay.inputs) || replay.inputs.length === 0 ||
      replay.inputs.length > MAX_REPLAY_FRAMES) {
    throw new Error("Invalid or unsupported replay header/frame count.");
  }
  const fourPorts = replay.format === "nhl94-browser-inputs-v4";
  const hasConfiguration = fourPorts || replay.format === "nhl94-browser-inputs-v3";
  const ports = fourPorts ? [0, 1, 2, 3] : [0, 1];
  const inputs: (ControllerInputs | [number, number])[] = [];
  for (const frame of replay.inputs) {
    if (!Array.isArray(frame) || frame.length !== ports.length || !ports.every((port) => validMask(frame[port]))) {
      throw new Error("Replay contains invalid controller inputs.");
    }
    inputs.push(fourPorts ? [frame[0]!, frame[1]!, frame[2]!, frame[3]!] : [frame[0]!, frame[1]!]);
  }
  const base = { coreBuild: replay.coreBuild, romHash: replay.romHash, inputs };
  if (replay.format === "nhl94-browser-inputs-v1") return { ...base, format: replay.format };
  const events: unknown = hasConfiguration ? replay.configurationEvents : replay.controllerEvents;
  if (typeof replay.startStateHash !== "string" || !/^[a-f0-9]{64}$/.test(replay.startStateHash) ||
      !Array.isArray(events) || events.length < 1 || events.length > MAX_CONTROLLER_EVENTS) {
    throw new Error("Invalid replay starting state or controller event list.");
  }
  const controllerEvents: ControllerEvent[] = [];
  const configurationEvents: ConfigurationEvent[] = [];
  let previousFrame = 0;
  for (const event of events) {
    if (typeof event !== "object" || event === null || typeof event.frame !== "number" ||
        !Number.isSafeInteger(event.frame) || event.frame < previousFrame || event.frame > inputs.length ||
        !Array.isArray(event.teams) || event.teams.length !== (fourPorts ? 4 : 2) ||
        !ports.every((port) => event.teams[port] === 0 || event.teams[port] === 1 || event.teams[port] === 2)) {
      throw new Error("Invalid replay controller assignment.");
    }
    const teams: ControllerEvent["teams"] = fourPorts ?
      [event.teams[0], event.teams[1], event.teams[2], event.teams[3]] : [event.teams[0], event.teams[1]];
    controllerEvents.push({ frame: event.frame, teams });
    if (hasConfiguration) {
      if (!Array.isArray(event.goalieModes) || event.goalieModes.length !== 2 ||
          !event.goalieModes.every((mode: unknown) => mode === 0 || mode === 1)) {
        throw new Error("Invalid replay goalie configuration.");
      }
      configurationEvents.push({
        frame: event.frame, teams,
        goalieModes: [event.goalieModes[0], event.goalieModes[1]],
      });
    }
    previousFrame = event.frame;
  }
  if (controllerEvents[0]!.frame !== 0) throw new Error("Replay must identify its initial controllers.");
  return replay.format === "nhl94-browser-inputs-v3" || replay.format === "nhl94-browser-inputs-v4" ?
    { ...base, format: replay.format, startStateHash: replay.startStateHash, configurationEvents } :
    { ...base, format: replay.format, startStateHash: replay.startStateHash, controllerEvents };
}
