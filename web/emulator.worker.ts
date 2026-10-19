import { type Bot } from "./bot";
import { createBot } from "./bot-factory";
import { DEFAULT_CLASSIC_CONFIG, parseClassicConfig, type ClassicConfig } from "./classic/config";
import { TeammateCoordinator } from "./classic/teammates";
import { assignedTeams, CONTROLLER_PORTS, controllerInputs, parseControllerSetup,
  type ControllerInputs, type ControllerPort, type ControllerSetup, type GoalieModes } from "./controllers";
import { Engine, type CoreFactory } from "./engine";
import type { FramePacket, WorkerRequest, WorkerResponse } from "./protocol";
import { MAX_CONTROLLER_EVENTS, MAX_REPLAY_FRAMES, parseReplay, type ConfigurationEvent, type Replay } from "./replay";
import { identifyRom, sha256 } from "./rom";
import { decodeDefaultState, DEFAULT_STATE_FILE, DEFAULT_STATE_HASH } from "./start-state";

let engine: Engine | undefined;
let factory: CoreFactory | undefined;
let wasm: Uint8Array<ArrayBuffer> | undefined;
let rom: Uint8Array<ArrayBuffer> | undefined;
let startState: Uint8Array<ArrayBuffer> | undefined;
let coreBuild = "", romHash = "";
let recording: ControllerInputs[] = [];
let playback: Replay | undefined;
let configurationEvents: ConfigurationEvent[] = [];
let playbackEvent = 0;
let classicConfig = DEFAULT_CLASSIC_CONFIG;
let defaultGoalieModes: GoalieModes = [0, 0];
let bots: (Bot | null)[] = [null, null, null, null];
let teammates: TeammateCoordinator[] = [];
let transientFrames = 0;
let setupKey = "";

async function restart(fromSave = true): Promise<void> {
  if (!factory || !wasm || !rom || !startState) throw new Error("Load a local ROM first.");
  engine?.dispose();
  engine = undefined;
  engine = await Engine.create(factory, wasm, rom);
  try {
    if (fromSave) engine.start(startState);
  } catch (error) {
    engine.dispose();
    engine = undefined;
    throw error;
  }
  recording = [];
  configurationEvents = [];
  playbackEvent = 0;
  for (const bot of bots) bot?.reset();
  defaultGoalieModes = engine.goalieModes();
  setupKey = "";
  teammates = [];
  transientFrames = 0;
}

function configure(controllers: ControllerSetup, config: ClassicConfig = classicConfig): void {
  if (!engine) throw new Error("Load a local ROM first.");
  if (playback) throw new Error("Live controller setup is disabled during replay.");
  const nextKey = JSON.stringify([controllers, config]);
  if (nextKey === setupKey) return;
  if (configurationEvents.length >= MAX_CONTROLLER_EVENTS) {
    throw new Error("Controller change limit reached. Export the replay and restart.");
  }
  const teams = assignedTeams(controllers);
  const modes: [0 | 1, 0 | 1] = [...defaultGoalieModes];
  for (const slot of CONTROLLER_PORTS) {
    if (controllers[slot].agent === "classic") {
      const team = controllers[slot].team;
      modes[team - 1] = config.ai_goalie && teams.filter((side) => side === team).length === 1 ? 0 : 1;
    }
  }
  engine.assignControllers(teams);
  engine.assignGoalies(modes);
  configurationEvents.push({ frame: engine.frame, teams, goalieModes: modes });
  classicConfig = config;
  bots = controllers.map((slot, port) => createBot(slot.agent, {
    ...config, ai_goalie: config.ai_goalie && teams.filter((team) => team === slot.team).length === 1 &&
      modes[port % 2] === 0 && modes[slot.team - 1] === 0,
  }));
  teammates = [1, 2].flatMap((team) => {
    const ports = CONTROLLER_PORTS.filter((port) => controllers[port].agent === "classic" && controllers[port].team === team);
    return ports.length >= 2 ? [new TeammateCoordinator(ports, config)] : [];
  });
  transientFrames = 0;
  setupKey = nextKey;
}

function applyReplayAssignments(): void {
  if (!engine || !playback || playback.format === "nhl94-browser-inputs-v1") return;
  const events = playback.format === "nhl94-browser-inputs-v3" || playback.format === "nhl94-browser-inputs-v4" ?
    playback.configurationEvents : playback.controllerEvents;
  while (playbackEvent < events.length && events[playbackEvent]!.frame === engine.frame) {
    const event = events[playbackEvent]!;
    engine.assignControllers(event.teams);
    if (playback.format === "nhl94-browser-inputs-v3" || playback.format === "nhl94-browser-inputs-v4") {
      engine.assignGoalies(playback.configurationEvents[playbackEvent]!.goalieModes);
    }
    playbackEvent++;
  }
}

function packet(audio = new Float32Array(0), inputs: ControllerInputs = [0, 0, 0, 0]): FramePacket {
  if (!engine) throw new Error("Emulator is not loaded.");
  const video = engine.video();
  return {
    frame: engine.frame, ...video, pixels: video.pixels.buffer, audio: audio.buffer,
    observation: engine.observe(), stats: engine.stats(), fps: engine.fps, sampleRate: engine.sampleRate,
    inputs, replaying: !!playback, finished: !!playback && engine.frame === playback.inputs.length,
  };
}

async function handle(request: WorkerRequest): Promise<WorkerResponse> {
  switch (request.type) {
    case "load": {
      const controllers = parseControllerSetup(request.controllers);
      const candidate = new Uint8Array(request.rom);
      const identity = await identifyRom(candidate);
      const base = new URL(request.coreBase);
      const imported: { default: CoreFactory } =
        await import(/* @vite-ignore */ new URL("genesis.js", base).href);
      const [response, stateResponse] = await Promise.all([
        fetch(new URL("genesis.wasm", base)), fetch(new URL(`../states/${DEFAULT_STATE_FILE}`, base)),
      ]);
      if (!response.ok) throw new Error(`WASM load failed: HTTP ${response.status}. Run npm run build:core.`);
      if (!stateResponse.ok) throw new Error(`Starting state load failed: HTTP ${stateResponse.status}.`);
      const [binaryBytes, stateBytes] = await Promise.all([response.arrayBuffer(), stateResponse.arrayBuffer()]);
      const binary = new Uint8Array(binaryBytes);
      const decodedState = await decodeDefaultState(new Uint8Array(stateBytes));
      factory = imported.default;
      wasm = binary;
      rom = candidate;
      startState = decodedState;
      romHash = identity.hash;
      coreBuild = await sha256(binary);
      playback = undefined;
      await restart();
      configure(controllers, parseClassicConfig(request.classicConfig === undefined ? classicConfig : request.classicConfig));
      return { id: request.id, type: "frame", packet: packet() };
    }
    case "restart": {
      const controllers = parseControllerSetup(request.controllers);
      playback = undefined;
      await restart();
      configure(controllers, parseClassicConfig(request.classicConfig === undefined ? classicConfig : request.classicConfig));
      return { id: request.id, type: "frame", packet: packet() };
    }
    case "configure":
      configure(parseControllerSetup(request.controllers), parseClassicConfig(request.classicConfig === undefined ? classicConfig : request.classicConfig));
      return { id: request.id, type: "frame", packet: packet() };
    case "replay": {
      const replay = parseReplay(request.value);
      if (replay.romHash !== romHash || replay.coreBuild !== coreBuild) {
        throw new Error("Replay ROM/core build differs from the loaded game.");
      }
      if (replay.format !== "nhl94-browser-inputs-v1" && replay.startStateHash !== DEFAULT_STATE_HASH) {
        throw new Error("Replay starting state differs from the bundled match.");
      }
      await restart(replay.format !== "nhl94-browser-inputs-v1");
      playback = replay;
      applyReplayAssignments();
      return { id: request.id, type: "frame", packet: packet() };
    }
    case "advance": {
      if (!engine) throw new Error("Load a local ROM first.");
      if (!Number.isInteger(request.frames) || request.frames < 1 || request.frames > 4) {
        throw new Error("Invalid frame request.");
      }
      const controllers = parseControllerSetup(request.controllers);
      if (!Array.isArray(request.humanInputs) || request.humanInputs.length !== 2) {
        throw new Error("Two human input sources are required.");
      }
      if (!playback) configure(controllers, parseClassicConfig(request.classicConfig === undefined ? classicConfig : request.classicConfig));
      const samples: Float32Array<ArrayBuffer>[] = [];
      let inputs: ControllerInputs = [0, 0, 0, 0];
      for (let index = 0; index < request.frames; index++) {
        if (playback) {
          applyReplayAssignments();
          const next = playback.inputs[engine.frame];
          if (!next) break;
          inputs = [next[0], next[1], next[2] ?? 0, next[3] ?? 0];
        } else {
          if (recording.length >= MAX_REPLAY_FRAMES) {
            throw new Error("Recording limit reached. Export the replay and restart.");
          }
          const observation = engine.observe();
          transientFrames = observation.transient ? transientFrames + 1 : 0;
          if (transientFrames > 120) throw new Error("Native controller ownership failed to settle for 120 frames.");
          const state = controllers.some((slot) => slot.agent === "classic") ? engine.classicState() : undefined;
          const coordinated = new Map<ControllerPort, number>();
          if (state) {
            for (const team of teammates) {
              for (const [port, action] of team.act(observation, state)) coordinated.set(port, action);
            }
          }
          inputs = controllerInputs(controllers, observation, request.humanInputs, bots, state, coordinated);
          if (observation.transient) inputs = [0, 0, 0, 0];
        }
        engine.step(...inputs);
        if (!playback) recording.push(inputs);
        samples.push(engine.audio());
      }
      applyReplayAssignments();
      const audio = new Float32Array(samples.reduce((length, sample) => length + sample.length, 0));
      let offset = 0;
      for (const sample of samples) { audio.set(sample, offset); offset += sample.length; }
      return { id: request.id, type: "frame", packet: packet(audio, inputs) };
    }
    case "export":
      if (!recording.length && !playback) throw new Error("Run at least one frame before exporting.");
      return { id: request.id, type: "export", text: JSON.stringify(playback ?? {
        format: "nhl94-browser-inputs-v4", coreBuild, romHash, inputs: recording,
        startStateHash: DEFAULT_STATE_HASH, configurationEvents,
      } satisfies Replay) };
  }
}

let queue = Promise.resolve();
self.addEventListener("message", (event: MessageEvent<WorkerRequest>) => {
  queue = queue.then(async () => {
    const request = event.data;
    try {
      const response = await handle(request);
      const transfers = response.type === "frame" ? [response.packet.pixels, response.packet.audio] : [];
      self.postMessage(response, { transfer: transfers });
    } catch (error) {
      console.error("Emulator request failed", error);
      self.postMessage({
        id: request.id, type: "error",
        message: error instanceof Error ? error.message : "Unexpected emulator failure.",
      } satisfies WorkerResponse);
    }
  });
});
