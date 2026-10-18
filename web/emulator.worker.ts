import { DiagnosticBot } from "./bot";
import { Engine, type CoreFactory } from "./engine";
import type { FramePacket, WorkerRequest, WorkerResponse } from "./protocol";
import { MAX_REPLAY_FRAMES, parseReplay, type Replay } from "./replay";
import { identifyRom, sha256 } from "./rom";

let engine: Engine | undefined;
let factory: CoreFactory | undefined;
let wasm: Uint8Array<ArrayBuffer> | undefined;
let rom: Uint8Array<ArrayBuffer> | undefined;
let coreBuild = "", romHash = "";
let recording: [number, number][] = [];
let playback: Replay | undefined;
const bot = new DiagnosticBot();

async function restart(): Promise<void> {
  if (!factory || !wasm || !rom) throw new Error("Load a local ROM first.");
  engine?.dispose();
  engine = undefined;
  engine = await Engine.create(factory, wasm, rom);
  recording = [];
  bot.reset();
}

function packet(audio = new Float32Array(0), inputs: [number, number] = [0, 0]): FramePacket {
  if (!engine) throw new Error("Emulator is not loaded.");
  const video = engine.video();
  return {
    frame: engine.frame, ...video, pixels: video.pixels.buffer, audio: audio.buffer,
    observation: engine.observe(), fps: engine.fps, sampleRate: engine.sampleRate,
    inputs, replaying: !!playback, finished: !!playback && engine.frame === playback.inputs.length,
  };
}

async function handle(request: WorkerRequest): Promise<WorkerResponse> {
  switch (request.type) {
    case "load": {
      const candidate = new Uint8Array(request.rom);
      const identity = await identifyRom(candidate);
      const base = new URL(request.coreBase);
      const imported: { default: CoreFactory } =
        await import(/* @vite-ignore */ new URL("genesis.js", base).href);
      const response = await fetch(new URL("genesis.wasm", base));
      if (!response.ok) throw new Error(`WASM load failed: HTTP ${response.status}. Run npm run build:core.`);
      const binary = new Uint8Array(await response.arrayBuffer());
      factory = imported.default;
      wasm = binary;
      rom = candidate;
      romHash = identity.hash;
      coreBuild = await sha256(binary);
      playback = undefined;
      await restart();
      return { id: request.id, type: "frame", packet: packet() };
    }
    case "restart":
      playback = undefined;
      await restart();
      return { id: request.id, type: "frame", packet: packet() };
    case "replay": {
      const replay = parseReplay(request.value);
      if (replay.romHash !== romHash || replay.coreBuild !== coreBuild) {
        throw new Error("Replay ROM/core build differs from the loaded game.");
      }
      await restart();
      playback = replay;
      return { id: request.id, type: "frame", packet: packet() };
    }
    case "advance": {
      if (!engine) throw new Error("Load a local ROM first.");
      if (!Number.isInteger(request.frames) || request.frames < 1 || request.frames > 4 ||
          (request.port !== 0 && request.port !== 1)) throw new Error("Invalid frame request.");
      const samples: Float32Array<ArrayBuffer>[] = [];
      let inputs: [number, number] = [0, 0];
      for (let index = 0; index < request.frames; index++) {
        if (playback) {
          const next = playback.inputs[engine.frame];
          if (!next) break;
          inputs = next;
        } else {
          if (recording.length >= MAX_REPLAY_FRAMES) {
            throw new Error("Recording limit reached. Export the replay and restart.");
          }
          const other = request.port === 0 ? 1 : 0;
          const generated = request.bot ? bot.act(engine.observe(), other) : 0;
          inputs = request.port === 0 ? [request.mask, generated] : [generated, request.mask];
        }
        engine.step(inputs[0], inputs[1]);
        if (!playback) recording.push(inputs);
        samples.push(engine.audio());
      }
      const audio = new Float32Array(samples.reduce((length, sample) => length + sample.length, 0));
      let offset = 0;
      for (const sample of samples) { audio.set(sample, offset); offset += sample.length; }
      return { id: request.id, type: "frame", packet: packet(audio, inputs) };
    }
    case "export":
      if (!recording.length && !playback) throw new Error("Run at least one frame before exporting.");
      return { id: request.id, type: "export", text: JSON.stringify(playback ?? {
        format: "nhl94-browser-inputs-v1", coreBuild, romHash, inputs: recording,
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
