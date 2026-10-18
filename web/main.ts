import "./style.css";
import { AudioOutput } from "./audio";
import { EmulatorClient } from "./client";
import { gamepadMask, KEY_BUTTONS, keyboardMask } from "./inputs";
import type { FramePacket, WorkerCommand } from "./protocol";
import { MAX_REPLAY_BYTES } from "./replay";
import { identifyRom, MAX_ROM_BYTES } from "./rom";

function element<T extends HTMLElement>(id: string, kind: new (...args: never[]) => T): T {
  const value = document.getElementById(id);
  if (!(value instanceof kind)) throw new Error(`Missing interface element: ${id}`);
  return value;
}

const canvas = element("screen", HTMLCanvasElement);
const context = canvas.getContext("2d", { alpha: false });
if (!context) throw new Error("Canvas 2D is unavailable in this browser.");
const display = context;
const romFile = element("rom", HTMLInputElement);
const replayFile = element("replay", HTMLInputElement);
const playButton = element("play", HTMLButtonElement);
const stepButton = element("step", HTMLButtonElement);
const restartButton = element("restart", HTMLButtonElement);
const exportButton = element("export", HTMLButtonElement);
const sound = element("sound", HTMLInputElement);
const bot = element("bot", HTMLInputElement);
const port = element("port", HTMLSelectElement);
const status = element("status", HTMLParagraphElement);
const errorBox = element("error", HTMLParagraphElement);
const observations = element("observations", HTMLPreElement);
const frameOutput = element("frame", HTMLOutputElement);
const placeholder = element("placeholder", HTMLParagraphElement);
const romInfo = element("rom-info", HTMLParagraphElement);
const audio = new AudioOutput();
const keys = new Set<string>();
const pointerKeys = new Set<string>();
let loaded = false, busy = false, advancing = false, running = false, replaying = false, finished = false;
let lastTime = 0, accumulated = 0, animation = 0, fps = 60, sampleRate = 44100;

function updateControls(): void {
  playButton.disabled = !loaded || busy || finished;
  stepButton.disabled = !loaded || busy || advancing || running || finished;
  restartButton.disabled = !loaded || busy || advancing;
  exportButton.disabled = !loaded || busy || advancing || Number(frameOutput.value) === 0;
  romFile.disabled = busy || advancing;
  replayFile.disabled = !loaded || busy || advancing;
  port.disabled = busy || replaying;
  bot.disabled = busy || replaying;
  playButton.textContent = running ? "Pause" : "Play";
}

function stop(message?: string): void {
  running = false;
  cancelAnimationFrame(animation);
  lastTime = 0;
  accumulated = 0;
  keys.clear();
  pointerKeys.clear();
  audio.clear();
  if (message) status.textContent = message;
  updateControls();
}

function report(error: unknown): void {
  console.error(error);
  stop("Stopped because of an error.");
  errorBox.textContent = error instanceof Error ? error.message : "Unexpected application failure.";
  errorBox.hidden = false;
}

const client = new EmulatorClient((error) => {
  loaded = false;
  busy = false;
  report(error);
});

function render(packet: FramePacket): void {
  fps = packet.fps;
  sampleRate = packet.sampleRate;
  replaying = packet.replaying;
  finished = packet.finished;
  frameOutput.value = String(packet.frame);
  if (packet.width && packet.height) {
    if (canvas.width !== packet.width || canvas.height !== packet.height) {
      canvas.width = packet.width;
      canvas.height = packet.height;
    }
    display.putImageData(new ImageData(new Uint8ClampedArray(packet.pixels), packet.width, packet.height), 0, 0);
    placeholder.hidden = true;
  } else {
    display.clearRect(0, 0, canvas.width, canvas.height);
    placeholder.hidden = false;
    placeholder.textContent = "ROM loaded. Press Play or Step one frame.";
  }
  observations.textContent = JSON.stringify({
    ...packet.observation, lastInputs: packet.inputs,
    nativeFps: Number(packet.fps.toFixed(4)), mode: replaying ? "input replay" : "live local",
  }, null, 2);
  if (sound.checked && running) audio.play(new Float32Array(packet.audio), sampleRate);
  if (finished) stop("Replay finished. Restart ROM to return to live controls.");
  updateControls();
}

async function requestFrame(command: WorkerCommand, transfers: Transferable[] = []): Promise<void> {
  const response = await client.request(command, transfers);
  if (response.type !== "frame") throw new Error("Expected an emulator frame response.");
  render(response.packet);
}

async function perform(action: () => Promise<void>): Promise<void> {
  if (busy) return;
  busy = true;
  errorBox.hidden = true;
  updateControls();
  try { await action(); }
  catch (error) { report(error); }
  finally { busy = false; updateControls(); }
}

function controls(): { mask: number; port: 0 | 1; bot: boolean } {
  const gamepad = Array.from(navigator.getGamepads()).find((value) => value?.mapping === "standard") ?? null;
  return {
    mask: keyboardMask(keys) | keyboardMask(pointerKeys) | gamepadMask(gamepad),
    port: port.value === "0" ? 0 : 1, bot: bot.checked,
  };
}

async function tick(now: number): Promise<void> {
  if (!running) return;
  if (lastTime === 0) lastTime = now;
  const interval = 1000 / fps;
  accumulated = Math.min(accumulated + now - lastTime, interval * 4);
  lastTime = now;
  const frames = Math.min(4, Math.floor(accumulated / interval));
  try {
    if (frames && !busy) {
      accumulated -= frames * interval;
      advancing = true;
      updateControls();
      try { await requestFrame({ type: "advance", frames, ...controls() }); }
      finally { advancing = false; updateControls(); }
    }
  } catch (error) { report(error); }
  if (running) animation = requestAnimationFrame((time) => { void tick(time); });
}

romFile.addEventListener("change", () => {
  void perform(async () => {
    stop("Loading local ROM and WASM...");
    const file = romFile.files?.[0];
    if (!file) throw new Error("No ROM selected.");
    if (file.size > MAX_ROM_BYTES) throw new Error("ROM exceeds the 4 MiB limit.");
    const bytes = new Uint8Array(await file.arrayBuffer());
    const identity = await identifyRom(bytes);
    await requestFrame({
      type: "load", rom: bytes.buffer,
      coreBase: new URL(`${import.meta.env.BASE_URL}core/`, location.href).href,
    }, [bytes.buffer]);
    loaded = true;
    romInfo.textContent = `${identity.title} | SHA-256 ${identity.hash}`;
    status.textContent = "ROM loaded locally. Press Play; use the normal game menus.";
  });
});

playButton.addEventListener("click", () => {
  if (running) { stop("Paused."); return; }
  void perform(async () => {
    if (sound.checked) await audio.enable(sampleRate);
    running = true;
    lastTime = 0;
    status.textContent = replaying ? "Playing recorded inputs." : "Running locally. Diagnostic bot is not Classic.";
    animation = requestAnimationFrame((time) => { void tick(time); });
    canvas.focus();
  });
});

stepButton.addEventListener("click", () => {
  void perform(async () => {
    await requestFrame({ type: "advance", frames: 1, ...controls() });
    if (!finished) status.textContent = "Advanced exactly one native frame.";
  });
});

restartButton.addEventListener("click", () => {
  void perform(async () => {
    stop("Restarting from ROM boot...");
    await requestFrame({ type: "restart" });
    status.textContent = "Restarted. Recording begins again from ROM boot.";
  });
});

sound.addEventListener("change", () => {
  if (!sound.checked) { audio.clear(); return; }
  void perform(async () => { if (running) await audio.enable(sampleRate); });
});

exportButton.addEventListener("click", () => {
  void perform(async () => {
    stop("Exporting input replay...");
    const response = await client.request({ type: "export" });
    if (response.type !== "export") throw new Error("Expected replay export.");
    const url = URL.createObjectURL(new Blob([response.text], { type: "application/json" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = "nhl94-input-replay.json";
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    status.textContent = "Replay downloaded. ROM and bot implementation are not included.";
  });
});

replayFile.addEventListener("change", () => {
  void perform(async () => {
    stop("Loading input replay...");
    const file = replayFile.files?.[0];
    if (!file) throw new Error("No replay selected.");
    if (file.size > MAX_REPLAY_BYTES) throw new Error("Replay exceeds the 4 MiB limit.");
    const value: unknown = JSON.parse(await file.text());
    await requestFrame({ type: "replay", value });
    status.textContent = "Replay loaded at ROM boot. Press Play.";
    replayFile.value = "";
  });
});

document.addEventListener("keydown", (event) => {
  if (!(event.code in KEY_BUTTONS) || event.target instanceof HTMLInputElement ||
      event.target instanceof HTMLSelectElement) return;
  event.preventDefault();
  keys.add(event.code);
});
document.addEventListener("keyup", (event) => { keys.delete(event.code); });
window.addEventListener("blur", () => { keys.clear(); pointerKeys.clear(); });
document.addEventListener("visibilitychange", () => {
  if (document.hidden && running) stop("Paused because the tab was hidden.");
});
port.addEventListener("change", () => { keys.clear(); pointerKeys.clear(); });
window.addEventListener("pagehide", () => { audio.clear(); client.terminate(); });

for (const button of document.querySelectorAll<HTMLButtonElement>("[data-code]")) {
  button.addEventListener("pointerdown", (event) => {
    const code = button.dataset.code;
    if (code) pointerKeys.add(code);
    button.setPointerCapture(event.pointerId);
  });
  const release = () => { if (button.dataset.code) pointerKeys.delete(button.dataset.code); };
  button.addEventListener("pointerup", release);
  button.addEventListener("pointercancel", release);
  button.addEventListener("lostpointercapture", release);
}
