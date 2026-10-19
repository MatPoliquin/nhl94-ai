import "./style.css";
import { AudioOutput } from "./audio";
import { EmulatorClient } from "./client";
import { KEY_BUTTONS, keyboardMask } from "./inputs";
import { GamepadInput } from "./gamepad";
import { GAME_STAT_ROWS } from "./game-stats";
import { AVAILABLE_BOTS } from "./bot";
import { controllerRouting, DEFAULT_CONTROLLERS, parseControllerSetup, type ControllerSetup } from "./controllers";
import type { Observation } from "./observations";
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
const controllerRows = [1, 2, 3, 4].map((number) => ({
  agent: element(`controller-${number}-agent`, HTMLSelectElement),
  team: element(`controller-${number}-team`, HTMLSelectElement),
}));
const controllerNote = element("controller-note", HTMLParagraphElement);
const controllerError = element("controller-error", HTMLParagraphElement);
for (const [index, row] of controllerRows.entries()) {
  row.agent.replaceChildren(
    new Option("Human", "human"),
    ...AVAILABLE_BOTS.map((bot) => new Option(bot.label, bot.id)),
    new Option("None", "none"),
  );
  row.agent.value = DEFAULT_CONTROLLERS[index]!.agent;
  row.team.value = String(DEFAULT_CONTROLLERS[index]!.team);
}
const status = element("status", HTMLParagraphElement);
const errorBox = element("error", HTMLParagraphElement);
const statsBody = element("game-stats-body", HTMLTableSectionElement);
const statsStatus = element("game-stats-status", HTMLParagraphElement);
const statsCells = GAME_STAT_ROWS.map((definition) => {
  const row = document.createElement("tr");
  row.dataset.stat = definition.id;
  const label = document.createElement("th");
  label.scope = "row";
  label.textContent = definition.label;
  const home = document.createElement("td"), away = document.createElement("td");
  home.textContent = away.textContent = "-";
  row.append(label, home, away);
  statsBody.append(row);
  return { definition, home, away };
});
const frameOutput = element("frame", HTMLOutputElement);
const placeholder = element("placeholder", HTMLParagraphElement);
const romInfo = element("rom-info", HTMLParagraphElement);
const audio = new AudioOutput();
const keys = new Set<string>();
const pointerKeys = new Set<string>();
let loaded = false, busy = false, advancing = false, running = false, replaying = false, finished = false;
let lastTime = 0, accumulated = 0, animation = 0, fps = 60, sampleRate = 44100;
let lastObservation: Observation | undefined;

function readControllers(): ControllerSetup {
  return parseControllerSetup(controllerRows.map((row) => ({
    agent: row.agent.value, team: Number(row.team.value),
  })));
}

function updateControls(): void {
  playButton.disabled = !loaded || busy || finished;
  stepButton.disabled = !loaded || busy || advancing || running || finished;
  restartButton.disabled = !loaded || busy || advancing;
  exportButton.disabled = !loaded || busy || advancing || Number(frameOutput.value) === 0;
  romFile.disabled = busy || advancing;
  replayFile.disabled = !loaded || busy || advancing;
  for (const [index, row] of controllerRows.entries()) {
    row.agent.disabled = index >= 2 || busy || replaying;
    row.team.disabled = index >= 2 || busy || replaying || row.agent.value === "none";
  }
  controllerError.hidden = true;
  if (replaying) {
    controllerNote.textContent = "Replay uses recorded controller inputs. Live controller setup is ignored.";
  } else {
    try {
      const setup = readControllers();
      const humans = setup.filter((slot) => slot.agent === "human").length;
      const inputNote = humans === 2 ? "Two humans: keyboard/on-screen input goes to controller 1; the selected gamepad goes to controller 2." :
        humans === 1 ? "Keyboard, on-screen controls and the selected gamepad go to the Human slot." :
          "No Human slot: keyboard and gamepad inputs are ignored. Unassigned teams use the game's CPU.";
      controllerNote.textContent = `${inputNote} ${controllerRouting(setup, lastObservation).message}`;
    } catch (error) {
      controllerError.textContent = error instanceof Error ? error.message : "Invalid controller setup.";
      controllerError.hidden = false;
      playButton.disabled = true;
      stepButton.disabled = true;
    }
  }
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

const gamepads = new GamepadInput({
  device: element("gamepad", HTMLSelectElement),
  setup: element("gamepad-setup", HTMLButtonElement),
  reset: element("gamepad-reset", HTMLButtonElement),
  status: element("gamepad-status", HTMLParagraphElement),
  live: element("gamepad-live", HTMLPreElement),
}, stop);

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
  lastObservation = packet.observation;
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
  for (const { definition, home, away } of statsCells) {
    const homeValue = definition.format(packet.stats[0]), awayValue = definition.format(packet.stats[1]);
    if (home.textContent !== homeValue) home.textContent = homeValue;
    if (away.textContent !== awayValue) away.textContent = awayValue;
  }
  statsStatus.textContent = replaying ? "Recorded match statistics." : "Live match statistics.";
  if (sound.checked && running) audio.play(new Float32Array(packet.audio), sampleRate);
  if (finished) stop("Replay finished. Restart match to return to live controls.");
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

function controls(): { humanInputs: readonly [number, number]; controllers: ControllerSetup } {
  return {
    humanInputs: [keyboardMask(keys) | keyboardMask(pointerKeys), gamepads.mask()],
    controllers: replaying ? DEFAULT_CONTROLLERS : readControllers(),
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
      controllers: readControllers(),
    }, [bytes.buffer]);
    loaded = true;
    romInfo.textContent = `${identity.title} | SHA-256 ${identity.hash}`;
    status.textContent = "ROM loaded locally. Sabres vs Mighty Ducks is ready. Press Play.";
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
    stop("Restoring the default match...");
    await requestFrame({ type: "restart", controllers: readControllers() });
    status.textContent = "Restarted the default Sabres vs Mighty Ducks match.";
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
    status.textContent = "Replay loaded at its recorded starting state. Press Play.";
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
for (const row of controllerRows) {
  for (const select of [row.agent, row.team]) {
    select.addEventListener("change", () => {
      stop("Controller setup changed. Press Play to resume.");
      if (loaded && !replaying) {
        void perform(async () => { await requestFrame({ type: "configure", controllers: readControllers() }); });
      }
    });
  }
}
window.addEventListener("pagehide", () => { audio.clear(); gamepads.dispose(); client.terminate(); });

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

updateControls();
