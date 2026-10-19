import {
  Button, GAMEPAD_ACTIONS, gamepadMask, newlyPressedInput,
  type GamepadAction, type GamepadBinding, type GamepadBindings, type GamepadReading,
} from "./inputs";

interface GamepadElements {
  device: HTMLSelectElement;
  setup: HTMLButtonElement;
  reset: HTMLButtonElement;
  status: HTMLParagraphElement;
  live: HTMLPreElement;
}

function snapshot(pad: GamepadReading): GamepadReading {
  return {
    connected: pad.connected, mapping: pad.mapping, axes: Array.from(pad.axes),
    buttons: pad.buttons.map((button) => ({
      pressed: button.pressed, touched: button.touched, value: button.value,
    })),
  };
}

function deviceKey(pad: Gamepad): string {
  return JSON.stringify([pad.index, pad.id, pad.mapping, pad.buttons.length, pad.axes.length]);
}

function bindingLabel(binding: GamepadBinding): string {
  return binding.kind === "button" ? `button ${binding.index}` :
    `axis ${binding.index} ${binding.direction < 0 ? "-" : "+"}`;
}

export class GamepadInput {
  private readonly profiles = new Map<string, GamepadBindings>();
  private selected = "";
  private selectedLabel = "";
  private autoSelect = true;
  private connected = false;
  private deviceList = "";
  private accessFailed = false;
  private previous: GamepadReading | null = null;
  private configuring: { step: number; bindings: Record<GamepadAction, GamepadBinding[]> } | null = null;
  private readonly timer: ReturnType<typeof setInterval>;

  constructor(private readonly ui: GamepadElements, private readonly pause: (reason: string) => void) {
    ui.device.addEventListener("change", () => {
      this.selected = ui.device.value;
      this.autoSelect = false;
      this.connected = false;
      this.previous = null;
      this.configuring = null;
      this.refresh();
    });
    ui.setup.addEventListener("click", () => {
      if (this.configuring) {
        this.configuring = null;
      } else {
        const pad = this.refresh();
        if (!pad) return;
        this.pause("Paused for gamepad setup.");
        this.previous = snapshot(pad);
        this.configuring = {
          step: 0, bindings: { UP: [], DOWN: [], LEFT: [], RIGHT: [], A: [], B: [], C: [], START: [] },
        };
      }
      this.refresh();
    });
    ui.reset.addEventListener("click", () => {
      this.profiles.delete(this.selected);
      this.configuring = null;
      this.refresh();
    });
    this.refresh();
    this.timer = setInterval(() => this.refresh(), 50);
  }

  mask(): number {
    const pad = this.refresh();
    return pad && !this.configuring ? gamepadMask(pad, this.profiles.get(this.selected)) : 0;
  }

  dispose(): void { clearInterval(this.timer); }

  private readPads(): Gamepad[] | null {
    let message: string;
    if (!isSecureContext) {
      message = "Gamepad access needs HTTPS or localhost.";
    } else if (typeof navigator.getGamepads !== "function") {
      message = "This browser does not expose the Gamepad API. Try a regular Firefox or Chrome window.";
    } else {
      try {
        const pads = Array.from(navigator.getGamepads()).filter((pad): pad is Gamepad => !!pad?.connected);
        this.accessFailed = false;
        return pads;
      } catch (error) {
        if (!this.accessFailed) console.error("Browser gamepad access failed", error);
        message = `Browser gamepad access failed: ${error instanceof Error ? error.message : String(error)}.`;
      }
    }
    this.accessFailed = true;
    this.status(message, "error");
    this.text(this.ui.live, "Gamepad input unavailable; keyboard controls still work.");
    return null;
  }

  private refresh(): Gamepad | null {
    const pads = this.readPads();
    if (pads && this.autoSelect && pads.length) {
      this.selected = deviceKey(pads[0]!);
      this.autoSelect = false;
    }
    const pad = pads?.find((candidate) => deviceKey(candidate) === this.selected) ?? null;
    if (pad) this.selectedLabel = pad.id;
    const list = JSON.stringify([pads?.map((pad) => [deviceKey(pad), pad.id]), this.selected, !!pad]);
    if (list !== this.deviceList) {
      const options = [new Option("Keyboard only", "")];
      for (const pad of pads ?? []) options.push(new Option(`${pad.index + 1}: ${pad.id}`, deviceKey(pad)));
      if (this.selected && !pad) options.push(new Option(`${this.selectedLabel} (disconnected)`, this.selected));
      this.ui.device.replaceChildren(...options);
      this.ui.device.value = this.selected;
      this.deviceList = list;
    }
    this.ui.device.disabled = !pads || (!pads.length && !this.selected);
    this.ui.setup.disabled = !pad;
    this.ui.reset.disabled = !pad || !this.profiles.has(this.selected);

    if (!pad) {
      if (this.connected) this.pause("Selected gamepad disconnected or became unavailable. Keyboard controls still work.");
      this.connected = false;
      this.previous = null;
      this.configuring = null;
      this.ui.setup.textContent = "Set up buttons / D-pad";
      if (pads) {
        this.status(this.selected ?
          "Selected gamepad disconnected. Reconnect it or select another device." :
          pads.length ? "Keyboard only. Select a gamepad to enable it." :
            "No gamepad exposed by the browser. Focus this page and press a controller button. Linux detection alone is not enough.",
        this.selected || !pads.length ? "warning" : "info");
        this.text(this.ui.live, "No selected gamepad input.");
      }
      return null;
    }
    this.connected = true;
    const previous = this.previous;
    this.previous = snapshot(pad);
    if (this.configuring && previous) {
      const binding = newlyPressedInput(previous, pad);
      if (binding) {
        const setup = this.configuring;
        const used = Object.values(setup.bindings).some((bindings) =>
          bindings.some((existing) => bindingLabel(existing) === bindingLabel(binding)));
        if (!used) {
          setup.bindings[GAMEPAD_ACTIONS[setup.step]!].push(binding);
          setup.step++;
          if (setup.step === GAMEPAD_ACTIONS.length) {
            this.profiles.set(this.selected, setup.bindings);
            this.configuring = null;
          }
        }
      }
    }
    const profile = this.profiles.get(this.selected);
    this.ui.setup.textContent = this.configuring ? "Cancel setup" : "Set up buttons / D-pad";
    this.ui.reset.disabled = !profile || !!this.configuring;
    this.status(this.configuring ?
      `Set ${GAMEPAD_ACTIONS[this.configuring.step]}: release the controls, then press that direction/button. Use a different input for each action.` :
      `${pad.id}: ${profile ? "custom mapping ready" : pad.mapping === "standard" ? "standard mapping ready" :
        "non-standard mapping; use Set up buttons / D-pad before playing"}.`,
    this.configuring ? "info" : profile || pad.mapping === "standard" ? "ready" : "warning");
    const mask = this.configuring ? 0 : gamepadMask(pad, profile);
    const pressed = pad.buttons.flatMap((button, index) => button.pressed ? [index] : []);
    const mapped = profile ? GAMEPAD_ACTIONS.map((action) =>
      `${action}: ${profile[action].map(bindingLabel).join(" / ")}`).join("\n") : "";
    this.text(this.ui.live, [
      `Browser mapping: ${pad.mapping || "non-standard"}`,
      `Buttons (${pad.buttons.length}) pressed: ${pressed.join(", ") || "none"}`,
      `Axes: ${pad.axes.map((value, index) => `${index}=${value.toFixed(2)}`).join(" ") || "none"}`,
      `Genesis input: ${GAMEPAD_ACTIONS.filter((action) => mask & Button[action]).join(" + ") || "neutral"}`,
      mapped,
    ].filter(Boolean).join("\n"));
    return pad;
  }

  private text(element: HTMLElement, text: string): void {
    if (element.textContent !== text) element.textContent = text;
  }

  private status(text: string, state: "ready" | "warning" | "info" | "error"): void {
    if (this.ui.status.dataset.state !== state) this.ui.status.dataset.state = state;
    this.text(this.ui.status, text);
  }
}
