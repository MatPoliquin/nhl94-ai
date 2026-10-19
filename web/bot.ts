import { Button } from "./inputs";
import type { Observation } from "./observations";

export const AVAILABLE_BOTS = [{
  id: "diagnostic",
  label: "Diagnostic bot",
  description: "Chases the puck with the selected player and briefly pulses C to shoot, boost or check. A simple emulator test bot, not Classic AI. It does not navigate game menus.",
}] as const;

export interface Bot {
  reset(): void;
  act(observation: Observation, port: 0 | 1): number;
}

export class DiagnosticBot implements Bot {
  reset(): void {}

  act(observation: Observation, port: 0 | 1): number {
    const controller = observation.controllers[port];
    if (!controller.player || observation.seconds === 0 || observation.period > 3) return 0;
    const dx = observation.puck.x - controller.player.x;
    const dy = observation.puck.y - controller.player.y;
    let buttons = 0;
    if (Math.abs(dx) > 5) buttons |= dx > 0 ? Button.RIGHT : Button.LEFT;
    if (Math.abs(dy) > 5) buttons |= dy > 0 ? Button.UP : Button.DOWN;
    if (observation.frame % 24 < 6) buttons |= Button.C;
    return buttons;
  }
}
