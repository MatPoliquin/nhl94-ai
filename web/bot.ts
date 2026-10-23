import { Button } from "./inputs";
import type { Observation } from "./observations";
import type { GameState } from "./classic/state";
import type { ControllerPort } from "./controllers";

export const AVAILABLE_BOTS = [{
  id: "rom",
  label: "ROM AI",
  description: "The game's built-in CPU AI. Releases this controller so teams without a Human or no ml bot controller use the original ROM AI. Does not use the Bot Configuration settings.",
}, {
  id: "classic",
  label: "no ml bot",
  description: "Genesis scripted bot with no machine learning: possession, passing, one-timers, defense and selective manual goalies. Two or three no ml bot teammates share carrier, receiving, support and coverage roles against the CPU or a human. Supports frequency, goalie and player-ratings settings; experimental Python policies are not included.",
}] as const;

export interface Bot {
  reset(): void;
  act(observation: Observation, port: ControllerPort, state?: GameState): number;
}

export class DiagnosticBot implements Bot {
  reset(): void {}

  act(observation: Observation, port: ControllerPort): number {
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
