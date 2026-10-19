import { Button } from "./inputs";
import type { Observation } from "./observations";
import type { GameState } from "./classic/state";

export const AVAILABLE_BOTS = [{
  id: "rom",
  label: "ROM AI",
  description: "The game's built-in CPU AI. Releases this controller so teams without a Human or Classic AI controller use the original ROM AI. Does not use the Bot Configuration settings.",
}, {
  id: "classic",
  label: "Classic AI",
  description: "TypeScript port of Genesis Classic: possession, passing, one-timers, defense and selective manual goalies. Supports the Classic frequency, goalie and player-ratings settings. Experimental Python policies and coordinated teammates are not included.",
}] as const;

export interface Bot {
  reset(): void;
  act(observation: Observation, port: 0 | 1, state?: GameState): number;
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
