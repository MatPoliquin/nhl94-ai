import type { Bot } from "../bot";
import type { ControllerPort } from "../controllers";
import { Button, validMask } from "../inputs";
import type { Observation } from "../observations";
import { ClassicController } from "./classic";
import { controlInterval, type ClassicConfig } from "./config";
import { controllerView } from "./decoder";
import type { Action, GameState } from "./state";

const MASKS = [Button.B, Button.A, 0, Button.START, Button.UP, Button.DOWN, Button.LEFT, Button.RIGHT, Button.C, 0, 0, 0] as const;

export function actionMask(action: Action): number {
  if (action.length !== 12) throw new Error("Classic must return exactly twelve controller channels.");
  let result = 0;
  for (const [index, value] of action.entries()) {
    if (value !== 0 && value !== 1) throw new Error("Classic returned a nonbinary controller action.");
    if (value && MASKS[index] === 0) throw new Error("Classic requested an unsupported Genesis button.");
    if (value) result |= MASKS[index]!;
  }
  if (!validMask(result)) throw new Error("Classic returned invalid Genesis controller inputs.");
  return result;
}

export class ClassicBot implements Bot {
  private controller: ClassicController;
  private remaining = 0;
  private held = 0;

  constructor(private readonly config: ClassicConfig) {
    this.controller = new ClassicController(config);
  }

  reset(): void {
    this.controller = new ClassicController(this.config);
    this.remaining = 0;
    this.held = 0;
  }

  act(observation: Observation, port: ControllerPort, state?: GameState): number {
    if (!state) throw new Error("Classic game-state decoding is unavailable.");
    if (this.remaining === 0) {
      const view = controllerView(state, port, observation.controllers[port].slot);
      this.held = actionMask(this.controller.predict_frame(view, 4));
      this.remaining = controlInterval(this.config);
    } else {
      this.controller.advance_held_frame();
    }
    this.remaining--;
    return this.held;
  }
}
