import type { Bot } from "./bot";
import { ClassicBot } from "./classic/bot";
import type { ClassicConfig } from "./classic/config";
import type { ControllerAgent } from "./controllers";

export function createBot(agent: ControllerAgent, config: ClassicConfig): Bot | null {
  return agent === "classic" ? new ClassicBot(config) : null;
}
