import { AVAILABLE_BOTS, type Bot } from "./bot";
import { validMask } from "./inputs";
import type { Observation } from "./observations";
import type { GameState } from "./classic/state";

export type ControllerAgent = "human" | "none" | typeof AVAILABLE_BOTS[number]["id"];
export interface ControllerSlot {
  readonly agent: ControllerAgent;
  readonly team: 1 | 2;
}
export type ControllerSetup = readonly [ControllerSlot, ControllerSlot, ControllerSlot, ControllerSlot];
export type ControllerPort = 0 | 1;
export type ControllerTeam = 0 | 1 | 2;
export type ControllerTeams = readonly [ControllerTeam, ControllerTeam];
export type GoalieModes = readonly [0 | 1, 0 | 1];

export function assignedTeams(setup: ControllerSetup): ControllerTeams {
  return [
    setup[0].agent === "none" || setup[0].agent === "rom" ? 0 : setup[0].team,
    setup[1].agent === "none" || setup[1].agent === "rom" ? 0 : setup[1].team,
  ];
}

export const DEFAULT_CONTROLLERS: ControllerSetup = [
  { agent: "human", team: 1 }, { agent: "classic", team: 2 },
  { agent: "none", team: 1 }, { agent: "none", team: 2 },
];

function isControllerSlot(value: unknown): value is ControllerSlot {
  if (typeof value !== "object" || value === null || !("agent" in value) || !("team" in value)) return false;
  return (value.team === 1 || value.team === 2) &&
    (value.agent === "human" || value.agent === "none" || AVAILABLE_BOTS.some((bot) => bot.id === value.agent));
}

export function parseControllerSetup(value: unknown): ControllerSetup {
  if (!Array.isArray(value) || value.length !== 4 || !value.every(isControllerSlot)) {
    throw new Error("Invalid controller setup. Choose Human, an available bot or None and a Home/Away team.");
  }
  const [first, second, third, fourth] = value;
  if (!first || !second || !third || !fourth) throw new Error("Four controller slots are required.");
  if (third.agent !== "none" || fourth.agent !== "none") {
    throw new Error("Controllers 3 and 4 are not supported yet. Leave them set to None.");
  }
  return [first, second, third, fourth];
}

export function controllerRouting(setup: ControllerSetup, observation?: Observation): {
  ports: readonly [ControllerPort, ControllerPort]; message: string;
} {
  const ports: [ControllerPort, ControllerPort] = [0, 1];
  const used = new Set<ControllerPort>();
  const teams = assignedTeams(setup);
  for (const slot of [0, 1] as const) {
    if (teams[slot] === 0) continue;
    const port = ([0, 1] as const).find((candidate) =>
      !used.has(candidate) && observation?.controllers[candidate].team === teams[slot]);
    if (port === undefined) {
      return {
        ports: [0, 1],
        message: "Team assignments will be applied to the loaded match.",
      };
    }
    ports[slot] = port;
    used.add(port);
  }
  return { ports, message: "" };
}

export function controllerInputs(
  setup: ControllerSetup, observation: Observation, humanInputs: readonly [number, number],
  bots: readonly [Bot | null, Bot | null], state?: GameState,
): [number, number] {
  if (!validMask(humanInputs[0]) || !validMask(humanInputs[1])) throw new Error("Invalid human controller inputs.");
  const inputs: [number, number] = [0, 0];
  const { ports } = controllerRouting(setup, observation);
  const humanCount = setup.filter((slot) => slot.agent === "human").length;
  for (const slot of [0, 1] as const) {
    const agent = setup[slot].agent, port = ports[slot];
    if (agent === "human") {
      inputs[port] = humanCount === 1 ? humanInputs[0] | humanInputs[1] : humanInputs[slot];
    } else if (agent === "classic") {
      const bot = bots[slot];
      if (!bot) throw new Error(`Bot on controller ${slot + 1} is not initialized.`);
      const action = bot.act(observation, port, state);
      if (!validMask(action)) throw new Error(`Bot on controller ${slot + 1} returned invalid buttons.`);
      inputs[port] = action;
    }
  }
  return inputs;
}
