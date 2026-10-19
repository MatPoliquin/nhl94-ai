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
export const CONTROLLER_PORTS = [0, 1, 2, 3] as const;
export type ControllerPort = typeof CONTROLLER_PORTS[number];
export type ControllerValues<T> = readonly [T, T, T, T];
export type ControllerInputs = [number, number, number, number];
export type ControllerTeam = 0 | 1 | 2;
export type ControllerTeams = ControllerValues<ControllerTeam>;
export type GoalieModes = readonly [0 | 1, 0 | 1];

export function assignedTeams(setup: ControllerSetup): ControllerTeams {
  const team = (slot: ControllerSlot) => slot.agent === "none" || slot.agent === "rom" ? 0 : slot.team;
  return [team(setup[0]), team(setup[1]), team(setup[2]), team(setup[3])];
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
  if (value.filter((slot) => slot.agent === "human").length > 2) {
    throw new Error("Up to two Human slots are supported: keyboard/on-screen controls and one selected gamepad.");
  }
  if ([1, 2].some((team) => value.filter((slot) => slot.agent === "classic" && slot.team === team).length > 3)) {
    throw new Error("no ml bot supports up to three bots on the same team.");
  }
  return [first, second, third, fourth];
}

export function controllerRouting(setup: ControllerSetup, observation?: Observation): {
  ports: ControllerValues<ControllerPort>; message: string;
} {
  const ports: [ControllerPort, ControllerPort, ControllerPort, ControllerPort] = [...CONTROLLER_PORTS];
  const used = new Set<ControllerPort>();
  const teams = assignedTeams(setup);
  for (const slot of CONTROLLER_PORTS) {
    if (teams[slot] === 0) continue;
    const port = CONTROLLER_PORTS.find((candidate) =>
      !used.has(candidate) && observation?.controllers[candidate]?.team === teams[slot]);
    if (port === undefined) {
      return {
        ports: CONTROLLER_PORTS,
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
  bots: readonly (Bot | null)[], state?: GameState,
  coordinatedInputs?: ReadonlyMap<ControllerPort, number>,
): ControllerInputs {
  if (!validMask(humanInputs[0]) || !validMask(humanInputs[1])) throw new Error("Invalid human controller inputs.");
  const inputs: ControllerInputs = [0, 0, 0, 0];
  const { ports } = controllerRouting(setup, observation);
  const humanCount = setup.filter((slot) => slot.agent === "human").length;
  let human = 0;
  for (const slot of CONTROLLER_PORTS) {
    const agent = setup[slot].agent, port = ports[slot];
    if (agent === "human") {
      inputs[port] = humanCount === 1 ? humanInputs[0] | humanInputs[1] : humanInputs[human++]!;
    } else if (agent === "classic") {
      const bot = bots[slot];
      if (!bot) throw new Error(`Bot on controller ${slot + 1} is not initialized.`);
      const action = coordinatedInputs?.get(port) ?? bot.act(observation, port, state);
      if (!validMask(action)) throw new Error(`Bot on controller ${slot + 1} returned invalid buttons.`);
      inputs[port] = action;
    }
  }
  return inputs;
}
