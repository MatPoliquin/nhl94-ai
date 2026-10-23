import source from "../../configs/classic_bot.json?raw";

export const DECISION_RATES = [60, 15, 10, 6] as const;
export interface ClassicConfig {
  readonly decision_hz: typeof DECISION_RATES[number];
  readonly ai_goalie: boolean;
  readonly player_ratings_consideration: boolean;
}

export function parseClassicConfig(value: unknown): ClassicConfig {
  if (typeof value !== "object" || value === null || Array.isArray(value) ||
      Object.keys(value).some((key) => !["decision_hz", "ai_goalie", "player_ratings_consideration"].includes(key))) {
    throw new Error("Classic config must be an object containing only supported Classic settings.");
  }
  const config = { decision_hz: 60, ai_goalie: true, player_ratings_consideration: true, ...value };
  const rate = DECISION_RATES.find((rate) => rate === config.decision_hz);
  if (rate === undefined || typeof config.ai_goalie !== "boolean" ||
      typeof config.player_ratings_consideration !== "boolean") {
    throw new Error("Classic requires decision_hz 60, 15, 10 or 6 and boolean goalie/ratings settings.");
  }
  return { decision_hz: rate, ai_goalie: config.ai_goalie, player_ratings_consideration: config.player_ratings_consideration };
}

export function parseClassicConfigText(text: string): ClassicConfig {
  const json = text.replace(/"(?:\\.|[^"\\])*"|\/\/[^\r\n]*|\/\*[\s\S]*?\*\//g,
    (token) => token.startsWith('"') ? token : token.replace(/[^\r\n]/g, " "));
  return parseClassicConfig(JSON.parse(json));
}

export const DEFAULT_CLASSIC_CONFIG = parseClassicConfigText(source);
export function controlInterval(config: ClassicConfig): number {
  return config.decision_hz === 60 ? 1 : 60 / config.decision_hz;
}
export function decisionInterval(config: ClassicConfig): number {
  return config.decision_hz === 60 ? 1 : 4;
}

const NEUTRAL_RATINGS = {
  speed: 15, agility: 15, shot_power: 15, shot_accuracy: 15, passing: 15, stick: 15,
  puck_control: 15, checking: 15, endurance: 15, weight: 64,
} as const;

import { GameState, Player, Team } from "./state";

function neutralPlayer(player: Player): Player {
  const result = Object.assign(new Player(), player);
  for (const key of Object.keys(NEUTRAL_RATINGS) as (keyof typeof NEUTRAL_RATINGS)[]) {
    if (result[key] !== null) result[key] = NEUTRAL_RATINGS[key];
  }
  if (player.input_state) {
    result.input_state = { ...player.input_state };
    const neutral: Record<string, number> = {
      ...NEUTRAL_RATINGS, glove_right: 15, stick_left: 15, stick_right: 15, shot_bias: 15,
      glove_left: 7, aggression: 7, offensive_delay: 7, defensive_delay: 7,
    };
    for (const key of Object.keys(result.input_state)) {
      if (key in neutral) result.input_state[key] = neutral[key]!;
    }
  }
  return result;
}

export function decisionState(state: GameState, considerRatings: boolean): GameState {
  if (considerRatings) return state;
  const result = Object.assign(new GameState(), state);
  for (const side of ["team1", "team2"] as const) {
    const original = state[side], team = Object.assign(new Team(original.controller), original);
    team.players = original.players.map(neutralPlayer);
    team.goalie = neutralPlayer(original.goalie);
    team.goalie_stats = {
      glove_left: 7, glove_right: 15, stick_left: 15, stick_right: 15, puck_control: 15,
      agility: 15, speed: 15, passing: 15, endurance: 15, weight: 64,
    };
    result[side] = team;
  }
  return result;
}
