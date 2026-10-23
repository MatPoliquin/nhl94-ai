import { bounded_projection } from "./motion";
import { GameState, Player, Team, type Vec2 } from "./state";

export function clone_forecast(state: GameState): GameState {
  const future = Object.assign(new GameState(), state);
  for (const side of ["team1", "team2"] as const) {
    const source = state[side], team = Object.assign(new Team(source.controller), source);
    team.players = source.players.map((player) => Object.assign(new Player(), player));
    team.goalie = Object.assign(new Player(), source.goalie);
    future[side] = team;
  }
  future.puck = Object.assign(new Player(), state.puck);
  return future;
}

export function set_position(player: Player, point: Vec2): void {
  [player.x, player.y] = point;
  player.precise_x = player.x; player.precise_y = player.y;
}

export function project_body(player: Player, frames: number, boards = true): void {
  const [point, uncertainty] = bounded_projection(player, frames, { boards });
  set_position(player, point); player.projection_uncertainty = uncertainty;
}
