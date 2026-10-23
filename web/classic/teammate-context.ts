import type { GameState, Vec2 } from "./state";

export class TeamControlContext {
  constructor(readonly slots: readonly number[], readonly known: boolean, readonly support_target: Vec2,
    readonly goalie_ownership = false) {}

  cpu_carrier(state: GameState): boolean {
    const owner = state.engine.puck_owner;
    return this.known && state.engine.puck_owner_known && state.team1.owns_scnum(owner) &&
      owner !== state.team1.goalie_scnum() && !this.slots.includes(owner);
  }
}
