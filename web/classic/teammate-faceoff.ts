import { Buttons, neutralAction, type Action, type GameState, type Player } from "./state";

export function faceoff_center(state: GameState): number | null {
  if (!state.engine.clock_stopped || state.engine.puck_owner >= 0 || state.time <= 0 ||
      state.engine.sflags & 0x100) return null;
  const participant = (player: Player) => player.assignment === 0x17 && player.role === 4 &&
    Math.hypot(player.x - state.puck.x, player.y - state.puck.y) <= 28;
  const candidates = state.team1.players.flatMap((player, index) =>
    participant(player) && !(player.unavailable & 4) && !player.is_falling ?
      [state.team1.skater_scnum_base() + index] : []);
  return candidates.length === 1 && state.team2.players.filter(participant).length === 1 ? candidates[0]! : null;
}

export class FaceoffInput {
  private controller: number | null = null;
  private since = 0;
  private releaseOnHandoff = false;

  choose(state: GameState, slots: readonly number[], frame: number, transient: boolean,
    previous: readonly Action[]): { controller: number | null; action: Action; center: number | null } {
    const center = transient ? null : faceoff_center(state);
    const controller = center !== null && slots.includes(center) ? slots.indexOf(center) : null;
    if (controller !== this.controller) {
      this.controller = controller; this.since = frame;
      this.releaseOnHandoff = controller !== null && !!previous[controller]![Buttons.INPUT_B];
    }
    const action = neutralAction();
    if (controller !== null && (frame !== this.since || !this.releaseOnHandoff)) {
      action[Buttons.INPUT_B] = Number(frame % 8 < 4);
    }
    return { controller, action, center };
  }
}
