import { pyRound } from "./numeric";
import type { GameState, Player, Team, Vec2 } from "./state";

function intersects(start: Vec2, end: Vec2, center: Vec2, radius: number): boolean {
  const dx = end[0] - start[0], dy = end[1] - start[1], length = dx * dx + dy * dy;
  const along = length ? Math.max(0, Math.min(1, ((center[0] - start[0]) * dx + (center[1] - start[1]) * dy) / length)) : 0;
  return (center[0] - start[0] - along * dx) ** 2 + (center[1] - start[1] - along * dy) ** 2 <= radius * radius;
}

export function refreshFeatures(state: GameState): void {
  const radius = (player: Player): number => player === state.team1.goalie || player === state.team2.goalie ?
    player.is_pad_stack ? 18 : Math.max(12, Math.min(state.engine.goalie_chk_body > 0 ? state.engine.goalie_chk_body : 12, 18)) : 14;
  const laneClear = (start: Vec2, end: Vec2, opponents: readonly Player[]): boolean => !opponents.some((player) =>
    Math.min(Math.abs(player.y - start[1]), Math.abs(player.y - end[1])) <= radius(player) + 8 &&
    intersects(start, end, [player.x, player.y], radius(player)));
  const inBand = (player: Player, team: Team): boolean => {
    const y = team.controller === 1 ? player.y : -player.y;
    return y >= 88 && y <= 264;
  };
  for (const team of [state.team1, state.team2]) {
    const opponents = team.controller === 1 ? state.team2 : state.team1;
    const passer = team.owns_scnum(state.engine.puck_owner) ?
      team.get_player_by_scnum(state.engine.puck_owner) : team.get_possession_player();
    const net = team.controller === 1 ? state.team2.net : state.team1.net;
    const targets = state._shot_target_points_for_net(net);
    const controlled = opponents.get_controlled_player();
    for (const player of [...team.players, team.goalie]) {
      player.dist_to_controlled_opp = controlled ? Math.hypot(player.x - controlled.x, player.y - controlled.y) : 0;
    }
    for (const player of team.players) {
      player.passing_lane_clear = false;
      player.one_timer_lane_good = false;
      player.clear_shot_lane = false;
      player.open_net_shot = false;
      if (passer && player !== passer) {
        const catchPoint: Vec2 = [player.x + pyRound(player.ori_x * 10), player.y + pyRound(player.ori_y * 10)];
        const dx = catchPoint[0] - passer.x, dy = catchPoint[1] - passer.y, length = Math.hypot(dx, dy);
        const angle = state.engine.pass_dir * Math.PI / 4;
        const matches = state.engine.pass_dir < 0 || state.engine.pass_dir > 7 ||
          length > 0 && (dx * Math.cos(angle) + dy * Math.sin(angle)) / length >= Math.cos(Math.PI / 4);
        const clear = matches && laneClear([passer.x, passer.y], catchPoint, [...opponents.players, opponents.goalie]);
        player.passing_lane_clear = clear;
        player.one_timer_lane_good = !!(clear && inBand(passer, team) && inBand(player, team) &&
          Math.abs(player.x) <= 71 && targets.some((point) => laneClear([player.x, player.y], point, opponents.players)));
      }
    }
    if (!passer || !team.players.includes(passer) || (team.controller === 1 ? passer.y : -passer.y) >= 264) continue;
    const goalie = opponents.goalie, gx = goalie.x + goalie.vx / 2, gy = goalie.y + goalie.vy / 2;
    const distance = Math.hypot(gx - passer.x, gy - passer.y);
    let shotTargets = targets;
    if (distance !== 0) {
      const metric = ((gx - passer.x) * (net.y - passer.y) - (gy - passer.y) * (net.left - passer.x) +
        (gx - passer.x) * (net.y - passer.y) - (gy - passer.y) * (net.right - passer.x)) / distance;
      if (Math.abs(metric) <= 44) {
        shotTargets = [
          [metric >= 0 ? net.right - 6 : net.left + 6, net.y],
          [Math.floor((net.left + net.right) / 2), net.y],
          [metric >= 0 ? net.left + 6 : net.right - 6, net.y],
        ];
      }
    }
    const clear = shotTargets.filter((point) => laneClear([passer.x, passer.y], point, opponents.players));
    passer.clear_shot_lane = clear.length > 0;
    passer.open_net_shot = clear.some((point) => !intersects([passer.x, passer.y], point, [goalie.x, goalie.y], radius(goalie)));
  }
}
