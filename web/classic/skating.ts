import { Player, type Vec2 } from "./state";
import { copySign, mod, pyRound, required, shiftRight } from "./numeric";
import { rom_direction, skating, VELOCITY_SCALE, velocity } from "./motion";
import { steering_direction } from "./target";

const DIRECTIONS: readonly Vec2[] = [[0, 200], [141, 141], [200, 0], [141, -141],
  [0, -200], [-141, -141], [-200, 0], [-141, 141]];

export function grounded_step(player: Player, pad: Vec2, { goalie = false, stop = false } = {}): Player {
  const future = Object.assign(new Player(), player);
  let [vx, vy] = velocity(player).map((component) => pyRound(component / VELOCITY_SCALE)) as [number, number];
  vx -= shiftRight(vx, 6) || Number(!!vx); vy -= shiftRight(vy, 6) || Number(!!vy);
  const x = player.precise_x ?? player.x, y = player.precise_y ?? player.y;
  future.precise_x = x + vx * VELOCITY_SCALE; future.precise_y = y + vy * VELOCITY_SCALE;
  future.x = Math.floor(future.precise_x); future.y = Math.floor(future.precise_y);
  let phase = mod(pyRound((player.facing_phase ?? player.facing ?? player.orientation) * 65536), 8 * 65536);
  let heading = Math.floor(phase / 65536);
  const [ux, uy] = pad, wanted = ux || uy ? mod(pyRound(Math.atan2(ux, uy) * 4 / Math.PI), 8) : 8;
  let accelerate = wanted < 8, braking = stop || goalie && wanted === 8;
  if (!goalie && wanted < 8) {
    const difference = mod(wanted - heading, 8);
    if (difference === 4) {
      const direction = vx || vy ? rom_direction(vx, vy) : 8;
      braking = direction !== 8 && mod(direction - heading + 2, 8) < 4;
      if (!braking) phase = mod(phase + (player.sprite_flipped_x ? -65536 : 65536), 8 * 65536);
      accelerate = false;
    } else if (difference) {
      const speed_word = shiftRight(vx * vx + vy * vy, 16);
      const increment = Math.max(0x180, 0x300 - speed_word) * (difference < 4 ? 16 : -16);
      phase = mod(phase + increment, 8 * 65536);
      heading = Math.floor(phase / 65536); accelerate = speed_word > 2;
    }
  }
  if (braking) {
    vx = Math.trunc(copySign(Math.max(0, Math.abs(vx) - 150), vx));
    vy = Math.trunc(copySign(Math.max(0, Math.abs(vy) - 150), vy));
  } else if (accelerate) {
    const agility = required(player.agility, "agility"), weight = required(player.weight, "weight");
    const factor = Math.floor((64 - Math.floor(weight / 4) + agility + player.movement_bonus + (goalie ? agility + 16 : 0)) / 2);
    const [dx, dy] = DIRECTIONS[goalie ? wanted : heading]!;
    const proposed: Vec2 = [vx + shiftRight(dx * factor, 5), vy + shiftRight(dy * factor, 5)];
    const [, limit] = skating(player);
    if (proposed[0] ** 2 + proposed[1] ** 2 <= pyRound((limit / VELOCITY_SCALE) ** 2)) [vx, vy] = proposed;
  }
  future.facing_phase = phase / 65536; future.facing = Math.floor(phase / 65536);
  future.motion_x = vx * VELOCITY_SCALE; future.motion_y = vy * VELOCITY_SCALE;
  return future;
}

export function* skating_route(player: Player, target: Vec2, frames: number): Generator<[number, Player]> {
  let carrier = Object.assign(new Player(), player);
  yield [0, carrier];
  for (let elapsed = 1; elapsed <= frames; elapsed++) {
    carrier = grounded_step(carrier, steering_direction(carrier, target)[0]);
    yield [elapsed, carrier];
  }
}
