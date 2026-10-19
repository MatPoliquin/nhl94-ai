import { GenesisRam } from "./ram";
import { RomGeometry } from "./rom-geometry";
import { EngineState, GameState, Net, Player, Stats, Team } from "./state";
import { refreshFeatures } from "./features";
import { CONTROLLER_PORTS, type ControllerPort, type ControllerValues } from "../controllers";
import type { Controller } from "../observations";

const VELOCITY_SCALE = 17 / 65536;

export class ClassicDecoder {
  private readonly geometry: RomGeometry;

  constructor(rom: Uint8Array) {
    this.geometry = new RomGeometry(rom);
  }

  decode(bytes: Uint8Array, controllers?: ControllerValues<Controller>): GameState {
    const ram = new GenesisRam(bytes), state = new GameState(), engine = state.engine;
    state.period = ram.u16(0xffc466) + 1;
    state.time = ram.u32(0xffc466);
    state.last_time = state.time;
    const puck = state.puck, base = 0xffb74a;
    puck.x = ram.i16(base); puck.y = ram.i16(base + 0x14);
    puck.vx = ram.i8(base + 0x28); puck.vy = ram.i8(base + 0x2a);
    puck.precise_x = ram.i32(base) / 65536;
    puck.precise_y = ram.i32(base + 0x14) / 65536;
    puck.motion_x = ram.i16(base + 0x28) * VELOCITY_SCALE;
    puck.motion_y = ram.i16(base + 0x2a) * VELOCITY_SCALE;
    puck.motion_z = ram.i16(base + 0x2c) * VELOCITY_SCALE;
    puck.height = ram.i32(base + 0x18) / 65536;
    puck.friction = ram.u8(base + 0x62) & 1 ? 511 / 512 : 63 / 64;
    engine.puck_owner = ram.i16(0xffb7aa);
    engine.puck_owner_known = true;
    engine.pass_target = ram.i16(0xffbee0);
    engine.last_puck_player = ram.i16(0xffbeda);
    engine.shot_player = ram.i16(0xffbed8);
    engine.pass_dir = ram.u16(0xffbedc);
    engine.pass_speed = ram.u16(0xffbede);
    engine.goalie_chk_body = ram.i16(0xffbf64);
    engine.sflags = ram.u16(0xffc2ec);
    engine.sflags2 = ram.u16(0xffc2ee);
    engine.ba_ps_flags = ram.u16(0xffc2f2);
    engine.word_ffc2f6 = ram.u16(0xffc2f6);
    engine.word_ffc2f8 = ram.u16(0xffc2f8);
    engine.word_ffc2fa = ram.u16(0xffc2fa);
    engine.offsides_enabled = !!(ram.u8(0xffc2ea) & 0x20);
    engine.clock_stopped = !!(ram.u8(0xffc2ea) & 1);
    engine.input_block = ram.u16(0xffd412);
    engine.camera = [-64 - ram.i16(0xffbd1e), -ram.i16(0xffbd20)];
    engine.goalie_modes = [ram.u16(0xffd05a), ram.u16(0xffd05c)];
    engine.goalie_hold_counts = [ram.u8(0xffbf06), ram.u8(0xffbf07)];
    engine.controller_teams = CONTROLLER_PORTS.map((port) => controllers?.[port].team ?? ram.u16(0xffc328 + port * 2));
    if (!engine.controller_teams[2] && !engine.controller_teams[3]) engine.controller_teams = engine.controller_teams.slice(0, 2);
    if (engine.controller_teams.length === 4) engine.goalie_hold_counts = CONTROLLER_PORTS.map((port) => ram.u8(0xffbf06 + port));
    engine.highlight = !!(ram.u8(0xffc2ea) & 0x10);
    engine.shot_mode_active = Number(!!(engine.sflags & 8));
    engine.shot_taken = Number(!!(engine.sflags2 & 16));
    engine.in_close_top_shelf = Number(!!(engine.word_ffc2f6 & 16));
    engine.one_timer_collision_mode = Number(!!(engine.word_ffc2f8 & 2));
    engine.breakaway_context = Number(!!(engine.word_ffc2fa & 16));
    engine.goalie_box_small = Number(engine.goalie_chk_body <= 12);
    state.is_shooting = !!(engine.sflags & 0x800);
    state.has_released_shot = !!(engine.sflags2 & 0x1000);
    state.is_shootout_active = !!(engine.ba_ps_flags & 0x400);
    const flags = ram.u8(0xffc2fc);
    const bonus = flags & 0x40 || ram.u8(0xffc2fe) & 2 ? 2 : 0;
    const teammates = [1, 2].some((side) => engine.controller_teams!.filter((team) => team === side).length > 1);
    for (const team of [state.team1, state.team2]) {
      const stats = 0xffc6ce + (team.controller - 1) * 0x364;
      team.stats.score = ram.u16(stats + 0x0c);
      team.stats.shots = ram.u16(stats);
      team.stats.bodychecks = ram.u16(stats + 0x10);
      team.stats.attackzone = ram.u32(stats + 8);
      team.stats.faceoffwon = ram.u16(team.controller === 1 ? stats + 0x0e : stats + 0x0c);
      team.stats.passing = ram.u16(stats + 0x14);
      team.stats.onetimer = ram.u32(stats + 0x35a);
      team.stats.fullstar_x = ram.i16(team.controller === 1 ? 0xffbf7c : 0xffbdb4);
      team.stats.fullstar_y = ram.i16(0xffbf7e);
      team.stats.emptystar_x = team.controller === 1 ? ram.i16(0xffbdd0) : 0;
      team.stats.emptystar_y = team.controller === 1 ? ram.i16(0xffbde4) : 0;
      team.last_stats = Object.assign(new Stats(), team.stats);
      team.pass_attempts = ram.u16(stats + 0x12);
      team.one_timer_attempts = ram.u16(stats + 0x35c);
      team.defense_goalie = team.goalie_scnum();
      let selected = -1;
      for (const port of CONTROLLER_PORTS) {
        if (engine.controller_teams[port] === team.controller) selected = controllers?.[port].slot ?? ram.i16(0xffc320 + port * 2);
      }
      team.defense_control = selected;
      team.control = selected < 0 ? -1 : selected === team.goalie_scnum() ? 0 : selected - team.skater_scnum_base() + 1;
      team.player_haspuck = team.owns_scnum(engine.puck_owner) && engine.puck_owner !== team.goalie_scnum();
      team.goalie_haspuck = engine.puck_owner === team.goalie_scnum();
      const direction = state.period % 2 === 0 ? -1 : 1;
      team.net.y = direction * (team.controller === 1 ? -264 : 264);
      for (const [index, player] of [...team.players, team.goalie].entries()) {
        const slot = team.skater_scnum_base() + index, address = 0xffb04a + slot * 0x80;
        this.player(player, ram, address, team, index === 5, flags, bonus);
        player.has_puck = Number(slot === engine.puck_owner);
        player.rel_puck_x = puck.x - player.x;
        player.rel_puck_y = puck.y - player.y;
        player.rel_puck_vx = puck.vx - player.vx;
        player.rel_puck_vy = puck.vy - player.vy;
        player.dist_to_puck = Math.hypot(player.rel_puck_x, player.rel_puck_y);
        if (index !== 5) this.geometry.update(player, ram, address, !!(engine.sflags & 0x8000), teammates);
      }
      team.update_controlled_relationships();
      const controlled = team.get_controlled_player();
      if (controlled) {
        team.net.rel_controlled_left = team.net.left - controlled.x;
        team.net.rel_controlled_right = team.net.right - controlled.x;
        team.net.rel_controlled_y = team.net.y - controlled.y;
      }
    }
    engine.controlled_is_shooter = Number(engine.shot_player >= 0 && engine.shot_player === state.team1.controlled_scnum());
    refreshFeatures(state);
    return state;
  }

  private player(player: Player, ram: GenesisRam, base: number, team: Team, goalie: boolean, flags: number, bonus: number): void {
    player.x = ram.i16(base); player.y = ram.i16(base + 0x14);
    player.vx = goalie ? 0 : ram.i8(base + 0x28);
    player.vy = goalie ? 0 : ram.i8(base + 0x2a);
    player.orientation = goalie ? 0 : ram.u8(base + 0x55);
    player.ori_x = goalie ? 0 : Math.cos(player.orientation * Math.PI / 4);
    player.ori_y = goalie ? 0 : Math.sin(player.orientation * Math.PI / 4);
    player.anim = ram.u16(base + 0x58);
    player.anim_frame = ram.u16(base + 0x5a);
    player.state_flags = ram.u8(base + 0x64);
    player.is_one_timer = Number(!!(player.state_flags & 8));
    player.is_breakaway = Number(!!(player.state_flags & 2));
    player.is_falling = Number(!!(player.state_flags & 32));
    player.is_pad_stack = Number(player.anim === 0x250 || player.anim === 0x2a2);
    player.is_dive = Number(player.anim === 0x2f4);
    player.has_anim = Number(player.anim !== 0);
    player.precise_x = ram.i32(base) / 65536;
    player.precise_y = ram.i32(base + 0x14) / 65536;
    player.motion_x = ram.i16(base + 0x28) * VELOCITY_SCALE;
    player.motion_y = ram.i16(base + 0x2a) * VELOCITY_SCALE;
    player.role = ram.i16(base + 0x34);
    player.facing = ram.u16(base + 0x54);
    player.facing_phase = goalie ? null : ram.u32(base + 0x54) / 65536;
    player.selection_flags = ram.u8(base + 0x62);
    player.unavailable = ram.u8(base + 0x63);
    player.movement_bonus = bonus;
    player.speed = ram.u8(base + 0x69);
    player.agility = ram.u8(base + 0x68);
    player.weight = ram.u8(base + 0x67);
    player.stick = ram.u8(base + 0x71);
    player.passing = ram.u8(base + 0x6e);
    const roster = ram.u8(base + 0x66);
    player.energy = flags & 0x10 ? 4096 : roster < 26 ?
      ram.u16(0xffc700 + (team.controller - 1) * 0x364 + roster * 2) : null;
    player.decision_timer = ram.i8(base + 0x40);
    player.steering = goalie ? ram.i8(base + 0x43) : ram.u8(base + 0x43);
    const assignment = ram.u16(base + 0x36);
    player.assignment = assignment < 8 ? ram.u8(base + 0x38 + assignment) : null;
    player.live_anim = player.anim;
    player.live_anim_frame = player.anim_frame;
    player.animation_timer = ram.i8(base + 0x5c);
    player.live_state_flags = goalie ? player.state_flags : null;
    player.cover_timer = goalie ? ram.i16(base + 0x48) : null;
    player.contact_player = ram.i16(base + 0x2e);
    player.contact_impact = ram.u16(base + 0x32);
    if (goalie) {
      player.puck_control = ram.u8(base + 0x6c);
      player.decision_interval = Math.floor(Math.max(0, ram.u8(base + 0x6b) - Number(!!(flags & 0x40))) / 4);
      player.cpu_tracking_active = !(player.selection_flags & 0x20 || player.unavailable & 2);
    } else {
      player.shot_accuracy = ram.u8(base + 0x6d);
      player.shot_power = ram.u8(base + 0x6c);
      player.handedness = ram.u8(base + 0x76);
      player.checking = ram.u8(base + 0x75);
      player.endurance = ram.u8(base + 0x72);
      player.burst_full_energy = !!(flags & 0x10);
      player.steering_timer = ram.i8(base + 0x42);
      player.cpu_target = [ram.i16(base + 0x44), ram.i16(base + 0x46)];
      player.support_zone = ram.i16(base + 0x48);
      player.cpu_response_available = !(player.selection_flags & 0x3a) && !(player.unavailable & 0x27);
    }
  }
}

export function controllerView(state: GameState, port: ControllerPort, slot: number): GameState {
  const side = state.engine.controller_teams?.[port];
  if (side !== 1 && side !== 2) throw new Error("Classic requires an assigned Home or Away controller.");
  const result = Object.assign(new GameState(), state);
  const team = side === 1 ? state.team1 : state.team2;
  const cloneTeam = (team: Team): Team => Object.assign(new Team(team.controller), team, {
    players: team.players.map((player) => Object.assign(new Player(), player)),
    goalie: Object.assign(new Player(), team.goalie), net: Object.assign(new Net(), team.net),
  });
  result.team1 = cloneTeam(team);
  result.team2 = cloneTeam(side === 1 ? state.team2 : state.team1);
  result.team1.defense_control = slot;
  result.team1.control = slot < 0 ? -1 : slot === team.goalie_scnum() ? 0 : slot - team.skater_scnum_base() + 1;
  result.team1.update_controlled_relationships();
  const selected = result.team1.get_controlled_player();
  if (selected) {
    result.team1.net.rel_controlled_left = result.team1.net.left - selected.x;
    result.team1.net.rel_controlled_right = result.team1.net.right - selected.x;
    result.team1.net.rel_controlled_y = result.team1.net.y - selected.y;
  }
  for (const player of [...result.team2.players, result.team2.goalie]) {
    player.dist_to_controlled_opp = selected ? Math.hypot(player.x - selected.x, player.y - selected.y) : 0;
  }
  if ((result.engine.controller_teams?.filter((team) => team === side).length ?? 0) > 1) {
    result.time = state.time & 0xffff;
    result.last_time = result.time;
  }
  result.engine = Object.assign(new EngineState(), state.engine);
  result.engine.controlled_is_shooter = Number(result.engine.shot_player >= 0 && result.engine.shot_player === slot);
  return result;
}
