export type Vec2 = readonly [number, number];
export type Action = [number, number, number, number, number, number, number, number, number, number, number, number];

export function neutralAction(): Action {
  return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
}

export const Buttons = {
  INPUT_B: 0, INPUT_A: 1, INPUT_MODE: 2, INPUT_START: 3,
  INPUT_UP: 4, INPUT_DOWN: 5, INPUT_LEFT: 6, INPUT_RIGHT: 7,
  INPUT_C: 8, INPUT_Y: 9, INPUT_X: 10, INPUT_Z: 11, INPUT_MAX: 12,
  SLOT_BAND_MAX_X: 71, SLOT_BAND_MIN_Y: 88,
} as const;

export class Player {
  x = 0;
  y = 0;
  vx = 0;
  vy = 0;
  anim = 0;
  anim_frame = 0;
  state_flags = 0;
  orientation = 0;
  ori_x = 0;
  ori_y = 0;
  rel_puck_x = 0;
  rel_puck_y = 0;
  rel_puck_vx = 0;
  rel_puck_vy = 0;
  rel_controlled_x = 0;
  rel_controlled_y = 0;
  rel_controlled_vx = 0;
  rel_controlled_vy = 0;
  dist_to_controlled = 0;
  dist_to_controlled_opp = 0;
  dist_to_puck = 0;
  is_controlled = 0;
  has_puck = 0;
  passing_lane_clear = false;
  one_timer_lane_good = false;
  clear_shot_lane = false;
  open_net_shot = false;
  is_one_timer = 0;
  is_breakaway = 0;
  is_falling = 0;
  is_pad_stack = 0;
  is_dive = 0;
  has_anim = 0;
  shot_accuracy: number | null = null;
  shot_power: number | null = null;
  puck_control: number | null = null;
  handedness: number | null = null;
  passing: number | null = null;
  stick_x: number | null = null;
  stick_y: number | null = null;
  shot_durations: readonly number[] | null = null;
  shot_offsets_y: Vec2 | null = null;
  shot_offsets_x: Vec2 | null = null;
  shot_projection_offsets_x: Vec2 | null = null;
  shot_projection_offsets_y: Vec2 | null = null;
  shot_projection_durations: readonly number[] | null = null;
  precise_x: number | null = null;
  precise_y: number | null = null;
  facing_phase: number | null = null;
  sprite_flipped_x: boolean | null = null;
  motion_x: number | null = null;
  motion_y: number | null = null;
  projection_uncertainty = 0;
  speed: number | null = null;
  agility: number | null = null;
  weight: number | null = null;
  energy: number | null = null;
  stick: number | null = null;
  checking: number | null = null;
  endurance: number | null = null;
  role: number | null = null;
  facing: number | null = null;
  selection_flags: number | null = null;
  unavailable = 0;
  movement_bonus = 0;
  burst_full_energy: boolean | null = null;
  height: number | null = null;
  motion_z: number | null = null;
  friction = 511 / 512;
  live_anim: number | null = null;
  live_anim_frame: number | null = null;
  animation_timer: number | null = null;
  live_state_flags: number | null = null;
  assignment: number | null = null;
  decision_timer: number | null = null;
  decision_interval: number | null = null;
  steering: number | null = null;
  cpu_tracking_active: boolean | null = null;
  steering_timer: number | null = null;
  cpu_target: Vec2 | null = null;
  support_zone: number | null = null;
  cpu_response_available = false;
  cover_timer: number | null = null;
  contact_player: number | null = null;
  contact_impact: number | null = null;
  input_state: Record<string, number> | null = null;
  input_state_known = false;
  readonly platform = "genesis";
  native_state_flags: number | null = null;
  one_timer_offsets: readonly Vec2[] | null = null;
}

export class Net {
  depth = 10;
  left = -19;
  right = 19;
  y = 0;
  rel_controlled_left = 0;
  rel_controlled_right = 0;
  rel_controlled_y = 0;
}

export class Stats {
  score = 0;
  shots = 0;
  bodychecks = 0;
  attackzone = 0;
  faceoffwon = 0;
  passing = 0;
  onetimer = 0;
  fullstar_x = 0;
  fullstar_y = 0;
  emptystar_x = 0;
  emptystar_y = 0;
}

export class GoalieStats {
  glove_left = 0;
  glove_right = 0;
  stick_left = 0;
  stick_right = 0;
  puck_control = 0;
  agility = 0;
  speed = 0;
  passing = 0;
  endurance = 0;
  weight = 0;
}

export class EngineState {
  puck_owner = -1;
  puck_owner_known = false;
  offsides_enabled: boolean | null = null;
  pass_target: number | null = null;
  last_puck_player: number | null = null;
  shot_player = -1;
  pass_dir = 0;
  pass_speed = 0;
  goalie_chk_body = 0;
  sflags = 0;
  sflags2 = 0;
  ba_ps_flags = 0;
  word_ffc2f6 = 0;
  word_ffc2f8 = 0;
  word_ffc2fa = 0;
  shot_mode_active = 0;
  shot_taken = 0;
  in_close_top_shelf = 0;
  one_timer_collision_mode = 0;
  breakaway_context = 0;
  controlled_is_shooter = 0;
  goalie_box_small = 0;
  goalie_modes: Vec2 | null = null;
  goalie_hold_counts: Vec2 | null = null;
  controller_teams: Vec2 | null = null;
  input_block: number | null = null;
  clock_stopped: boolean | null = null;
  camera: Vec2 | null = null;
  faceoff = false;

  get native_one_timer_shot(): boolean {
    return !!(this.word_ffc2f8 & 0x0200);
  }
}

export class Team {
  readonly num_players = 5;
  players = Array.from({ length: 5 }, () => new Player());
  goalie = new Player();
  stats = new Stats();
  last_stats = new Stats();
  goalie_stats = new GoalieStats();
  net = new Net();
  control = -1;
  player_haspuck = false;
  goalie_haspuck = false;
  defense_control: number | null = null;
  defense_goalie: number | null = null;
  pass_attempts: number | null = null;
  one_timer_attempts: number | null = null;

  constructor(public readonly controller: number) {}

  get_controlled_player(): Player | null {
    return this.control < 0 ? null : this.control === 0 ? this.goalie : this.players[this.control - 1] ?? null;
  }

  controlled_scnum(): number {
    return this.control < 0 ? -1 : this.control === 0 ? this.goalie_scnum() : this.skater_scnum_base() + this.control - 1;
  }

  skater_scnum_base(): number { return this.controller === 1 ? 0 : 6; }
  goalie_scnum(): number { return this.controller === 1 ? 5 : 11; }
  owns_scnum(slot: number): boolean {
    return slot >= this.skater_scnum_base() && slot <= this.goalie_scnum();
  }

  get_player_by_scnum(slot: number): Player | null {
    return slot === this.goalie_scnum() ? this.goalie :
      this.players[slot - this.skater_scnum_base()] ?? null;
  }

  get_possession_player(): Player | null {
    return this.goalie_haspuck ? this.goalie : this.players.find((player) => this.has_puck(player.x, player.y)) ?? null;
  }

  has_puck(x: number, y: number): boolean {
    return Math.abs(x - this.stats.fullstar_x) < 3 && Math.abs(y - this.stats.fullstar_y) < 3;
  }

  update_controlled_relationships(): void {
    const selected = this.get_controlled_player();
    for (const player of [...this.players, this.goalie]) {
      player.rel_controlled_x = selected ? player.x - selected.x : 0;
      player.rel_controlled_y = selected ? player.y - selected.y : 0;
      player.rel_controlled_vx = selected ? player.vx - selected.vx : 0;
      player.rel_controlled_vy = selected ? player.vy - selected.vy : 0;
      player.dist_to_controlled = Math.hypot(player.rel_controlled_x, player.rel_controlled_y);
    }
  }
}

export class GameState {
  team1 = new Team(1);
  team2 = new Team(2);
  puck = new Player();
  engine = new EngineState();
  period = 1;
  time = 0;
  last_time = 0;
  is_shootout_active = false;
  is_shooting = false;
  has_released_shot = false;
  readonly rink = { net_y: 264, post_x: 19, net_depth: 10, player_x: 120, player_y: 270, puck_x: 130, puck_y: 270, max_velocity: 50 };

  _shot_target_points_for_net(net: Net): readonly Vec2[] {
    return [[Math.floor((net.left + net.right) / 2), net.y], [net.left + 6, net.y], [net.right - 6, net.y]];
  }
}
