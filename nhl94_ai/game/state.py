"""
NHL94 Game State
"""

from nhl94_ai.game.features import TacticalFeatures
from nhl94_ai.game.ram import decode_shot_count
import math
from nhl94_ai.game.constants import GameConsts
from typing import Dict, Any
from copy import deepcopy
from dataclasses import dataclass
import time

@dataclass
class Net:
    depth: int = GameConsts.NET_DEPTH
    left: int = GameConsts.P1_NET_LEFT_POLL
    right: int = GameConsts.P1_NET_RIGHT_POLL
    y: int = 0 #center of net

    rel_controlled_left: int = 0
    rel_controlled_right: int = 0
    rel_controlled_y: int = 0 #center of net

@dataclass
class Player:
    x: int = 0
    y: int = 0
    vx: int = 0
    vy: int = 0
    anim: int = 0
    anim_frame: int = 0
    state_flags: int = 0
    orientation: int = 0  # New field for orientation (0-7)
    ori_x: float = 0.0    # Normalized x component of orientation
    ori_y: float = 0.0    # Normalized y component of orientation
    rel_puck_x: float = 0.0  # Relative puck x position
    rel_puck_y: float = 0.0  # Relative puck y position
    rel_puck_vx: float = 0.0 # Relative puck x velocity
    rel_puck_vy: float = 0.0 # Relative puck y velocity
    rel_controlled_x: float = 0.0  # New: Relative to controlled player x position
    rel_controlled_y: float = 0.0  # New: Relative to controlled player y position
    rel_controlled_vx: float = 0.0 # New: Relative to controlled player x velocity
    rel_controlled_vy: float = 0.0 # New: Relative to controlled player y velocity
    dist_to_controlled: float = 0.0  # New: Distance to controlled player
    dist_to_controlled_opp: float = 0.0  # New: Distance to controlled opponent player
    dist_to_puck: float = 0.0  # New: Distance to puck
    is_controlled: float = 0.0
    has_puck: float = 0.0
    passing_lane_clear: bool = False
    one_timer_lane_good: bool = False
    clear_shot_lane: bool = False
    open_net_shot: bool = False
    is_one_timer: float = 0.0
    is_breakaway: float = 0.0
    is_falling: float = 0.0
    is_pad_stack: float = 0.0
    is_dive: float = 0.0
    has_anim: float = 0.0
    # Optional live ROM attribute; deliberately absent from neural encodings.
    shot_accuracy: int | None = None
    shot_power: int | None = None
    puck_control: int | None = None
    handedness: int | None = None
    passing: int | None = None
    stick_x: int | None = None
    stick_y: int | None = None
    shot_durations: tuple[int, ...] | None = None
    shot_offsets_y: tuple[int, int] | None = None
    shot_offsets_x: tuple[int, int] | None = None
    shot_projection_offsets_x: tuple[int, int] | None = None
    shot_projection_offsets_y: tuple[int, int] | None = None
    shot_projection_durations: tuple[int, ...] | None = None
    precise_x: float | None = None
    precise_y: float | None = None
    facing_phase: float | None = None
    sprite_flipped_x: bool | None = None
    # Position integration uses signed velocity * 17 per elapsed game tick.
    # These optional values are not part of the historical neural encoding.
    motion_x: float | None = None
    motion_y: float | None = None
    # Bounded forecast uncertainty, not RAM telemetry or a model input.
    projection_uncertainty: float = 0.0
    # Scripted defense telemetry is never included in normalized policy inputs.
    speed: int | None = None
    agility: int | None = None
    weight: int | None = None
    energy: int | None = None
    stick: int | None = None
    checking: int | None = None
    endurance: int | None = None
    role: int | None = None
    facing: int | None = None
    selection_flags: int | None = None
    unavailable: int = 0
    movement_bonus: int = 0
    burst_full_energy: bool | None = None
    height: float | None = None
    motion_z: float | None = None
    friction: float = 511 / 512
    live_anim: int | None = None
    live_anim_frame: int | None = None
    animation_timer: int | None = None
    live_state_flags: int | None = None
    assignment: int | None = None
    decision_timer: int | None = None
    decision_interval: int | None = None
    steering: int | None = None
    cpu_tracking_active: bool | None = None
    steering_timer: int | None = None
    cpu_target: tuple[int, int] | None = None
    support_zone: int | None = None
    cpu_response_available: bool = False
    cover_timer: int | None = None
    contact_player: int | None = None
    contact_impact: int | None = None
    input_state: dict[str, int] | None = None
    input_state_known: bool = False
    platform: str = 'genesis'
    native_state_flags: int | None = None
    one_timer_offsets: tuple[tuple[int, int], ...] | None = None

    def debug_print(self, prefix="Player"):
        print(f"{prefix} - x: {self.x}, y: {self.y}, vx: {self.vx}, vy: {self.vy}, "
            f"anim: {hex(self.anim)}, anim_frame: {self.anim_frame}, state_flags: {bin(self.state_flags)}, "
            f"orientation: {self.orientation}, ori_vec: ({self.ori_x:.2f}, {self.ori_y:.2f})\n"
            f"  rel_puck: ({self.rel_puck_x:.1f}, {self.rel_puck_y:.1f}) "
            f"rel_puck_vel: ({self.rel_puck_vx:.1f}, {self.rel_puck_vy:.1f})\n"
            f"  rel_controlled: ({self.rel_controlled_x:.1f}, {self.rel_controlled_y:.1f}) "
            f"rel_controlled_vel: ({self.rel_controlled_vx:.1f}, {self.rel_controlled_vy:.1f})\n"
            f"  dist_to_controlled: {self.dist_to_controlled:.1f}, "
            f"dist_to_controlled_opp: {self.dist_to_controlled_opp:.1f}, "
            f"dist_to_puck: {self.dist_to_puck:.1f}\n"
            f"  controlled: {self.is_controlled}, has_puck: {self.has_puck}\n"
            f"  one_timer: {self.is_one_timer}, breakaway: {self.is_breakaway}, "
            f"falling: {self.is_falling}, pad_stack: {self.is_pad_stack}, dive: {self.is_dive}\n"
            f"  passing_lane_clear: {self.passing_lane_clear}, "
            f"one_timer_lane_good: {self.one_timer_lane_good}, "
            f"clear_shot_lane: {self.clear_shot_lane}, "
            f"open_net_shot: {self.open_net_shot}")

@dataclass
class Stats:
    score: int = 0
    shots: int = 0
    bodychecks: int = 0
    attackzone: int = 0
    faceoffwon: int = 0
    passing: int = 0
    onetimer: int = 0
    fullstar_x: int = 0
    fullstar_y: int = 0
    emptystar_x: int = 0
    emptystar_y: int = 0

    def debug_print(self, prefix="Stats"):
        print(f"{prefix} - score: {self.score}, shots: {self.shots}, bodychecks: {self.bodychecks}, "
              f"attackzone: {self.attackzone}, faceoffwon: {self.faceoffwon}, passing: {self.passing}, onetimer: {self.onetimer},"
              f"fullstar: ({self.fullstar_x},{self.fullstar_y}), emptystar: ({self.emptystar_x},{self.emptystar_y})")


@dataclass
class GoalieStats:
    glove_left: float = 0.0
    glove_right: float = 0.0
    stick_left: float = 0.0
    stick_right: float = 0.0
    puck_control: float = 0.0
    agility: float = 0.0
    speed: float = 0.0
    passing: float = 0.0
    endurance: float = 0.0
    weight: float = 0.0

    def debug_print(self, prefix="GoalieStats"):
        print(
            f"{prefix} - glove_left: {self.glove_left}, glove_right: {self.glove_right}, "
            f"stick_left: {self.stick_left}, stick_right: {self.stick_right}, "
            f"puck_control: {self.puck_control}, agility: {self.agility}, speed: {self.speed}, "
            f"passing: {self.passing}, endurance: {self.endurance}, weight: {self.weight}"
        )


@dataclass
class EngineState:
    @property
    def native_one_timer_shot(self):
        """ROM byte bit 1, when the integration reads FFC2F8 as big-endian u16.

        Keep the historical normalized one_timer_collision_mode field unchanged
        for saved skater-policy compatibility; new goalie code uses this bit.
        """
        return bool(self.word_ffc2f8 & 0x0200)

    puck_owner: int = -1
    puck_owner_known: bool = False
    offsides_enabled: bool | None = None
    pass_target: int | None = None
    last_puck_player: int | None = None
    shot_player: int = -1
    pass_dir: int = 0
    pass_speed: int = 0
    goalie_chk_body: int = 0
    sflags: int = 0
    sflags2: int = 0
    ba_ps_flags: int = 0
    word_ffc2f6: int = 0
    word_ffc2f8: int = 0
    word_ffc2fa: int = 0
    shot_mode_active: float = 0.0
    shot_taken: float = 0.0
    in_close_top_shelf: float = 0.0
    one_timer_collision_mode: float = 0.0
    breakaway_context: float = 0.0
    controlled_is_shooter: float = 0.0
    goalie_box_small: float = 0.0
    goalie_modes: tuple | None = None
    goalie_hold_counts: tuple | None = None
    controller_teams: tuple | None = None
    input_block: int | None = None
    clock_stopped: bool | None = None
    camera: tuple | None = None
    faceoff: bool = False

    def debug_print(self, prefix="EngineState"):
        print(
            f"{prefix} - puck_owner: {self.puck_owner}, shot_player: {self.shot_player}, "
            f"pass_dir: {self.pass_dir}, pass_speed: {self.pass_speed}, goalie_chk_body: {self.goalie_chk_body}\n"
            f"  shot_mode_active: {self.shot_mode_active}, shot_taken: {self.shot_taken}, "
            f"in_close_top_shelf: {self.in_close_top_shelf}, one_timer_collision_mode: {self.one_timer_collision_mode},\n"
            f"  breakaway_context: {self.breakaway_context}, controlled_is_shooter: {self.controlled_is_shooter}, "
            f"goalie_box_small: {self.goalie_box_small}"
        )

class Team():
    HAS_PUCK_TRESHOLD = 3
    PAD_STACK_ANIMS = {0x250, 0x2A2}
    DIVE_ANIM = 0x2F4

    def __init__(self, controller: int, num_players: int, platform=None):
        from nhl94_ai.game.platforms import get_backend
        self.platform = platform or get_backend()
        self.rink = self.platform.rink
        self.controller = controller
        self.stats = Stats()
        self.last_stats = Stats()
        self.goalie_stats = GoalieStats()
        self.num_players = num_players

        self.players = [Player() for _ in range(num_players)]
        self.goalie = Player()

        self.ram_var_prefix = 'p1_' if controller == 1 else 'p2_'
        self.ram_var_goalie_prefix = 'g1_' if controller == 1 else 'g2_'
        self.control = 0  # -1: no selection; 0: goalie; 1..N: skater.
        self.player_haspuck = False
        self.goalie_haspuck = False
        self.defense_control = None
        self.defense_goalie = None
        self.pass_attempts = None
        self.one_timer_attempts = None

        # for model input
        self.nz_players = [Player() for _ in range(num_players)]
        self.nz_goalie = Player()
        self.nz_player_haspuck = 0.0
        self.nz_goalie_haspuck = 0.0
        self.nz_goalie_stats = GoalieStats()

        self.net = Net(depth=self.rink.net_depth, left=-self.rink.post_x, right=self.rink.post_x)
        self.nz_net = Net()

        net_left: tuple = (0.0, 0.0)        # Normalized absolute left post
        nz_net_right: tuple = (0.0, 0.0)       # Normalized absolute right post
        nz_net_center: tuple = (0.0, 0.0)      # Normalized absolute center
        nz_net_left_rel: tuple = (0.0, 0.0)    # Relative to controlled player
        nz_net_right_rel: tuple = (0.0, 0.0)
        nz_net_center_rel: tuple = (0.0, 0.0)

    def get_controlled_player(self) -> Player | None:
        if self.control == -1:
            return None
        return self.goalie if self.control == 0 else self.players[self.control - 1]

    def controlled_scnum(self) -> int:
        if self.control == -1:
            return -1
        if self.control == 0:
            return 5 if self.controller == 1 else 11
        base_scnum = 0 if self.controller == 1 else 6
        return base_scnum + (self.control - 1)

    def skater_scnum_base(self) -> int:
        return 0 if self.controller == 1 else 6

    def goalie_scnum(self) -> int:
        return 5 if self.controller == 1 else 11

    def owns_scnum(self, scnum: int) -> bool:
        skater_base = self.skater_scnum_base()
        return skater_base <= scnum < (skater_base + self.num_players) or scnum == self.goalie_scnum()

    def get_player_by_scnum(self, scnum: int) -> Player | None:
        if scnum == self.goalie_scnum():
            return self.goalie

        skater_base = self.skater_scnum_base()
        skater_index = scnum - skater_base
        if 0 <= skater_index < self.num_players:
            return self.players[skater_index]

        return None

    def get_possession_player(self) -> Player | None:
        if self.goalie_haspuck:
            return self.goalie

        for player in self.players:
            if self.has_puck(player.x, player.y):
                return player

        return None

    def update_controlled_relationships(self):
        controlled = self.get_controlled_player()
        for player in [*self.players, self.goalie]:
            if controlled is None:
                player.rel_controlled_x = player.rel_controlled_y = 0
                player.rel_controlled_vx = player.rel_controlled_vy = 0
                player.dist_to_controlled = 0
            else:
                player.rel_controlled_x = player.x - controlled.x
                player.rel_controlled_y = player.y - controlled.y
                player.rel_controlled_vx = player.vx - controlled.vx
                player.rel_controlled_vy = player.vy - controlled.vy
                player.dist_to_controlled = GameConsts.Distance(
                    (player.x, player.y), (controlled.x, controlled.y),
                )

    def _load_hidden_player_fields(self, player: Player, info: Dict[str, Any], prefix: str) -> None:
        self.platform.decode_player(player, info, prefix)

        nz_net_left: tuple = (0.0, 0.0)        # Normalized absolute left post
        nz_net_right: tuple = (0.0, 0.0)       # Normalized absolute right post
        nz_net_center: tuple = (0.0, 0.0)      # Normalized absolute center
        nz_net_left_rel: tuple = (0.0, 0.0)    # Relative to controlled player
        nz_net_right_rel: tuple = (0.0, 0.0)
        nz_net_center_rel: tuple = (0.0, 0.0)

    def _load_goalie_stats(self, info: Dict[str, Any]) -> None:
        prefix = f"{self.ram_var_goalie_prefix}goalie_"
        self.goalie_stats.glove_left = info.get(f"{prefix}glove_left", 0) or 0
        self.goalie_stats.glove_right = info.get(f"{prefix}glove_right", 0) or 0
        self.goalie_stats.stick_left = info.get(f"{prefix}stick_left", 0) or 0
        self.goalie_stats.stick_right = info.get(f"{prefix}stick_right", 0) or 0
        self.goalie_stats.puck_control = info.get(f"{prefix}puck_control", 0) or 0
        self.goalie_stats.agility = info.get(f"{prefix}agility", 0) or 0
        self.goalie_stats.speed = info.get(f"{prefix}speed", 0) or 0
        self.goalie_stats.passing = info.get(f"{prefix}passing", 0) or 0
        self.goalie_stats.endurance = info.get(f"{prefix}endurance", 0) or 0
        self.goalie_stats.weight = info.get(f"{prefix}weight", 0) or 0

    def _load_defense_fields(self, info):
        self.platform.load_team_telemetry(self, info)

    def has_puck(self, pos_x, pos_y):
        return (abs(pos_x - self.stats.fullstar_x) < self.HAS_PUCK_TRESHOLD and abs(pos_y - self.stats.fullstar_y) < self.HAS_PUCK_TRESHOLD)

    def has_control(self, pos_x, pos_y):
        return (abs(pos_x - self.stats.emptystar_x) < self.HAS_PUCK_TRESHOLD and abs(pos_y - self.stats.emptystar_y) < self.HAS_PUCK_TRESHOLD)

    def begin_frame(self, info: Dict[str, Any], puck_x: int, puck_y: int, puck_vx: int, puck_vy: int) -> None:
        # General state
        self.stats.score = info.get(f"{self.ram_var_prefix}score")
        self.stats.shots = decode_shot_count(info[f"{self.ram_var_prefix}shots"])
        self.stats.bodychecks = info.get(f"{self.ram_var_prefix}bodychecks")
        self.stats.attackzone = info.get(f"{self.ram_var_prefix}attackzone")
        self.stats.faceoffwon = info.get(f"{self.ram_var_prefix}faceoffwon")
        self.stats.passing = info.get(f"{self.ram_var_prefix}passing")
        self.stats.onetimer = info.get(f"{self.ram_var_prefix}onetimer")

        # special case for team 1 as the stable-retro ram var name don't have the prefix
        if self.controller == 1:
            self.stats.fullstar_x = info.get("fullstar_x")
            self.stats.fullstar_y = info.get("fullstar_y")
        else:
            self.stats.fullstar_x = info.get(f"{self.ram_var_prefix}fullstar_x")
            self.stats.fullstar_y = info.get(f"{self.ram_var_prefix}fullstar_y")

        self.stats.emptystar_x = info.get(f"{self.ram_var_prefix}emptystar_x", 0)
        self.stats.emptystar_y = info.get(f"{self.ram_var_prefix}emptystar_y", 0)
        self._load_goalie_stats(info)

        # Goalie
        self.goalie.x = info.get(f"{self.ram_var_goalie_prefix}x")
        self.goalie.y = info.get(f"{self.ram_var_goalie_prefix}y")
        self.goalie.vx = info.get(f"{self.ram_var_goalie_prefix}vel_x", 0)
        self.goalie.vy = info.get(f"{self.ram_var_goalie_prefix}vel_y", 0)
        self._load_hidden_player_fields(self.goalie, info, f"{self.ram_var_goalie_prefix}")
        input_players = info.get('model_input_players', {})
        goalie_slot = self.skater_scnum_base() + self.num_players
        self.goalie.input_state_known = goalie_slot in input_players
        self.goalie.input_state = input_players.get(goalie_slot)

        # Players
        for p in range(0, self.num_players):
            if p == 0:
                self.players[p].x = info.get(f"{self.ram_var_prefix}x")
                self.players[p].y = info.get(f"{self.ram_var_prefix}y")
                self.players[p].vx = info.get(f"{self.ram_var_prefix}vel_x")
                self.players[p].vy = info.get(f"{self.ram_var_prefix}vel_y")
                self.players[p].orientation = info.get(f"{self.ram_var_prefix}ori", 0)
                self._load_hidden_player_fields(self.players[p], info, self.ram_var_prefix)
            else:
                pi = p + 1
                self.players[p].x = info.get(f"{self.ram_var_prefix}{pi}_x")
                self.players[p].y = info.get(f"{self.ram_var_prefix}{pi}_y")
                self.players[p].vx = info.get(f"{self.ram_var_prefix}{pi}_vel_x")
                self.players[p].vy = info.get(f"{self.ram_var_prefix}{pi}_vel_y")
                self.players[p].orientation = info.get(f"{self.ram_var_prefix}{pi}_ori", 0)
                self._load_hidden_player_fields(self.players[p], info, f"{self.ram_var_prefix}{pi}_")

            # Convert orientation to vector
            slot = self.skater_scnum_base() + p
            self.players[p].input_state_known = slot in input_players
            self.players[p].input_state = input_players.get(slot)
            angle = self.players[p].orientation * (2 * math.pi / 8)
            self.players[p].ori_x = math.cos(angle)
            self.players[p].ori_y = math.sin(angle)

            # Calculate relative puck position and velocity
            self.players[p].rel_puck_x = puck_x - self.players[p].x
            self.players[p].rel_puck_y = puck_y - self.players[p].y
            self.players[p].rel_puck_vx = puck_vx - self.players[p].vx
            self.players[p].rel_puck_vy = puck_vy - self.players[p].vy

        # Calculate relative puck position for goalie
        self.goalie.rel_puck_x = puck_x - self.goalie.x
        self.goalie.rel_puck_y = puck_y - self.goalie.y
        self.goalie.rel_puck_vx = puck_vx - self.goalie.vx
        self.goalie.rel_puck_vy = puck_vy - self.goalie.vy

        # Knowing if the player has the puck is tricky since the fullstar in the game is not aligned with the player every frame
        # There is an offset of up to 2 sometimes
        self.player_haspuck = False
        for p in range(0, self.num_players):
            if self.has_puck(self.players[p].x, self.players[p].y):
                self.player_haspuck = True

        self.goalie_haspuck = self.has_puck(self.goalie.x, self.goalie.y)

        # Check which player is being controlled
        if self.goalie_haspuck:
            self.control = 0
        elif self.player_haspuck:
            for p in range(0, self.num_players):
                if self.has_puck(self.players[p].x, self.players[p].y):
                    self.control = p + 1
        else:
            for p in range(0, self.num_players):
                if self.has_control(self.players[p].x, self.players[p].y):
                    self.control = p + 1

        # Explicit controller fields opt into authoritative slot handling.
        control_slot = info.get(f'{self.ram_var_prefix}control_slot')
        if control_slot is not None:
            base = self.skater_scnum_base()
            if control_slot < 0:
                self.control = -1
            elif base <= control_slot <= base + self.num_players:
                self.control = 0 if control_slot == self.goalie_scnum() else control_slot - base + 1
            else:
                raise ValueError(f'Invalid controller slot: {control_slot}')

        self.update_controlled_relationships()
        for player in [*self.players, self.goalie]:
            player.dist_to_puck = GameConsts.Distance((player.x, player.y), (puck_x, puck_y))

    def Normalize(self):
        # Normalize for model input
        for p in range(0, self.num_players):
            self.nz_players[p].x = self.players[p].x / self.rink.player_x
            self.nz_players[p].y = self.players[p].y / self.rink.player_y
            self.nz_players[p].vx = self.players[p].vx / self.rink.max_velocity
            self.nz_players[p].vy = self.players[p].vy / self.rink.max_velocity
            # Orientation vector is already normalized (-1 to 1)
            self.nz_players[p].ori_x = self.players[p].ori_x
            self.nz_players[p].ori_y = self.players[p].ori_y
            # Normalize relative puck position and velocity
            self.nz_players[p].rel_puck_x = self.players[p].rel_puck_x / self.rink.puck_x
            self.nz_players[p].rel_puck_y = self.players[p].rel_puck_y / self.rink.puck_y
            self.nz_players[p].rel_puck_vx = self.players[p].rel_puck_vx / self.rink.max_velocity
            self.nz_players[p].rel_puck_vy = self.players[p].rel_puck_vy / self.rink.max_velocity
            # Normalize relative controlled player position and velocity
            self.nz_players[p].rel_controlled_x = self.players[p].rel_controlled_x / self.rink.player_x
            self.nz_players[p].rel_controlled_y = self.players[p].rel_controlled_y / self.rink.player_y
            self.nz_players[p].rel_controlled_vx = self.players[p].rel_controlled_vx / self.rink.max_velocity
            self.nz_players[p].rel_controlled_vy = self.players[p].rel_controlled_vy / self.rink.max_velocity
            # Normalize distances
            self.nz_players[p].dist_to_controlled = self.players[p].dist_to_controlled / self.rink.player_x
            self.nz_players[p].dist_to_puck = self.players[p].dist_to_puck / self.rink.puck_x
            self.nz_players[p].is_controlled = 1.0 if self.control == p + 1 else 0.0
            self.nz_players[p].has_puck = float(self.players[p].has_puck)
            self.nz_players[p].is_one_timer = self.players[p].is_one_timer
            self.nz_players[p].is_breakaway = self.players[p].is_breakaway
            self.nz_players[p].is_falling = self.players[p].is_falling
            self.nz_players[p].is_pad_stack = self.players[p].is_pad_stack
            self.nz_players[p].is_dive = self.players[p].is_dive
            self.nz_players[p].anim_frame = min(abs(self.players[p].anim_frame) / 32.0, 1.0)
            self.nz_players[p].has_anim = self.players[p].has_anim

        self.nz_goalie.x = self.goalie.x / self.rink.player_x
        self.nz_goalie.y = self.goalie.y / self.rink.player_y
        self.nz_goalie.vx = self.goalie.vx / self.rink.max_velocity
        self.nz_goalie.vy = self.goalie.vy / self.rink.max_velocity
        # Normalize relative puck position and velocity for goalie
        self.nz_goalie.rel_puck_x = self.goalie.rel_puck_x / self.rink.puck_x
        self.nz_goalie.rel_puck_y = self.goalie.rel_puck_y / self.rink.puck_y
        self.nz_goalie.rel_puck_vx = self.goalie.rel_puck_vx / self.rink.max_velocity
        self.nz_goalie.rel_puck_vy = self.goalie.rel_puck_vy / self.rink.max_velocity
        # Normalize relative controlled player position and velocity for goalie
        self.nz_goalie.rel_controlled_x = self.goalie.rel_controlled_x / self.rink.player_x
        self.nz_goalie.rel_controlled_y = self.goalie.rel_controlled_y / self.rink.player_y
        self.nz_goalie.rel_controlled_vx = self.goalie.rel_controlled_vx / self.rink.max_velocity
        self.nz_goalie.rel_controlled_vy = self.goalie.rel_controlled_vy / self.rink.max_velocity
        # Normalize distances for goalie
        self.nz_goalie.dist_to_controlled = self.goalie.dist_to_controlled / self.rink.player_x
        self.nz_goalie.dist_to_puck = self.goalie.dist_to_puck / self.rink.puck_x
        self.nz_goalie.is_controlled = 1.0 if self.control == 0 else 0.0
        self.nz_goalie.has_puck = float(self.goalie.has_puck)
        self.nz_goalie.is_one_timer = self.goalie.is_one_timer
        self.nz_goalie.is_breakaway = self.goalie.is_breakaway
        self.nz_goalie.is_falling = self.goalie.is_falling
        self.nz_goalie.is_pad_stack = self.goalie.is_pad_stack
        self.nz_goalie.is_dive = self.goalie.is_dive
        self.nz_goalie.anim_frame = min(abs(self.goalie.anim_frame) / 32.0, 1.0)
        self.nz_goalie.has_anim = self.goalie.has_anim

        self.nz_goalie_stats.glove_left = self.goalie_stats.glove_left / GameConsts.MAX_GOALIE_GLOVE_LEFT
        self.nz_goalie_stats.glove_right = self.goalie_stats.glove_right / GameConsts.MAX_GOALIE_RATING
        self.nz_goalie_stats.stick_left = self.goalie_stats.stick_left / GameConsts.MAX_GOALIE_RATING
        self.nz_goalie_stats.stick_right = self.goalie_stats.stick_right / GameConsts.MAX_GOALIE_RATING
        self.nz_goalie_stats.puck_control = self.goalie_stats.puck_control / GameConsts.MAX_GOALIE_RATING
        self.nz_goalie_stats.agility = self.goalie_stats.agility / GameConsts.MAX_GOALIE_RATING
        self.nz_goalie_stats.speed = self.goalie_stats.speed / GameConsts.MAX_GOALIE_RATING
        self.nz_goalie_stats.passing = self.goalie_stats.passing / GameConsts.MAX_GOALIE_RATING
        self.nz_goalie_stats.endurance = self.goalie_stats.endurance / GameConsts.MAX_GOALIE_RATING
        self.nz_goalie_stats.weight = self.goalie_stats.weight / GameConsts.MAX_GOALIE_WEIGHT

        # 0.0 and 1.0 switched around due to current models trained that way
        self.nz_player_haspuck = 1.0 if self.player_haspuck else 0.0
        self.nz_goalie_haspuck = 1.0 if self.goalie_haspuck else 0.0

        # Normalize
        for p in range(0, self.num_players):
            self.nz_players[p].dist_to_controlled_opp = self.players[p].dist_to_controlled_opp / self.rink.player_x
            self.nz_players[p].passing_lane_clear = float(self.players[p].passing_lane_clear)  # Convert bool to 0.0/1.0
            self.nz_players[p].one_timer_lane_good = float(self.players[p].one_timer_lane_good)
            self.nz_players[p].clear_shot_lane = float(self.players[p].clear_shot_lane)
            self.nz_players[p].open_net_shot = float(self.players[p].open_net_shot)

        # Normalize net positions
        # Absolute positions
        self.nz_net.left = self.net.left / self.rink.puck_x
        self.nz_net.right = self.net.right / self.rink.puck_x
        self.nz_net.y = self.net.y / self.rink.puck_y
        self.nz_net.depth = self.net.depth / self.rink.puck_y

        # Relative positions
        self.nz_net.rel_controlled_left = self.net.rel_controlled_left / self.rink.puck_x
        self.nz_net.rel_controlled_right = self.net.rel_controlled_right / self.rink.puck_x
        self.nz_net.rel_controlled_y = self.net.rel_controlled_y / self.rink.puck_y

    def end_frame(self) -> None:
        self.last_stats = deepcopy(self.stats)

    def debug_print(self):
        print(f"\nTeam controller: {self.controller}")
        print(f"Team player prefix: {self.ram_var_prefix}")
        self.stats.debug_print("Stats")
        self.last_stats.debug_print("Last Stats")
        self.goalie_stats.debug_print("Goalie Stats")
        print(f"Number of players: {self.num_players}")
        print(f"Control: {self.control}")
        print(f"Player has puck: {self.player_haspuck}")
        print(f"Goalie has puck: {self.goalie_haspuck}")

        # Print all players with full details
        for idx, player in enumerate(self.players):
            player.debug_print(f"Player {idx}")

        # Print goalie
        self.goalie.debug_print("Goalie")

        # Print normalized values
        print("\nNormalized values:")
        print(f"NZ Player has puck: {self.nz_player_haspuck}")
        print(f"NZ Goalie has puck: {self.nz_goalie_haspuck}")
        for idx, player in enumerate(self.nz_players):
            print(f"NZ Player {idx}: "
                f"pos=({player.x:.2f},{player.y:.2f}) "
                f"vel=({player.vx:.2f},{player.vy:.2f}) "
                f"ori=({player.ori_x:.2f},{player.ori_y:.2f})")
        self.nz_goalie.debug_print("NZ Goalie")


class NHL94GameState(TacticalFeatures):
    def __init__(self, numPlayers, platform=None):
        from nhl94_ai.game.platforms import get_backend
        self.platform = platform or get_backend()
        self.rink = self.platform.rink
        self.team1 = Team(1, numPlayers, self.platform)
        self.team2 = Team(2, numPlayers, self.platform)
        self.puck = Player()
        self.engine = EngineState()
        self.is_shootout_active = False
        self.is_shooting = False
        self.has_released_shot = False
        self.c_pressed = False
        self.c_frames_held = 0
        self.period = 1
        self.time = 0
        self.last_time = 0
        self.numPlayers = numPlayers

        self.action = [0] * 6 # Up, Down, Left, Right, B, C

        # Slapshot tracking
        self.slapshot_frames_held = 0
        self.SLAPSHOT_HOLD_FRAMES = 60  # Max frames to hold for slapshot

        #For model input
        self.nz_puck = Player()

    def _update_engine_state(self, info):
        self.platform.update_engine_state(self, info)

    def _clear_possession_flags(self) -> None:
        for team in [self.team1, self.team2]:
            for player in team.players:
                player.has_puck = 0.0
            team.goalie.has_puck = 0.0

    def _set_possession_flag_from_owner(self) -> bool:
        puck_owner = self.engine.puck_owner
        for team in [self.team1, self.team2]:
            if team.owns_scnum(puck_owner):
                player = team.get_player_by_scnum(puck_owner)
                if player is not None:
                    player.has_puck = 1.0
                    return True
        return False

    def _set_possession_flag_from_stars(self) -> None:
        for team in [self.team1, self.team2]:
            if team.goalie_haspuck:
                team.goalie.has_puck = 1.0
                return

            for player in team.players:
                if team.has_puck(player.x, player.y):
                    player.has_puck = 1.0
                    return

    def _update_possession_flags(self) -> None:
        self._clear_possession_flags()
        if not self._set_possession_flag_from_owner():
            self._set_possession_flag_from_stars()

    # Flip the variables
    def Flip(self):
        self.team1, self.team2 = self.team2, self.team1
        return


    def update_nets(self):
        # Public periods are one-based; teams change ends each period. Use
        # physical team identity so an away view can also refresh its geometry.
        direction = -1 if self.period % 2 == 0 else 1
        for team in [self.team1, self.team2]:
            home = team.controller == 1
            team.net.y = direction * (-self.rink.net_y if home else self.rink.net_y)
            team.net.left = -self.rink.post_x
            team.net.right = self.rink.post_x
            controlled = team.get_controlled_player()
            team.net.rel_controlled_left = 0 if controlled is None else team.net.left - controlled.x
            team.net.rel_controlled_right = 0 if controlled is None else team.net.right - controlled.x
            team.net.rel_controlled_y = 0 if controlled is None else team.net.y - controlled.y


    def BeginFrame(self, info, action):
        info = self.platform.decode_info(info)
        self.action = action
        self.c_pressed = bool(action[5])
        self.c_frames_held = self.c_frames_held + 1 if self.c_pressed else 0

        # Handle slapshot frames
        if action[5]:  # C button pressed
            if self.slapshot_frames_held == 0:
                self.slapshot_frames_held = 1
            else:
                self.slapshot_frames_held += 1
                if self.slapshot_frames_held >= self.SLAPSHOT_HOLD_FRAMES:
                    self.slapshot_frames_held = 0  # Reset after max hold
        else:
            self.slapshot_frames_held = 0  # Reset if C not pressed


        # The ROM counts periods from zero. Preserve the public one-based
        # convention without collapsing the first and second periods together.
        self.period = int(info.get("period") or 0) + 1
        self.time = info.get("time")

        #Puck
        self.puck.x = info.get("puck_x")
        self.puck.y = info.get("puck_y")
        self.puck.vx = info.get("puck_vel_x")
        self.puck.vy = info.get("puck_vel_y")
        for axis in ('x', 'y'):
            position = info.get('defense_puck_precise_' + axis)
            setattr(self.puck, 'precise_' + axis, None if position is None else position / 65536)
        for axis in ('x', 'y', 'z'):
            velocity = info.get('defense_puck_v' + axis)
            setattr(self.puck, 'motion_' + axis, None if velocity is None else velocity * 17 / 65536)
        height = info.get('defense_puck_z')
        self.puck.height = None if height is None else height / 65536
        self.puck.friction = 511 / 512 if info.get('defense_puck_flags', 1) & 1 else 63 / 64

        #Teams
        self.team1.begin_frame(info, self.puck.x, self.puck.y, self.puck.vx, self.puck.vy)
        self.team2.begin_frame(info, self.puck.x, self.puck.y, self.puck.vx, self.puck.vy)
        self.team1._load_defense_fields(info)
        self.team2._load_defense_fields(info)
        self._update_engine_state(info)
        self.platform.update_state(self, info)
        if 'p1_control_slot' in info:
            self._clear_possession_flags()
            self._set_possession_flag_from_owner()
            for team in (self.team1, self.team2):
                team.goalie_haspuck = self.engine.puck_owner == team.goalie_scnum()
                team.player_haspuck = team.owns_scnum(self.engine.puck_owner) and not team.goalie_haspuck
        else:
            self._update_possession_flags()

        self.update_nets()

        # Compute passing lanes AFTER both teams are updated
        self._update_passing_lanes()
        self._update_shot_lanes()
        self._update_opponent_controlled_distances()

        self.team1.Normalize()
        self.team2.Normalize()

        #Normalize Puck
        self.nz_puck.x = self.puck.x / self.rink.puck_x
        self.nz_puck.y = self.puck.y / self.rink.puck_y
        self.nz_puck.vx = self.puck.vx / self.rink.max_velocity
        self.nz_puck.vy = self.puck.vy / self.rink.max_velocity

    def EndFrame(self):
        self.last_time = self.time

        self.team1.end_frame()
        self.team2.end_frame()

        #self.debug_print()
        #time.sleep(5)

    def debug_print(self):
        print("\n" + "="*80)
        print(f"Game time: {self.time}, Last time: {self.last_time}")
        print(f"Slapshot frames held: {self.slapshot_frames_held}/{self.SLAPSHOT_HOLD_FRAMES}")

        # Puck info
        print("\nPuck State:")
        self.puck.debug_print("Raw Puck")
        self.nz_puck.debug_print("NZ Puck")

        # Team states
        print("\n" + "="*40 + " Team 1 State " + "="*40)
        self.team1.debug_print()

        print("\n" + "="*40 + " Team 2 State " + "="*40)
        self.team2.debug_print()

        # Action state
        action_names = ["Up", "Down", "Left", "Right", "B", "C"]
        print("\nCurrent Actions:")
        for name, val in zip(action_names, self.action):
            print(f"{name}: {'ON' if val else 'OFF'}", end=" | ")
        print()
        self.engine.debug_print()
