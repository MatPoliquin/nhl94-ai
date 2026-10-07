"""
NHL94 Debug Display with Player Input
"""

from copy import deepcopy

import pygame
import numpy as np
from nhl94_ai.game.constants import GameConsts
from pygame import gfxdraw
from nhl94_ai.game.state import NHL94GameState
from nhl94_ai.ui.decision_panel import BACKGROUND, CYAN, GRAY, WHITE, DecisionInspector
from nhl94_ai.agents.decisions import catalogue_snapshot

class NHL94DebugDisplay:
    CANVAS_WIDTH = 1920
    CANVAS_HEIGHT = 1080
    ACTION_WIDTH = 740
    DEBUG_HEIGHT = 300
    GAME_WIDTH = 1040
    GAME_HEIGHT = 780
    RINK_SIDE_PADDING = 10
    HUD_PADDING = 20
    HUD_LINE_HEIGHT = 20
    HUD_HEADER_HEIGHT = 30
    HUD_SECTION_GAP = 10

    # Colors
    COLOR_WHITE = (255, 255, 255)
    COLOR_RED = (255, 0, 0)
    COLOR_BLUE = (0, 0, 255)
    COLOR_GREEN = (0, 255, 0)
    COLOR_YELLOW = (255, 255, 0)
    COLOR_CYAN = (0, 255, 255)
    COLOR_ICE = (200, 230, 255)
    COLOR_LINE = (150, 150, 150)
    COLOR_BLACK = (0, 0, 0)
    COLOR_PURPLE = (128, 0, 128)
    COLOR_ORANGE = (255, 165, 0)

    # Display parameters
    PLAYER_RADIUS = 4
    PUCK_RADIUS = 2
    VELOCITY_SCALE = 1.5
    ORIENTATION_LENGTH = 8

    def __init__(self, env, args, total_params, nn_type, button_names):
        self.env = env
        self.args = args

        self.game_state = env.get_attr("game_state")[0]

        pygame.init()
        self.screen_height = self.CANVAS_HEIGHT

        self.scale_x = self.DEBUG_HEIGHT / (2 * GameConsts.MAX_PUCK_Y)
        self.scale_y = self.scale_x
        self.rink_pixel_width = int(round(2 * GameConsts.MAX_PUCK_X * self.scale_x))
        self.DEBUG_WIDTH = self.rink_pixel_width + (2 * self.RINK_SIDE_PADDING)
        self.rink_rect = pygame.Rect(
            self.RINK_SIDE_PADDING,
            0,
            self.rink_pixel_width,
            self.DEBUG_HEIGHT
        )
        self.game_rect = pygame.Rect(
            self.ACTION_WIDTH + (self.CANVAS_WIDTH - self.ACTION_WIDTH - self.GAME_WIDTH) // 2,
            0, self.GAME_WIDTH, self.GAME_HEIGHT)
        self.mini_rink_rect = pygame.Rect(self.ACTION_WIDTH, self.GAME_HEIGHT, self.DEBUG_WIDTH, self.DEBUG_HEIGHT)
        self.stats_rect = pygame.Rect(self.mini_rink_rect.right, self.GAME_HEIGHT,
                                      self.CANVAS_WIDTH - self.mini_rink_rect.right, self.DEBUG_HEIGHT)

        desktop_width, desktop_height = pygame.display.get_desktop_sizes()[0]
        window_size = (min(self.CANVAS_WIDTH, max(640, desktop_width - 40)),
                       min(self.CANVAS_HEIGHT, max(360, desktop_height - 80)))
        self.window = pygame.display.set_mode(window_size, pygame.RESIZABLE)
        self.screen = pygame.Surface((self.CANVAS_WIDTH, self.CANVAS_HEIGHT))
        self.presentation_rect = self.window.get_rect()
        self.font = pygame.font.SysFont('Arial', 16)
        self.big_font = pygame.font.SysFont('Arial', 24)
        self.mini_font = pygame.font.SysFont('Arial', 11)

        # Create surfaces
        self.debug_surf = pygame.Surface((self.DEBUG_WIDTH, self.DEBUG_HEIGHT))
        self.game_surf = pygame.Surface((self.GAME_WIDTH, self.GAME_HEIGHT))
        self._frame_surface = None
        self._presentation_surface = None

        # Input state
        self.player_actions = [0] * GameConsts.INPUT_MAX
        self.action_probabilities = None
        self.model_params = None
        self.model_in_use = None
        self.classic_defense = {}
        self.classic_offense = {}
        self.classic_goalie = {}
        self.inspector = DecisionInspector()
        self.paused = False
        self.playback_frames = 0
        self.score_evaluation = None
        self.score_evaluation_frame = 0
        self._last_frame = None
        self._last_render_info = [{}]
        self._last_action = None
        self.key_action_map = {
            pygame.K_UP: GameConsts.INPUT_UP,
            pygame.K_DOWN: GameConsts.INPUT_DOWN,
            pygame.K_LEFT: GameConsts.INPUT_LEFT,
            pygame.K_RIGHT: GameConsts.INPUT_RIGHT,
            pygame.K_z: GameConsts.INPUT_A,
            pygame.K_x: GameConsts.INPUT_B,
            pygame.K_c: GameConsts.INPUT_C,
            pygame.K_a: GameConsts.INPUT_X,
            pygame.K_s: GameConsts.INPUT_Y,
            pygame.K_d: GameConsts.INPUT_Z,
            pygame.K_RETURN: GameConsts.INPUT_START,
            pygame.K_TAB: GameConsts.INPUT_MODE
        }

        # Control mode (AI or human)
        self.human_control = args.mode in ('player_vs_game', 'player_vs_model')
        self.control_help = [
            "CONTROLS:",
            "Arrows: Move",
            "Z: A Button (Clear/Hold)",
            "X: B Button (Pass/Switch/Poke)",
            "C: C Button (Shoot/Burst/Check)",
            "A/S/D: X/Y/Z Buttons",
            "TAB: Mode Button",
            "ENTER: Start Button",
            "F1: Toggle P2 Keyboard" if args.mode == 'player_vs_model' else "F1: Toggle AI/Human Control",
            "ESC: Quit",
            "SPACE / P: Pause or resume playback",
            "1: Toggle Passing Lanes",
            "2: Toggle One-Timer Lanes",
            "3: Toggle Velocities",
            "4: Toggle Orientations",
            "5: Toggle Distances",
            "6: Toggle Clear Shot Lanes",
            "7: Toggle Open Net Shots",
            "8: Toggle AI Planner Overlays",
            "9: Toggle Teammate Scores",
            "F2: Save Screenshot",
            "Mouse wheel / PgUp / PgDn: Scroll actions"
        ]

        # Visualization toggles
        self.show_passing_lanes = False
        self.show_one_timer_lanes = False
        self.show_clear_shot_lanes = False
        self.show_open_net_shots = False
        self.show_velocities = False
        self.show_orientations = False
        self.show_distances = False
        self.show_planner_overlay = getattr(args, 'action_type', '').upper() == 'TARGET_POSITION'
        self.show_teammate_scores = True
        print('\nNHL94 debug controls (shortcuts remain active while paused):\n' + '\n'.join(self.control_help))
        if getattr(args, 'action_type', '').upper() == 'TARGET_POSITION':
            print('TARGET_POSITION playback is AI-only; F1/button overrides are unavailable.')

    def set_ai_sys_info(self, ai_sys):
        if ai_sys is None:
            return
        self.action_probabilities = getattr(ai_sys, 'display_probs', None)
        self.model_params = getattr(ai_sys, 'model_num_params', None)
        self.model_in_use = getattr(ai_sys, 'model_in_use', None)
        self.classic_defense = getattr(ai_sys, 'last_diagnostics', {}).get('classic_defense', {})
        self.classic_offense = getattr(ai_sys, 'last_diagnostics', {}).get('classic_offense', {})
        self.classic_goalie = getattr(ai_sys, 'last_diagnostics', {}).get('classic_goalie', {})
        snapshot = getattr(ai_sys, 'last_diagnostics', {}).get('decision_inspector')
        if snapshot is None:
            team = self.game_state.team2 if getattr(self.args, 'side', 'home') == 'away' else self.game_state.team1
            snapshot = catalogue_snapshot(team, getattr(self.args, 'nn', 'Agent'), self.playback_frames + 1,
                                          getattr(self.args, 'action_type', 'FILTERED'))
        self.inspector.update(snapshot, self.playback_frames + 1)
        scores = self.classic_offense.get('teammate_scores', ())
        frame = self.classic_offense.get('evaluation_frame')
        if frame is not None and any(row['pass']['status'] != 'not-evaluated' for row in scores):
            previous = self.score_evaluation
            key = frame, self.classic_offense.get('evaluation_carrier')
            if previous is None or key != (previous['evaluation_frame'], previous.get('evaluation_carrier')):
                self.score_evaluation = deepcopy(self.classic_offense)
                self.score_evaluation_frame = self.playback_frames + 1

    def _normalize_env_action(self, action):
        if getattr(self.args, 'action_type', '').upper() == 'TARGET_POSITION':
            from nhl94_ai.env.target_control import validate_target
            values = np.asarray(action, dtype=np.float32)
            if values.shape == (1, 2):
                values = values[0]
            return [validate_target(values).tolist()]
        expected_action_size = 6 if getattr(self.args, 'action_type', '').upper() == 'HOCKEY_INTENT_DPAD' else GameConsts.INPUT_MAX

        if action is None:
            return [[0] * expected_action_size]

        if isinstance(action, np.ndarray):
            action = action.tolist()

        if isinstance(action, list) and len(action) == expected_action_size and not isinstance(action[0], (list, np.ndarray)):
            return [[int(a) for a in action]]

        if isinstance(action, list) and len(action) == 1:
            first = action[0]
            if isinstance(first, np.ndarray):
                first = first.tolist()
            if isinstance(first, list) and len(first) == expected_action_size:
                return [[int(a) for a in first]]

        if isinstance(action, list) and len(action) == 2:
            normalized = []
            for player_action in action:
                if isinstance(player_action, np.ndarray):
                    player_action = player_action.tolist()
                if not isinstance(player_action, list) or len(player_action) != expected_action_size:
                    break
                normalized.append([int(a) for a in player_action])
            if len(normalized) == 2:
                return normalized

        return [[0] * expected_action_size]

    def _flatten_action_for_display(self, action):
        normalized = self._normalize_env_action(action)
        if normalized and len(normalized[0]) == GameConsts.INPUT_MAX:
            return normalized[0]
        return [0] * GameConsts.INPUT_MAX

    def close(self):
        self.env.close()
        pygame.quit()

    def reset(self, **kwargs):
        self._clear_diagnostics()
        self.paused = False
        self._last_frame = None
        self._last_render_info = [{}]
        self._last_action = None
        result = self.env.reset(**kwargs)
        self.game_state = self.env.get_attr("game_state")[0]
        return result

    def _clear_diagnostics(self):
        self.classic_defense = {}
        self.classic_offense = {}
        self.classic_goalie = {}
        self.score_evaluation = None
        self.score_evaluation_frame = 0
        self.playback_frames = 0
        self.inspector.reset()

    def process_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.close()
                raise SystemExit("User requested exit")
            if event.type == pygame.VIDEORESIZE:
                self.window = pygame.display.set_mode((max(320, event.w), max(180, event.h)), pygame.RESIZABLE)
                continue
            if event.type == pygame.MOUSEWHEEL:
                if self._logical_mouse_position()[0] < self.ACTION_WIDTH:
                    self.inspector.scroll_by(-event.y * 69)
                continue
            if event.type != pygame.KEYDOWN:
                continue
            if event.key in (pygame.K_SPACE, pygame.K_p):
                if not getattr(event, 'repeat', False):
                    self.paused = not self.paused
                    print('Playback paused.' if self.paused else 'Playback resumed.')
            elif event.key == pygame.K_F1:
                if getattr(self.args, 'action_type', '').upper() == 'TARGET_POSITION':
                    print('TARGET_POSITION playback is AI-only; human button override is not supported.')
                else:
                    self.human_control = not self.human_control
                    print(f'{"P2 keyboard" if self.args.mode == "player_vs_model" else "Human control"}: '
                          f'{"ON" if self.human_control else "OFF"}')
            elif event.key == pygame.K_F2:
                pygame.image.save(self.screen, "debug_screenshot.png")
            elif event.key in (pygame.K_PAGEUP, pygame.K_PAGEDOWN):
                self.inspector.scroll_by(-345 if event.key == pygame.K_PAGEUP else 345)
            else:
                toggles = {
                    pygame.K_1: 'show_passing_lanes', pygame.K_2: 'show_one_timer_lanes',
                    pygame.K_3: 'show_velocities', pygame.K_4: 'show_orientations',
                    pygame.K_5: 'show_distances', pygame.K_6: 'show_clear_shot_lanes',
                    pygame.K_7: 'show_open_net_shots', pygame.K_8: 'show_planner_overlay',
                    pygame.K_9: 'show_teammate_scores',
                }
                attribute = toggles.get(event.key)
                if attribute is not None:
                    setattr(self, attribute, not getattr(self, attribute))
                    print(f'{attribute.removeprefix("show_").replace("_", " ")}: '
                          f'{"ON" if getattr(self, attribute) else "OFF"}')

    def wait_until_running(self):
        """Service inspection controls without advancing the emulator or policy."""
        was_paused = self.paused
        self.process_events()
        was_paused |= self.paused
        while self.paused:
            if self._last_frame is None:
                self._last_frame = np.array(self.env.render(), copy=True)
            self.draw_frame(self._last_frame, self._last_render_info, self._last_action)
            pygame.time.wait(16)
            self.process_events()
        if was_paused and self._last_frame is not None:
            self.draw_frame(self._last_frame, self._last_render_info, self._last_action)
        return was_paused

    def get_human_input(self):
        """Get input from keyboard and map to game actions"""
        pygame.event.pump()
        keys = pygame.key.get_pressed()
        actions = [0] * GameConsts.INPUT_MAX

        for key, action in self.key_action_map.items():
            if keys[key]:
                actions[action] = 1

        return actions

    def step(self, action=None):
        """Step the environment with either AI or human input"""

        self.wait_until_running()
        human_vs_model = self.args.mode == 'player_vs_model'
        if human_vs_model:
            ai_action = np.asarray(action)
            if ai_action.shape == (1, GameConsts.INPUT_MAX):
                ai_action = ai_action[0]
            if ai_action.shape != (GameConsts.INPUT_MAX,) or not np.all((ai_action == 0) | (ai_action == 1)):
                raise ValueError('player_vs_model expects one 12-button AI action')
            self.player_actions = self.get_human_input() if self.human_control else [0] * GameConsts.INPUT_MAX
            env_action = [np.concatenate((ai_action, self.player_actions)).astype(np.int8)]
        else:
            if self.human_control:
                action = self.get_human_input()
            env_action = self._normalize_env_action(action)

        obs, rew, done, info = self.env.step(env_action)
        self.playback_frames += 1
        self.game_state = self.env.get_attr("game_state")[0]

        # Draw the frame
        framebuffer = self.env.render()
        if np.any(done):
            self._clear_diagnostics()
        render_info = [{}] if np.any(done) else info
        if self.classic_goalie.get('target') is not None and (human_vs_model or not self.human_control):
            render_info = [dict(render_info[0], classic_goalie=self.classic_goalie)]
        elif self.classic_defense and (human_vs_model or not self.human_control):
            render_info = [dict(render_info[0], classic_defense=self.classic_defense)]
        elif self.classic_offense and (human_vs_model or not self.human_control):
            render_info = [dict(render_info[0], classic_offense=self.classic_offense)]
        self._last_frame = np.array(framebuffer, copy=True)
        self._last_render_info = deepcopy(render_info)
        applied = env_action[0][:GameConsts.INPUT_MAX] if human_vs_model else env_action[0]
        self._last_action = deepcopy(applied)
        self.draw_frame(self._last_frame, self._last_render_info, applied)
        self.process_events()

        return obs, rew, done, info

    def _transform_coords(self, x, y):
        """Transform game coordinates to debug display coordinates with proper bounds checking"""
        tx = self.rink_rect.centerx + x * self.scale_x
        ty = self.rink_rect.centery - y * self.scale_y

        return int(round(tx)), int(round(ty))

    def _draw_rink(self):
        """Draw the hockey rink with center lines and zones"""
        self.debug_surf.fill(self.COLOR_ICE)

        rink_rect = self.rink_rect
        pygame.draw.rect(self.debug_surf, self.COLOR_LINE, rink_rect, 2)

        # Draw horizontal rink markings using the same Y thresholds as game logic.
        rink_lines = (
            (GameConsts.DEFENSEZONE_POS_Y, self.COLOR_BLUE),
            (0, self.COLOR_RED),
            (GameConsts.ATACKZONE_POS_Y, self.COLOR_BLUE),
        )
        for line_y, color in rink_lines:
            _, screen_y = self._transform_coords(0, line_y)
            pygame.draw.line(
                self.debug_surf, color,
                (rink_rect.left, screen_y),
                (rink_rect.right, screen_y),
                2
            )

        # Faceoff circles (simplified)
        pygame.draw.circle(
            self.debug_surf, self.COLOR_LINE,
            rink_rect.center, max(3, round(15 * self.scale_x)), 1
        )

        # Draw nets at opposite ends
        # Team 1 (Red) net at bottom
        left_x, left_y = self._transform_coords(GameConsts.P1_NET_LEFT_POLL, GameConsts.P1_NET_Y)
        right_x, right_y = self._transform_coords(GameConsts.P1_NET_RIGHT_POLL, GameConsts.P1_NET_Y)
        back_y = GameConsts.P1_NET_Y + GameConsts.NET_DEPTH  # Note: + instead of - because Y increases downward
        back_left_x, back_left_y = self._transform_coords(GameConsts.P1_NET_LEFT_POLL, back_y)
        back_right_x, back_right_y = self._transform_coords(GameConsts.P1_NET_RIGHT_POLL, back_y)

        pygame.draw.polygon(
            self.debug_surf, self.COLOR_RED,
            [(left_x, left_y), (right_x, right_y),
            (back_right_x, back_right_y), (back_left_x, back_left_y)],
            2
        )

        # Team 2 (Blue) net at top
        left_x, left_y = self._transform_coords(GameConsts.P2_NET_LEFT_POLL, GameConsts.P2_NET_Y)
        right_x, right_y = self._transform_coords(GameConsts.P2_NET_RIGHT_POLL, GameConsts.P2_NET_Y)
        back_y = GameConsts.P2_NET_Y - GameConsts.NET_DEPTH  # Note: - because Y decreases upward
        back_left_x, back_left_y = self._transform_coords(GameConsts.P2_NET_LEFT_POLL, back_y)
        back_right_x, back_right_y = self._transform_coords(GameConsts.P2_NET_RIGHT_POLL, back_y)

        pygame.draw.polygon(
            self.debug_surf, self.COLOR_BLUE,
            [(left_x, left_y), (right_x, right_y),
            (back_right_x, back_right_y), (back_left_x, back_left_y)],
            2
        )

    def _draw_player(self, player, color, is_goalie=False):
        """Draw a player with position, orientation and velocity"""
        x, y = self._transform_coords(player.x, player.y)

        # Draw player circle
        radius = self.PLAYER_RADIUS + 2 if is_goalie else self.PLAYER_RADIUS
        pygame.draw.circle(self.debug_surf, color, (x, y), radius)

        # Draw orientation line if enabled
        if self.show_orientations:
            end_x = x + player.ori_x * self.ORIENTATION_LENGTH
            end_y = y + player.ori_y * self.ORIENTATION_LENGTH
            pygame.draw.line(self.debug_surf, self.COLOR_WHITE, (x, y), (end_x, end_y), 2)

        # Draw velocity vector if enabled
        if self.show_velocities:
            vel_x = x + player.vx * self.VELOCITY_SCALE
            vel_y = y + player.vy * self.VELOCITY_SCALE
            pygame.draw.line(self.debug_surf, self.COLOR_YELLOW, (x, y), (vel_x, vel_y), 2)

            # Draw small arrowhead for velocity
            pygame.draw.line(self.debug_surf, self.COLOR_YELLOW, (vel_x, vel_y),
                            (vel_x - 5, vel_y - 5), 2)
            pygame.draw.line(self.debug_surf, self.COLOR_YELLOW, (vel_x, vel_y),
                            (vel_x - 5, vel_y + 5), 2)

        return x, y

    def _draw_puck(self, puck):
        """Draw the puck with position and velocity"""
        x, y = self._transform_coords(puck.x, puck.y)

        # Draw puck
        pygame.draw.circle(self.debug_surf, self.COLOR_WHITE, (x, y), self.PUCK_RADIUS)

        # Draw velocity vector if enabled
        if self.show_velocities:
            vel_x = x + puck.vx * self.VELOCITY_SCALE
            vel_y = y + puck.vy * self.VELOCITY_SCALE
            pygame.draw.line(self.debug_surf, self.COLOR_RED, (x, y), (vel_x, vel_y), 2)

            # Draw small arrowhead for velocity
            pygame.draw.line(self.debug_surf, self.COLOR_RED, (vel_x, vel_y),
                            (vel_x - 3, vel_y - 3), 2)
            pygame.draw.line(self.debug_surf, self.COLOR_RED, (vel_x, vel_y),
                            (vel_x - 3, vel_y + 3), 2)

    def _draw_passing_lanes(self, team, color):
        """Draw passing lanes only for the player controlling the puck"""
        if not (self.show_passing_lanes or self.show_one_timer_lanes):
            return

        passer = self.game_state._get_passer_for_team(team)
        if passer is None:
            return

        opponents = self.game_state.team2 if team.controller == 1 else self.game_state.team1

        # Draw line to each teammate
        for player in team.players:
            if player is not passer:
                start_x, start_y = self._transform_coords(passer.x, passer.y)
                catch_x, catch_y = self.game_state._receiver_catch_point(player)
                end_x, end_y = self._transform_coords(catch_x, catch_y)

                if player.one_timer_lane_good and self.show_one_timer_lanes:
                    lane_color = self.COLOR_ORANGE
                    lane_width = 5
                    marker_color = self.COLOR_YELLOW
                    marker_radius = 7
                elif self.show_passing_lanes:
                    lane_color = self.COLOR_GREEN if player.passing_lane_clear else self.COLOR_RED
                    lane_width = 3
                    marker_color = color
                    marker_radius = 5
                else:
                    continue

                pygame.draw.line(self.debug_surf, lane_color, (start_x, start_y), (end_x, end_y), lane_width)

                # Mark strong one-timer options distinctly from generic passing lanes.
                mid_x = (start_x + end_x) // 2
                mid_y = (start_y + end_y) // 2
                pygame.draw.circle(self.debug_surf, marker_color, (mid_x, mid_y), marker_radius)

                if player.one_timer_lane_good and self.show_one_timer_lanes:
                    shot_target = self.game_state._get_clear_one_timer_shot_target(player, team, opponents)
                    if shot_target is not None:
                        shot_start_x, shot_start_y = self._transform_coords(player.x, player.y)
                        shot_end_x, shot_end_y = self._transform_coords(*shot_target)
                        pygame.draw.line(
                            self.debug_surf,
                            self.COLOR_YELLOW,
                            (shot_start_x, shot_start_y),
                            (shot_end_x, shot_end_y),
                            3,
                        )
                        pygame.draw.circle(
                            self.debug_surf,
                            self.COLOR_YELLOW,
                            (shot_end_x, shot_end_y),
                            4,
                        )

    def _draw_clear_shot_lane(self, team, color):
        if not self.show_clear_shot_lanes:
            return

        shooter = self.game_state._get_puck_controlled_skater_for_team(team)
        if shooter is None:
            return

        opponents = self.game_state.team2 if team.controller == 1 else self.game_state.team1
        if shooter.clear_shot_lane:
            shot_target = self.game_state._get_clear_shot_lane_target(shooter, team, opponents)
            lane_color = self.COLOR_GREEN
            lane_width = 4
            marker_color = self.COLOR_YELLOW
        else:
            shot_target = self.game_state._get_preferred_shot_lane_target(shooter, team, opponents)
            lane_color = self.COLOR_RED
            lane_width = 2
            marker_color = self.COLOR_RED

        if shot_target is None:
            return

        start_x, start_y = self._transform_coords(shooter.x, shooter.y)
        end_x, end_y = self._transform_coords(*shot_target)
        pygame.draw.line(self.debug_surf, lane_color, (start_x, start_y), (end_x, end_y), lane_width)
        pygame.draw.circle(self.debug_surf, marker_color, (end_x, end_y), 5)
        pygame.draw.circle(self.debug_surf, color, (start_x, start_y), self.PLAYER_RADIUS + 3, 2)

    def _draw_open_net_shot(self, team, color):
        if not self.show_open_net_shots:
            return

        shooter = self.game_state._get_puck_controlled_skater_for_team(team)
        if shooter is None:
            return

        opponents = self.game_state.team2 if team.controller == 1 else self.game_state.team1
        goalie = opponents.goalie
        goalie_x, goalie_y = self._transform_coords(goalie.x, goalie.y)
        goalie_radius = int(round(self.game_state._goalie_shot_collision_radius(goalie) * self.scale_x))
        pygame.draw.circle(self.debug_surf, self.COLOR_CYAN, (goalie_x, goalie_y), goalie_radius, 2)

        if shooter.open_net_shot:
            shot_target = self.game_state._get_open_net_shot_target(shooter, team, opponents)
            lane_color = self.COLOR_GREEN
            lane_width = 4
            marker_color = self.COLOR_YELLOW
        else:
            shot_target = (
                self.game_state._get_clear_shot_lane_target(shooter, team, opponents)
                or self.game_state._get_preferred_shot_lane_target(shooter, team, opponents)
            )
            lane_color = self.COLOR_RED
            lane_width = 2
            marker_color = self.COLOR_RED

        if shot_target is None:
            return

        start_x, start_y = self._transform_coords(shooter.x, shooter.y)
        end_x, end_y = self._transform_coords(*shot_target)
        pygame.draw.line(self.debug_surf, lane_color, (start_x, start_y), (end_x, end_y), lane_width)
        pygame.draw.circle(self.debug_surf, marker_color, (end_x, end_y), 5)
        pygame.draw.circle(self.debug_surf, color, (start_x, start_y), self.PLAYER_RADIUS + 6, 2)

    def _draw_controlled_distances(self, team, color):
        """Draw distances from controlled player to teammates"""
        if not self.show_distances:
            return

        controlled_player = team.get_controlled_player()
        if controlled_player is None:
            return

        # Draw distance to each teammate
        for i, player in enumerate(team.players):
            if i != (team.control - 1) or team.control == 0:  # Skip self if not goalie
                start_x, start_y = self._transform_coords(controlled_player.x, controlled_player.y)
                end_x, end_y = self._transform_coords(player.x, player.y)

                # Draw distance line
                pygame.draw.line(self.debug_surf, self.COLOR_PURPLE, (start_x, start_y), (end_x, end_y), 1)

                # Calculate midpoint for distance text
                mid_x = (start_x + end_x) // 2
                mid_y = (start_y + end_y) // 2

                # Display the distance value
                distance = GameConsts.Distance(
                    (controlled_player.x, controlled_player.y),
                    (player.x, player.y)
                )
                text = self.font.render(f"{distance:.1f}", True, self.COLOR_WHITE)
                self.debug_surf.blit(text, (mid_x - text.get_width()//2, mid_y - text.get_height()//2))

    def _draw_stats(self, team1, team2):
        """Compact stats beside the mini rink; controls belong in the console."""
        pygame.draw.rect(self.screen, BACKGROUND, self.stats_rect)
        info_x = self.stats_rect.x + self.HUD_PADDING
        info_y = self.stats_rect.y + self.HUD_PADDING
        header_text = f"NHL '94 - {'HUMAN' if self.human_control else 'AI'} CONTROL"
        if self.args.mode == 'player_vs_model':
            header_text = "NHL '94 - P1 AI / P2 KEYBOARD"
        if self.paused:
            header_text += " - PAUSED"
        self.screen.blit(self.big_font.render(header_text, True, WHITE), (info_x, info_y))
        stats_y = info_y + self.HUD_HEADER_HEIGHT
        columns = (info_x, info_x + 180, info_x + 330)
        for label, left in zip(('Statistic', 'TEAM 1', 'TEAM 2'), columns):
            self.screen.blit(self.font.render(label, True, CYAN), (left, stats_y))
        for left, color in zip(columns[1:], (self.COLOR_RED, self.COLOR_BLUE)):
            pygame.draw.rect(self.screen, color, (left - 12, stats_y + 6, 6, 6))
        fields = (('Score', 'score'), ('Shots', 'shots'), ('Checks', 'bodychecks'),
                  ('Attack', 'attackzone'), ('Faceoffs', 'faceoffwon'),
                  ('Passing', 'passing'), ('One-timers', 'onetimer'))
        rows = [(label, str(getattr(team1.stats, field)), str(getattr(team2.stats, field)))
                for label, field in fields]
        rows.append(('Puck', *('Player' if team.player_haspuck else 'Goalie' if team.goalie_haspuck else 'None'
                               for team in (team1, team2))))
        for index, values in enumerate(rows, 1):
            for text, left in zip(values, columns):
                self.screen.blit(self.font.render(text, True, WHITE),
                                 (left, stats_y + index * self.HUD_LINE_HEIGHT))
        if self.show_teammate_scores and self.score_evaluation is not None:
            scores = self.score_evaluation
            historical = self._scores_historical()
            age = max(0, self.playback_frames - self.score_evaluation_frame)
            retained = scores.get('retained_value')
            keep_value = '--' if retained is None else f'{retained:.1f}'
            score_text = [
                f"{'LAST' if historical else 'CURRENT'} offense evaluation; {age} emulator frames ago",
                f"Carrier: {scores['evaluation_carrier']}; keep-puck value: {keep_value}",
                "P=pass | Pos/carry=position",
                "Finish=executable shot | OT=one-timer",
            ]
            for index, text in enumerate(score_text):
                self.screen.blit(self.font.render(text, True, GRAY if historical else CYAN),
                                 (info_x + 510, stats_y + index * self.HUD_LINE_HEIGHT))

    def _scores_historical(self):
        return (self.score_evaluation['evaluation_frame'] != self.classic_offense.get('evaluation_frame')
                or self.game_state.engine.puck_owner != self.score_evaluation['evaluation_carrier']
                or self.classic_goalie.get('target') is not None or bool(self.classic_defense)
                or self.human_control and self.args.mode != 'player_vs_model')

    def draw_frame(self, frame_img, info, action=None):
        self.screen.fill(self.COLOR_BLACK)

        # Clear surfaces
        self.debug_surf.fill(self.COLOR_BLACK)
        self.game_surf.fill(self.COLOR_BLACK)

        # Draw the rink and game elements
        self._draw_rink()

        # Draw players and passing lanes for both teams
        team1 = self.game_state.team1
        team2 = self.game_state.team2

        # Draw team 1 (red)
        for player in team1.players:
            self._draw_player(player, self.COLOR_RED)
        self._draw_player(team1.goalie, self.COLOR_RED, is_goalie=True)
        if self.show_passing_lanes or self.show_one_timer_lanes:
            self._draw_passing_lanes(team1, self.COLOR_RED)
        if self.show_clear_shot_lanes:
            self._draw_clear_shot_lane(team1, self.COLOR_RED)
        if self.show_open_net_shots:
            self._draw_open_net_shot(team1, self.COLOR_RED)
        if self.show_distances:
            self._draw_controlled_distances(team1, self.COLOR_RED)

        # Draw team 2 (blue)
        for player in team2.players:
            self._draw_player(player, self.COLOR_BLUE)
        self._draw_player(team2.goalie, self.COLOR_BLUE, is_goalie=True)
        if self.show_passing_lanes or self.show_one_timer_lanes:
            self._draw_passing_lanes(team2, self.COLOR_BLUE)
        if self.show_clear_shot_lanes:
            self._draw_clear_shot_lane(team2, self.COLOR_BLUE)
        if self.show_open_net_shots:
            self._draw_open_net_shot(team2, self.COLOR_BLUE)
        if self.show_distances:
            self._draw_controlled_distances(team2, self.COLOR_BLUE)

        # Draw puck
        self._draw_puck(self.game_state.puck)

        # Draw stats and debug info
        self._draw_stats(team1, team2)

        # Draw action info if available
        frame_info = info[0] if isinstance(info, (list, tuple)) else info
        from nhl94_ai.ui.targets import select_target
        diagnostics = select_target(frame_info)
        if self.show_planner_overlay and diagnostics:
            from nhl94_ai.ui.targets import draw_game_target, draw_target_overlay
            draw_target_overlay(self.debug_surf, self._transform_coords, self.game_state,
                                dict(diagnostics, teammate_scores=[]), self.mini_font, compact=True)
        if self.show_teammate_scores and self.score_evaluation is not None:
            from nhl94_ai.ui.targets import draw_teammate_scores
            draw_teammate_scores(
                self.debug_surf, self._transform_coords, self.game_state,
                dict(self.score_evaluation, historical=self._scores_historical()), self.mini_font, compact=True)

        # Draw the actual game frame
        emu_screen = np.transpose(frame_img, (1, 0, 2))
        frame_size = emu_screen.shape[:2]
        if self._frame_surface is None or self._frame_surface.get_size() != frame_size:
            self._frame_surface = pygame.Surface(frame_size)
        pygame.surfarray.blit_array(self._frame_surface, emu_screen)
        pygame.transform.scale(self._frame_surface, self.game_surf.get_size(), self.game_surf)
        if self.show_planner_overlay and diagnostics:
            draw_game_target(self.game_surf, self.game_surf.get_rect(), frame_size, frame_info)

        # Combine both surfaces on the screen
        self.screen.blit(self.debug_surf, self.mini_rink_rect)
        self.screen.blit(self.game_surf, self.game_rect)
        actual = None if action is None else np.asarray(action).reshape(-1).tolist()
        self.inspector.draw(
            self.screen, pygame.Rect(0, 0, self.ACTION_WIDTH, self.CANVAS_HEIGHT),
            self.font, self.big_font, self.playback_frames, actual_action=actual,
            human_override=self.human_control and self.args.mode != 'player_vs_model',
            mouse_position=self._logical_mouse_position())
        if self.paused:
            text = self.big_font.render("PAUSED", True, self.COLOR_YELLOW, self.COLOR_BLACK)
            self.screen.blit(text, (self.game_rect.x + 12, self.game_rect.y + 12))

        self._present()
        pygame.display.flip()

    def _logical_mouse_position(self):
        x, y = pygame.mouse.get_pos()
        rect = self.presentation_rect
        return ((x - rect.x) * self.CANVAS_WIDTH / rect.width,
                (y - rect.y) * self.CANVAS_HEIGHT / rect.height)

    def _present(self):
        width, height = self.window.get_size()
        scale = min(width / self.CANVAS_WIDTH, height / self.CANVAS_HEIGHT)
        size = max(1, round(self.CANVAS_WIDTH * scale)), max(1, round(self.CANVAS_HEIGHT * scale))
        self.presentation_rect = pygame.Rect((0, 0), size)
        self.presentation_rect.center = self.window.get_rect().center
        self.window.fill(self.COLOR_BLACK)
        if size == self.screen.get_size():
            image = self.screen
        else:
            if self._presentation_surface is None or self._presentation_surface.get_size() != size:
                self._presentation_surface = pygame.Surface(size)
            image = pygame.transform.smoothscale(self.screen, size, self._presentation_surface)
        self.window.blit(image, self.presentation_rect)
